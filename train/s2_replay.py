"""S2.3 Tier 2: replay a condensed trajectory on a real checkout through the harness tool code. Runs in WSL (stdlib only).

  python3 train/s2_replay.py --jobs /mnt/d/data/s2/pilot/jobs.jsonl --out /mnt/d/data/s2/pilot/replayed.jsonl \
      --cache /mnt/d/data/s2/tarballs [--limit 50] [--keep-going]

Each job is replayed in a private mount namespace (`unshare -Urm`, no root needed) where a fresh tmpfs directory is bind-mounted at /workspace, so
the real tool code sees the same paths the scorer's container does. Steps, per job:
  1. fetch the repository at base_commit from GitHub's tarball endpoint (cached), unpack it, write pytest.ini / conftest.py / git excludes the way
     the scorer's sandbox_setup.py does, commit the baseline;
  2. the first user message is the scorer's build_agent_prompt (workspace tree read from the checkout);
  3. discovery (a grep with line numbers), read_file windows, the teacher's edits through the real edit_file, the teacher's verification, submit_patch;
  4. a sample is kept only if every edit applied with strategy `exact` and the resulting diff equals the teacher's change to those files.
Observations that are NOT produced by running the tool on the checkout (listed per sample in `synthesized`): the discovery command (chosen here from a
term in the issue, because the teacher's greps print no line numbers) is run for real, so its output is real; the verification output is the teacher's
recorded output, since the repository's dependencies are not installed here.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "agent" / "vendor"))
import scorer_port as P  # noqa: E402
from tools import Budget, Tools  # noqa: E402

SETUP_PY = HERE.parent / "rules-raw" / "dataset" / "sandbox_setup.py"
WORKSPACE = "/workspace"
SIX_TOOLS = ["read_file", "edit_file", "write_file", "run_command", "get_status", "submit_patch"]   # the v2rec agent.yaml, in declared order
BUDGET = dict(tool_calls=24, time_minutes=5.5, turns=40, command_timeout=90)                      # agent/submission_e4b_v2rec/eval_config.yaml
WINDOW_BEFORE, WINDOW_LEN, CAP = 20, 100, 150
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
STOP = {"self", "none", "true", "false", "return", "import", "class", "from", "with", "this", "that", "when", "then", "should", "would", "could",
        "error", "value", "string", "function", "method", "issue", "code", "file", "files", "test", "tests", "python", "using", "expected",
        "actual", "bug", "description", "steps", "reproduce", "version", "output", "result", "results", "instead", "because", "there", "which",
        "where", "while", "have", "been", "does", "doesn", "also", "like", "just", "only", "into", "than", "them", "they", "their", "will"}


class LocalBox:
    """The Container interface of agent/vendor/sandbox.py over the bind-mounted /workspace of this process."""

    def sh(self, cmd, timeout=600):
        r = subprocess.run(["/bin/bash", "-c", cmd], cwd=WORKSPACE, capture_output=True, timeout=timeout)
        return r.returncode, r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace")

    def exec_scorer(self, cmd, timeout):
        wrapped = f"timeout -k 5s {timeout} /bin/bash -c {shlex.quote(cmd)}"
        t0 = time.time()
        r = subprocess.run(["/bin/bash", "-c", wrapped], cwd=WORKSPACE, capture_output=True, timeout=timeout + 30)
        if r.returncode == 124 and time.time() - t0 >= max(0.0, timeout - 1.0):
            raise subprocess.TimeoutExpired(cmd, timeout)
        return r.returncode, r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace")

    def get_bytes(self, path):
        try:
            return Path(path).read_bytes()
        except OSError:
            return None

    def put_bytes(self, path, data):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)

    def get_text(self, path):
        b = self.get_bytes(path)
        return None if b is None else b.decode("utf-8", "replace")

    def put_text(self, path, content):
        self.put_bytes(path, content.encode("utf-8"))


def fetch(repo: str, sha: str, cache: Path, timeout: int = 120) -> Path:
    """One attempt, no retries: GitHub builds the archive on the fly and serves it at a few hundred KB/s per connection, so a big repository can
    hold a shard for minutes (a measured 11% of the jobs took 65% of the time). A run uses a short timeout first and a second, quieter pass
    (`--retry-fetch`, long timeout) picks up whatever failed."""
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / f"{repo.replace('/', '__')}__{sha[:12]}.tar.gz"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(".part")
    # a repository that takes longer than this (or is larger than the cap, when the server says so) is not worth the disk and the time
    r = subprocess.run(["curl", "-sSL", "--fail", "--connect-timeout", "20", "-m", str(timeout),
                        "--max-filesize", "150000000", "-o", str(tmp), f"https://codeload.github.com/{repo}/tar.gz/{sha}"],
                       capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"fetch failed ({r.returncode}): {r.stderr.strip()[:200]}")
    tmp.rename(dest)
    return dest


def setup_checkout(tarball: Path):
    """What the scorer's sandbox does before the agent starts (HARNESS_README 4.2, sandbox_setup.py): unpack, git excludes, pytest.ini,
    conftest.py, then the baseline commit."""
    for item in Path(WORKSPACE).iterdir():
        shutil.rmtree(item, ignore_errors=True) if item.is_dir() else item.unlink()
    if tarball.stat().st_size > 150_000_000:
        raise RuntimeError("repo_too_large")
    subprocess.run(["tar", "-xzf", str(tarball), "-C", WORKSPACE, "--strip-components=1", "--no-same-owner"], check=True, capture_output=True)
    spec = importlib.util.spec_from_file_location("sandbox_setup", SETUP_PY)
    ss = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ss)
    ws = Path(WORKSPACE)
    subprocess.run(["git", "init", "-q"], cwd=WORKSPACE, check=True, capture_output=True)
    ss.setup_git_exclude(ws)
    ss.configure_workspace_pytest_ini(ws)
    ss.configure_workspace_conftest(ws)
    env = dict(os.environ, GIT_AUTHOR_NAME="s2", GIT_AUTHOR_EMAIL="s2@example.com", GIT_COMMITTER_NAME="s2", GIT_COMMITTER_EMAIL="s2@example.com")
    for cmd in (["git", "add", "-A"], ["git", "commit", "-q", "-m", "baseline", "--allow-empty"], ["git", "tag", "_swegemma_baseline"]):
        subprocess.run(cmd, cwd=WORKSPACE, check=True, capture_output=True, env=env)


def find_span(text: str, old: str):
    """(first_line, last_line), 1-based, of the single occurrence of `old` in `text`; None if absent or not unique."""
    if not old or text.count(old) != 1:
        return None
    i = text.index(old)
    first = text.count("\n", 0, i) + 1
    return first, first + old.count("\n")


def term_candidates(job: dict, edit_text: dict[str, str], first_edit: dict) -> list[tuple[str, int]]:
    """Search terms from the issue that occur in the edited file close to the first edit, best first: [(term, nearest_hit_line_in_file)].
    Order: written in backticks in the issue, then identifier-like (underscore or inner capital), then fewest hits in the file."""
    path = first_edit["path"]
    text = edit_text[path]
    span = find_span(text, first_edit["old"])
    if span is None:
        return []
    lines = text.split("\n")
    lo, hi = max(0, span[0] - 1 - 30), min(len(lines), span[1] + 30)
    issue = job["problem_statement"]
    near = "\n".join(lines[lo:hi])
    cands = []
    for t in dict.fromkeys(job["disc"]["terms"] + TOKEN.findall(issue)):
        if t.lower() in STOP or len(t) < 4 or t not in issue or t not in near:
            continue
        hits = [n + 1 for n, ln in enumerate(lines) if t in ln]
        if not hits:
            continue
        nearest = min(hits, key=lambda n: abs(n - span[0]))
        backticked = f"`{t}" in issue or f"{t}`" in issue
        identifier = "_" in t or bool(re.search(r"[a-z][A-Z]", t))
        cands.append((0 if backticked else 1, 0 if identifier else 1, len(hits), abs(nearest - span[0]), t, nearest))
    cands.sort()
    return [(c[4], c[5]) for c in cands]


def read_args(text: str, spans: list[tuple[int, int]], hit_line: int | None):
    """Arguments of read_file for one file: none for a short file, else a window from about 20 lines above the search hit (or the first edit)."""
    n = text.count("\n") + 1
    if n <= CAP and len(text) <= 9000:
        return {}, None
    first, last = min(s[0] for s in spans), max(s[1] for s in spans)
    anchor = min(hit_line or first, first)
    start = max(1, anchor - WINDOW_BEFORE)
    end = max(start + WINDOW_LEN - 1, last + 3)
    if end - start + 1 > CAP:
        return None, "window_too_big"
    end = min(end, n)
    return {"start_line": start, "end_line": end}, None


def replay(job: dict, tar: Path):
    c = LocalBox()
    setup_checkout(tar)
    budget = Budget(**BUDGET)
    tools = Tools(c, budget, baseline_ref="_swegemma_baseline")
    tools.available = list(SIX_TOOLS)
    texts = {}
    for e in job["edits"]:
        if e["path"] not in texts:
            b = c.get_bytes(f"{WORKSPACE}/{e['path']}")
            if b is None:
                return None, "file_missing_at_base_commit"
            texts[e["path"]] = b.decode("utf-8", "replace")
    first = job["edits"][0]
    if find_span(texts[first["path"]], first["old"]) is None:
        return None, "old_string_not_unique_or_absent"
    rc, tree_out, _ = c.sh(P.build_workspace_tree_command())
    user = P.build_agent_prompt(job["repo"], job["problem_statement"], "", time_minutes=budget.time_minutes, tool_calls=budget.tool_calls,
                                turns=budget.turns, command_timeout_seconds=budget.command_timeout, max_stdout_chars=tools.max_stdout,
                                max_file_lines=tools.max_file_lines, max_file_chars=tools.max_file_chars,
                                workspace_tree=tree_out.strip() if rc == 0 else "", enable_sandbox_testing=True,
                                declared_tools=SIX_TOOLS, graph_available=False)
    msgs = [{"role": "user", "content": user}]
    synthesized, n = [], 0

    def do(name, args):
        nonlocal n
        cid = f"chatcmpl-tool-{n}"
        n += 1
        msgs.append({"role": "assistant", "tool_calls": [{"type": "function", "id": cid,
                                                           "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]})
        res = tools.call(name, args)
        msgs.append({"role": "tool", "tool_call_id": cid, "content": res})
        return res

    # groups of edits that one read covers
    groups: dict[int, list[dict]] = {}
    for e in job["edits"]:
        groups.setdefault(e["group"], []).append(e)
    small_and_named = all(texts[g[0]["path"]].count("\n") < CAP and len(texts[g[0]["path"]]) <= 9000 for g in groups.values()) \
        and job["disc"]["kind"] == "issue"
    hit = None
    if not small_and_named:
        recursive = job["disc"]["kind"] != "issue"
        top = first["path"].split("/")[0] if "/" in first["path"] else "."
        pattern = (r"(?:^|\./)" + re.escape(first["path"]) + ":") if recursive else r"^\d+:"
        chosen = None
        for term, hit_line in term_candidates(job, texts, first)[:8]:
            cmd = f'grep -rn "{term}" --include="*.py" {top}' if recursive else f'grep -n "{term}" {first["path"]}'
            rc, so, _ = c.sh(cmd)
            n_lines = so.count("\n")
            # a targeted search: the edited file is among the hits and the output is short enough to read
            if rc == 0 and re.search(pattern, so, re.M) and n_lines <= (25 if recursive else 12):
                chosen = (cmd, hit_line)
                break
        if chosen is None:
            return None, "no_targeted_search_term"
        do("run_command", {"command": chosen[0]})
        synthesized.append("discovery_command")
        hit = chosen[1]
    applied = []
    for gid in sorted(groups):
        es = groups[gid]
        path = es[0]["path"]
        spans = [find_span(texts[path], e["old"]) for e in es]
        if any(s is None for s in spans):
            return None, "old_string_not_unique_or_absent"
        args, why = read_args(texts[path], spans, hit if path == first["path"] else None)
        if args is None:
            return None, why
        res = do("read_file", dict({"filepath": path}, **args))
        d = json.loads(res)
        if d.get("status") not in ("ok", "success") and "content" not in d:
            return None, "read_failed"
        shown = d.get("content") or d.get("text") or ""
        for e in es:
            if e["old"] not in shown:
                return None, "old_string_not_in_read_window"
        for e in es:
            res = do("edit_file", {"filepath": e["path"], "old_string": e["old"], "new_string": e["new"]})
            d = json.loads(res)
            if d.get("status") != "ok" or d.get("strategy") != "exact":
                return None, f"edit_not_exact:{d.get('strategy') or d.get('error_type')}"
            applied.append(d)
    v = job["verif"]
    cid = f"chatcmpl-tool-{n}"
    n += 1
    msgs.append({"role": "assistant", "tool_calls": [{"type": "function", "id": cid,
                                                       "function": {"name": "run_command", "arguments": json.dumps({"command": v["command"]}, ensure_ascii=False)}}]})
    msgs.append({"role": "tool", "tool_call_id": cid, "content": P.ok_response(stdout=v["output"][: tools.max_stdout], stderr="", exit_code=0)})
    budget.used += 1
    synthesized.append("verification_output")
    res = do("submit_patch", {})
    d = json.loads(res)
    if d.get("status") != "ok":
        return None, "submit_failed"
    # the diff must be the teacher's change to the source files
    ok, why = diff_matches(job, tools.submitted_patch)
    if not ok:
        return None, why
    files = ", ".join(sorted({e["path"] for e in job["edits"]}))
    msgs.append({"role": "assistant", "content": f"I changed {files} and submitted the patch."})
    return {"messages": msgs, "synthesized": synthesized, "tool_calls": budget.used, "edit_strategies": [a.get("strategy") for a in applied],
            "tree_lines": tree_out.count("\n") + 1}, ""


def diff_matches(job: dict, ours: str):
    """Every hunk we made must appear in the teacher's patch for the same file (the teacher's patch may hold more: tests, other hunks)."""
    def hunks(p):
        out, cur = {}, None
        for ln in p.split("\n"):
            m = re.match(r"diff --git a/(\S+) b/\S+", ln)
            if m:
                cur = m.group(1)
                out.setdefault(cur, [])
            elif cur and (ln.startswith("+") or ln.startswith("-")) and not ln.startswith(("+++", "---")):
                out[cur].append(ln)
        return out
    mine, theirs = hunks(ours), hunks(job["model_patch"])
    srcs = {e["path"] for e in job["edits"]}
    if set(mine) != srcs:
        return False, f"diff_files_differ:{sorted(mine)}"
    for f in srcs:
        # order-insensitive multiset comparison of changed lines
        if Counter(mine[f]) != Counter(theirs.get(f, [])):
            return False, "diff_differs_from_teacher"
    return True, ""


def child(args) -> int:
    """Runs inside the namespace: /workspace is already the bind mount."""
    job = json.loads(Path(args.job_file).read_text(encoding="utf-8"))
    tar = Path(args.tarball)
    out = {"instance_id": job["instance_id"], "repo": job["repo"], "license": job["license"], "teacher": job["teacher"]}
    try:
        res, why = replay(job, tar)
    except Exception as e:  # noqa: BLE001
        res, why = None, f"exception:{type(e).__name__}:{str(e)[:160]}"
    if res is None:
        out["dropped"] = why
    else:
        out.update(res)
    Path(args.result_file).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return 0


# A drop that may be an accident (disk, a killed child) is not an answer, so a resumed run tries it again; a kept record and a drop the rules gave
# are. A failed download is settled for the first pass (it was a slow or large repository) and retried only with --retry-fetch. A tarball over the
# size cap fails the same way every time (curl exit 63), so it is settled in every pass.
TRANSIENT = ("child_failed", "exception:")


def read_records(path: Path):
    """Records of a shard file; the last line of a run that was killed mid-write may be cut off and is skipped."""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            yield json.loads(line)
        except ValueError:
            continue


def settled(out_dir: Path, retry_fetch: bool = False) -> set:
    transient = TRANSIENT + (("fetch:",) if retry_fetch else ())
    done = set()
    for p in out_dir.glob("replayed.shard*.jsonl"):
        for r in read_records(p):
            why = r.get("dropped") or ""
            if "messages" in r or not why.startswith(transient) or "repo_too_large" in why or "(63)" in why:
                done.add(r["instance_id"])
    return done


def merge(out_dir: Path) -> int:
    """replayed.jsonl from the shard files: one record per instance; a kept record is never replaced by a drop, otherwise the later one wins."""
    best = {}
    for p in sorted(out_dir.glob("replayed.shard*.jsonl")):
        for r in read_records(p):
            old = best.get(r["instance_id"])
            if old is None or "messages" in r or "messages" not in old:
                best[r["instance_id"]] = r
    with open(out_dir / "replayed.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for r in best.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"merged {len(best)} records into {out_dir / 'replayed.jsonl'}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs")
    ap.add_argument("--out")
    ap.add_argument("--merge", help="directory of replayed.shard*.jsonl: write replayed.jsonl and stop")
    ap.add_argument("--resume", action="store_true", help="append to --out and skip the jobs that any replayed.shard*.jsonl beside it has settled")
    ap.add_argument("--retry-fetch", action="store_true", help="with --resume: also try again the jobs whose download failed (use a long --fetch-timeout)")
    ap.add_argument("--fetch-timeout", type=int, default=120, help="seconds a repository download may take")
    ap.add_argument("--cache")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n: replay only jobs whose index is i modulo n (run n workers with distinct --out files)")
    ap.add_argument("--tmp", default="/dev/shm/s2")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--job-file")
    ap.add_argument("--tarball")
    ap.add_argument("--result-file")
    a = ap.parse_args(argv)
    if a.child:
        return child(a)
    if a.merge:
        return merge(Path(a.merge))
    jobs = [json.loads(l) for l in open(a.jobs, encoding="utf-8") if l.strip()]
    if a.limit:
        jobs = jobs[: a.limit]
    si, sn = (int(x) for x in a.shard.split("/"))
    jobs = [j for k, j in enumerate(jobs) if k % sn == si]       # by position in the whole list, before anything is skipped, so shards never overlap
    out = Path(a.out)
    if a.resume:
        done = settled(out.parent, a.retry_fetch)
        print(f"resume: {len(done)} instances settled, {sum(j['instance_id'] in done for j in jobs)} of this shard's {len(jobs)} skipped", flush=True)
        jobs = [j for j in jobs if j["instance_id"] not in done]
        if out.exists() and out.stat().st_size and not out.read_bytes().endswith(b"\n"):
            with open(out, "ab") as fh0:
                fh0.write(b"\n")
    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    with open(out, "a" if a.resume else "w", encoding="utf-8", newline="\n") as fh:
        for k, job in enumerate(jobs):
            t0 = time.time()
            try:
                tar = fetch(job["repo"], job["base_commit"], Path(a.cache), a.fetch_timeout)
            except Exception as e:  # noqa: BLE001
                rec = {"instance_id": job["instance_id"], "dropped": f"fetch:{str(e)[:120]}"}
                stats[rec["dropped"].split(":")[0]] += 1
                fh.write(json.dumps(rec) + "\n")
                print(k, job["instance_id"], rec["dropped"], flush=True)
                continue
            work = tmp / f"w{os.getpid()}"
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            jf, rf = tmp / f"job{os.getpid()}.json", tmp / f"res{os.getpid()}.json"
            jf.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
            rf.unlink(missing_ok=True)
            inner = (f"mount --bind {work} {WORKSPACE} && cd / && python3 {Path(__file__).resolve()} --child "
                     f"--job-file {jf} --tarball {tar} --result-file {rf}")
            try:
                subprocess.run(["unshare", "-Urm", "bash", "-c", inner], check=False, capture_output=True, timeout=900)
            except subprocess.TimeoutExpired:
                pass
            rec = json.loads(rf.read_text(encoding="utf-8")) if rf.exists() else {"instance_id": job["instance_id"], "dropped": "child_failed"}
            shutil.rmtree(work, ignore_errors=True)
            stats["kept" if "messages" in rec else rec.get("dropped", "?")[:40]] += 1
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(k, job["instance_id"], "kept" if "messages" in rec else rec.get("dropped"), f"{time.time() - t0:.1f}s", flush=True)
    print(json.dumps(dict(stats), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
