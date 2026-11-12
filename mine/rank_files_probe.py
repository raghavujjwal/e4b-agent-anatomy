"""Could a no-model file ranker put the gold file in front of the agent?  (CPU only; the offline test of a "find relevant files" tool)

  python mine/rank_files_probe.py --tasks D:/data/kaggle_comp/tasks.jsonl --repos D:/data/repos --runs-funnel D:/data/analysis/funnel_all6.json --out D:/data/analysis/rank_files.json

For each public task the candidates are the non-test .py files of the repo at base_commit (git objects of a clone, nothing is checked out). The query is the issue text. Three rankers,
fixed before the first run (no tuning on the results):
  content   BM25 over identifier tokens of the file text (snake/camel case split, the whole identifier kept)
  path      BM25 over the identifier tokens of the file path only
  symbols   BM25 over the path plus the names of the classes/functions/constants the file defines
Measured: recall@1/3/5/10 of the non-test gold files. Single-file tasks (one non-test gold file) are the ones that count; multi-file tasks are reported as "any gold file in top k".
Compared with what the agent did on its own (docs/MEASUREMENTS.md, 396 runs): it saw the gold file in 56.7% of single-file runs, 21% on issues under 300 characters, 74% on longer ones.
Pre-registered rule: a "find relevant files" tool is worth building if the best ranker reaches recall@5 >= 75% on single-file tasks AND >= 50% on single-file tasks with issues under 300 characters.
If it reaches recall@5 below 60% overall the idea is dropped. In between: report, decide by hand.
"""
from __future__ import annotations
import argparse
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from localization_funnel import gold_files, TEST_RE  # noqa: E402

STOP = set("the and for are with that this from have has not but you your when what which they them then than will would should could can into out all any use used using also one two get set was were been being its it's there their these those how why where does did done doing just like more most some such only other about over under after before while because need needs want see make made".split())
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
DEF = re.compile(r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_]\w*)|^([A-Za-z_]\w*)\s*[:=]", re.M)


def tokens(text):
    out = []
    for ident in IDENT.findall(text):
        low = ident.lower()
        parts = [p.lower() for p in CAMEL.findall(ident.replace("_", " "))]
        for t in ([low] if len(parts) > 1 else []) + parts:
            if len(t) >= 3 and t not in STOP:
                out.append(t)
    return out


class Bm25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(docs)) if docs else 1.0
        df = Counter(t for tf in self.tf for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query):
        q = Counter(query)
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            for t, qc in q.items():
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg)) * (1 + math.log(qc))
            out.append(s)
        return out


def ranked(paths, scores):
    return [p for _s, p in sorted(zip(scores, paths), key=lambda x: (-x[0], x[1]))]


def repo_files(repo_dir, commit):
    """{path: text} of the non-test .py files at commit, read through one git cat-file --batch call."""
    ls = subprocess.run(["git", "-C", str(repo_dir), "ls-tree", "-r", commit], capture_output=True, text=True, check=True).stdout.splitlines()
    want = []
    for line in ls:
        meta, path = line.split("\t", 1)
        if path.endswith(".py") and not TEST_RE.search(path) and meta.split()[1] == "blob":
            want.append((meta.split()[2], path))
    proc = subprocess.Popen(["git", "-C", str(repo_dir), "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out = {}
    for sha, path in want:
        proc.stdin.write((sha + "\n").encode())
        proc.stdin.flush()
        hdr = proc.stdout.readline().split()
        size = int(hdr[2])
        out[path] = proc.stdout.read(size).decode("utf-8", "replace")
        proc.stdout.read(1)
    proc.stdin.close()
    proc.wait()
    return out


def rankers(files):
    paths = sorted(files)
    docs = {
        "content": [tokens(files[p]) for p in paths],
        "path": [tokens(p) for p in paths],
        "symbols": [tokens(p) + tokens(" ".join(a or b for a, b in DEF.findall(files[p]))) for p in paths],
    }
    return paths, {k: Bm25(v) for k, v in docs.items()}


def analyse(tasks, repos, funnel_tasks):
    rows = []
    cache = {}
    for t in tasks:
        gold = gold_files(t["patch"])
        repo_dir = Path(repos) / t["repo"].split("/")[1]
        if not repo_dir.exists() or not gold:
            continue
        files = repo_files(repo_dir, t["base_commit"])
        gold_in = {g for g in gold if g in files}
        if not gold_in:
            continue
        paths, bm = rankers(files)
        q = tokens(t["problem_statement"])
        row = {"task": t["instance_id"], "n_gold": len(gold), "n_cand": len(paths), "short": len(t["problem_statement"]) < 300, "dev": t["instance_id"] in funnel_tasks, "ranks": {}}
        for name, m in bm.items():
            order = ranked(paths, m.scores(q))
            row["ranks"][name] = sorted(order.index(g) + 1 for g in gold_in)
        rows.append(row)
    return rows


def recall(rows, name, k):
    single = [r for r in rows if r["n_gold"] == 1]
    return sum(1 for r in single if r["ranks"][name] and r["ranks"][name][0] <= k) / max(len(single), 1)


def report(rows):
    out = {"tasks": len(rows), "single_file_tasks": sum(r["n_gold"] == 1 for r in rows)}
    for name in ("content", "path", "symbols"):
        res = {}
        for label, sel in (("all", rows), ("dev66", [r for r in rows if r["dev"]]), ("short<300", [r for r in rows if r["short"]]), ("long", [r for r in rows if not r["short"]])):
            single = [r for r in sel if r["n_gold"] == 1]
            res[label] = {"n": len(single), **{f"recall@{k}": round(100 * sum(1 for r in single if r["ranks"][name][0] <= k) / max(len(single), 1), 1) for k in (1, 3, 5, 10)}}
        multi = [r for r in rows if r["n_gold"] > 1]
        res["multi_any_gold@5"] = {"n": len(multi), "%": round(100 * sum(1 for r in multi if r["ranks"][name] and r["ranks"][name][0] <= 5) / max(len(multi), 1), 1)}
        out[name] = res
    best = max(("content", "path", "symbols"), key=lambda n: recall(rows, n, 5))
    short = [r for r in rows if r["short"] and r["n_gold"] == 1]
    r5, s5 = recall(rows, best, 5), (sum(1 for r in short if r["ranks"][best][0] <= 5) / max(len(short), 1))
    out["best_ranker"] = best
    out["verdict"] = ("BUILD (rule met)" if r5 >= 0.75 and s5 >= 0.5 else "DROP" if r5 < 0.60 else "BORDERLINE: decide by hand") + f" [recall@5 {100 * r5:.1f}% single-file, {100 * s5:.1f}% on short issues]"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--repos", required=True, help="directory with one clone per repo name (fastapi, rich, requests, httpx)")
    ap.add_argument("--runs-funnel", required=True, help="funnel json from localization_funnel.py (names the dev66 tasks)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    tasks = [json.loads(l) for l in open(a.tasks, encoding="utf-8")]
    dev = {r["task"] for r in json.load(open(a.runs_funnel, encoding="utf-8"))["rows"]}
    rows = analyse(tasks, a.repos, dev)
    rep = report(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"report": rep, "rows": rows}, open(a.out, "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    sys.exit(main())
