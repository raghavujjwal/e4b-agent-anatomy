"""Ceiling of a programmatic recovery for the refused edit_file calls of the corrected control runs, on FULL files.
Replay: base-commit file (git show) + the episode's earlier applied edits through the scorer-port matcher; the predicted outcome of every edit is compared with the recorded one
(agreement = the replay is faithful). For the refused edits whose replay agrees:
  ambiguity  -> the model's anchor occurs N times; which occurrences did the model SEE (inside a read window still in the prompt)? For those: the shortest extension (lines above/below)
                that makes the anchor unique in the whole file, and whether every extra line was inside the window the model had seen.
  not found  -> the closest block of the lines the model saw (similarity), and whether copying that block verbatim would apply (unique in the file).
Also: does the located place overlap the gold patch hunk (same file, within 10 lines, base coordinates mapped through the replay)?
No backslash is typed in this file."""
import os
import collections, difflib, json, subprocess, sys, time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "agent" / "vendor"))
import scorer_port as P

NL, CR = chr(10), chr(13)
SP = os.environ.get("WORK_DIR", "work") + "/"
eps = json.load(open(SP + "episodes.json", encoding="utf-8"))
edits_all = json.load(open(SP + "edits_all.json", encoding="utf-8"))
corp_by = {(r["run"], r["iid"], r["chain"], r["turn"]): r.get("corp", "") for r in edits_all if "corp" in r}
tasks = {}
for line in open(os.environ.get("TASKS_JSONL", "data/tasks.jsonl"), encoding="utf-8"):
    t = json.loads(line)
    tasks[t["instance_id"]] = t
REPO = {"fastapi/fastapi": "fastapi", "psf/requests": "requests", "encode/httpx": "httpx", "Textualize/rich": "rich"}
_cache = {}


def clean(p):
    p = p or ""
    for pre in ("/workspace/", "./", "/"):
        if p.startswith(pre):
            p = p[len(pre):]
    return p


def base_file(repo, commit, path):
    key = (repo, commit, path)
    if key not in _cache:
        p = subprocess.run(["git", "-C", os.environ.get("REPOS_DIR", "data/repos") + "/" + REPO[repo], "show", commit + ":" + path], capture_output=True)
        _cache[key] = p.stdout.decode("utf-8", errors="replace") if p.returncode == 0 else None
    return _cache[key]


def gold_hunks(patch):
    out, cur = {}, None
    for l in patch.split(NL):
        if l.startswith("--- a/"):
            cur = l[6:].strip()
            out.setdefault(cur, [])
        elif l.startswith("--- /dev/null"):
            cur = None
        elif l.startswith("@@") and cur is not None:
            a, _, b = l.split()[1][1:].partition(",")
            out[cur].append((int(a), int(b) if b else 1))
    return out


def kind_of(resp):
    if resp.get("status") == "ok":
        return "ok"
    m = resp.get("error_message", "")
    if "occurs" in m and "times" in m:
        return "multiple"
    if "flexibly matches" in m:
        return "multiple_flex"
    if "not found" in m:
        return "not_found"
    if "identical" in m:
        return "unchanged"
    return "other"


def predict(content, old, new, am):
    if not old:
        return "other", None
    res = P.apply_replacement_core(content, old, new, am)
    if not res.error_message:
        return "ok", res.new_content
    if res.occurrences > 1:
        return ("multiple" if res.strategy == "exact" else "multiple_flex"), None
    if res.error_message.startswith("No changes"):
        return "unchanged", None
    return "not_found", None


def find_all(content, s):
    out, i = [], 0
    while s:
        j = content.find(s, i)
        if j < 0:
            break
        out.append(j)
        i = j + len(s)
    return out


def line_starts(content):
    st, pos = [0], 0
    for l in content.split(NL)[:-1]:
        pos += len(l) + 1
        st.append(pos)
    return st


def line_of(starts, pos):
    lo, hi = 0, len(starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= pos:
            lo = mid
        else:
            hi = mid - 1
    return lo


def visible_spans(content, starts, windows, path, corp, chain, turn):
    """(first_line, last_line) of every read window of this file that is still in the prompt and can be located in the current text."""
    spans = []
    for w in windows:
        if w["path"] != path or not w["text"]:
            continue
        in_prompt = (w["text"][:400] in corp) if corp else (w["chain"] == chain and w["turn"] < turn)
        if not in_prompt:
            continue
        exp = starts[min(max(w["start"] - 1, 0), len(starts) - 1)]
        cands = find_all(content, w["text"])
        if not cands:
            continue
        p = min(cands, key=lambda x: abs(x - exp))
        spans.append((line_of(starts, p), line_of(starts, p + max(len(w["text"]) - 1, 0))))
    return spans


def inside(spans, lo, hi):
    return any(a <= lo and hi <= b for a, b in spans)


def shortest_extension(lines, content, s, e, spans, need_visible, cap=40):
    for t in range(0, cap + 1):
        for u in range(0, t + 1):
            d = t - u
            lo, hi = s - u, e + d
            if lo < 0 or hi >= len(lines):
                continue
            if need_visible and not inside(spans, lo, hi):
                continue
            if content.count(NL.join(lines[lo:hi + 1])) == 1:
                return t, u, d
    return None


def map_to_base(base, cur):
    """current line index -> base line index (None for changed lines)"""
    sm = difflib.SequenceMatcher(None, base.split(NL), cur.split(NL), autojunk=False)
    m = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                m[j1 + k] = i1 + k
    return m


def near_gold(hunks, path, base, cur, cur_line_lo, cur_line_hi, margin=10):
    hs = hunks.get(path)
    if not hs or base is None:
        return None
    m = map_to_base(base, cur)
    bl = [m[i] for i in range(cur_line_lo, cur_line_hi + 1) if i in m]
    if not bl:
        return False
    lo, hi = min(bl) + 1, max(bl) + 1
    return any(lo <= a + n - 1 + margin and a - margin <= hi for a, n in hs)


stats = collections.Counter()
amb_rows, nf_rows = [], []
t0 = time.time()
for e in eps:
    t = tasks[e["iid"]]
    repo, commit = t["repo"], t["base_commit"]
    hunks = gold_hunks(t["patch"])
    state, windows = {}, []
    for c in e["calls"]:
        a, r = c.get("args") or {}, c.get("resp") or {}
        if c["tool"] == "read_file" and r.get("status") == "ok":
            windows.append(dict(chain=c["chain"], turn=c["turn"], path=clean(a.get("filepath")), start=r.get("start_line") or 1, end=r.get("end_line") or 1, text=r.get("content", "")))
            continue
        if c["tool"] != "edit_file" or not a or "old_string" not in a or not r:
            continue
        path = clean(a.get("filepath"))
        if path not in state:
            state[path] = base_file(repo, commit, path)
        content = state[path]
        stats["edit calls with a response"] += 1
        if content is None:
            stats["file unavailable / unreliable"] += 1
            continue
        if CR in content:
            stats["CRLF file skipped"] += 1
            continue
        old, new = a.get("old_string", ""), a.get("new_string", "")
        am = P.coerce_allow_multiple(a.get("allow_multiple", False))
        rec = kind_of(r)
        pred, new_content = predict(content, old, new, am)
        agree = rec == pred
        stats["replay agrees"] += agree
        stats["replay disagrees"] += (not agree)
        if rec == "ok":
            state[path] = new_content if pred == "ok" else None
            continue
        if not agree or rec not in ("multiple", "not_found"):
            continue
        corp = corp_by.get((e["run"], e["iid"], c["chain"], c["turn"]), "")
        lines = content.split(NL)
        starts = line_starts(content)
        spans = visible_spans(content, starts, windows, path, corp, c["chain"], c["turn"])
        base = base_file(repo, commit, path)
        if rec == "multiple":
            pos = find_all(content, old)
            vis = [p for p in pos if inside(spans, line_of(starts, p), line_of(starts, p + max(len(old) - 1, 0)))]
            row = dict(iid=e["iid"], path=path, n_occ=len(pos), n_vis=len(vis), anchor_lines=len(old.split(NL)))
            if len(vis) >= 1:
                res = []
                for p in vis:
                    s, en = line_of(starts, p), line_of(starts, p + max(len(old) - 1, 0))
                    ext_vis = shortest_extension(lines, content, s, en, spans, True)
                    ext_any = shortest_extension(lines, content, s, en, spans, False)
                    res.append((ext_vis, ext_any, near_gold(hunks, path, base, content, s, en)))
                row["res"] = res
            amb_rows.append(row)
        else:
            ol = [l for l in old.split(NL)]
            best, bi = 0.0, None
            for (a1, b1) in spans:
                for i in range(a1, max(a1, b1 - len(ol) + 1) + 1):
                    blk = NL.join(lines[i:i + len(ol)])
                    sm = difflib.SequenceMatcher(None, blk, old, autojunk=False)
                    if sm.real_quick_ratio() < best or sm.quick_ratio() < best:
                        continue
                    rt = sm.ratio()
                    if rt > best:
                        best, bi = rt, i
            row = dict(iid=e["iid"], path=path, best=best, n_vis_spans=len(spans), anchor_lines=len(ol))
            if bi is not None and best >= 0.85:
                blk = NL.join(lines[bi:bi + len(ol)])
                row["block_unique"] = content.count(blk) == 1
                row["applies"] = predict(content, blk, new, am)[0] == "ok"
                row["near_gold"] = near_gold(hunks, path, base, content, bi, bi + len(ol) - 1)
            nf_rows.append(row)
        if (stats["refused analysed"] % 25) == 0:
            print("  ...%d refused edits analysed, %.0f s" % (stats["refused analysed"], time.time() - t0), flush=True)
        stats["refused analysed"] += 1

print()
print("REPLAY:", dict(stats))
print()
print("AMBIGUITY failures with an agreeing replay: %d" % len(amb_rows))
cnt = collections.Counter()
ext = collections.Counter()
for r in amb_rows:
    if r["n_vis"] == 0:
        cnt["the model saw NO occurrence in a window still in the prompt"] += 1
    elif r["n_vis"] == 1:
        cnt["exactly ONE occurrence was visible (intent is clear)"] += 1
    else:
        cnt["several occurrences were visible (intent unclear)"] += 1
for k, v in cnt.most_common():
    print("  %3d  %s" % (v, k))
print("  occurrences in the file: median %d; visible: %s" % (sorted(r["n_occ"] for r in amb_rows)[len(amb_rows) // 2], dict(collections.Counter(r["n_vis"] for r in amb_rows))))


def bucket(t):
    if t is None:
        return "none within 40 lines"
    return "<=2 lines" if t[0] <= 2 else ("<=5" if t[0] <= 5 else ("<=10" if t[0] <= 10 else ("<=20" if t[0] <= 20 else "<=40")))


clear = [r for r in amb_rows if r["n_vis"] == 1]
print()
print("clear-intent ambiguity failures (n=%d): shortest extension that makes the anchor unique" % len(clear))
print("   needing every added line to be inside the window the model saw:", dict(collections.Counter(bucket(r["res"][0][0]) for r in clear)))
print("   ignoring visibility (any line of the file):                    ", dict(collections.Counter(bucket(r["res"][0][1]) for r in clear)))
ng = [r["res"][0][2] for r in clear]
print("   aimed near the gold hunk (same file, within 10 lines):", dict(collections.Counter(str(x) for x in ng)))
multi = [r for r in amb_rows if r["n_vis"] >= 2]
print("several-visible (n=%d): every visible occurrence has a visible unique extension <=10 lines:" % len(multi),
      sum(1 for r in multi if all(x[0] is not None and x[0][0] <= 10 for x in r["res"])))

print()
print("NOT FOUND failures with an agreeing replay: %d" % len(nf_rows))
sims = [r["best"] for r in nf_rows]
print("  no visible window located: %d" % sum(1 for r in nf_rows if r["n_vis_spans"] == 0))
print("  best similarity to a block of the lines the model saw: >=0.95: %d | 0.85-0.95: %d | 0.6-0.85: %d | <0.6: %d" % (
    sum(s >= .95 for s in sims), sum(.85 <= s < .95 for s in sims), sum(.6 <= s < .85 for s in sims), sum(s < .6 for s in sims)))
near = [r for r in nf_rows if "applies" in r]
print("  near-misses (>=0.85): %d | copying the closest block verbatim applies: %d | block not unique in the file: %d" % (
    len(near), sum(r["applies"] for r in near), sum(not r["block_unique"] for r in near)))
print("  near-misses aimed near the gold hunk:", dict(collections.Counter(str(r.get("near_gold")) for r in near)))
json.dump(dict(amb=amb_rows, nf=nf_rows, stats=dict(stats)), open(SP + "recovery_ceiling.json", "w"), default=str)
