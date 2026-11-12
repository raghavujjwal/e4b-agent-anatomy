"""How well can a plain BM25 over the repo's .py files, queried with the FULL task statement, rank the gold (non-test) file? An upper-ish bound for any localization assist
that is fed the whole statement. Stdlib only; blobs streamed with git cat-file --batch at each task's base commit."""
import os
import collections, json, math, subprocess, sys, time

NL = chr(10)
tasks = [json.loads(l) for l in open(os.environ.get("TASKS_JSONL", "data/tasks.jsonl"), encoding="utf-8")]
REPO = {"fastapi/fastapi": "fastapi", "psf/requests": "requests", "encode/httpx": "httpx", "Textualize/rich": "rich"}
dev = set()
SP = os.environ.get("WORK_DIR", "work") + "/"
for r in json.load(open(SP + "funnel_rows.json")):
    dev.add(r["iid"])


def words(s):
    out, cur = [], []
    for ch in s:
        if ch.isalnum() or ch == "_":
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def subtokens(w):
    toks = []
    for part in w.split("_"):
        if not part:
            continue
        cur = part[0]
        for a, b in zip(part, part[1:]):
            if (a.islower() or a.isdigit()) and b.isupper():
                toks.append(cur)
                cur = b
            else:
                cur += b
        toks.append(cur)
    return [t.lower() for t in toks if len(t) >= 3]


def tokens(s, with_full=True):
    out = []
    for w in words(s):
        if with_full and len(w) >= 5 and ("_" in w or any(a.islower() and b.isupper() for a, b in zip(w, w[1:]))):
            out.append(w.lower())
        out.extend(subtokens(w))
    return out


def py_blobs(repo_dir, commit):
    ls = subprocess.run(["git", "-C", repo_dir, "ls-tree", "-r", commit], capture_output=True).stdout.decode("utf-8", "replace").split(NL)
    items = []
    for l in ls:
        if not l.strip():
            continue
        meta, path = l.split("\t", 1)
        sha = meta.split()[2]
        if path.endswith(".py"):
            items.append((path, sha))
    p = subprocess.Popen(["git", "-C", repo_dir, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out = {}
    for path, sha in items:
        p.stdin.write((sha + NL).encode())
        p.stdin.flush()
        head = p.stdout.readline().decode().split()
        size = int(head[2])
        data = p.stdout.read(size)
        p.stdout.read(1)
        out[path] = data.decode("utf-8", "replace")
    p.stdin.close()
    p.wait()
    return out


def gold_files(patch):
    return [l[6:].strip() for l in patch.split(NL) if l.startswith("--- a/")]


def rank_gold(files, query, gold):
    docs = {}
    for path, text in files.items():
        toks = tokens(text) + 3 * tokens(path)
        docs[path] = collections.Counter(toks)
    N = len(docs)
    df = collections.Counter()
    for c in docs.values():
        for t in c:
            df[t] += 1
    avgdl = sum(sum(c.values()) for c in docs.values()) / max(N, 1)
    q = set(tokens(query))
    scores = {}
    for path, c in docs.items():
        dl = sum(c.values())
        s = 0.0
        for t in q:
            f = c.get(t, 0)
            if f:
                idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * dl / avgdl))
        scores[path] = s
    order = sorted(scores, key=lambda p: -scores[p])
    ranks = [order.index(g) + 1 for g in gold if g in order]
    return (min(ranks) if ranks else None), N


res = {"dev": [], "all": []}
t0 = time.time()
cache = {}
for t in tasks:
    gold = [g for g in gold_files(t["patch"]) if not g.startswith("tests/") and g.endswith(".py")]
    if not gold:
        continue
    key = (t["repo"], t["base_commit"])
    if key not in cache:
        cache[key] = py_blobs(os.environ.get("REPOS_DIR", "data/repos") + "/" + REPO[t["repo"]], t["base_commit"])
    files = {p: x for p, x in cache[key].items() if not p.startswith("tests/") and "/tests/" not in p and not p.startswith("test_")}
    q = t.get("problem_statement") or ""
    rk, n = rank_gold(files, q, gold)
    res["all"].append((t["instance_id"], rk, n))
    if t["instance_id"] in dev:
        res["dev"].append((t["instance_id"], rk, n))
    if len(cache) > 6:
        cache.pop(next(iter(cache)))
print("done in %.0f s" % (time.time() - t0))
for name in ("dev", "all"):
    xs = res[name]
    hit = lambda k: sum(1 for _, r, _ in xs if r is not None and r <= k)
    print("%-4s tasks with a gold .py file: %3d | top-1 %d (%.0f%%) | top-3 %d (%.0f%%) | top-5 %d (%.0f%%) | top-10 %d (%.0f%%) | median files %d" % (
        name, len(xs), hit(1), 100 * hit(1) / len(xs), hit(3), 100 * hit(3) / len(xs), hit(5), 100 * hit(5) / len(xs), hit(10), 100 * hit(10) / len(xs), sorted(n for _, _, n in xs)[len(xs) // 2]))
json.dump(res, open(SP + "retrieval_bound.json", "w"))
