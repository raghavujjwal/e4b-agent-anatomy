"""Rank the Python files of the workspace by how well they match a piece of text (the issue). Standard library only.

  --query TEXT   the issue text (title and the most specific sentences)
  --max N        how many files to list (default 10)

BM25 over identifier tokens (snake and camel case split, the whole identifier kept), in two views: the path plus the names the file defines (class, def, module-level
names), and the full file text. The top 5 of each view are listed, the first view first. Offline, on 90 single-file tasks, the gold file was among them 72% of the time
(mine/rank_files_probe.py in the repository that holds this skill).
"""
import argparse
import math
import os
import re
import sys
from collections import Counter

STOP = set("the and for are with that this from have has not but you your when what which they them then than will would should could can into out all any use used using also one two get set was were been being its it's there their these those how why where does did done doing just like more most some such only other about over under after before while because need needs want see make made".split())
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
DEF = re.compile(r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_]\w*)|^([A-Za-z_]\w*)\s*[:=]", re.M)
TEST_RE = re.compile(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*$|_test\.py$|(^|/)conftest\.py$")
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".tox", "build", "dist", "site-packages", ".eggs"}
MAX_BYTES = 400_000
PER_VIEW = 5


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


def python_files(root):
    files = {}
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d not in SKIP_DIRS and not d.endswith(".egg-info"))
        for fn in sorted(fns):
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if not fn.endswith(".py") or TEST_RE.search(rel):
                continue
            try:
                if os.path.getsize(full) > MAX_BYTES:
                    continue
                files[rel] = open(full, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
    return files


def candidates(files, query, limit=10):
    """[(path, matching names)] for the top PER_VIEW of the symbol view, then the top PER_VIEW of the content view, without repeats."""
    paths = sorted(files)
    q = tokens(query)
    if not paths or not q:
        return []
    defs = {p: [a or b for a, b in DEF.findall(files[p])] for p in paths}
    sym = Bm25([tokens(p) + tokens(" ".join(defs[p])) for p in paths])
    con = Bm25([tokens(files[p]) for p in paths])
    order = ranked(paths, sym.scores(q))[:PER_VIEW] + ranked(paths, con.scores(q))[:PER_VIEW]
    qset = set(q)
    out = []
    for p in dict.fromkeys(order):
        names = [n for n in dict.fromkeys(defs[p]) if qset & set(tokens(n))][:4]
        out.append((p, names))
    return out[:limit]


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--max", type=int, default=10)
    a = ap.parse_args()
    files = python_files(os.getcwd())
    cands = candidates(files, a.query, a.max)
    if not cands:
        print("ERROR: nothing to rank (no Python files, or the query has no usable words). Search with run_command instead.")
        sys.exit(1)
    print(f"Files most likely to need the change ({len(files)} Python files searched), best first:")
    for i, (p, names) in enumerate(cands, 1):
        print(f"{i}. {p}" + (f"   [{', '.join(names)}]" if names else ""))
    print("Open the first one that fits with read_file. The change is usually inside one of these files.")


if __name__ == "__main__":
    main()
