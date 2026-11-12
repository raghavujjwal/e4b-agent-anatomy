"""Pack a submission directory into a Kaggle submission.zip (agent.yaml at the archive root).

  python agent/make_submission_zip.py agent/submission_e4b_v2rec --out D:/data/submissions/e4b_v2rec.zip

Files are added in sorted order with a fixed timestamp and forward-slash names, so the same directory always gives the same
archive (same sha256). README.md files and caches are left out: only what the scorer reads goes in.
"""
from __future__ import annotations
import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

SKIP_NAMES = {"README.md", ".DS_Store", "Thumbs.db"}
SKIP_DIRS = {"__pycache__", ".git"}
FIXED_TIME = (2026, 1, 1, 0, 0, 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    src, out = Path(a.src).resolve(), Path(a.out)
    if not (src / "agent.yaml").is_file():
        print(f"no agent.yaml in {src}")
        return 1
    files = sorted(p for p in src.rglob("*")
                   if p.is_file() and p.name not in SKIP_NAMES and not (set(p.relative_to(src).parts) & SKIP_DIRS)
                   and not p.name.endswith(".pyc"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            info = zipfile.ZipInfo(p.relative_to(src).as_posix(), FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, p.read_bytes())
    for p in files:
        print("  " + p.relative_to(src).as_posix())
    print(f"{out}  {out.stat().st_size} bytes  sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
