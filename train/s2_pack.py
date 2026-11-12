"""Pack a built S2 training set (and optionally the training code) for Drive and Colab (notebook 07).

  python -I -X utf8 train/s2_pack.py --sft D:/data/s2/full3/sft.jsonl --card D:/data/s2/full3/DATA_CARD.md --out D:/data/s2/full3/sft_v2.zip \
      [--code-out D:/data/s2/full3/train_code_v2.zip]

sft_v2.zip holds sft.jsonl (the rendered text left out: ids and loss masks only, about a tenth of the size), DATA_CARD.md and manifest.json (counts, the sha256 of
sft.jsonl, the commit of the code that built it). train_code_v2.zip holds the three files `train_lora.py` imports (so a run needs no GitHub push and no token) and a
manifest with their sha256 and the list of those that differ from the commit. Nothing else leaves the PC.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = ["train/train_lora.py", "train/render.py", "train/split.py"]
PAIRS_ENTRIES = ("train/edit_pairs.py", "train/pref_lora.py", "train/format_probe.py", "train/build_replay_data.py")
PAIRS_EXTRA = ("train/prompt_spec_v2rec.json", "agent/vendor/scorer_port.py")           # loaded by path, not imported from train/


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def pack_data(sft: Path, card: Path, out: Path) -> dict:
    lines, n_tok, n_loss, repos = [], 0, 0, set()
    for line in open(sft, encoding="utf-8"):
        o = json.loads(line)
        o.pop("text", None)
        n_tok += o["n_tokens"]
        n_loss += o["n_loss_tokens"]
        repos.add(o["repo"])
        lines.append(json.dumps(o, ensure_ascii=False))
    body = ("\n".join(lines) + "\n").encode("utf-8")
    manifest = {"sequences": len(lines), "repositories": len(repos), "tokens": n_tok, "loss_tokens": n_loss, "sha256_sft_jsonl": hashlib.sha256(body).hexdigest(),
                "built_with_commit": git("rev-parse", "HEAD"), "format": "T0 one-pass: input_ids + loss_mask per conversation (train/s2_render.py)"}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("sft_v2/sft.jsonl", body)
        z.write(card, "sft_v2/DATA_CARD.md")
        z.writestr("sft_v2/manifest.json", json.dumps(manifest, indent=1) + "\n")
    return manifest


def local_imports(path: Path) -> set[str]:
    """Names of sibling modules (files next to `path`) that the file imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return {n for n in names if (path.parent / f"{n}.py").exists()}


def pack_code(out: Path, root: Path = ROOT, entries: tuple[str, ...] = ("train/train_lora.py",), extra: tuple[str, ...] = ()) -> dict:
    """`entries` and what they import from train/; `extra`: files that are loaded by path (a data file, a module from another folder), packed as they are."""
    needed = set(entries) | set(extra)
    todo = list(entries)
    while todo:                                                      # follow the sibling imports, so a new one cannot be forgotten
        for n in local_imports(root / todo.pop()):
            if f"train/{n}.py" not in needed:
                needed.add(f"train/{n}.py")
                todo.append(f"train/{n}.py")
    files = sorted(needed)
    manifest = {"files": {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files}, "commit": git("rev-parse", "HEAD"),
                "differ_from_commit": [f for f in files if git("status", "--porcelain", "--", f)]}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(root / f, f)
        z.writestr("code_manifest.json", json.dumps(manifest, indent=1) + "\n")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", help="with --card and --out: pack the training set")
    ap.add_argument("--card")
    ap.add_argument("--out")
    ap.add_argument("--code-out", help="also write a zip with the training code (no GitHub push needed to run notebook 07)")
    ap.add_argument("--s4-code-out", help="write a zip with the S4 gate code (train/s4_gates.py, train/build_replay_data.py and what they import; notebook 08). Needs no data arguments")
    ap.add_argument("--pairs-code-out", help="write a zip with the code of the edit-pairs study (train/edit_pairs.py, pref_lora.py, format_probe.py, build_replay_data.py, what they import, "
                                             "the prompt spec and agent/vendor/scorer_port.py; notebooks 09 and 10). Needs no data arguments")
    a = ap.parse_args(argv)
    if bool(a.sft) != bool(a.card) or bool(a.sft) != bool(a.out):
        ap.error("--sft, --card and --out go together")
    if not (a.sft or a.code_out or a.s4_code_out or a.pairs_code_out):
        ap.error("nothing to pack")
    if a.sft:
        manifest = pack_data(Path(a.sft), Path(a.card), Path(a.out))
        print(json.dumps(manifest, indent=1))
        print(a.out, Path(a.out).stat().st_size // 1024, "KiB")
    if a.code_out:
        code = pack_code(Path(a.code_out))
        print(json.dumps(code, indent=1))
        print(a.code_out, Path(a.code_out).stat().st_size // 1024, "KiB")
        if code["differ_from_commit"]:
            print("note: these files differ from the commit above (uncommitted changes):", code["differ_from_commit"])
    if a.s4_code_out:
        code = pack_code(Path(a.s4_code_out), entries=("train/s4_gates.py", "train/build_replay_data.py"))
        print(json.dumps(code, indent=1))
        print(a.s4_code_out, Path(a.s4_code_out).stat().st_size // 1024, "KiB")
        if code["differ_from_commit"]:
            print("note: these files differ from the commit above (uncommitted changes):", code["differ_from_commit"])
    if a.pairs_code_out:
        code = pack_code(Path(a.pairs_code_out), entries=PAIRS_ENTRIES, extra=PAIRS_EXTRA)
        print(json.dumps(code, indent=1))
        print(a.pairs_code_out, Path(a.pairs_code_out).stat().st_size // 1024, "KiB")
        if code["differ_from_commit"]:
            print("note: these files differ from the commit above (uncommitted changes):", code["differ_from_commit"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
