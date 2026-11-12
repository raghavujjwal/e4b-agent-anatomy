"""notebooks/09 and 10 (the edit-pairs study): every code cell compiles, every name is defined by an earlier cell, and every command-line flag a cell passes to a script of this
repository exists in one of them. A typo in a notebook costs Colab units, and there is no GPU here to find it, so the checks that can be made on the PC are made."""
import ast
import builtins
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ["09_edit_pairs_sample_and_train", "10_edit_pairs_eval_and_gates"]
SCRIPTS = ["edit_pairs", "pref_lora", "s4_gates", "build_replay_data", "format_probe", "train_lora"]
EXTERNAL = {"--python", "--branch", "--single-branch", "--detach", "--served-model-name", "--port", "--max-model-len", "--gpu-memory-utilization", "--enable-auto-tool-choice",
            "--tool-call-parser", "--reasoning-parser", "--reasoning-config", "--enable-lora", "--max-loras", "--max-lora-rank", "--lora-modules", "--query-gpu", "--oneline"}


def cells(name, kind="code"):
    nb = json.load(open(ROOT / "notebooks" / f"{name}.ipynb", encoding="utf-8"))
    assert nb["nbformat"] == 4
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == kind]


def python_of(src):
    return re.sub(r"(?m)^\s*%pip .*$", "pass", src)                                          # magics are not Python


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_every_name_a_cell_uses_is_defined_by_it_or_an_earlier_cell(name):
    defined = set(dir(builtins)) | {"get_ipython"}
    for i, src in enumerate(cells(name)):
        tree = ast.parse(python_of(src), f"{name} cell {i}")
        for node in ast.walk(tree):                         # a name bound anywhere in the cell counts for the whole cell: functions run after they are defined
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                defined.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, ast.arg):
                defined.add(node.arg)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                defined |= {(a.asname or a.name).split(".")[0] for a in node.names}
            elif isinstance(node, ast.ExceptHandler) and node.name:
                defined.add(node.name)
        missing = sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in defined})
        assert not missing, f"{name} cell {i} uses {missing}, which no cell has defined by then"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_every_flag_a_cell_passes_exists_in_a_script_of_the_repository(name):
    known = set()
    for s in SCRIPTS:
        known |= set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', (ROOT / "train" / f"{s}.py").read_text(encoding="utf-8")))
    used = set()
    for src in cells(name):
        used |= set(re.findall(r'"(--[a-z][a-z0-9-]*)(?:=[^"]*)?"', src))
    assert used, "no flags found: the pattern or the notebook changed"
    assert not (used - known - EXTERNAL), f"{name} passes flags that no script defines: {sorted(used - known - EXTERNAL)}"


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_the_files_a_cell_asserts_are_in_the_code_bundle(name):
    """The cell that unpacks pairs_code.zip checks for some files; `s2_pack.py --pairs-code-out` must put them there."""
    import sys
    sys.path.insert(0, str(ROOT / "train"))
    import s2_pack as P
    packed = set(P.PAIRS_ENTRIES) | set(P.PAIRS_EXTRA)
    for src in cells(name):
        m = re.search(r"for f in \(((?:\"[^\"]+\",?\s*)+)\):\s*\n\s*assert os\.path\.exists\(f\"\{CODE_DIR\}/\{f\}\"\)", src)
        if m:
            files = re.findall(r"\"([^\"]+)\"", m.group(1))
            assert files
            # the entries' imports (train_lora.py, s4_gates.py, ...) are bundled too, so compare with the real bundle, not only the entry list
            assert all((ROOT / f).exists() for f in files), files
            assert set(files) <= packed | {f"train/{n}.py" for e in packed if e.endswith(".py") and e.startswith("train/") for n in P.local_imports(ROOT / e)}, files


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_the_notebook_releases_the_gpu_at_the_end_and_never_prints_the_token(name):
    code = cells(name)
    assert "runtime.unassign()" in code[-1] and "RELEASE = True" in code[-1]
    assert not any("GITHUB_TOKEN" in c and "print(" in c and "replace(gh" not in c for c in code if "os.environ[\"GITHUB_TOKEN\"] =" not in c)
