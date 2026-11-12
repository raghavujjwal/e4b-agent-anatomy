"""notebooks/03 with the lab pools: a typo or a missing arm folder costs a GPU session, so what can be checked on the PC is checked here.
The config cell is run as Python (it only assigns); the run cell is checked for the pieces that protect the unseen pools."""
import ast
import builtins
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "mine"))
from test_notebooks_pairs import cells, python_of  # noqa: E402
import make_lab_pack as L  # noqa: E402

NB = "03_serve_and_run_e4b"
POOLS = {"dev66", "rich", "tune", "unseen"}


def find(marker):
    """The one code cell that contains `marker` (cells are found by what they do, not by position)."""
    hits = [src for src in cells(NB) if marker in src]
    assert len(hits) == 1, f"{len(hits)} cells contain {marker!r}"
    return hits[0]


@pytest.fixture(scope="module")
def config():
    ns = {}
    exec(compile(python_of(find("RUNS = [")), f"{NB} config cell", "exec"), ns)      # assignments only: no Colab, no network
    return ns


def test_every_name_a_cell_uses_is_defined_by_it_or_an_earlier_cell():
    defined = set(dir(builtins)) | {"get_ipython"}
    for i, src in enumerate(cells(NB)):
        tree = ast.parse(python_of(src), f"{NB} cell {i}")
        for node in ast.walk(tree):
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
        assert not missing, f"{NB} cell {i} uses {missing}, which no cell has defined by then"


def test_the_pool_is_known_and_every_active_arm_exists(config):
    assert config["POOL"] in POOLS
    names = [n for n, _ in config["RUNS"]]
    assert len(names) == len(set(names)), "two runs with one name would overwrite each other's results file"
    for name, folder in config["RUNS"]:
        assert (ROOT / folder / "agent.yaml").exists(), f"{name}: {folder} has no agent.yaml"


def test_a_lab_run_only_uses_arms_the_pack_carries(config):
    """On Colab the clone is at the pushed commit; an arm folder that is local only arrives through lab_pack.zip or not at all."""
    if config["POOL"] not in ("tune", "unseen"):
        pytest.skip("not a lab pool")
    for name, folder in config["RUNS"]:
        assert Path(folder).name in L.DEFAULT_ARMS, f"{name}: {folder} is not in make_lab_pack.DEFAULT_ARMS, so Colab would not have it"


def test_the_unseen_pools_never_print_a_per_task_line():
    run_cell, table_cell = find("agent/run_eval.py"), find("outcome classes")
    assert 'AGG_ONLY = POOL in ("rich", "unseen")' in run_cell and "not AGG_ONLY" in table_cell
    pattern = eval(re.search(r're\.match\((r"[^"]+"), line\)', run_cell).group(1))
    per_task = "[3/70] RESOLVED  click_2840             resolved calls=9 turns=11 52.3s "            # the shape agent/run_eval.py prints (lines 80-83)
    assert re.match(pattern, per_task)
    for keep in ("70 selected, 0 done, 70 to run | backend=openai", "resolved 5/70", "===== lab_tune_ctl1 finished in 31.2 min", "Traceback (most recent call last):"):
        assert not re.match(pattern, keep), f"the filter would hide: {keep}"


def test_the_pack_and_the_notebook_agree_on_the_file_names(config):
    run_cell = find("agent/run_eval.py")
    for member in ("lab_manifest.json", "tasks.jsonl", "snapshots", "env_map.json", "usable_tasks.json", "submissions"):
        assert member in run_cell, f"the run cell never reads {member}, which the pack contains"
    assert config["LAB_PACK"].endswith("/lab_pack.zip")


def test_the_pipeline_arms_are_never_run_on_a_stale_clone(config):
    """Colab clones the pushed branch; the pipeline arms need the runner that models a multi-agent session as the scorer's ADK does (runner.view). The run cell
    stops, with the reason, when the clone still has the old one -- and the local runner has it."""
    run_cell = find("agent/run_eval.py")
    assert 'if any("_pipe_" in sub for _, sub in RUNS):' in run_cell and 'assert "def view(" in open(f"{CODE_DIR}/agent/vendor/runner.py").read()' in run_cell
    assert run_cell.index('"_pipe_" in sub') < run_cell.index("for name, sub in RUNS:\n    out, trace")          # before the first run starts
    assert "def view(" in (ROOT / "agent" / "vendor" / "runner.py").read_text(encoding="utf-8")
    ladder = [folder for _, folder in config["RUNS"] if "_pipe_" in folder]
    assert ladder == ["agent/submission_e4b_pipe_s1", "agent/submission_e4b_pipe_s2", "agent/submission_e4b_pipe_s3"], ladder


def test_the_ladder_runs_the_control_first_and_last(config):
    """Protocol rule 9/16: the same-session controls bracket the rungs, in the order of the arms' own nesting."""
    folders = [folder for _, folder in config["RUNS"]]
    assert folders == ["agent/submission_e4b_v2rec", "agent/submission_e4b_pipe_s1", "agent/submission_e4b_pipe_s2", "agent/submission_e4b_pipe_s3",
                       "agent/submission_e4b_v2rec"]
    assert [n for n, _ in config["RUNS"]] == ["lab_tune_ctl2", "lab_tune_pipe_s1_1", "lab_tune_pipe_s2_1", "lab_tune_pipe_s3_1", "lab_tune_ctl3"]
    assert config["POOL"] == "tune" and config["TASK_IDS"] is None
