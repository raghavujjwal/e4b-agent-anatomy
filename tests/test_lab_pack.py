"""mine/make_lab_pack.py: the pack must be complete, split by the fixed rule, and refuse to ship a half-built task."""
import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mine"))
import make_lab_pack as L  # noqa: E402


def _repos():
    """One repository of each side, found with the real rule (so the test follows the rule, not a copy of its answer)."""
    found = {}
    for k in range(200):
        r = f"owner/repo{k}"
        found.setdefault(L.side(r), r)
        if len(found) == 2:
            return found["tune"], found["unseen"]
    raise AssertionError("no repository of both sides in 200 tries")


def _mined(tmp_path, ids_repos, skip_snapshot=None):
    d = tmp_path / "mined"
    (d / "snapshots").mkdir(parents=True, exist_ok=True)
    (d / "tasks.jsonl").write_text("".join(json.dumps({"instance_id": i, "repo": r, "patch": "p"}) + "\n" for i, r in ids_repos), encoding="utf-8")
    for i, _ in ids_repos:
        if i != skip_snapshot:
            (d / "snapshots" / f"{i}.tgz").write_bytes(b"tgz-" + i.encode())
    return d


def _arm(tmp_path, name, sampling="temperature: 1.0\n"):
    a = tmp_path / "agent" / name
    (a / "configs").mkdir(parents=True)
    (a / "agent.yaml").write_text("name: x\n", encoding="utf-8")
    (a / "configs" / "sampling.yaml").write_text(sampling, encoding="utf-8")
    (a / "__pycache__").mkdir()
    (a / "__pycache__" / "x.pyc").write_bytes(b"junk")
    return a


def test_pack_has_everything_and_splits_by_repository(tmp_path):
    tune, unseen = _repos()
    pairs = [("a_1", tune), ("a_2", tune), ("b_1", unseen)]
    arm = _arm(tmp_path, "arm1")
    out = tmp_path / "lab" / "pack.zip"
    m = L.pack(_mined(tmp_path, pairs), [i for i, _ in pairs], {i: "3.13" for i, _ in pairs}, [arm], out)
    assert m["pools"] == {"tune": 2, "unseen": 1} and m["tasks"] == 3
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        assert {"lab_manifest.json", "tasks.jsonl", "env_map.json", "usable_tasks.json", "snapshots/a_1.tgz", "snapshots/b_1.tgz",
                "submissions/arm1/agent.yaml", "submissions/arm1/configs/sampling.yaml"} <= names
        assert not any("__pycache__" in n for n in names)
        u = json.loads(z.read("usable_tasks.json"))
        assert u["pools"] == {"tune": ["a_1", "a_2"], "unseen": ["b_1"]} and u["usable"] == ["a_1", "a_2", "b_1"]
        assert z.read("snapshots/b_1.tgz") == b"tgz-b_1"
        assert json.loads(z.read("lab_manifest.json"))["arms"] == m["arms"]


def test_an_incomplete_task_stops_the_build(tmp_path):
    tune, _ = _repos()
    pairs = [("a_1", tune), ("a_2", tune)]
    arm = _arm(tmp_path, "arm1")
    with pytest.raises(SystemExit) as e:
        L.pack(_mined(tmp_path, pairs, skip_snapshot="a_2"), ["a_1", "a_2"], {"a_1": "3.13", "a_2": "3.13"}, [arm], tmp_path / "p.zip")
    assert "a_2: no snapshot" in str(e.value)
    with pytest.raises(SystemExit) as e:
        L.pack(_mined(tmp_path, pairs), ["a_1", "a_2"], {"a_1": "3.13"}, [arm], tmp_path / "p.zip")
    assert "a_2: no interpreter in env_map" in str(e.value)


def test_the_arm_digest_follows_the_content_not_the_folder_name(tmp_path):
    one, two = _arm(tmp_path, "one"), _arm(tmp_path, "two")
    assert L.arm_digest(one) == L.arm_digest(two)
    (two / "configs" / "sampling.yaml").write_text("temperature: 0.7\n", encoding="utf-8")
    assert L.arm_digest(one) != L.arm_digest(two)
