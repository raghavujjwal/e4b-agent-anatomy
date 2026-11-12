"""train/s2_pack.py: what goes to Drive must be complete and must not carry more than it should."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
import s2_pack as P  # noqa: E402


def test_the_data_pack_drops_the_text_and_records_counts_and_hash(tmp_path):
    rows = [{"instance_id": f"i{k}", "repo": f"o{k % 2}/r", "input_ids": [1, 2, 3], "loss_mask": [0, 1, 1], "n_tokens": 3, "n_loss_tokens": 2, "text": "SECRET TEXT"} for k in range(4)]
    sft, card, out = tmp_path / "sft.jsonl", tmp_path / "card.md", tmp_path / "sft_v2.zip"
    sft.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    card.write_text("# card\n", encoding="utf-8")
    m = P.pack_data(sft, card, out)
    assert (m["sequences"], m["repositories"], m["tokens"], m["loss_tokens"]) == (4, 2, 12, 8)
    with zipfile.ZipFile(out) as z:
        assert sorted(z.namelist()) == ["sft_v2/DATA_CARD.md", "sft_v2/manifest.json", "sft_v2/sft.jsonl"]
        body = z.read("sft_v2/sft.jsonl")
        assert b"SECRET TEXT" not in body and m["sha256_sft_jsonl"] == hashlib.sha256(body).hexdigest()
        assert all("text" not in json.loads(l) for l in body.splitlines())


def test_the_code_bundle_holds_every_module_training_imports(tmp_path):
    out = tmp_path / "code.zip"
    m = P.pack_code(out)
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        assert names == set(m["files"]) | {"code_manifest.json"}
        assert {"train/train_lora.py", "train/render.py", "train/split.py"} <= names
        for f, digest in m["files"].items():
            assert hashlib.sha256(z.read(f)).hexdigest() == digest
    # nothing outside the bundle may be imported from the same folder by any file in it (a new import in train_lora.py must reach the bundle)
    for f in m["files"]:
        assert {f"train/{n}.py" for n in P.local_imports(ROOT / f)} <= set(m["files"]), f


def test_the_pairs_bundle_carries_the_study_code_the_spec_and_the_matcher(tmp_path):
    out = tmp_path / "pairs_code.zip"
    assert P.main(["--pairs-code-out", str(out)]) == 0
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        m = json.loads(z.read("code_manifest.json"))
        assert hashlib.sha256(z.read("train/prompt_spec_v2rec.json")).hexdigest() == m["files"]["train/prompt_spec_v2rec.json"]
    assert names == set(m["files"]) | {"code_manifest.json"}
    assert {"train/edit_pairs.py", "train/pref_lora.py", "train/format_probe.py", "train/build_replay_data.py", "train/s4_gates.py", "train/train_lora.py", "train/served.py",
            "train/prompt_spec_v2rec.json", "agent/vendor/scorer_port.py"} <= names
    for f in m["files"]:
        if f.endswith(".py") and f.startswith("train/"):
            assert {f"train/{n}.py" for n in P.local_imports(ROOT / f)} <= set(m["files"]), f
    # the module loaded by path must be where edit_pairs.py looks for it: REPO/agent/vendor next to train/
    assert "agent" in (ROOT / "train" / "edit_pairs.py").read_text(encoding="utf-8")


def test_the_s4_bundle_holds_the_gate_code_and_what_it_imports(tmp_path):
    out = tmp_path / "s4_code.zip"
    assert P.main(["--s4-code-out", str(out)]) == 0
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        m = json.loads(z.read("code_manifest.json"))
    assert names == set(m["files"]) | {"code_manifest.json"}
    assert {"train/s4_gates.py", "train/build_replay_data.py", "train/step_check.py"} <= names
    assert "train/train_lora.py" not in names                       # the S4 bundle does not carry the training code
    for f in m["files"]:
        assert {f"train/{n}.py" for n in P.local_imports(ROOT / f)} <= set(m["files"]), f
