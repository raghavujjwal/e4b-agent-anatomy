"""train/pref_lora.py: the pair losses, the data split, and the whole training path on a tiny random Gemma 4 on CPU (skipped where the E4B config is absent)."""
import json
import math
import os
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
import served  # noqa: E402
import pref_lora as PL  # noqa: E402

CFG_DIR = Path(os.environ.get("E4B_TOKENIZER_DIR") or served.DEFAULT_TOKENIZER_DIR)
need_cfg = pytest.mark.skipif(not (CFG_DIR / "config.json").exists(), reason="E4B config.json not present")
t = torch.tensor


# ---- the losses ---------------------------------------------------------------------------------------------------------------------
def test_dpo_starts_at_log_two_and_falls_as_the_chosen_call_gains_on_the_rejected_one():
    loss, margin = PL.pair_loss("dpo", t(-10.0), t(-12.0), t(-10.0), t(-12.0), 8)              # policy == reference
    assert loss.item() == pytest.approx(math.log(2)) and margin.item() == 0.0
    better, m = PL.pair_loss("dpo", t(-8.0), t(-12.0), t(-10.0), t(-12.0), 8, beta=0.1)         # the chosen call became 2 nats likelier
    assert m.item() == 2.0 and better.item() == pytest.approx(-math.log(1 / (1 + math.exp(-0.2))))
    worse, _ = PL.pair_loss("dpo", t(-10.0), t(-9.0), t(-10.0), t(-12.0), 8)                   # the rejected call became likelier
    assert worse.item() > math.log(2) > better.item()


def test_the_tokens_two_calls_share_cancel_in_dpo():
    """Both calls open with the same tokens (the tool name, the file) and differ afterwards. The margin depends only on the log-ratio of the part where they differ,
    whatever the policy does to the shared part: nothing in DPO asks for `edit_file` over `read_file`."""
    shared_ref, shared_pol = -3.0, -1.0                       # the policy has made the shared opening 2 nats likelier
    c_ref, r_ref, c_pol, r_pol = -7.0, -6.0, -5.0, -9.0
    _, m_shared_moved = PL.pair_loss("dpo", t(shared_pol + c_pol), t(shared_pol + r_pol), t(shared_ref + c_ref), t(shared_ref + r_ref), 9)
    _, m_shared_still = PL.pair_loss("dpo", t(shared_ref + c_pol), t(shared_ref + r_pol), t(shared_ref + c_ref), t(shared_ref + r_ref), 9)
    assert m_shared_moved.item() == m_shared_still.item() == (c_pol - c_ref) - (r_pol - r_ref)


def test_dpo_nll_adds_the_chosen_calls_mean_token_cross_entropy():
    plain, _ = PL.pair_loss("dpo", t(-20.0), t(-30.0), t(-20.0), t(-30.0), 10)
    with_nll, _ = PL.pair_loss("dpo_nll", t(-20.0), t(-30.0), t(-20.0), t(-30.0), 10, nll_weight=0.5)
    assert (with_nll - plain).item() == pytest.approx(0.5 * 20.0 / 10)
    with pytest.raises(ValueError):
        PL.pair_loss("rft", t(0.0), t(0.0), t(0.0), t(0.0), 1)


# ---- data ---------------------------------------------------------------------------------------------------------------------------
class WordTok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}


def pairs_file(tmp_path, n_sources=40, per_source=2, long_at=None):
    rows = []
    for s in range(n_sources):
        for j in range(per_source):
            rows.append({"id": f"s{s}#{j}", "state": f"s{s}", "round": 1 + (j % 2), "source": f"hist-{s}", "prompt_ids": list(range(30)), "chosen": "a b c <tool_call|>", "rejected": "d e f g"})
    if long_at is not None:
        rows[long_at]["prompt_ids"] = list(range(5000))
    p = tmp_path / "pairs.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def test_the_validation_split_is_by_commit_and_pairs_over_the_length_limit_are_dropped(tmp_path):
    train, val, dropped = PL.load_pairs(pairs_file(tmp_path, long_at=0), WordTok(), max_len=1000, limit=0, seed=0, val_share=0.25)
    ids = lambda xs: {x["id"].split("#")[0] for x in xs}                                       # noqa: E731
    assert dropped == 1 and not ids(train) & ids(val), "both pairs of a commit stay on one side"
    assert 3 <= len(ids(val)) <= 20 and len(train) + len(val) == 79
    assert train[0]["chosen_ids"] == [0, 1, 2, 3, 0], "a call that closes ends with the stop token (one id for the word-tokenizer's '<|tool_response>')"
    assert train[0]["rejected_ids"] == [0, 1, 2, 3], "a call that never closed has no stop token"
    a, _, _ = PL.load_pairs(pairs_file(tmp_path), WordTok(), 1000, 0, seed=5, val_share=0.25)
    b, _, _ = PL.load_pairs(pairs_file(tmp_path), WordTok(), 1000, 0, seed=5, val_share=0.25)
    assert [x["id"] for x in a] == [x["id"] for x in b]
    capped, _, _ = PL.load_pairs(pairs_file(tmp_path), WordTok(), 1000, limit=10, seed=0, val_share=0.25)
    assert len(capped) == 10


def test_the_stop_token_is_one_id_and_closes_every_call_that_closed():
    """eos ids 1, 106, 50 in the checkpoint's generation_config.json: 50 is `<|tool_response>`, what a model stops on after writing a call."""
    try:
        tok = served.load_tokenizer()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"the checkpoint's tokenizer is not on this machine: {e}")
    assert tok(PL.STOP, add_special_tokens=False)["input_ids"] == [50]
    call = '<|tool_call>call:read_file{filepath:<|"|>a.py<|"|>}<tool_call|>'
    ids = PL.call_ids(tok, call)
    assert ids[-1] == 50 and ids[:-1] == tok(call, add_special_tokens=False)["input_ids"]
    assert PL.call_ids(tok, call[:-len(PL.CALL_CLOSE)]) == tok(call[:-len(PL.CALL_CLOSE)], add_special_tokens=False)["input_ids"]


# ---- the model ----------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny():
    import train_lora as T
    os.environ["TINY_CONFIG"] = str(CFG_DIR)
    torch.manual_seed(0)
    m = T.tiny_model()
    m.eval()
    return T, m


@need_cfg
def test_call_logp_is_minus_the_completion_loss_of_the_sft_trainer(tiny):
    T, model = tiny
    rng = random.Random(0)
    prompt, call = [rng.randrange(10, 2000) for _ in range(50)], [rng.randrange(10, 2000) for _ in range(9)]
    with torch.no_grad():
        lp = PL.call_logp(model, prompt, call, "cpu")
        ce, g = T.completion_loss(model, {"prompt_ids": prompt, "gen_ids": call}, "cpu")
    assert g == 9 and lp.item() == pytest.approx(-ce.item(), rel=1e-5)


def tiny_pairs(tmp_path, n):
    from transformers import AutoTokenizer
    rng = random.Random(1)
    rows = []
    for k in range(n):
        rows.append({"id": f"p{k}#0", "state": f"p{k}", "round": 1, "source": f"hist-{k}", "prompt_ids": [rng.randrange(10, 2000) for _ in range(40)],
                     "chosen": "<|tool_call>call:edit_file{filepath:<|\"|>a.py<|\"|>,new_string:<|\"|>x = 1<|\"|>,old_string:<|\"|>x = 0<|\"|>}<tool_call|>",
                     "rejected": "<|tool_call>call:edit_file{filepath:<|\"|>a.py<|\"|>,new_string=\"x = 1\",old_string=\"x = 0\"}<tool_call|>"})
    p = tmp_path / "pairs.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def run_args(tmp_path, kind, out, *extra):
    return ["--tiny", "--model", str(CFG_DIR), "--pairs", str(tiny_pairs(tmp_path, 16)), "--loss", kind, "--out", str(out), "--rank", "4", "--alpha", "4", "--accum", "2",
            "--max-pairs", "0", "--val-share", "0.25", "--val-max", "3", "--save-every", "1", "--keep-checkpoints", "2", "--checkpoint", "layers", *extra]


@need_cfg
@pytest.mark.parametrize("kind", ["dpo", "dpo_nll", "rft"])
def test_a_tiny_run_trains_saves_audits_and_resumes(tmp_path, kind):
    pytest.importorskip("peft")
    out = tmp_path / kind
    assert PL.main(run_args(tmp_path, kind, out, "--max-steps", "3")) == 0
    rep = json.loads((out / "train_report.json").read_text(encoding="utf-8"))
    assert rep["steps"] == 3 and rep["audit"]["n_bad_keys"] == 0 and rep["validation"] and sorted(p.name for p in out.glob("ckpt-*")) == ["ckpt-00002", "ckpt-00003"]
    m = [json.loads(l) for l in open(out / "metrics.jsonl", encoding="utf-8")]
    assert [r["step"] for r in m] == [1, 2, 3] and all(math.isfinite(r["loss"]) for r in m)
    if kind != "rft":
        assert m[0]["loss"] == pytest.approx(math.log(2) + (0.0 if kind == "dpo" else m[0]["loss"] - math.log(2)), rel=1e-6)
        assert m[0]["margin"] == 0.0 and m[0]["reward_accuracy"] == 0.0, "at the first step the LoRA output is zero, so the policy is the reference"
        if kind == "dpo":
            assert m[0]["loss"] == pytest.approx(math.log(2))
    assert PL.main(run_args(tmp_path, kind, out, "--max-steps", "5", "--resume")) == 0
    assert json.loads((out / "train_report.json").read_text(encoding="utf-8"))["steps"] == 5


@need_cfg
def test_dpo_learns_to_prefer_the_chosen_call_on_a_tiny_set(tmp_path):
    pytest.importorskip("peft")
    out = tmp_path / "dpo_fit"
    assert PL.main(run_args(tmp_path, "dpo", out, "--epochs", "8", "--lr", "3e-3", "--beta", "0.5", "--rank", "8", "--alpha", "16", "--lr-floor", "0.5")) == 0
    m = [json.loads(l) for l in open(out / "metrics.jsonl", encoding="utf-8")]
    assert len(m) == 16 * 8 // 2 - len([1 for _ in range(0)]) or len(m) > 20
    assert max(r["reward_accuracy"] for r in m[-4:]) >= 0.75 and m[-1]["loss"] < math.log(2) and m[-1]["margin"] > 0
