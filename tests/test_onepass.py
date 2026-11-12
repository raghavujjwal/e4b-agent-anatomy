"""train/train_lora.py, one-pass format (plan S3.0 test 2): one forward pass over a whole rendered conversation must give the loss of the
separate per-turn passes, because attention is causal and the targets are the same tokens.

Uses a tiny random Gemma 4 built on CPU from the E4B config (E4B_TOKENIZER_DIR or the default Kaggle copy); skipped where that is absent.
"""
import json
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
from split import is_validation  # noqa: E402

CFG_DIR = Path(os.environ.get("E4B_TOKENIZER_DIR") or served.DEFAULT_TOKENIZER_DIR)
need_cfg = pytest.mark.skipif(not (CFG_DIR / "config.json").exists(), reason="E4B config.json not present")


@pytest.fixture(scope="module")
def tiny():
    import train_lora as T
    os.environ["TINY_CONFIG"] = str(CFG_DIR)
    torch.manual_seed(0)
    m = T.tiny_model()
    m.eval()
    return T, m


@need_cfg
def test_onepass_loss_equals_the_sum_of_per_turn_losses(tiny):
    T, model = tiny
    rng = random.Random(0)
    n = 160
    ids = [rng.randrange(10, 2000) for _ in range(n)]
    spans = [(30, 45), (80, 83), (120, 160)]
    mask = [0] * n
    for s, e in spans:
        for t in range(s, e):
            mask[t] = 1
    with torch.no_grad():
        one, n1 = T.onepass_loss(model, {"input_ids": ids, "loss_mask": mask}, "cpu")
        total, n2 = 0.0, 0
        for s, e in spans:
            loss, g = T.completion_loss(model, {"prompt_ids": ids[:s], "gen_ids": ids[s:e]}, "cpu")
            total, n2 = total + loss.item(), n2 + g
    assert n1 == n2 == 15 + 3 + 40
    assert one.item() == pytest.approx(total, rel=1e-4)


@need_cfg
def test_onepass_loss_ignores_tokens_outside_the_mask(tiny):
    """Changing a token that is context only (a tool result) after the last target must not change the loss; changing one before it must."""
    T, model = tiny
    rng = random.Random(1)
    ids = [rng.randrange(10, 2000) for _ in range(120)]
    mask = [0] * 120
    for t in range(40, 50):
        mask[t] = 1
    base = {"input_ids": ids, "loss_mask": mask}
    after = dict(base, input_ids=ids[:100] + [ids[100] + 1] + ids[101:])
    before = dict(base, input_ids=ids[:20] + [ids[20] + 1] + ids[21:])
    with torch.no_grad():
        l0 = T.onepass_loss(model, base, "cpu")[0].item()
        assert T.onepass_loss(model, after, "cpu")[0].item() == pytest.approx(l0, rel=1e-5)
        assert T.onepass_loss(model, before, "cpu")[0].item() != pytest.approx(l0, rel=1e-5)


@need_cfg
def test_checkpointing_the_layers_by_hand_changes_no_gradient():
    """Plan S3.0 test 3 relies on this: the wrapper may only trade compute for memory. Includes the layers that reuse another layer's K/V."""
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    import train_lora as T
    os.environ["TINY_CONFIG"] = str(CFG_DIR)
    torch.manual_seed(0)
    model = get_peft_model(T.tiny_model(), LoraConfig(r=4, lora_alpha=4, lora_dropout=0.0, target_modules=T.TARGET_RE, task_type="CAUSAL_LM"))
    for n, p in model.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.05)               # B starts at zero, which would hide every gradient of A
    model.train()
    rng = random.Random(2)
    ids = [rng.randrange(10, 2000) for _ in range(96)]
    ex = {"input_ids": ids, "loss_mask": [0] * 30 + [1] * 40 + [0] * 26}

    def grads():
        model.zero_grad(set_to_none=True)
        loss, _ = T.onepass_loss(model, ex, "cpu")
        loss.backward()
        return loss.item(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    loss0, g0 = grads()
    assert T.checkpoint_layers(model) == 6
    loss1, g1 = grads()
    assert loss1 == pytest.approx(loss0, rel=1e-5)
    assert g0.keys() == g1.keys() and len(g0) > 0
    assert sum(float(v.abs().sum()) for n, v in g0.items() if "lora_A" in n) > 0
    for n in g0:
        assert torch.allclose(g0[n], g1[n], rtol=1e-3, atol=1e-6), n


@need_cfg
def test_probe_reports_each_length(tiny, capsys):
    T, model = tiny
    sample = {"input_ids": list(range(10, 90)), "loss_mask": [0] * 80}
    assert T.probe(model, [sample], [100, 150], "cpu") == 0
    out = capsys.readouterr().out
    assert "probe 100 tokens" in out and "probe 150 tokens" in out and "tokens/s" in out


def test_onepass_split_is_by_repository_and_respects_max_len(tmp_path):
    import train_lora as T
    rows = []
    for k in range(60):
        repo = f"owner{k % 20}/repo{k % 20}"
        rows.append({"instance_id": f"i{k}", "repo": repo, "input_ids": list(range(50 + (k % 3) * 100)), "loss_mask": [0, 1] * (25 + (k % 3) * 50)})
    p = tmp_path / "sft.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    train, val, dropped = T.load_onepass(str(p), max_len=160, limit=0, seed=0, val_share=0.25)
    assert dropped == sum(1 for r in rows if len(r["input_ids"]) > 160)
    assert {e["repo"] for e in train}.isdisjoint({e["repo"] for e in val})
    assert val and train
    assert T.n_targets(train[0]) == sum(train[0]["loss_mask"])
    assert all(is_validation(e["repo"], 0.25) for e in val) and not any(is_validation(e["repo"], 0.25) for e in train)


def test_lr_schedule_warms_up_then_follows_a_cosine_to_the_floor():
    import train_lora as T
    total, warm = 200, 10
    assert T.lr_factor(0, total, warm) == pytest.approx(1 / warm)
    assert T.lr_factor(warm - 1, total, warm) == pytest.approx(1.0, abs=0.01)
    assert T.lr_factor(total, total, warm) == pytest.approx(0.0, abs=1e-9)                      # old behaviour: decays to zero
    assert T.lr_factor(total, total, warm, floor=0.1) == pytest.approx(0.1)                     # plan S3.2: cosine to 10%
    assert T.lr_factor(total // 2, total, warm, floor=0.1) == pytest.approx(0.55)
    assert all(T.lr_factor(s + 1, total, warm, 0.1) <= T.lr_factor(s, total, warm, 0.1) for s in range(warm, total))


@need_cfg
def test_a_tiny_run_keeps_the_last_checkpoints_logs_validation_loss_and_resumes(tmp_path):
    """The whole training path on a tiny Gemma 4 on CPU: onepass data, hand-wrapped layers, cosine to a floor, kept checkpoints, validation at every save,
    optimizer state, resume. The same code runs on the Colab card; only the memory and the speed differ."""
    pytest.importorskip("peft")
    import train_lora as T
    rng = random.Random(3)
    rows = []
    for k in range(16):
        n = 100 + 10 * (k % 3)
        rows.append({"instance_id": f"i{k}", "repo": f"o{k}/r{k}", "input_ids": [rng.randrange(10, 2000) for _ in range(n)], "loss_mask": [0] * 60 + [1] * (n - 60)})
    data, out = tmp_path / "data", tmp_path / "adapter"
    data.mkdir()
    (data / "sft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    base = ["--tiny", "--model", str(CFG_DIR), "--format", "onepass", "--data", str(data), "--out", str(out), "--rank", "4", "--alpha", "4", "--accum", "2",
            "--max-len", "300", "--max-samples", "0", "--val-share", "0.25", "--val-samples", "4", "--save-every", "1", "--keep-checkpoints", "2",
            "--val-at-save", "--lr-floor", "0.1", "--checkpoint", "layers", "--epochs", "1"]
    assert T.main(base + ["--max-steps", "3"]) == 0
    rep = json.loads((out / "train_report.json").read_text(encoding="utf-8"))
    assert rep["steps"] == 3 and [h["step"] for h in rep["val_history"]] == [1, 2, 3] and rep["audit"]["n_bad_keys"] == 0
    assert sorted(p.name for p in out.glob("ckpt-*")) == ["ckpt-00002", "ckpt-00003"]
    assert (out / "state" / "optimizer.pt").exists() and (out / "ckpt-00003" / "adapter_model.safetensors").exists()
    assert T.main(base + ["--max-steps", "5", "--resume"]) == 0
    rep = json.loads((out / "train_report.json").read_text(encoding="utf-8"))
    assert rep["steps"] == 5 and [h["step"] for h in rep["val_history"]] == [4, 5]               # the resumed run continues from step 3
    assert sorted(p.name for p in out.glob("ckpt-*")) == ["ckpt-00004", "ckpt-00005"]


def test_the_validation_rule_is_stable_and_ignores_case():
    assert is_validation("Owner/Repo", 0.5) == is_validation("owner/repo", 0.5)
    assert not any(is_validation(f"o/r{k}", 0.0) for k in range(200))
    assert all(is_validation(f"o/r{k}", 1.01) for k in range(200))
    share = sum(is_validation(f"owner{k}/repo{k}", 0.05) for k in range(4000)) / 4000
    assert 0.03 < share < 0.07, share
    # pinned: changing the rule would silently move repositories between training and validation and make runs incomparable
    assert [k for k in range(40) if is_validation(f"o/r{k}", 0.2)] == [8, 9, 10, 13, 16, 26, 28, 31]
