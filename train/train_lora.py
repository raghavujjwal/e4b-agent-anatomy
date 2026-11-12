"""LoRA SFT of Gemma 4 E4B on the edit-step samples of train/make_edit_data.py (loss on the completion tokens only).

  python train/train_lora.py --model /content/models/gemma-4-e4b-it-qat-q4_0-unquantized --data /content/edit_sft --out /content/drive/MyDrive/gemma4-adapters/edit_v1
  python train/train_lora.py --tiny --data D:/data/edit_sft --out D:/data/adapters/tiny      # CPU smoke test with a small random model

Base: the unquantized QAT weights (bf16). The served model is the W4A16 version of the same weights, so the adapter is a correction on
nearly the same matrices (QLoRA tool-internalization found 4-bit and 8-bit bases to differ only marginally). Target modules are the ones the
public LoRA canary loaded into the scorer's vLLM (language-model attention and MLP projections; vision and audio towers untouched), and the
saved adapter is audited against that key pattern. Recipe from the QLoRA tool-internalization paper: rank 32, alpha 64, all linear layers.
Adapter checkpoints are written every --save-every steps, so a Colab disconnect loses little; --resume continues from the last one, with the
optimizer state (kept in <out>/state/optimizer.pt, outside the adapter files).

One-pass format (2026-11-04, plan S3.1): `--format onepass --data <folder with sft.jsonl from train/s2_render.py>` trains on whole rendered conversations
(format T0, no thought channel), one forward pass each, loss only on the model's own calls and closing text; validation is by repository. Per-turn
samples (`--replay-dir`, the edit-step data) can be mixed in; the loss function is chosen per example.
  python train/train_lora.py --model <bf16 base> --format onepass --data D:/data/s2/full --rank 64 --alpha 64 --lr 1.5e-4 --epochs 2 \
      --accum 16 --max-len 12288 --max-samples 0 --save-every 50 --out <adapter dir>
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render  # noqa: E402
from split import is_validation  # noqa: E402

TARGET_RE = r"model\.language_model\.layers\.\d+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
VLLM_KEY_RE = re.compile(r"^base_model\.model\.model\.language_model\.layers\.\d+\.(self_attn|mlp)\."
                         r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_[AB]\.weight$")


def load_replay(path, limit, edit_repeat, seed, max_len):
    """Own-trace turns (train/build_replay_data.py): exact prompt and completion ids. Successful edit turns can be repeated."""
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if len(r["prompt_ids"]) + len(r["gen_ids"]) > max_len:
            continue
        ex = {"id": f"replay-{r['run']}-{r['instance_id']}-{r['turn']}", "prompt_ids": r["prompt_ids"], "gen_ids": r["gen_ids"]}
        out += [ex] * (edit_repeat if r["tool"] in ("edit_file", "write_file") else 1)
    random.Random(seed).shuffle(out)
    return out[:limit] if limit else out


def build_examples(tok, data_dir, split, formats, thought, max_len, limit, seed):
    out, dropped = [], {}
    if formats == ["none"]:
        return out, dropped
    recs = list(render.load_split(data_dir, split, formats))
    random.Random(seed).shuffle(recs)          # shuffle before rendering: rendering is the slow part, so stop at `limit`
    for rec, tools in recs:
        if limit and len(out) >= limit:
            break
        ex, why = render.render(tok, rec, tools, thought)
        if ex is None:
            dropped[why] = dropped.get(why, 0) + 1
        elif len(ex["prompt_ids"]) + len(ex["gen_ids"]) > max_len:
            dropped["too long"] = dropped.get("too long", 0) + 1
        else:
            out.append({"id": rec["id"], **ex})
    return out, dropped


def tiny_model():
    from transformers import AutoConfig, AutoModelForImageTextToText
    cfg = AutoConfig.from_pretrained(os.environ["TINY_CONFIG"])
    t = cfg.text_config
    n = 6
    # the last num_kv_shared_layers layers reuse K/V of an earlier layer of the same type, so each type must appear before them
    t.num_hidden_layers, t.num_kv_shared_layers = n, 2
    t.layer_types = ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "full_attention"]
    t.hidden_size, t.intermediate_size, t.hidden_size_per_layer_input = 64, 128, 16
    t.num_attention_heads, t.num_key_value_heads = 4, 2
    for k in ("head_dim", "global_head_dim"):
        if hasattr(t, k):
            setattr(t, k, 16)
    for sub in (cfg.vision_config, cfg.audio_config):
        if sub is not None:
            for k in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim"):
                if hasattr(sub, k):
                    setattr(sub, k, {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 1, "num_attention_heads": 4,
                                     "num_key_value_heads": 4, "head_dim": 16}[k])
    return AutoModelForImageTextToText.from_config(cfg, dtype=torch.float32)


def completion_loss(model, ex, device):
    ids = torch.tensor([ex["prompt_ids"] + ex["gen_ids"]], device=device)
    g = len(ex["gen_ids"])
    # mm_token_type_ids (all text) is required by the model's mask builder in training mode for the variants that use it
    # use_cache=False: `model.config.use_cache = False` only reaches the top-level config, and the text model builds a DynamicCache from its own
    # (default True) setting, which keeps every layer's keys and values for nothing and breaks recomputation (checkpoint_layers) by appending twice
    logits = model(input_ids=ids, mm_token_type_ids=torch.zeros_like(ids), logits_to_keep=g + 1, use_cache=False).logits[0, :-1].float()   # softcapping is inside the model's forward
    tgt = torch.tensor(ex["gen_ids"], device=logits.device)
    return torch.nn.functional.cross_entropy(logits, tgt, reduction="sum"), g


def lr_factor(s: int, total: int, warm: int, floor: float = 0.0) -> float:
    """Linear warm-up over `warm` steps, then a cosine from 1 down to `floor` (plan S3.2: 5% warm-up, cosine to 10%; floor 0 is the old behaviour)."""
    return min(1.0, (s + 1) / warm) * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))


def checkpoint_layers(model) -> int:
    """Recompute each decoder layer in the backward pass instead of storing its activations (plan S3.0 test 3). A competitor measured that
    gradient_checkpointing_enable() saves nothing on E4B, and a 12,000-token sequence stores tens of GB without checkpointing, so the layers of the
    language model are wrapped by hand. Returns the number wrapped. The gradients equal those of the unwrapped model (tests/test_onepass.py)."""
    from torch.utils.checkpoint import checkpoint

    def wrap(layer):
        inner = layer.forward
        layer.forward = lambda *a, **k: checkpoint(inner, *a, use_reentrant=False, **k)

    n = 0
    for name, mod in model.named_modules():
        if name.endswith("language_model.layers") and isinstance(mod, torch.nn.ModuleList):
            for layer in mod:
                wrap(layer)
                n += 1
    return n


def onepass_loss(model, ex, device):
    """One forward pass over a whole rendered conversation (train/s2_render.py: `input_ids`, `loss_mask`); the loss is the sum of the cross entropy
    over the target tokens (the model's own calls and closing text), nothing else. Under causal attention this equals one sample per turn.
    Logits are computed only where there is a target: the 262,144-wide output layer dominates memory otherwise. The model applies its own
    final_logit_softcapping inside forward when `logits_to_keep` is a tensor of positions."""
    ids = torch.tensor([ex["input_ids"]], device=device)
    mask = torch.tensor(ex["loss_mask"], device=device)
    keep = (mask[1:] == 1).nonzero().flatten()           # the state at position t predicts token t+1
    logits = model(input_ids=ids, mm_token_type_ids=torch.zeros_like(ids), logits_to_keep=keep, use_cache=False).logits[0].float()
    return torch.nn.functional.cross_entropy(logits, ids[0, keep + 1], reduction="sum"), int(keep.numel())


def example_loss(model, ex, device):
    return onepass_loss(model, ex, device) if "input_ids" in ex else completion_loss(model, ex, device)


def n_targets(ex):
    return sum(ex["loss_mask"]) if "input_ids" in ex else len(ex["gen_ids"])


def load_onepass(path, max_len, limit, seed, val_share):
    """train/s2_render.py output. The split is by repository, never by sample (the rule is train/split.py, shared with the data card), so validation
    loss measures repositories the adapter has not seen."""
    train, val, dropped = [], [], 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if len(r["input_ids"]) > max_len:
            dropped += 1
            continue
        ex = {"id": f"s2-{r['instance_id']}", "input_ids": r["input_ids"], "loss_mask": r["loss_mask"], "repo": r["repo"]}
        (val if is_validation(r["repo"], val_share) else train).append(ex)
    random.Random(seed).shuffle(train)
    return (train[:limit] if limit else train), val, dropped


def probe(model, train, lens, device):
    """Plan S3.0 tests 3 and 4: peak memory and tokens per second of one forward and backward pass at each length, on tiled real tokens (the content
    does not change the cost). Run it once per --checkpoint mode on the Colab card and choose the mode for the long run from the numbers."""
    src = next(e for e in train if "input_ids" in e)["input_ids"]
    cuda = str(device).startswith("cuda")
    for n in lens:
        ex = {"input_ids": (src * (n // len(src) + 1))[:n], "loss_mask": [0] * (n - n // 10) + [1] * (n // 10)}
        for _ in range(2):                              # the first pass pays for kernel selection and allocator growth; report the second
            if cuda:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t0 = time.time()
            loss, g = onepass_loss(model, ex, device)
            loss.backward()
            if cuda:
                torch.cuda.synchronize()
            dt = time.time() - t0
            model.zero_grad(set_to_none=True)
        mem = f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} of {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB, " if cuda else ""
        print(f"probe {n} tokens: {mem}{n / dt:.0f} tokens/s, {dt:.1f} s per sequence, loss {loss.item() / g:.3f} per target token", flush=True)
    return 0


@torch.no_grad()
def mean_loss(model, exs, device):
    model.eval()
    tot = n = 0
    for ex in exs:
        l, g = example_loss(model, ex, device)
        tot, n = tot + l.item(), n + g
    model.train()
    return tot / max(n, 1)


def audit(out_dir):
    from safetensors import safe_open
    p = os.path.join(out_dir, "adapter_model.safetensors")
    with safe_open(p, "pt") as f:
        keys = list(f.keys())
    bad = [k for k in keys if not VLLM_KEY_RE.match(k)]
    conf = json.load(open(os.path.join(out_dir, "adapter_config.json")))
    rep = {"tensors": len(keys), "n_bad_keys": len(bad), "bad_examples": bad[:5], "rank": conf["r"], "size_MiB": round(os.path.getsize(p) / 2**20, 1)}
    print("adapter audit:", json.dumps(rep), flush=True)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", help="directory of the unquantized checkpoint (weights and tokenizer)")
    ap.add_argument("--tiny", action="store_true", help="small random model on CPU; needs --model for the tokenizer directory only")
    ap.add_argument("--data", help="train/make_edit_data.py output directory (not needed with --format none)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", action="append", default=[], help="edit_file and/or lines (default edit_file); none = only the replay data")
    ap.add_argument("--replay-dir", help="train/build_replay_data.py output: turns_train.jsonl is added to the training set, turns_holdout.jsonl gives validation turns")
    ap.add_argument("--replay-edit-repeat", type=int, default=1, help="repeat each successful edit turn of the replay data this many times")
    ap.add_argument("--replay-max", type=int, default=0, help="use at most this many replay turns (0 = all)")
    ap.add_argument("--replay-max-len", type=int, default=12000)
    ap.add_argument("--thought", choices=["empty", "copy", "none"], default="empty",
                    help="thought channel of per-turn samples; `none` (format T0) is what --format onepass trains, where the thought is already absent from the rendered text")
    ap.add_argument("--val-share", type=float, default=0.05, help="share of repositories held out for validation (--format onepass)")
    ap.add_argument("--probe-lens", help="comma-separated sequence lengths: measure peak memory and tokens/s of a forward+backward pass at each (needs --format onepass), then exit")
    ap.add_argument("--checkpoint", choices=["hf", "layers", "none"], default="hf",
                    help="activation checkpointing: hf = gradient_checkpointing_enable (saves nothing on E4B per a competitor's measurement), "
                         "layers = wrap every decoder layer by hand (use for sequences of 8,000+ tokens), none")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=16, help="samples per optimizer step")
    ap.add_argument("--max-samples", type=int, default=4000, help="training samples used (0 = all)")
    ap.add_argument("--val-samples", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (smoke tests)")
    ap.add_argument("--max-minutes", type=float, default=0, help="save and stop after this wall-clock time (0 = no limit)")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--lr-floor", type=float, default=0.0, help="the cosine ends at this fraction of --lr (the plan uses 0.1; 0 = decay to zero as in edit_v1)")
    ap.add_argument("--keep-checkpoints", type=int, default=0, help="also keep the adapter of the last N saves in <out>/ckpt-<step> (0 = only the latest, as before)")
    ap.add_argument("--val-at-save", action="store_true", help="measure the validation loss at every save and list it in train_report.json (gate G6)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    formats = a.format or ["edit_file"]
    onepass = "onepass" in formats                                   # D:/data/s2/.../sft.jsonl from train/s2_render.py, found in --data
    formats = [f for f in formats if f != "onepass"] or ["none"]
    if onepass and formats != ["none"]:
        ap.error("--format onepass reads sft.jsonl from --data; mix per-turn data in with --replay-dir, not with another --format")
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    tok = AutoTokenizer.from_pretrained(a.model)
    train, drop_t = build_examples(tok, a.data, "train", formats, a.thought, a.max_len, a.max_samples, a.seed)
    val, drop_v = build_examples(tok, a.data, "validation", formats, a.thought, a.max_len, a.val_samples, a.seed)
    if onepass:
        op, opv, n_long = load_onepass(f"{a.data}/sft.jsonl", a.max_len, a.max_samples, a.seed, a.val_share)
        print(f"onepass: {len(op)} training and {len(opv)} validation sequences ({n_long} over --max-len)", flush=True)
        train, val = train + op, val + (opv[: a.val_samples] if a.val_samples else opv)
        random.Random(a.seed).shuffle(train)
    if a.replay_dir:
        rp = load_replay(f"{a.replay_dir}/turns_train.jsonl", a.replay_max, a.replay_edit_repeat, a.seed, a.replay_max_len)
        print(f"replay: {len(rp)} turns added", flush=True)
        train = train + rp
        random.Random(a.seed).shuffle(train)
        val = val + load_replay(f"{a.replay_dir}/turns_holdout.jsonl", a.val_samples, 1, a.seed, a.replay_max_len)
    print(f"train {len(train)} (dropped {drop_t}), validation {len(val)} (dropped {drop_v}), "
          f"loss tokens/sample {sum(n_targets(e) for e in train) / max(len(train), 1):.0f}", flush=True)

    t0 = time.time()
    if a.tiny:
        os.environ["TINY_CONFIG"] = a.model
        model, device = tiny_model(), "cpu"
    else:
        from transformers import AutoModelForImageTextToText
        # No attn_implementation argument: forcing "sdpa" on the whole model fails because E4B's audio tower does not support it
        # (the 31B model has no audio tower, which is why the teammates' trainer could force it). The default picks per sub-model.
        model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
        device = "cuda"
        print(f"loaded in {time.time() - t0:.0f}s; {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    model.config.use_cache = False
    if a.checkpoint == "hf" and not a.tiny:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    start_step = 0
    if a.resume and (out / "adapter_model.safetensors").exists() and (out / "state.json").exists():
        model = PeftModel.from_pretrained(model, str(out), is_trainable=True)
        start_step = json.load(open(out / "state.json"))["step"]
        print(f"resumed adapter at step {start_step}", flush=True)
    else:
        cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none", target_modules=TARGET_RE, task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    if a.checkpoint == "layers":
        print(f"activation checkpointing by hand on {checkpoint_layers(model)} decoder layers", flush=True)
    model.train()
    if a.probe_lens:
        return probe(model, train, [int(x) for x in a.probe_lens.split(",")], device)

    base_val = mean_loss(model, val, device) if val and not start_step else None
    if base_val is not None:
        print(f"validation loss before training: {base_val:.4f}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    opt_file = out / "state" / "optimizer.pt"          # outside the adapter files, so it is never packed into a submission
    if start_step and opt_file.exists():
        opt.load_state_dict(torch.load(opt_file, map_location=device, weights_only=True))      # plain tensors and numbers: nothing needs unpickling
        print("optimizer state restored", flush=True)
    total = max(1, math.ceil(len(train) * a.epochs / a.accum))
    if a.max_steps:
        total = min(total, a.max_steps)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total, warm, a.lr_floor))
    for _ in range(start_step):
        sched.step()
    order = []
    while len(order) < total * a.accum:
        idx = list(range(len(train)))
        random.Random(a.seed + len(order)).shuffle(idx)
        order += idx

    val_hist, saved = [], [None]

    def checkpoint(step):
        model.save_pretrained(str(out))
        json.dump({"step": step, "total": total}, open(out / "state.json", "w"))
        opt_file.parent.mkdir(exist_ok=True)
        torch.save(opt.state_dict(), opt_file)
        if a.keep_checkpoints:                     # plan S3.2: keep the last few so a later step can be dropped if the offline gates (S4) fail
            model.save_pretrained(str(out / f"ckpt-{step:05d}"))
            for old in sorted(out.glob("ckpt-*"))[:-a.keep_checkpoints]:
                shutil.rmtree(old, ignore_errors=True)
        if a.val_at_save and val:                  # gate G6: the chosen checkpoint's validation loss against the best one
            vl = mean_loss(model, val, device)
            val_hist.append({"step": step, "val_loss": vl})
            print(f"step {step}: validation loss {vl:.4f}", flush=True)
        saved[0] = step

    t0, run, run_n, step, seen = time.time(), 0.0, 0, start_step, 0
    while step < total:
        batch = order[step * a.accum:(step + 1) * a.accum]
        n_tok = sum(n_targets(train[i]) for i in batch)
        for i in batch:
            loss, g = example_loss(model, train[i], device)
            (loss / n_tok).backward()
            run, run_n = run + loss.item(), run_n + g
            seen += len(train[i]["input_ids"]) if "input_ids" in train[i] else len(train[i]["prompt_ids"]) + len(train[i]["gen_ids"])
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 5 == 0 or step == total:
            el = time.time() - t0
            mem = f"  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if device == "cuda" else ""
            print(f"step {step}/{total}  train loss {run / run_n:.4f}  lr {sched.get_last_lr()[0]:.2e}  {el / 60:.1f} min  "
                  f"eta {el / (step - start_step) * (total - step) / 60:.0f} min  {seen / max(el, 1e-9):.0f} tok/s{mem}", flush=True)
            run, run_n = 0.0, 0
        if step % a.save_every == 0:
            checkpoint(step)
        if a.max_minutes and (time.time() - t0) / 60 > a.max_minutes:
            print(f"stopping at step {step}: --max-minutes reached", flush=True)
            break
    if saved[0] != step:
        checkpoint(step)
    final_val = val_hist[-1]["val_loss"] if val_hist and val_hist[-1]["step"] == step else (mean_loss(model, val, device) if val else None)
    if final_val is not None:
        print(f"validation loss after training: {final_val:.4f}" + (f" (before {base_val:.4f})" if base_val is not None else ""), flush=True)
    rep = audit(str(out))
    json.dump({"steps": step, "total_steps": total, "val_loss_before": base_val, "val_loss_after": final_val, "val_history": val_hist, "audit": rep,
               "args": vars(a), "train_samples": len(train)}, open(out / "train_report.json", "w"), indent=1)
    return 0 if rep["n_bad_keys"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
