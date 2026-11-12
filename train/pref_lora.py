"""LoRA preference training on the on-policy edit pairs of train/edit_pairs.py: DPO, RFT and DPO + NLL on the same pairs.

  python train/pref_lora.py --model /content/models/gemma-4-e4b-it-qat-q4_0-unquantized --pairs /content/pairs/pairs.jsonl --loss dpo --out /content/drive/MyDrive/gemma4-adapters/pair_dpo
  python train/pref_lora.py --tiny --model <dir with config.json> --pairs pairs.jsonl --loss dpo --out /tmp/tiny      # CPU smoke test with a small random model

A pair is a prompt (state plus the thought both calls share) and two calls. Only the call tokens carry loss; the prompt, thought included, is context.
  dpo      -log sigmoid(beta * ((log pi(c) - log ref(c)) - (log pi(r) - log ref(r)))). The tokens the two calls share cancel, so nothing in this loss pushes the model
           toward `edit_file` over `read_file`; it moves only where the calls differ. The reference is the model with the adapter switched off.
  rft      cross entropy on the chosen calls only (rejection-sampling fine-tuning on the same data; no reference, no rejected call).
  dpo_nll  dpo + nll_weight * (the chosen call's mean token cross entropy), the usual guard against both likelihoods falling together.
Everything else follows train/train_lora.py (LoRA on the language model's attention and MLP projections, rank 32 / alpha 64, hand-wrapped layer checkpointing, cosine to a
floor, adapter audit against vLLM's key pattern, checkpoints and resume). docs/PLAN_EDIT_PAIRS.md fixes the settings of each arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_lora as T  # noqa: E402

LOSSES = ("dpo", "rft", "dpo_nll")
STOP = "<|tool_response>"          # what the model stops on after a call (eos id 50 of generation_config.json): the server leaves it out of the text, the SFT data ends with it (train/render.py)
CALL_CLOSE = "<tool_call|>"


def call_ids(tok, text: str) -> list[int]:
    """Token ids of a call as the model wrote it: the text, and for a call that closes, the stop token it ended on. (A call that ran into the length limit has no stop.)"""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if text.endswith(CALL_CLOSE):
        stop = tok(STOP, add_special_tokens=False)["input_ids"]
        if len(stop) != 1:
            raise ValueError(f"{STOP!r} is not one token for this tokenizer: {stop}")
        ids = ids + stop
    return ids


def pair_loss(kind: str, pc, pr, rc, rr, len_c: int, beta: float = 0.1, nll_weight: float = 1.0):
    """Loss of one pair from the summed log-probabilities of the chosen (c) and rejected (r) call under the policy (p) and the reference (r).
    Returns (loss, margin) where margin = (pc - rc) - (pr - rr), unscaled."""
    margin = (pc - rc) - (pr - rr)
    if kind == "dpo":
        return -F.logsigmoid(beta * margin), margin
    if kind == "dpo_nll":
        return -F.logsigmoid(beta * margin) + nll_weight * (-pc / len_c), margin
    raise ValueError(f"no per-pair loss for {kind!r}")


def call_logp(model, prompt_ids: list[int], call_ids: list[int], device):
    """Summed log-probability of the call tokens given the prompt. Logits are computed only where there is a target (the output layer is 262,144 wide)."""
    ids = torch.tensor([prompt_ids + call_ids], device=device)
    g = len(call_ids)
    logits = model(input_ids=ids, mm_token_type_ids=torch.zeros_like(ids), logits_to_keep=g + 1, use_cache=False).logits[0, :-1].float()
    tgt = torch.tensor(call_ids, device=logits.device)
    return -F.cross_entropy(logits, tgt, reduction="sum")


def load_pairs(path, tok, max_len: int, limit: int, seed: int, val_share: float):
    """(train, validation) pairs with token ids; validation is chosen by commit (a hash of `source`), never by pair, so no prompt is on both sides."""
    train, val, dropped = [], [], 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        p = json.loads(line)
        c, r = call_ids(tok, p["chosen"]), call_ids(tok, p["rejected"])
        if len(p["prompt_ids"]) + max(len(c), len(r)) > max_len:
            dropped += 1
            continue
        ex = {"id": p["id"], "prompt_ids": p["prompt_ids"], "chosen_ids": c, "rejected_ids": r, "round": p.get("round", 1)}
        h = int(hashlib.sha1(p["source"].encode()).hexdigest(), 16) % 1000
        (val if h < val_share * 1000 else train).append(ex)
    random.Random(seed).shuffle(train)
    return (train[:limit] if limit else train), val, dropped


@torch.no_grad()
def evaluate(model, pairs, kind: str, beta: float, device) -> dict:
    """Held-out pairs: how often the implicit reward prefers the chosen call, the mean margin, and the chosen call's mean token cross entropy."""
    model.eval()
    acc = marg = nll = n_tok = 0.0
    for ex in pairs:
        pc = call_logp(model, ex["prompt_ids"], ex["chosen_ids"], device).item()
        if kind == "rft":
            nll, n_tok = nll - pc, n_tok + len(ex["chosen_ids"])
            continue
        pr = call_logp(model, ex["prompt_ids"], ex["rejected_ids"], device).item()
        with model.disable_adapter():
            rc = call_logp(model, ex["prompt_ids"], ex["chosen_ids"], device).item()
            rr = call_logp(model, ex["prompt_ids"], ex["rejected_ids"], device).item()
        m = (pc - rc) - (pr - rr)
        acc, marg, nll, n_tok = acc + (m > 0), marg + m, nll - pc, n_tok + len(ex["chosen_ids"])
    model.train()
    n = max(len(pairs), 1)
    return {"n": len(pairs), "reward_accuracy": acc / n if kind != "rft" else None, "mean_margin": marg / n if kind != "rft" else None, "chosen_nll_per_token": nll / max(n_tok, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True, help="directory of the unquantized checkpoint (weights and tokenizer); with --tiny only its config and tokenizer are read")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--pairs", required=True, help="pairs.jsonl of train/edit_pairs.py")
    ap.add_argument("--loss", choices=LOSSES, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--nll-weight", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=None, help="default: 5e-5 for dpo and dpo_nll, 1e-4 for rft (the SFT setting)")
    ap.add_argument("--lr-floor", type=float, default=0.1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=8, help="pairs per optimizer step")
    ap.add_argument("--max-pairs", type=int, default=1600, help="training pairs used (0 = all); the same for every arm")
    ap.add_argument("--val-share", type=float, default=0.05, help="share of commits held out")
    ap.add_argument("--val-max", type=int, default=60)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (smoke tests)")
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--keep-checkpoints", type=int, default=3)
    ap.add_argument("--checkpoint", choices=["hf", "layers", "none"], default="layers")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    lr = a.lr if a.lr is not None else (1e-4 if a.loss == "rft" else 5e-5)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    tok = AutoTokenizer.from_pretrained(a.model)
    train, val, dropped = load_pairs(a.pairs, tok, a.max_len, a.max_pairs, a.seed, a.val_share)
    val = val[:a.val_max]
    print(f"loss {a.loss}: {len(train)} training pairs, {len(val)} validation pairs ({dropped} over --max-len); lr {lr:g}, beta {a.beta}", flush=True)

    t0 = time.time()
    if a.tiny:
        os.environ["TINY_CONFIG"] = a.model
        model, device = T.tiny_model(), "cpu"
    else:
        from transformers import AutoModelForImageTextToText
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
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none", target_modules=T.TARGET_RE, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    if a.checkpoint == "layers":
        print(f"activation checkpointing by hand on {T.checkpoint_layers(model)} decoder layers", flush=True)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.99))
    opt_file = out / "state" / "optimizer.pt"
    if start_step and opt_file.exists():
        opt.load_state_dict(torch.load(opt_file, map_location=device, weights_only=True))
    total = max(1, math.ceil(len(train) * a.epochs / a.accum))
    if a.max_steps:
        total = min(total, a.max_steps)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: T.lr_factor(s, total, warm, a.lr_floor))
    for _ in range(start_step):
        sched.step()
    order = []
    while len(order) < total * a.accum:
        idx = list(range(len(train)))
        random.Random(a.seed + len(order)).shuffle(idx)
        order += idx

    hist, saved = [], [None]
    metrics = open(out / "metrics.jsonl", "a", encoding="utf-8")

    def checkpoint(step):
        model.save_pretrained(str(out))
        json.dump({"step": step, "total": total}, open(out / "state.json", "w"))
        opt_file.parent.mkdir(exist_ok=True)
        torch.save(opt.state_dict(), opt_file)
        if a.keep_checkpoints:
            model.save_pretrained(str(out / f"ckpt-{step:05d}"))
            for old in sorted(out.glob("ckpt-*"))[:-a.keep_checkpoints]:
                shutil.rmtree(old, ignore_errors=True)
        if val:
            ev = evaluate(model, val, a.loss, a.beta, device)
            hist.append({"step": step, **ev})
            print(f"step {step}: validation {json.dumps(ev)}", flush=True)
        saved[0] = step

    run = {"loss": 0.0, "acc": 0.0, "margin": 0.0, "dc": 0.0, "dr": 0.0, "n": 0, "tok": 0}
    t0, step = time.time(), start_step
    while step < total:
        batch = [train[i] for i in order[step * a.accum:(step + 1) * a.accum]]
        n_tok = sum(len(ex["chosen_ids"]) for ex in batch)
        for ex in batch:
            if a.loss == "rft":
                loss = -call_logp(model, ex["prompt_ids"], ex["chosen_ids"], device)
                (loss / n_tok).backward()
                run["loss"], run["tok"] = run["loss"] + loss.item(), run["tok"] + len(ex["chosen_ids"])
            else:
                with torch.no_grad(), model.disable_adapter():
                    rc = call_logp(model, ex["prompt_ids"], ex["chosen_ids"], device)
                    rr = call_logp(model, ex["prompt_ids"], ex["rejected_ids"], device)
                pc = call_logp(model, ex["prompt_ids"], ex["chosen_ids"], device)
                pr = call_logp(model, ex["prompt_ids"], ex["rejected_ids"], device)
                loss, margin = pair_loss(a.loss, pc, pr, rc, rr, len(ex["chosen_ids"]), a.beta, a.nll_weight)
                (loss / len(batch)).backward()
                run["loss"] += loss.item()
                run["acc"] += float(margin.item() > 0)
                run["margin"] += margin.item()
                run["dc"] += (pc - rc).item()
                run["dr"] += (pr - rr).item()
            run["n"] += 1
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        n = max(run["n"], 1)
        rec = {"step": step, "loss": run["loss"] / (max(run["tok"], 1) if a.loss == "rft" else n), "lr": sched.get_last_lr()[0], "grad_norm": float(gn)}
        if a.loss != "rft":
            rec.update(reward_accuracy=run["acc"] / n, margin=run["margin"] / n, chosen_logratio=run["dc"] / n, rejected_logratio=run["dr"] / n)
        metrics.write(json.dumps(rec) + "\n")
        metrics.flush()
        if step % 5 == 0 or step == total:
            el = time.time() - t0
            print(f"step {step}/{total}  " + "  ".join(f"{k} {v:.4g}" for k, v in rec.items() if k != "step")
                  + f"  {el:.0f}s ({el / (step - start_step):.1f} s/step)  eta {el / (step - start_step) * (total - step) / 60:.0f} min", flush=True)
        run = {k: 0.0 for k in run}
        run["n"], run["tok"] = 0, 0
        if step % a.save_every == 0:
            checkpoint(step)
        if a.max_minutes and (time.time() - t0) / 60 > a.max_minutes:
            print(f"stopping at step {step}: --max-minutes reached", flush=True)
            break
    if saved[0] != step:
        checkpoint(step)
    rep = T.audit(str(out))
    json.dump({"loss": a.loss, "steps": step, "total_steps": total, "lr": lr, "beta": a.beta, "validation": hist, "audit": rep, "args": vars(a), "train_pairs": len(train)},
              open(out / "train_report.json", "w"), indent=1)
    return 0 if rep["n_bad_keys"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
