"""Does E4B, shown the tool's error, see what was wrong with its own failed edit?  (GPU; the first step of testing SDPO, arXiv 2601.20802.)

  python train/sdpo_probe.py --probe D:/data/sdpo_probe/probe.jsonl --model <unquantized checkpoint dir> --out D:/data/sdpo_probe/result

Each sample of train/build_sdpo_probe.py is scored twice with the SAME model (no training, no adapter):
  student  log p(gen | prompt)                                   what the agent saw when it acted
  teacher  log p(gen | prompt + a user turn with the feedback)   what it would think of the same output after seeing the tool's answer
The advantage of token t is A_t = log p_teacher(y_t) - log p_student(y_t) (SDPO's sampled-token form): negative = the teacher finds that token less likely.
The teacher prompt closes the model turn, adds one user turn "The following is the result of your previous action: <feedback> ... choose the action again" and opens
a new model turn; the generation then follows unchanged. A control repeats this with the feedback of a DIFFERENT failed edit (fb_mismatch).

Pre-registered verdict (fixed before the first run; the numbers are chosen so that a teacher that has not read its error cannot pass):
  A  credit assignment, kind notfound (needs >= 30 samples): d = mean A over the tokens known to be wrong - mean A over the other old_string tokens, per sample.
       PASS if d < 0 in at least 65% of samples with the matching feedback AND in at most 55% with the mismatched feedback.
  B  loop suppression, kind repeat_fail (needs >= 20): s = sum of A over the tool-call tokens of the repeated call.
       PASS if median s <= -2.0 nats with the matching feedback AND the median s of ok_edit samples is within +-1.0 (the teacher is not just pushing everything down).
  PASS = A and B; PARTIAL = one of them; FAIL = neither; INCONCLUSIVE = too few samples for A or B.
Reading: PASS means the in-context ability the paper relies on exists in E4B and SDPO-style training on recorded turns is worth building (~10-12 Colab units); FAIL means it
does not (the paper found the same for models below about 4B) and the adapter route should be dropped.
"""
from __future__ import annotations
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PREAMBLE = "The following is the result of your previous action:\n"
POSTAMBLE = "\n\nNow choose the action again."
A_SHARE_MATCHED, A_SHARE_MISMATCHED, B_MEDIAN, B_OK_BAND = 0.65, 0.55, -2.0, 1.0
A_MIN, B_MIN = 30, 20


def teacher_ids(tok, prompt_ids, feedback):
    """prompt, then: end the model turn, a user turn with the feedback, a new model turn (the generation is appended by the caller)."""
    extra = tok(f"<turn|>\n<|turn>user\n{PREAMBLE}{feedback}{POSTAMBLE}<turn|>\n<|turn>model\n", add_special_tokens=False)["input_ids"]
    return prompt_ids + extra


def gen_logprobs(model, torch, prefix_ids, gen_ids):
    ids = torch.tensor([prefix_ids + gen_ids], device=model.device)
    g = len(gen_ids)
    logits = model(input_ids=ids, mm_token_type_ids=torch.zeros_like(ids), logits_to_keep=g + 1).logits[0, :-1].float()
    lp = torch.log_softmax(logits, dim=-1)
    tgt = torch.tensor(gen_ids, device=lp.device)
    return lp.gather(1, tgt[:, None])[:, 0].cpu().tolist()


def stats_of(adv, sample):
    s, e = sample["call_tok"]
    row = {"id": sample["id"], "kind": sample["kind"], "n_gen": len(adv), "call_sum": sum(adv[s:e]), "all_mean": sum(adv) / len(adv)}
    if sample.get("wrong_tok"):
        w = [adv[i] for i in sample["wrong_tok"]]
        r = [adv[i] for i in sample["right_tok"]]
        row["d"] = sum(w) / len(w) - sum(r) / len(r)
        row["wrong_mean"], row["right_mean"] = sum(w) / len(w), sum(r) / len(r)
    return row


def median(xs):
    return statistics.median(xs) if xs else None


def summarize(rows):
    """rows: [{kind, matched: {call_sum, d?}, mismatched: {...}|None}] -> (report dict, verdict string)."""
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    rep = {}
    nf = [r for r in by["notfound"] if "d" in r["matched"] and r.get("mismatched") and "d" in r["mismatched"]]
    if nf:
        rep["A"] = {"n": len(nf), "share_d_neg_matched": sum(r["matched"]["d"] < 0 for r in nf) / len(nf),
                    "share_d_neg_mismatched": sum(r["mismatched"]["d"] < 0 for r in nf) / len(nf),
                    "median_d_matched": median([r["matched"]["d"] for r in nf]), "median_d_mismatched": median([r["mismatched"]["d"] for r in nf]),
                    "median_wrong_token_adv_matched": median([r["matched"]["wrong_mean"] for r in nf]),
                    "median_right_token_adv_matched": median([r["matched"]["right_mean"] for r in nf])}
    for kind in ("repeat_fail", "first_fail", "ok_edit"):
        rs = by[kind]
        if rs:
            rep[kind] = {"n": len(rs), "median_call_sum_matched": median([r["matched"]["call_sum"] for r in rs]),
                         "median_call_sum_mismatched": median([r["mismatched"]["call_sum"] for r in rs if r.get("mismatched")]) if any(r.get("mismatched") for r in rs) else None}
    a_ok = b_ok = None
    if "A" in rep and rep["A"]["n"] >= A_MIN:
        a_ok = rep["A"]["share_d_neg_matched"] >= A_SHARE_MATCHED and rep["A"]["share_d_neg_mismatched"] <= A_SHARE_MISMATCHED
    if "repeat_fail" in rep and rep["repeat_fail"]["n"] >= B_MIN and "ok_edit" in rep:
        b_ok = rep["repeat_fail"]["median_call_sum_matched"] <= B_MEDIAN and abs(rep["ok_edit"]["median_call_sum_matched"]) <= B_OK_BAND
    rep["A_pass"], rep["B_pass"] = a_ok, b_ok
    if a_ok is None or b_ok is None:
        missing = [n for n, v in (("A", a_ok), ("B", b_ok)) if v is None]
        return rep, "INCONCLUSIVE (too few samples for " + " and ".join(missing) + ")"
    if a_ok and b_ok:
        return rep, "PASS"
    if a_ok or b_ok:
        return rep, "PARTIAL (" + ("A" if a_ok else "B") + " only)"
    return rep, "FAIL"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--model", required=True, help="the unquantized E4B checkpoint (bf16), also used for the tokenizer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="score at most this many samples (0 = all)")
    a = ap.parse_args(argv)
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    # No attn_implementation argument: forcing "sdpa" on the whole model fails because E4B's audio tower does not support it
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    samples = [json.loads(l) for l in open(a.probe, encoding="utf-8")]
    if a.limit:
        samples = samples[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad(), open(out / "rows.jsonl", "w", encoding="utf-8") as fh:
        for i, s in enumerate(samples):
            stu = gen_logprobs(model, torch, s["prompt_ids"], s["gen_ids"])
            tea = gen_logprobs(model, torch, teacher_ids(tok, s["prompt_ids"], s["fb"]), s["gen_ids"])
            row = {"kind": s["kind"], "matched": stats_of([t - u for t, u in zip(tea, stu)], s), "mismatched": None}
            if s.get("fb_mismatch"):
                mis = gen_logprobs(model, torch, teacher_ids(tok, s["prompt_ids"], s["fb_mismatch"]), s["gen_ids"])
                row["mismatched"] = stats_of([t - u for t, u in zip(mis, stu)], s)
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            if i % 25 == 0:
                print(f"{i + 1}/{len(samples)} {s['kind']}", flush=True)
    rep, verdict = summarize(rows)
    print(json.dumps(rep, indent=1))
    print("VERDICT:", verdict)
    json.dump({"report": rep, "verdict": verdict}, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
