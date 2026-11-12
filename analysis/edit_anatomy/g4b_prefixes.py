"""G4b prefixes: how many failed-edit prefixes show no applied edit in the prompt? Analysis helper.
usage: python -I -X utf8 g4b_prefixes.py <check_failed_edit.jsonl> [--show]
"""
import collections
import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "train"))
import build_replay_data as B  # noqa: E402
import served  # noqa: E402

tok = served.load_tokenizer()
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
flags = [B.applied_edit_visible(tok, r["prompt_ids"]) for r in rows]
no = [r for r, f in zip(rows, flags) if not f]
print(f"{len(rows)} failed-edit prefixes from {len({r['instance_id'] for r in rows})} tasks; an applied edit is visible in {sum(flags)}, not visible in {len(no)} ({len({r['instance_id'] for r in no})} tasks)")
print("recorded next tool where NO applied edit is visible:", dict(collections.Counter(r["tool"] for r in no)))
print("recorded next tool where one IS visible:          ", dict(collections.Counter(r["tool"] for r, f in zip(rows, flags) if f)))
comp = [("<|turn>model" in tok.decode(r["prompt_ids"], skip_special_tokens=False)[:3000]) for r in rows]
print("prompts that look compacted (a model turn before any user turn):", sum(comp))
if "--show" in sys.argv:
    for want in (True, False):
        for r, f in zip(rows, flags):
            if f == want:
                t = tok.decode(r["prompt_ids"], skip_special_tokens=False)
                print(f"\n==== applied_edit_visible={want} | {r['instance_id']} turn {r['turn']} | prompt {len(r['prompt_ids'])} tokens | edit responses in prompt:")
                for m in B.RESP_BLOCK.finditer(t):
                    if m.group(1) in B.EDIT_TOOLS:
                        print("    ", m.group(1), m.group(2)[:110].replace("\n", " "))
                print("   ... prompt ends:", repr(t[-420:]))
                break
