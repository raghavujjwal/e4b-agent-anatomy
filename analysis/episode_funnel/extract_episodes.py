"""Dump every tool call of the corrected-harness control runs with its response, in episode order, to one JSON file.
Per call: chain, turn, index inside the generation, tool, parsed arguments, the response (status, error text, and for read_file the window
start/end/total lines and its text), the token count of the generation, and the first 2,500 characters of the generation for edit calls.
No backslash is typed in this file (BS = chr(92))."""
import os
import collections, json, sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "train"))
import build_replay_data as B
import s4_gates as G
import served

t0 = time.time()
tok = served.load_tokenizer()
print("tokenizer loaded in %.0f s" % (time.time() - t0), flush=True)
Q = chr(34)
VALUE_OPEN, VALUE_CLOSE = 'value:<|' + Q + '|>', '<|' + Q + '|>'
OUT = os.environ.get("WORK_DIR", "work") + "/episodes.json"


def body_json(body):
    if body.startswith(VALUE_OPEN):
        body = body[len(VALUE_OPEN):]
    if body.endswith(VALUE_CLOSE):
        body = body[:-len(VALUE_CLOSE)]
    return json.loads(body)


def short_args(name, args):
    if not isinstance(args, dict):
        return None
    if name == "run_command":
        return {"command": str(args.get("command", ""))[:600]}
    if name == "write_file":
        return {"filepath": args.get("filepath"), "content_len": len(str(args.get("content", "")))}
    return args


def response_of(block):
    name, body = block.groups()
    out = {"tool": name}
    try:
        d = body_json(body)
    except Exception:
        out["raw"] = body[:300]
        out["status"] = "rejected" if body.startswith(B.REJECTED_PREFIX) else "unparsed"
        return out
    out["status"] = d.get("status")
    if d.get("status") != "ok":
        out["error_type"] = d.get("error_type")
        out["error_message"] = str(d.get("error_message", ""))[:400]
    elif name == "read_file":
        out.update(start_line=d.get("start_line"), end_line=d.get("end_line"), total_lines=d.get("total_lines"),
                   is_truncated=d.get("is_truncated"), content=d.get("content", ""))
    elif name == "edit_file":
        out.update(occurrences=d.get("occurrences"), strategy=d.get("strategy"))
    elif name == "run_command":
        out["stdout"] = str(d.get("stdout", ""))[:300]
    return out


results = {}
for run in ("ctl_v2rec1", "ctl_v2rec2"):
    for line in open(os.environ.get("CTL_DIR", "data/ctl_files") + "/" + run + ".jsonl", encoding="utf-8"):
        r = json.loads(line)
        results[(run, r["instance_id"])] = {k: r.get(k) for k in ("resolved", "outcome_class", "patch_bytes", "edited_files", "stop_reason", "tool_calls", "required_source")}

episodes = []
for run in ("ctl_v2rec1", "ctl_v2rec2"):
    for line in open(os.environ.get("CTL_DIR", "data/ctl_files") + "/" + run + "_trace.jsonl", encoding="utf-8"):
        tr = json.loads(line)
        iid = tr.get("instance_id")
        calls = []
        for ci, turns in enumerate(B.chains_of(tr)):
            for k in range(len(turns)):
                gen = tok.decode(turns[k][1], skip_special_tokens=False)
                matches = list(G.CALL_BLOCK.finditer(gen))
                n_open = gen.count("<|tool_call>")
                if n_open == 0:
                    continue
                blocks = []
                if k + 1 < len(turns):
                    blocks = list(B.RESP_BLOCK.finditer(tok.decode(turns[k + 1][4], skip_special_tokens=False)))
                for i in range(n_open):
                    rec = {"chain": ci, "turn": k, "i": i, "n_calls_in_gen": n_open, "gen_tokens": len(turns[k][1])}
                    if i < len(matches):
                        name, body = matches[i].groups()
                        rec["tool"] = name
                        rec["args"] = short_args(name, G.parse_args(body))
                    else:
                        rec["tool"] = "?"
                        rec["args"] = None
                    if len(blocks) >= n_open:
                        resp = response_of(blocks[len(blocks) - n_open + i])
                        if resp["tool"] == rec["tool"]:
                            rec["resp"] = resp
                    if rec["tool"] == "edit_file":
                        rec["gen"] = gen[:2500]
                    calls.append(rec)
        episodes.append({"run": run, "iid": iid, "result": results.get((run, iid)), "calls": calls})
        print(run, iid, len(calls), "calls; elapsed %.0f s" % (time.time() - t0), flush=True)
json.dump(episodes, open(OUT, "w", encoding="utf-8"))
print("wrote", OUT, len(episodes), "episodes,", sum(len(e["calls"]) for e in episodes), "calls;",
      dict(collections.Counter(c["tool"] for e in episodes for c in e["calls"])), flush=True)
