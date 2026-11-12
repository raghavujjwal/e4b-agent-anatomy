"""train/edit_pairs.py: prompt spec, labels, the shared-thought sampler, pairs and the held-out evaluation.

The tokenizer is a stand-in (one id per character) wherever the real chat template is not the point; the tests that render with the real template skip when the
checkpoint's tokenizer is not on the machine. The server is a fake on localhost.
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
import edit_pairs as E  # noqa: E402

Q = '<|"|>'
WINDOW = "def a():\n    return 1\n\n\ndef b():\n    return 2\n\n\ndef c():\n    return 3\n"      # the lines: 0 def a, 1 return 1, 4 def b, 5 return 2, 8 def c, 9 return 3
GOLD = {"filepath": "m.py", "old_string": "    return 2", "new_string": "    return 22"}
EXPECTED = WINDOW.replace("    return 2", "    return 22")
GOLD_SPAN = E.changed_span(WINDOW, EXPECTED)


def call(**kw):
    kw.setdefault("filepath", "m.py")
    body = ",".join(f"{k}:{Q}{v}{Q}" if isinstance(v, str) else f"{k}:{str(v).lower()}" for k, v in sorted(kw.items()))
    return f"<|tool_call>call:edit_file{{{body}}}<tool_call|>"


def label(text):
    return E.label_sample(text, WINDOW, GOLD, EXPECTED, GOLD_SPAN)


# ---- the prompt spec ----------------------------------------------------------------------------------------------------------------
def test_the_spec_is_the_captured_request_with_three_holes(tmp_path):
    spec = E.export_spec(out=tmp_path / "spec.json")
    req = json.load(open(E.FIXTURE, encoding="utf-8"))["requests"][0]
    assert spec["system"] == req["messages"][0]["content"] and spec["tools"] == req["tools"] and spec["enable_thinking"] is True
    assert all(spec["user_template"].count(p) == 1 for p in ("{repo}", "{statement}", "{tree}"))
    # filling the holes with the captured task's own values gives the captured message back
    assert E.fill(spec["user_template"], repo="demo/repo", statement="foo() should return 2, not 1.", tree="./README.md\n./pkg\n./pkg/__init__.py\n./pkg/mod.py") == req["messages"][1]["content"]
    assert json.load(open(E.SPEC_FILE, encoding="utf-8")) == json.load(open(tmp_path / "spec.json", encoding="utf-8")), "train/prompt_spec_v2rec.json is stale: rerun export-spec"


def test_fill_does_not_rescan_what_it_inserted():
    assert E.fill("{repo}|{statement}|{tree}", repo="r", statement="{tree} {repo} {x}", tree="t") == "r|{tree} {repo} {x}|t"


def test_tree_of_lists_the_directories_down_to_the_file():
    assert E.tree_of("src/click/core.py") == "./README.md\n./src\n./src/click\n./src/click/core.py"
    assert E.tree_of("setup.py") == "./README.md\n./setup.py"
    assert E.tree_of("a/b/c/d/e.py").splitlines()[-1] == "./a/b/c"


# ---- spans and labels ---------------------------------------------------------------------------------------------------------------
def test_changed_span_and_near():
    assert E.changed_span("a\nb\nc", "a\nb\nc") is None
    assert E.changed_span("a\nb\nc\nd", "a\nX\nc\nd") == (1, 1)                    # one line replaced
    assert E.changed_span("a\nb\nc\nd", "a\nb\nY\nZ\nc\nd") == (2, 2)               # two lines inserted in front of c: the line they go before
    assert E.changed_span("a\nb\nc\nd", "a\nd") == (1, 2)                           # two lines removed
    assert E.near((5, 5), (5, 5)) and E.near((9, 9), (5, 5), margin=4) and not E.near((9, 9), (5, 5), margin=3)
    assert E.near((0, 6), (8, 8)) and not E.near(None, (1, 1)) and not E.near((1, 1), None)


def test_a_call_that_applies_where_the_gold_edit_goes_is_chosen_and_reproduces():
    s = label(call(old_string="    return 2", new_string="    return 22"))
    assert (s["status"], s["tier"], s["changed"], s["localized"], s["reproduces"]) == ("applied", "exact", True, True, True)
    assert E.is_chosen(s) and not E.is_rejected(s)


def test_a_call_that_applies_far_from_the_gold_edit_is_neither_chosen_nor_rejected():
    s = label(call(old_string="    return 3", new_string="    return 33"))
    assert s["status"] == "applied" and s["localized"] is False and not s["reproduces"]
    assert not E.is_chosen(s) and not E.is_rejected(s)
    near_ = label(call(old_string="def b():", new_string="def b(x=0):"))                 # line 4 against line 5: within the margin, and not the gold edit
    assert near_["localized"] is True and near_["reproduces"] is False and E.is_chosen(near_)


def test_a_whitespace_rescue_is_applied_but_not_at_the_exact_tier():
    s = label(call(old_string="def b():\n  return 2", new_string="def b():\n    return 22"))
    assert s["status"] == "applied" and s["tier"] == "flexible" and not E.is_chosen(s) and not E.is_rejected(s)


def test_what_the_tool_refuses_is_rejected():
    assert label(call(old_string="    return 5", new_string="x"))["status"] == "not_found"
    s = label(call(old_string="return", new_string="yield"))
    assert s["status"] == "ambiguous" and E.is_rejected(s)
    assert label(call(old_string="return", new_string="yield", allow_multiple=True))["status"] == "applied"        # asked for, so it applies
    assert label(call(old_string="    return 2", new_string="    return 2"))["status"] == "unchanged"
    assert label(call(old_string="", new_string="x"))["status"] == "empty_old"
    assert all(E.is_rejected(label(call(old_string=o, new_string=n))) for o, n in (("zzz", "x"), ("return", "y"), ("a", "a"), ("", "q")))


def test_other_files_other_tools_and_several_calls_draw_no_preference():
    assert label(call(filepath="other.py", old_string="    return 2", new_string="x"))["status"] == "wrong_file"
    assert label("<|tool_call>call:read_file{filepath:" + Q + "m.py" + Q + "}<tool_call|>")["status"] == "wrong_tool"
    assert label(call(old_string="    return 2", new_string="a") + call(old_string="    return 3", new_string="b"))["status"] == "multi_call"
    assert label("I think the fix is in place.")["status"] == "no_call"
    for text in (call(filepath="other.py", old_string="    return 2", new_string="x"), "<|tool_call>call:read_file{}<tool_call|>", "no call"):
        s = label(text)
        assert not E.is_chosen(s) and not E.is_rejected(s)


def test_a_call_the_template_cannot_read_or_that_lacks_an_argument_is_malformed():
    assert label('<|tool_call>call:edit_file{filepath:' + Q + 'm.py' + Q + ',old_string="    return 2",new_string="x"}<tool_call|>')["status"] == "malformed"
    assert label('<|tool_call>call:edit_file{filepath:' + Q + 'm.py' + Q + ',new_string:' + Q + 'x' + Q + '}<tool_call|>') == {"status": "malformed", "why": "missing old_string"}
    assert label('<|tool_call>call:edit_file{filepath:' + Q + 'm.py' + Q + ',old_string:' + Q + 'a')["status"] == "malformed"           # never closed
    assert E.is_rejected(label('<|tool_call>call:edit_file{old_string="x"}<tool_call|>'))


def test_other_unreadable_arguments_are_unparsed_and_draw_no_preference():
    # the server's tool parser ran 14 of the 37 calls of these styles in the control runs: not certain enough to call them rejected
    for body in (f"filepath:{Q}m.py{Q},new_string: \"x\",old_string:{Q}a{Q}",                        # JSON quotes after a space
                 f'filepath:{Q}m.py{Q},new_string:"""x""",old_string:{Q}a{Q}',                        # triple quotes
                 f"filepath:{Q}m.py{Q},new_string:None,old_string:{Q}a{Q}",                           # a bare literal
                 f"filepath:{Q}m.py{Q},new_string:{Q}x{Q},old_string:5"):                              # not a string
        s = label(f"<|tool_call>call:edit_file{{{body}}}<tool_call|>")
        assert s["status"] == "unparsed", body
        assert not E.is_chosen(s) and not E.is_rejected(s)


# ---- the harness's own error text, thoughts, call text -----------------------------------------------------------------------------
def test_tool_error_message_is_the_harnesss_own():
    assert E.tool_error_message(WINDOW, {"filepath": "m.py", "old_string": "zzz", "new_string": "x"}).startswith("Failed to replace: old_string not found.")
    assert E.tool_error_message(WINDOW, {"filepath": "m.py", "old_string": "return", "new_string": "y"}) == "Target string occurs 3 times in m.py. Please provide more surrounding context."
    with pytest.raises(ValueError):
        E.tool_error_message(WINDOW, {"filepath": "m.py", "old_string": "    return 2", "new_string": "x"})


def test_call_text_and_thought_of():
    t = "<|channel>thought\nline one\nline two\n<channel|>" + call(old_string="a", new_string="b") + "<|tool_response>"
    assert E.call_text(t) == call(old_string="a", new_string="b")
    assert E.call_text("no call here") == ""
    assert E.call_text("<|tool_call>call:edit_file{old_string:" + Q + "a<|tool_response>") == "<|tool_call>call:edit_file{old_string:" + Q + "a"
    assert E.thought_of(t[:t.index("<|tool_call>")]) == "line one\nline two\n"
    assert E.thought_of("") == "" and E.thought_of(call(old_string="a", new_string="b")) == ""


# ---- pairs --------------------------------------------------------------------------------------------------------------------------
class WordTok:
    """One id per whitespace-separated word."""
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}


def smp(status, text, **kw):
    base = {"status": status, "call": text, "args": {}}
    if status == "applied":
        base.update(tier="exact", changed=True, localized=True, reproduces=False)
    base.update(kw)
    return base


def state_row(samples, sid="s1", rnd=1):
    mixed = E.is_mixed(samples)
    return {"id": sid, "round": rnd, "repo": "r/r", "file": "m.py", "source": "x", "samples": samples, "mixed": mixed, "prompt_ids": [1, 2, 3] if mixed else None}


def test_pairs_follow_the_rules():
    rows = [state_row([smp("applied", "A1"), smp("applied", "A2"), smp("applied", "A2"), smp("applied", "A3"), smp("not_found", "N1"), smp("not_found", "N2"),
                       smp("malformed", "M1"), smp("ambiguous", "G1"), smp("applied", "far", localized=False), smp("applied", "flex", tier="flexible"), smp("wrong_tool", "W")]),
            state_row([smp("applied", "B1"), smp("applied", "B2")], sid="s2"),                                     # nothing rejected: no pair
            state_row([smp("not_found", "C1"), smp("not_found", "C2")], sid="s3"),                                 # nothing chosen: no pair
            state_row([smp("applied", "D1"), smp("not_found", "long call " * 50), smp("malformed", "D3")], sid="s4", rnd=2)]
    pairs, stats = E.build_pairs(rows, WordTok(), max_pairs=2, max_call_tokens=20, seed=0)
    by = {}
    for p in pairs:
        by.setdefault(p["state"], []).append(p)
    assert sorted(by) == ["s1", "s4"] and len(by["s1"]) == 2 and len(by["s4"]) == 1
    s1 = by["s1"]
    assert len({p["chosen"] for p in s1}) == 2 and {p["chosen"] for p in s1} <= {"A1", "A2", "A3"}            # distinct chosen calls, the duplicate A2 counted once
    assert len({p["rejected"] for p in s1}) == 2 and {p["rejected"] for p in s1} <= {"N1", "N2", "M1", "G1"}
    assert len({p["rejected_status"] for p in s1}) == 2, "the two rejected calls come from different statuses (round-robin), not both from the commonest one"
    assert by["s4"][0]["rejected"] == "D3" and by["s4"][0]["round"] == 2                                      # the over-long refused call was left out
    assert all(p["prompt_ids"] == [1, 2, 3] for p in pairs) and all(p["chosen"] not in ("far", "flex") for p in pairs)
    assert stats["pairs"] == 3 and stats["states_mixed"] == 2 and stats["pairs_by_round"] == {1: 2, 2: 1}
    assert stats["status_by_round"]["1"]["applied"] == 8 and stats["status_by_round"]["1"]["wrong_tool"] == 1


def test_unparsed_calls_are_negatives_only_when_asked_for():
    row = state_row([smp("applied", "A1"), smp("unparsed", "U1")])
    assert row["mixed"] is True and row["prompt_ids"] == [1, 2, 3]                # the sampler keeps the prompt of such a state ...
    assert E.build_pairs([row], WordTok())[0] == []                               # ... but it gives no pair unless asked to
    pairs, stats = E.build_pairs([row], WordTok(), unparsed_rejected=True)
    assert [(p["chosen"], p["rejected"], p["rejected_status"]) for p in pairs] == [("A1", "U1", "unparsed")] and stats["rejected_status"] == {"unparsed": 1}


def test_pairs_are_the_same_for_the_same_seed():
    rows = [state_row([smp("applied", f"A{i}") for i in range(5)] + [smp("not_found", f"N{i}") for i in range(5)])]
    a, _ = E.build_pairs(rows, WordTok(), seed=3)
    b, _ = E.build_pairs(rows, WordTok(), seed=3)
    assert a == b


def eval_row(good, reproduces=7.0, edit_call=70.0, malformed=1.0, diff=None, ci=None, ref="base"):
    row = {"good_%": good, "reproduces_%": reproduces, "edit_call_%": edit_call, "malformed_%": malformed}
    if diff is not None:
        row[f"good_vs_{ref}"] = {"n_prefixes": 300, "diff_points": diff, "ci95_points": ci}
    return row


def gates_row(**verdicts):
    return {"gates": {g: ([] if verdicts.get(g, "PASS") == "PASS" else ["G failed: x"] if verdicts[g] == "FAIL" else ["INSUFFICIENT: few"]) for g in E.GATES}}


def test_decide_applies_the_outcome_rules_of_the_plan():
    evals = {"base": eval_row(50.0),
             "win": eval_row(66.0, diff=16.0, ci=[9.0, 23.0]),                                    # primary met, nothing else wrong
             "tiny": eval_row(58.0, diff=8.0, ci=[2.0, 14.0]),                                   # below the 10-point line
             "noisy": eval_row(66.0, diff=16.0, ci=[-1.0, 31.0]),                                # the interval reaches 0
             "eager": eval_row(80.0, edit_call=85.0, diff=30.0, ci=[20.0, 40.0]),                # edits at 15 points more of the steps: a guard
             "forgetful": eval_row(66.0, reproduces=4.0, diff=16.0, ci=[9.0, 23.0]),             # reproduces falls 3 points
             "sloppy": eval_row(66.0, malformed=1.5, diff=16.0, ci=[9.0, 23.0]),
             "gated": eval_row(66.0, diff=16.0, ci=[9.0, 23.0]),
             "unchecked": eval_row(66.0, diff=16.0, ci=[9.0, 23.0])}
    gates = {"win": gates_row(), "tiny": gates_row(), "noisy": gates_row(), "eager": gates_row(), "forgetful": gates_row(), "sloppy": gates_row(),
             "gated": gates_row(G3="FAIL", G4b="INSUFFICIENT")}
    r = E.decide(evals, gates, "base")
    assert r["win"]["outcome"] == "primary met, guards and gates pass" and r["win"]["agent_run_allowed"] is True
    assert r["tiny"]["outcome"] == "primary not met" and r["noisy"]["outcome"] == "primary not met"
    assert r["tiny"]["agent_run_allowed"] is False
    assert r["eager"]["outcome"] == "primary met, but guard edit_call" and r["forgetful"]["outcome"] == "primary met, but guard reproduces"
    assert r["sloppy"]["outcome"] == "primary met, but guard malformed"
    assert r["gated"]["outcome"] == "primary met, but gate G3 FAIL, gate G4b INSUFFICIENT" and r["gated"]["agent_run_allowed"] is False
    assert r["unchecked"]["outcome"] == "primary met, but gate G3 missing, gate G4 missing, gate G4b missing, gate G5 missing", "no gate result is not a pass"
    assert "base" not in r


def test_decide_reads_the_files_and_prints_one_line_per_arm(tmp_path, capsys):
    evals = {"base": eval_row(50.0), "arm": eval_row(66.0, diff=16.0, ci=[9.0, 23.0])}
    (tmp_path / "e.json").write_text(json.dumps(evals), encoding="utf-8")
    (tmp_path / "g.json").write_text(json.dumps({"arm": gates_row()}), encoding="utf-8")
    assert E.main(["decide", "--eval", str(tmp_path / "e.json"), "--gates", str(tmp_path / "g.json"), "--ref", "base", "--out", str(tmp_path / "d.json")]) == 0
    assert "primary met, guards and gates pass  (an agent-level run is allowed)" in capsys.readouterr().out
    assert json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))["arm"]["primary"] is True


def test_pack_data_holds_the_records_the_sampler_would_draw_and_nothing_else(tmp_path):
    import zipfile
    data = tmp_path / "edit_sft"
    data.mkdir()
    for split, n in (("train", 60), ("validation", 20)):
        recs = [{"id": f"{split}-{k}", "format": "edit_file" if k % 4 else "lines", "payload": "x" * 50} for k in range(n)]
        (data / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    pack = tmp_path / "edit_states.zip"
    assert E.main(["pack-data", "--data", str(data), "--out", str(pack), "--n-train", "20", "--n-validation", "7", "--seed", "3"]) == 0
    with zipfile.ZipFile(pack) as z:
        assert sorted(z.namelist()) == ["pack_manifest.json", "train.jsonl", "validation.jsonl"]
        z.extractall(tmp_path / "unpacked")
        assert json.loads(z.read("pack_manifest.json")) == {"seed": 3, "records": {"train": 20, "validation": 7}}
    for split, n in (("train", 20), ("validation", 7)):
        want = {r["id"] for r in E.draw_records(data, split, n, 3)}
        got = {r["id"] for r in E.draw_records(tmp_path / "unpacked", split, n, 3)}                    # the notebook draws again, from the pack
        assert got == want and len(got) == n and all(r["format"] == "edit_file" for r in E.draw_records(tmp_path / "unpacked", split, n, 3))
    assert {r["id"] for r in E.draw_records(data, "train", 20, 4)} != {r["id"] for r in E.draw_records(data, "train", 20, 3)}, "another seed draws other records"


# ---- the sampler against a fake server ---------------------------------------------------------------------------------------------
class CharTok:
    """One id per character; enough for a server that reads the prompt back."""
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


REQUESTS = []
THOUGHT = "<|channel>thought\nCOPY THE LINE<channel|>"


class Fake(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["top_k"] == 64 and body["skip_special_tokens"] is False and body["temperature"] == 1.0 and "seed" in body
        prompt = "".join(chr(i) for i in body["prompt"])
        REQUESTS.append({"n": body["n"], "prompt": prompt, "max_tokens": body["max_tokens"]})
        if "COPY THE LINE" not in prompt:                                          # no thought yet: one full generation
            texts = [THOUGHT + call(old_string="    return 5", new_string="x")]    # the first sample fails: not found
        else:                                                                      # the thought is in the prompt: calls only
            pool = [call(old_string="    return 2", new_string="    return 22"), call(old_string="return", new_string="y"), '<|tool_call>call:edit_file{old_string="q"}<tool_call|>',
                    call(old_string="    return 2", new_string="    return 222")]
            texts = [pool[i % len(pool)] for i in range(body["n"])]
        data = json.dumps({"choices": [{"index": i, "text": t, "finish_reason": "stop"} for i, t in enumerate(texts)], "usage": {"completion_tokens": 9}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture()
def server():
    REQUESTS.clear()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def a_state():
    return {"id": "st1", "round": 1, "repo": "r/r", "file": "m.py", "source": "x", "messages": [], "prompt_ids": [ord(c) for c in "PROMPT>"], "window": WINDOW, "gold": GOLD,
            "expected": EXPECTED, "gold_span": GOLD_SPAN}


def args(server, k=5):
    return SimpleNamespace(base_url=server, model="m", k=k, max_tokens_full=100, max_tokens_call=50)


def test_the_k_calls_of_a_state_share_the_thought_of_the_first_generation(server):
    row = E.sample_state(args(server, k=5), CharTok(), a_state())
    assert [r["n"] for r in REQUESTS] == [1, 4] and REQUESTS[1]["max_tokens"] == 50
    assert REQUESTS[1]["prompt"] == "PROMPT>" + THOUGHT                            # the second request is the first prompt plus the thought, nothing else
    assert row["head"] == THOUGHT and len(row["samples"]) == 5
    assert [s["status"] for s in row["samples"]] == ["not_found", "applied", "ambiguous", "malformed", "applied"]
    assert row["mixed"] is True and row["prompt_ids"] == [ord(c) for c in "PROMPT>" + THOUGHT] and row["n_prompt"] == len("PROMPT>" + THOUGHT)
    assert all(s["call"].startswith("<|tool_call>") and s["call"].endswith("<tool_call|>") for s in row["samples"][:3])


def test_a_prefix_whose_samples_do_not_differ_in_outcome_keeps_no_prompt(server):
    row = E.sample_state(args(server, k=1), CharTok(), a_state())
    assert len(REQUESTS) == 1 and row["mixed"] is False and row["prompt_ids"] is None and [s["status"] for s in row["samples"]] == ["not_found"]


# ---- the held-out evaluation ----------------------------------------------------------------------------------------------------------
def test_summarize_eval_counts_shares_of_all_samples():
    rows = [label(call(old_string="    return 2", new_string="    return 22")), label(call(old_string="    return 5", new_string="x")), label("text"),
            label(call(old_string="    return 3", new_string="    return 33")), label("<|tool_call>call:read_file{}<tool_call|>")]
    s = E.summarize_eval(rows)
    assert (s["n"], s["good_%"], s["applied_any_tier_%"], s["reproduces_%"], s["not_found_%"], s["no_call_%"], s["wrong_tool_%"], s["edit_call_%"]) == (5, 20.0, 40.0, 20.0, 20.0, 20.0, 20.0, 60.0)
    assert s["localized_given_applied_%"] == 50.0


def test_paired_bootstrap_reports_the_difference_and_an_interval_over_prefixes():
    base = {f"p{i}": [0, 0] for i in range(40)}
    better = {f"p{i}": [1, 1] if i < 20 else [0, 0] for i in range(40)}
    r = E.paired_bootstrap(base, better, iters=500)
    assert r["n_prefixes"] == 40 and r["diff_points"] == 50.0 and 30.0 < r["ci95_points"][0] <= r["ci95_points"][1] < 70.0
    same = E.paired_bootstrap(base, base, iters=200)
    assert same["diff_points"] == 0.0 and same["ci95_points"] == [0.0, 0.0]
    assert E.paired_bootstrap({}, {}) == {"n_prefixes": 0}


# ---- the real chat template ---------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_tok():
    pytest.importorskip("transformers")
    import served
    try:
        return served.load_tokenizer()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"the checkpoint's tokenizer is not on this machine: {e}")


def sft_record():
    window = json.dumps({"status": "ok", "filepath": "m.py", "content": WINDOW, "start_line": 1, "end_line": 10, "total_lines": 10, "is_truncated": False})
    return {"id": "t1", "format": "edit_file", "meta": {"repo": "demo/repo", "file": "m.py", "source": "hist-1"}, "messages": [
        {"role": "system", "content": "OLD SYSTEM PROMPT"},
        {"role": "user", "content": "You are evaluating a software engineering task for repository demo/repo.\n\nProblem Statement:\nb() should return 22 {not 2}"},
        {"role": "assistant", "content": None, "tool_calls": [{"type": "function", "id": "c1", "function": {"name": "read_file", "arguments": json.dumps({"filepath": "m.py", "start_line": 1, "end_line": 10})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": window},
        {"role": "assistant", "content": None, "tool_calls": [{"type": "function", "id": "c2", "function": {"name": "edit_file", "arguments": json.dumps(GOLD)}}]}]}


def test_a_state_is_rendered_the_way_the_scorer_renders_the_shipped_agent(real_tok):
    spec = E.load_spec()
    st = E.make_state(sft_record(), spec, real_tok)
    text = real_tok.decode(st["prompt_ids"], skip_special_tokens=False)
    assert text.startswith("<bos><|turn>system\n<|think|>\n") and "OLD SYSTEM PROMPT" not in text
    assert spec["system"].split("\n")[0] in text and text.count("<|tool>declaration:") == 6
    assert "b() should return 22 {not 2}" in text and "Task Budget" in text and "./README.md\n./m.py" in text
    assert text.endswith("<tool_response|>") and "<|turn>model\n<|tool_call>call:read_file{" in text                 # the model continues its own turn after the read result
    assert st["window"] == WINDOW and st["gold_span"] == GOLD_SPAN and st["expected"] == EXPECTED


def test_the_state_after_a_refused_edit_carries_the_thought_the_call_and_the_error(real_tok):
    spec = E.load_spec()
    st = E.make_state(sft_record(), spec, real_tok)
    bad = {"status": "not_found", "args": {"filepath": "m.py", "old_string": "    return 5", "new_string": "x"}}
    st2 = E.failure_state(st, "<|channel>thought\nCopy the return line.<channel|>", bad, spec, real_tok)
    text = real_tok.decode(st2["prompt_ids"], skip_special_tokens=False)
    assert st2["id"] == "t1/r2" and st2["round"] == 2 and st2["prompt_ids"][:len(st["prompt_ids"])] == st["prompt_ids"]
    tail = text[len(real_tok.decode(st["prompt_ids"], skip_special_tokens=False)):]
    assert tail.startswith("<|channel>thought\nCopy the return line.") and "<channel|><|tool_call>call:edit_file{" in tail
    assert 'old_string:<|"|>    return 5<|"|>' in tail
    assert tail.endswith('"error_message": "Failed to replace: old_string not found. Ensure you\'re not escaping content incorrectly and check whitespace, indentation, and context."}<|"|>}<tool_response|>')


# ---- the whole program --------------------------------------------------------------------------------------------------------------
class FakeTemplateTok(CharTok):
    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=False, enable_thinking=True, **kw):
        return "PROMPT>" + "|".join(m["role"] + ":" + json.dumps(m.get("content")) for m in messages)


def test_sample_writes_one_row_per_state_resumes_and_adds_round_two(server, tmp_path, monkeypatch, capsys):
    import served
    monkeypatch.setattr(served, "load_tokenizer", lambda *a, **k: FakeTemplateTok())
    data = tmp_path / "data"
    data.mkdir()
    recs = []
    for i in range(3):
        r = sft_record()
        r["id"] = f"t{i}"
        recs.append(r)
    (data / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    out = tmp_path / "out"
    argv = ["sample", "--data", str(data), "--base-url", server, "--model", "m", "--out", str(out), "--n", "3", "--k", "5", "--workers", "2", "--round2", "--no-verify"]
    assert E.main(argv) == 0
    rows = [json.loads(l) for l in open(out / "samples.jsonl", encoding="utf-8")]
    assert sorted((r["id"], r["round"]) for r in rows) == sorted([(f"t{i}", 1) for i in range(3)] + [(f"t{i}/r2", 2) for i in range(3)])
    assert all(r["mixed"] for r in rows) and all(len(r["samples"]) == 5 for r in rows)
    n_req = len(REQUESTS)
    assert E.main(argv) == 0                                                       # nothing left to do: no new request, no new row
    assert len(REQUESTS) == n_req and len(open(out / "samples.jsonl", encoding="utf-8").readlines()) == 6
    pairs_dir = tmp_path / "pairs"
    assert E.main(["pairs", "--samples", str(out / "samples.jsonl"), "--out", str(pairs_dir), "--no-verify"]) == 0
    pairs = [json.loads(l) for l in open(pairs_dir / "pairs.jsonl", encoding="utf-8")]
    assert len(pairs) == 12 and {p["round"] for p in pairs} == {1, 2}                # 2 per state: two distinct chosen calls and two refused ones exist among the five
    assert all(p["prompt_ids"] and p["chosen"].startswith("<|tool_call>") for p in pairs)
    st = json.load(open(pairs_dir / "pairs_stats.json"))
    assert st["pairs"] == 12 and st["states_mixed"] == 6
    capsys.readouterr()
