"""train/build_replay_data.py: which turns become training turns, G3 prefixes and G4 (failed-edit) prefixes.

Synthetic traces and a fake tokenizer (an id is a chunk of text), so nothing here needs the model's files.
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
pytest.importorskip("transformers")
import build_replay_data as B  # noqa: E402

Q = '<|"|>'
CHUNK = {
    1: "SYSTEM AND TASK ",
    2: f'<|tool_call>call:edit_file{{filepath:{Q}a.py{Q},new_string:{Q}b{Q},old_string:{Q}a{Q}}}<tool_call|>',
    3: f'<|tool_response>response:edit_file{{value:{Q}{{"status": "error", "message": "not found"}}{Q}}}<tool_response|>',
    4: f'<|tool_call>call:read_file{{filepath:{Q}a.py{Q}}}<tool_call|>',
    5: f'<|tool_response>response:read_file{{value:{Q}{{"status": "ok"}}{Q}}}<tool_response|>',
    6: f'<|tool_call>call:edit_file{{filepath:{Q}a.py{Q},new_string:{Q}b{Q},old_string:{Q}a2{Q}}}<tool_call|>',
    7: f'<|tool_response>response:edit_file{{value:{Q}{{"status": "ok"}}{Q}}}<tool_response|>',
    8: "<|tool_call>call:submit_patch{}<tool_call|>",
    9: "I think the fix is in place.",                       # a text-only turn: the model stops before submit_patch
    10: "<nudge: you have not submitted>",
    # ADK's answer to a call whose mandatory parameter it could not read: the tool never ran (no "status" in the envelope)
    11: f'<|tool_response>response:edit_file{{value:{Q}{{"error": "Invoking `edit_file()` failed as the following mandatory input parameters are not present:\\nold_string\\nYou could retry."}}{Q}}}<tool_response|>',
    12: f'<|tool_call>call:edit_file{{filepath:{Q}a.py{Q},new_string="b"}}<tool_call|>',      # a string the template's parser cannot read
}
CHUNK[13] = CHUNK[2] + CHUNK[6]                                                              # two edit calls in one generation


class FakeTok:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(CHUNK[i] for i in ids)


# one episode: a failing edit, a read, a good edit, a submit. Prompt k = previous prompt[:prefix_len] + suffix.
TURNS = [{"prefix_len": 0, "suffix": [1], "gen": [2], "finish": "stop"},
         {"prefix_len": 1, "suffix": [2, 3], "gen": [4], "finish": "stop"},
         {"prefix_len": 3, "suffix": [4, 5], "gen": [6], "finish": "stop"},
         {"prefix_len": 5, "suffix": [6, 7], "gen": [8], "finish": "stop"}]


def is_holdout(iid):
    return hashlib.sha256(("replay-split-v1/" + iid).encode()).digest()[0] % 5 == 0


def two_tasks():
    held = next(f"held{i}" for i in range(1000) if is_holdout(f"held{i}"))
    train = next(f"train{i}" for i in range(1000) if not is_holdout(f"train{i}"))
    return held, train


def build(tmp_path, monkeypatch, *extra, events=None):
    """events: the trace events of every episode (default: one `conversation` event holding TURNS)."""
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTok())
    held, train = two_tasks()
    run = tmp_path / "run1"
    events = events or [{"event": "conversation", "turn_tokens": TURNS, "messages": []}]
    with open(f"{run}_trace.jsonl", "w", encoding="utf-8") as fh:
        for iid in (held, train):
            fh.write(json.dumps({"instance_id": iid, "trace": events}) + "\n")
    out = tmp_path / "out"
    B.main(["--runs", str(run), "--tokenizer", "unused", "--out", str(out), *extra])
    rows = lambda name: [json.loads(l) for l in open(out / name, encoding="utf-8")]
    return held, train, rows


def test_the_failed_edit_becomes_a_g4_prefix_and_is_not_a_training_turn(tmp_path, monkeypatch, capsys):
    held, train, rows = build(tmp_path, monkeypatch)
    capsys.readouterr()
    failed = rows("check_failed_edit.jsonl")
    assert sorted(r["instance_id"] for r in failed) == sorted([held, train])                    # --g4-tasks all is the default
    r = failed[0]
    assert r["turn"] == 1 and r["prompt_ids"] == [1, 2, 3]                                       # the state after the failure: its prompt ends in the error result
    assert r["failed_call"] == {"name": "edit_file", "args": {"filepath": "a.py", "new_string": "b", "old_string": "a"}}
    assert r["recorded_next"] == {"name": "read_file", "args": {"filepath": "a.py"}} and r["tool"] == "read_file"
    assert "holdout" not in r
    assert r["applied_edit_visible"] is False                                                    # nothing applied before this failure
    kept = rows("turns_train.jsonl") + rows("turns_holdout.jsonl")
    assert sorted((k["instance_id"], k["turn"], k["tool"]) for k in kept) == sorted(
        [(i, t, tool) for i in (held, train) for t, tool in ((1, "read_file"), (2, "edit_file"), (3, "submit_patch"))])        # turn 0 (the failed edit) is gone
    stats = json.load(open(tmp_path / "out" / "stats.json"))
    assert stats["failed_edit_prefixes"] == 2 and stats["failed_edit_checked"] == 2 and stats["failed_edit_tasks"] == 2


def test_applied_edit_visible_reads_only_the_result_of_an_edit_that_applied():
    tok = FakeTok()
    assert B.applied_edit_visible(tok, [1, 2, 3]) is False               # the only edit failed
    assert B.applied_edit_visible(tok, [1, 6, 7, 2, 3]) is True          # an edit applied, a later one failed
    assert B.applied_edit_visible(tok, [1, 4, 5, 2, 3]) is False         # a read with status ok is not an edit
    assert B.applied_edit_visible(tok, [1, 2, 11]) is False              # a call the framework refused never ran
    assert B.applied_edit_visible(tok, [1, 8]) is False                  # no response at all


def test_a_failed_edit_after_an_applied_one_is_flagged_so(tmp_path, monkeypatch, capsys):
    turns = [{"prefix_len": 0, "suffix": [1], "gen": [6], "finish": "stop"},                 # an edit that applies (result 7)
             {"prefix_len": 1, "suffix": [6, 7], "gen": [2], "finish": "stop"},              # then one that fails (result 3)
             {"prefix_len": 3, "suffix": [2, 3], "gen": [8], "finish": "stop"}]
    held, train, rows = build(tmp_path, monkeypatch, events=[{"event": "conversation", "turn_tokens": turns, "messages": []}])
    capsys.readouterr()
    failed = rows("check_failed_edit.jsonl")
    assert len(failed) == 2 and all(r["prompt_ids"] == [1, 6, 7, 2, 3] and r["applied_edit_visible"] is True for r in failed)


def test_g4_tasks_holdout_keeps_only_the_held_out_tasks(tmp_path, monkeypatch, capsys):
    held, train, rows = build(tmp_path, monkeypatch, "--g4-tasks", "holdout")
    capsys.readouterr()
    assert [r["instance_id"] for r in rows("check_failed_edit.jsonl")] == [held]


def test_g3_tasks_chooses_which_tasks_check_jsonl_is_drawn_from(tmp_path, monkeypatch, capsys):
    held, train, rows = build(tmp_path, monkeypatch)                                              # default: holdout
    capsys.readouterr()
    assert {r["instance_id"] for r in rows("check.jsonl")} == {held}
    (tmp_path / "all").mkdir()
    held, train, rows = build(tmp_path / "all", monkeypatch, "--g3-tasks", "all")
    capsys.readouterr()
    assert {r["instance_id"] for r in rows("check.jsonl")} == {held, train}
    assert {r["tool"] for r in rows("check.jsonl")} == {"read_file", "edit_file", "submit_patch"}
    assert all("gen_ids" in r for r in rows("check.jsonl"))


def test_check_failed_edit_caps_the_number_of_prefixes(tmp_path, monkeypatch, capsys):
    held, train, rows = build(tmp_path, monkeypatch, "--check-failed-edit", "1")
    capsys.readouterr()
    assert len(rows("check_failed_edit.jsonl")) == 1
    stats = json.load(open(tmp_path / "out" / "stats.json"))
    assert stats["failed_edit_prefixes"] == 2 and stats["failed_edit_checked"] == 1               # the count of what exists is reported next to what was written


# An episode the model ends on text before submit_patch gets a nudge: the agent's loop runs again and logs a second `conversation` event whose delta chain starts over
# (prefix_len 0, the whole history in the suffix). The builder used to read only the last event, which kept the short tail and dropped the exploration and the edits.
NUDGED = [
    {"event": "conversation", "agent": "swe_agent", "messages": [], "turn_tokens": [
        {"prefix_len": 0, "suffix": [1], "gen": [2], "finish": "stop"},                     # failing edit
        {"prefix_len": 1, "suffix": [2, 3], "gen": [4], "finish": "stop"},                  # re-read
        {"prefix_len": 3, "suffix": [4, 5], "gen": [9], "finish": "stop"}]},                # text only: the run ends, no result follows
    {"event": "nudge", "n": 0},
    {"event": "conversation", "agent": "swe_agent", "messages": [], "turn_tokens": [
        {"prefix_len": 0, "suffix": [1, 2, 3, 4, 5, 9, 10], "gen": [8], "finish": "stop"}]},   # after the nudge: submit
]


def test_every_run_of_the_loop_is_read_not_only_the_last(tmp_path, monkeypatch, capsys):
    held, train, rows = build(tmp_path, monkeypatch, "--g3-tasks", "all", events=NUDGED)
    capsys.readouterr()
    kept = rows("turns_train.jsonl") + rows("turns_holdout.jsonl")
    assert sorted((k["instance_id"], k["turn"], k["tool"]) for k in kept) == sorted(
        [(i, 1, "read_file") for i in (held, train)] + [(i, 3, "submit_patch") for i in (held, train)])      # turns count on across the two runs; the text-only turn 2 is not a tool decision
    assert next(k for k in kept if k["turn"] == 3)["prompt_ids"] == [1, 2, 3, 4, 5, 9, 10]                    # the second run's first prompt is its own chain, not prefixed by the first run
    failed = rows("check_failed_edit.jsonl")
    assert sorted((r["instance_id"], r["turn"]) for r in failed) == sorted([(held, 1), (train, 1)])         # the failure in run 1 is found inside run 1
    assert all(r["prompt_ids"] == [1, 2, 3] and r["recorded_next"]["name"] == "read_file" for r in failed)
    stats = json.load(open(tmp_path / "out" / "stats.json"))
    assert stats["turns"] == 8 and stats["extra_runs"] == 2 and stats["no_tool_call"] == 2


def test_the_result_of_a_runs_last_edit_is_not_looked_up_in_the_next_run(tmp_path, monkeypatch, capsys):
    # run 1 ends on an edit whose result never reached the model in that run; run 2 starts from the whole history. The edit's status is unknown, not read from run 2.
    events = [
        {"event": "conversation", "agent": "swe_agent", "messages": [], "turn_tokens": [
            {"prefix_len": 0, "suffix": [1], "gen": [2], "finish": "stop"}]},
        {"event": "conversation", "agent": "swe_agent", "messages": [], "turn_tokens": [
            {"prefix_len": 0, "suffix": [1, 2, 3], "gen": [8], "finish": "stop"}]},
    ]
    held, train, rows = build(tmp_path, monkeypatch, events=events)
    capsys.readouterr()
    assert rows("check_failed_edit.jsonl") == []
    stats = json.load(open(tmp_path / "out" / "stats.json"))
    assert stats["dropped_edit_file_unknown"] == 2


def test_another_agents_conversation_is_not_the_roots(tmp_path, monkeypatch, capsys):
    helper = {"event": "conversation", "agent": "helper", "messages": [], "turn_tokens": [{"prefix_len": 0, "suffix": [1], "gen": [4], "finish": "stop"}]}
    held, train, rows = build(tmp_path, monkeypatch, "--g3-tasks", "all", events=[helper] + NUDGED)
    capsys.readouterr()
    assert json.load(open(tmp_path / "out" / "stats.json"))["turns"] == 8                                  # the helper's one turn per episode is not counted


# A call the framework refused before the tool ran (a mandatory parameter it could not read) is neither a failed edit nor a training turn. In the 2026-11-06 control runs
# 56 of 504 edit calls were of this kind. Their prefixes go to check_rejected.jsonl for the report-only format probe.
REJECTED = [{"prefix_len": 0, "suffix": [1], "gen": [12], "finish": "stop"},                 # new_string="b": not the template's syntax
            {"prefix_len": 1, "suffix": [12, 11], "gen": [8], "finish": "stop"}]              # the rejection, then submit


def test_a_rejected_call_is_its_own_class_and_its_prefix_is_kept_for_the_probe(tmp_path, monkeypatch, capsys):
    events = [{"event": "conversation", "agent": "swe_agent", "messages": [], "turn_tokens": REJECTED}]
    held, train, rows = build(tmp_path, monkeypatch, events=events)
    capsys.readouterr()
    probe = rows("check_rejected.jsonl")
    assert sorted(r["instance_id"] for r in probe) == sorted([held, train])
    r = probe[0]
    assert (r["turn"], r["tool"], r["prompt_ids"], r["gen_ids"]) == (0, "edit_file", [1], [12])      # the state BEFORE the call, and the call itself
    assert rows("check_failed_edit.jsonl") == []                                                   # nothing ran, so it is not a failed edit
    kept = rows("turns_train.jsonl") + rows("turns_holdout.jsonl")
    assert sorted((k["instance_id"], k["turn"], k["tool"]) for k in kept) == sorted([(i, 1, "submit_patch") for i in (held, train)])
    stats = json.load(open(tmp_path / "out" / "stats.json"))
    assert stats["dropped_edit_file_rejected"] == 2 and "dropped_edit_file_unknown" not in stats
    assert (stats["rejected_prefixes"], stats["rejected_checked"], stats["rejected_tasks"]) == (2, 2, 2)


def test_check_rejected_caps_the_number_of_prefixes(tmp_path, monkeypatch, capsys):
    events = [{"event": "conversation", "agent": "swe_agent", "messages": [], "turn_tokens": REJECTED}]
    held, train, rows = build(tmp_path, monkeypatch, "--check-rejected", "1", events=events)
    capsys.readouterr()
    assert len(rows("check_rejected.jsonl")) == 1
    stats = json.load(open(tmp_path / "out" / "stats.json"))
    assert stats["rejected_prefixes"] == 2 and stats["rejected_checked"] == 1


# The same episode as TURNS, but the last prompt is encoded with a short common prefix (something early in the history was re-rendered), so its suffix repeats the
# older turns and their results in front of the newest one. The first response in that suffix is the FAILED edit of turn 0; the edit made at turn 2 applied.
REPLAYED = TURNS[:3] + [{"prefix_len": 1, "suffix": [2, 3, 4, 5, 6, 7], "gen": [8], "finish": "stop"}]


def test_the_newest_response_is_read_when_the_suffix_repeats_older_ones(tmp_path, monkeypatch, capsys):
    events = [{"event": "conversation", "agent": "swe_agent", "messages": [], "turn_tokens": REPLAYED}]
    held, train, rows = build(tmp_path, monkeypatch, events=events)
    capsys.readouterr()
    kept = rows("turns_train.jsonl") + rows("turns_holdout.jsonl")
    assert sorted((k["instance_id"], k["turn"], k["tool"]) for k in kept) == sorted(
        [(i, t, tool) for i in (held, train) for t, tool in ((1, "read_file"), (2, "edit_file"), (3, "submit_patch"))])         # turn 2's edit applied: it stays
    assert [r["turn"] for r in rows("check_failed_edit.jsonl")] == [1, 1]                           # only the failure of turn 0 (its state is turn 1's prompt)
    stats = json.load(open(tmp_path / "out" / "stats.json"))
    assert stats["dropped_edit_file_error"] == 2 and "dropped_edit_file_unknown" not in stats


def test_a_generation_with_two_calls_is_read_from_its_responses_in_order(tmp_path, monkeypatch, capsys):
    # the first of two edit calls failed, the second applied: the answer for the first is the second block from the end, not the last one
    turns = [{"prefix_len": 0, "suffix": [1], "gen": [13], "finish": "stop"},
             {"prefix_len": 1, "suffix": [13, 3, 7], "gen": [8], "finish": "stop"}]
    events = [{"event": "conversation", "agent": "swe_agent", "messages": [], "turn_tokens": turns}]
    held, train, rows = build(tmp_path, monkeypatch, events=events)
    capsys.readouterr()
    failed = rows("check_failed_edit.jsonl")
    assert len(failed) == 2 and failed[0]["failed_call"]["args"]["old_string"] == "a" and failed[0]["recorded_next"]["name"] == "submit_patch"
    assert json.load(open(tmp_path / "out" / "stats.json"))["dropped_edit_file_error"] == 2
