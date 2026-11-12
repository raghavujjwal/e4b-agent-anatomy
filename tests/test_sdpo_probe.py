"""train/build_sdpo_probe.py and train/sdpo_probe.py: token spans, wrong-character marking and the verdict, without a model."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "train"))
sys.path.insert(0, str(HERE / "mine"))
import build_sdpo_probe as bp  # noqa: E402
import sdpo_probe as sp  # noqa: E402


def test_tokens_overlapping():
    ends = [3, 5, 9, 10]          # tokens cover [0,3) [3,5) [5,9) [9,10)
    assert bp.tokens_overlapping(ends, 4, 6) == [1, 2]
    assert bp.tokens_overlapping(ends, 0, 3) == [0]
    assert bp.tokens_overlapping(ends, 9, 10) == [3]


def test_wrong_chars_marks_only_what_differs():
    old = "    return data.get('x')"
    win = '    return data.get("x")'
    bad = bp.wrong_chars(old, win)
    assert {old.index("'"), old.rindex("'")} == bad


def test_classify_tokens_splits_wrong_and_right():
    ends = [4, 8, 12]             # three 4-character tokens
    wrong, right = bp.classify_tokens(ends, 0, "abcdefghijkl", {5})
    assert wrong == [1] and right == [0, 2]


def test_best_window_finds_the_near_miss():
    text = "a = 1\ndef f(x):\n    return x + 1\nb = 2"
    sim, win = bp.best_window("def f(y):\n    return y + 1", [text])
    assert sim > 0.8 and win.startswith("def f(x)")


def row(kind, call_sum, d=None, mis_call_sum=None, mis_d=None):
    m = {"call_sum": call_sum}
    mm = {"call_sum": mis_call_sum if mis_call_sum is not None else 0.0}
    if d is not None:
        m.update(d=d, wrong_mean=-1.0, right_mean=0.0)
        mm.update(d=mis_d, wrong_mean=0.0, right_mean=0.0)
    return {"kind": kind, "matched": m, "mismatched": mm}


def test_verdict_pass_fail_and_inconclusive():
    good = ([row("notfound", 0.0, d=-0.5, mis_d=0.1) for _ in range(40)] + [row("repeat_fail", -5.0) for _ in range(25)] + [row("ok_edit", 0.1) for _ in range(20)])
    assert sp.summarize(good)[1] == "PASS"
    # a teacher that pushes every call down just as much with the wrong feedback does not pass A
    bad_a = [row("notfound", 0.0, d=-0.5, mis_d=-0.5) for _ in range(40)] + [row("repeat_fail", -5.0) for _ in range(25)] + [row("ok_edit", 0.1) for _ in range(20)]
    assert sp.summarize(bad_a)[1] == "PARTIAL (B only)"
    # pushing ok edits down as well is not loop suppression
    bad_b = [row("notfound", 0.0, d=-0.5, mis_d=0.1) for _ in range(40)] + [row("repeat_fail", -5.0) for _ in range(25)] + [row("ok_edit", -4.0) for _ in range(20)]
    assert sp.summarize(bad_b)[1] == "PARTIAL (A only)"
    flat = [row("notfound", 0.0, d=0.2, mis_d=0.2) for _ in range(40)] + [row("repeat_fail", 0.0) for _ in range(25)] + [row("ok_edit", 0.0) for _ in range(20)]
    assert sp.summarize(flat)[1] == "FAIL"
    assert sp.summarize(good[:10])[1].startswith("INCONCLUSIVE")


def test_stats_of_uses_the_marked_tokens():
    adv = [0.0, -2.0, 0.0, 0.0, -1.0, 0.0]
    s = {"id": "x", "kind": "notfound", "call_tok": [1, 5], "wrong_tok": [1], "right_tok": [2, 3]}
    r = sp.stats_of(adv, s)
    assert r["call_sum"] == -3.0 and r["d"] == -2.0
