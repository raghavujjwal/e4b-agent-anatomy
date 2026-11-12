"""Re-record tests/fixtures/scorer_requests_pipe_s2*.json: the requests the scorer's own stack (swegemma 0.2.11 + adk-submission 0.2.13 + google-adk 1.36.1 +
LiteLLM) sends for the S2 tree of the pipeline ladder, for every script in mine/fidelity_scripts/script_pipeline*.json.

  python -I -X utf8 mine/record_pipe_fixtures.py [--python D:/envs/scorer/Scripts/python.exe]

tests/test_pipeline_arms.py replays the same scripts through the vendored runner (agent/vendor/runner.py) and requires identical request bodies. The fixtures hold
the S2 prompt, so they must be recorded again after every change of mine/make_pipeline_arms.py (the test of request 0 says so). Needs the scorer venv (see
mine/capture_scorer_requests.py); nothing here talks to a model.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "mine" / "fidelity_scripts"
FIX = ROOT / "tests" / "fixtures"
ARM = ROOT / "agent" / "submission_e4b_pipe_s2"
TODAY = "2026-11-09"


def capture(python: str, script: str, outer: bool, out: Path, port: int) -> dict:
    cmd = [python, "-I", "-X", "utf8", str(ROOT / "mine" / "capture_scorer_requests.py"), "--submission", str(ARM), "--script", str(SCRIPTS / script),
           "--out", str(out), "--port", str(port)] + (["--outer-loop"] if outer else [])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if r.returncode:
        raise SystemExit(f"capture of {script} failed:\n{r.stdout[-1500:]}\n{r.stderr[-2500:]}")
    print(r.stdout.strip().splitlines()[-1])
    return json.loads(out.read_text(encoding="utf-8"))


def bodies(cap: dict) -> list:
    return [r["body"] for r in cap["requests"]]


def write(name: str, obj: dict) -> None:
    with (FIX / name).open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print("wrote", name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--python", default="D:/envs/scorer/Scripts/python.exe", help="the scorer venv's interpreter")
    a = ap.parse_args(argv)
    base = json.loads((FIX / "scorer_requests_v2rec.json").read_text(encoding="utf-8"))          # the workspace the capture tool assumes
    shared = {"ws_files": base["ws_files"], "tree": base["tree"], "diff": base["diff"]}
    script = lambda f: json.loads((SCRIPTS / f).read_text(encoding="utf-8"))                      # noqa: E731
    with tempfile.TemporaryDirectory(prefix="pipe_fix_") as tmp:
        tmp = Path(tmp)
        main_cap = capture(a.python, "script_pipeline.json", False, tmp / "main.json", 18765)
        write("scorer_requests_pipe_s2.json", {
            "_comment": (f"Recorded {TODAY} by mine/record_pipe_fixtures.py (the scorer's own code, versions below) on agent/submission_e4b_pipe_s2 with "
                         "mine/fidelity_scripts/script_pipeline.json against a mock server and a fake sandbox. Requests 0-3 are the stage `investigate` (three tools, the "
                         "scorer's first message), requests 4-8 the stage `fix` (include_contents none: ADK hands it the last reply of `investigate` as a user message "
                         "of two text parts, 'For context:' and '[investigate] said: ...'). tests/test_pipeline_arms.py replays the script through the vendored runner."),
            "versions": main_cap["versions"], "config": main_cap["config"], "script": script("script_pipeline.json"), **shared, "requests": bodies(main_cap)})

        empty = capture(a.python, "script_pipeline_empty.json", False, tmp / "empty.json", 18766)
        thoughts = capture(a.python, "script_pipeline_thoughts_only.json", False, tmp / "thoughts.json", 18767)
        if bodies(empty) != bodies(thoughts):
            raise SystemExit("an empty reply and a thoughts-only reply of stage 1 no longer give the same requests; the fixture assumes they do")
        write("scorer_requests_pipe_s2_empty.json", {
            "_comment": (f"Recorded {TODAY} by mine/record_pipe_fixtures.py (the scorer's own code, versions below) on agent/submission_e4b_pipe_s2 with "
                         "mine/fidelity_scripts/script_pipeline_empty.json: stage `investigate` replies with EMPTY content and no tool call. The real stack leaves no event "
                         "for it, so `fix` (include_contents none) starts from the latest USER message, the scorer's first message, and {findings?} renders empty. A stage 1 "
                         "that ends with THOUGHTS ONLY (reasoning, empty content; script_pipeline_thoughts_only.json, the shape of a reply cut by the token limit inside "
                         "the thinking) gave byte-identical request bodies, so `script_thoughts_only` is replayed against the same `requests`."),
            "versions": empty["versions"], "config": empty["config"], "script": script("script_pipeline_empty.json"),
            "script_thoughts_only": script("script_pipeline_thoughts_only.json"), **shared, "requests": bodies(empty)})

        nudge = capture(a.python, "script_pipeline_nudge.json", True, tmp / "nudge.json", 18768)
        if nudge["passes"] != 2:
            raise SystemExit(f"the nudge script should take two passes, took {nudge['passes']}")
        write("scorer_requests_pipe_s2_nudge.json", {
            "_comment": (f"Recorded {TODAY} by mine/record_pipe_fixtures.py (mine/capture_scorer_requests.py --outer-loop: the scorer's own code, versions below, including "
                         "its continuation loop) on agent/submission_e4b_pipe_s2 with mine/fidelity_scripts/script_pipeline_nudge.json. PASS 1 ends after stage 2's edit with "
                         "text and no submit_patch (requests 0-5); the scorer then sends its plain nudge as a new user message of the same session and the SequentialAgent "
                         "root starts over: requests 6-7 are stage `investigate` on PASS 2 (the whole session: the first message, its own history and note, stage 2's events "
                         "as 'For context' user messages, the nudge), requests 8-10 stage `fix` on pass 2 (a fresh start from the new note only). tests/test_pipeline_arms.py "
                         "replays the script through the vendored runner."),
            "versions": nudge["versions"], "config": nudge["config"], "script": script("script_pipeline_nudge.json"), **shared, "requests": bodies(nudge)})

        cases = {}
        for key, f, outer, port, why in (
                ("error_then_silence", "script_pipeline_edge_error.json", False, 18769,
                 "stage `investigate` calls a tool it was not given (the scorer answers with its own error) and then ends with an empty reply: the latest event of another "
                 "agent is that tool RESULT, so `fix` starts from '[investigate] `edit_file` tool returned result: ...' alone"),
                ("multi_call_and_thoughts", "script_pipeline_multi.json", True, 18770,
                 "two tool calls in one reply, text together with calls, thoughts on both stages, then a nudge: on pass 2 `investigate` sees `fix`'s events as 'For context' "
                 "messages without its thoughts, with its own thoughts still attached to its own turns")):
            cap = capture(a.python, f, outer, tmp / f"{key}.json", port)
            cases[key] = {"why": why, "outer_loop": outer, "script": script(f), "requests": bodies(cap)}
    write("scorer_requests_pipe_s2_cases.json", {
        "_comment": (f"Recorded {TODAY} by mine/record_pipe_fixtures.py (the scorer's own code, versions below) on agent/submission_e4b_pipe_s2, one entry of `cases` per "
                     "script in mine/fidelity_scripts. `outer_loop` = captured with --outer-loop (the scorer's continuation loop). tests/test_pipeline_arms.py replays each "
                     "script through the vendored runner and requires identical request bodies."),
        "versions": main_cap["versions"], "config": main_cap["config"], **shared, "cases": cases})
    return 0


if __name__ == "__main__":
    sys.exit(main())
