#!/usr/bin/env python3
"""Container B: the grading half of the harness, reimplemented locally.

verify(task, agent_patch) -> resolved: bool

Mirrors HARNESS_README section 8.2 step for step:
  fresh snapshot -> eval_baseline commit -> 4-pass patch application ->
  reset every PROTECTED path the agent patch touched -> apply test_patch ->
  hermetic pytest with JUnit XML -> validate that every required node passed un-skipped.

Needs no model. That is the point: it is a pure function of (snapshot, patch, test_patch),
so it can be validated on its own with gold patches and empty patches before anything
depends on it.
"""
from __future__ import annotations
import argparse, json, os, re, shlex, subprocess, sys, time, xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sandbox import (Container, bootstrap, apply_patch, patch_files, is_protected,
                     test_functions)


def pytest_targets(test_patch: str) -> list[str]:
    return [f for f in patch_files(test_patch) if f.endswith(".py")]


def validate_junit(xml_text: str, required: list[str]):
    """passed>0, failures==0, errors==0, and every required node present and not skipped."""
    if not xml_text:
        return False, "no junit xml"
    try:
        root = ET.fromstring(xml_text)
    except Exception as e:
        return False, f"unparseable junit xml: {e}"
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    cases, failures, errors = [], 0, 0
    for s in suites:
        failures += int(s.get("failures") or 0)
        errors += int(s.get("errors") or 0)
        cases += list(s.iter("testcase"))
    if not cases:
        return False, "junit xml contains no testcases"
    if failures or errors:
        # Name them. "failures=1" is not a diagnosis, and the one task that regressed under
        # --parallel 3 (rich_3472) cost a re-run purely because this message would not say
        # WHICH test went red.
        who = []
        for c in cases:
            for ch in c:
                if ch.tag in ("failure", "error"):
                    msg = (ch.get("message") or "").strip().replace("\n", " ")[:90]
                    who.append(f"{c.get('classname','')}::{c.get('name','')} [{ch.tag}: {msg}]")
                    break
        return False, (f"junit reports failures={failures} errors={errors}: "
                       + "; ".join(who[:3]) + (f" (+{len(who)-3} more)" if len(who) > 3 else ""))
    status = {}
    for c in cases:
        name = c.get("name") or ""
        bad = [ch.tag for ch in c if ch.tag in ("failure", "error", "skipped")]
        status[name] = bad[0] if bad else "passed"
    passed = sum(1 for v in status.values() if v == "passed")
    if passed == 0:
        return False, "no passing testcases"
    for req in required:
        hits = [v for k, v in status.items() if k == req or k.startswith(req + "[")]
        if not hits:
            return False, f"required test {req} never ran"
        if any(h != "passed" for h in hits):
            return False, f"required test {req} did not pass ({set(hits)})"
    return True, f"{passed} passed, all {len(required)} required node(s) ok"


def verify(task: dict, agent_patch: str, snapshots_dir: str, keep=False, verbose=False,
           image=None):
    t0 = time.time()
    res = {"instance_id": task["instance_id"], "resolved": False, "error": None,
           "stage": None, "apply_pass": None, "detail": None}
    tgz = os.path.join(snapshots_dir, f"{task['instance_id']}.tgz")
    if not os.path.exists(tgz):
        res["error"] = f"snapshot missing: {tgz}"
        return res
    c = Container("evalB", image=image)
    try:
        res["stage"] = "bootstrap"
        log = bootstrap(c, tgz, task.get("repo", ""), "eval_baseline", reqs=task.get("frozen_reqs"))
        if verbose:
            print("   bootstrap:", json.dumps(log)[:300])

        if agent_patch.strip():
            res["stage"] = "apply_agent_patch"
            ok, which = apply_patch(c, agent_patch)
            res["apply_pass"] = which
            if not ok:
                res["error"] = "Failed to apply agent patch"
                return res

            # Reset every protected path the agent patch touched, before test_patch lands.
            prot = [f for f in patch_files(agent_patch) if is_protected(f)]
            if prot:
                res["detail"] = f"reset {len(prot)} protected path(s): {prot[:5]}"
                q = " ".join(shlex.quote(p) for p in prot)
                c.sh(f"cd /workspace && git checkout HEAD -- {q} 2>/dev/null || true")
                c.sh(f"cd /workspace && git clean -f -- {q} 2>/dev/null || true")

        res["stage"] = "apply_test_patch"
        ok, which = apply_patch(c, task["test_patch"])
        if not ok:
            res["error"] = "Failed to apply test_patch (harness-side)"
            return res

        # test deps / pytest.ini / conftest refreshed after test_patch, as the harness does
        # after test_patch lands, refresh paths/ini only -- deps were installed in bootstrap
        c.sh(f"python3 /sandbox/setup.py --fast-path {shlex.quote(task.get('repo',''))} "
             f">/dev/null 2>&1 || true", timeout=900)

        res["stage"] = "pytest"
        targets = pytest_targets(task["test_patch"])
        if not targets:
            res["error"] = "no pytest targets in test_patch"
            return res
        xml = f"/tmp/_swegemma_junit_{task['instance_id']}.xml"
        cmd = ("cd /workspace && PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 python3 -s -m pytest "
               + " ".join(shlex.quote(t) for t in targets)
               + f" --junitxml={xml} -p no:anyio -o timeout=0 "
               + "-o python_classes='Test* *Test' -q")
        # The harness passes -o timeout=0, which DISABLES pytest-timeout, and some tasks'
        # tests loop forever without the fix: rich_4006 ran 12 minutes in empty mode before
        # being killed, while it resolves in seconds with the gold patch. The real harness
        # bounds this with the agent's session budget; we bound it here and classify it, so a
        # hanging task is never silently charged to the agent as a wrong patch.
        try:
            rc, so, se = c.sh(cmd, timeout=600)
        except subprocess.TimeoutExpired:
            res["error"] = "tests hung (>600s) -- this task's tests do not terminate in this state"
            res["stage"] = "pytest_hang"
            res["resolved"] = False
            res["seconds"] = round(time.time() - t0, 1)
            return res
        res["exit_code"] = rc
        tail = (so + se)[-1200:]
        required, req_src = test_functions(task["test_patch"])
        res["required"] = required
        res["required_source"] = req_src
        xml_text = c.get_text(xml)
        ok, why = validate_junit(xml_text or "", required)
        res["detail"] = f"{why} [required via {req_src}]"
        res["resolved"] = bool(rc == 0 and ok)
        if not res["resolved"]:
            res["error"] = res["error"] or (f"exit={rc}; {why}")
            # Keep the tail in the record, not just on the terminal: a failure diagnosed hours
            # later from a .jsonl is the normal case, and re-running to see it costs minutes.
            res["pytest_tail"] = tail
            if verbose:
                print("   pytest tail:", tail[-600:])
        res["seconds"] = round(time.time() - t0, 1)
        return res
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        return res
    finally:
        if not keep:
            c.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--task-id", action="append", default=None)
    ap.add_argument("--n", type=int, default=0, help="first N tasks")
    ap.add_argument("--mode", choices=["gold", "empty", "file"], default="gold",
                    help="gold = apply the reference patch (MUST resolve); "
                         "empty = apply nothing (MUST NOT resolve); file = use --patch")
    ap.add_argument("--patch", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--image", default=None, help="sandbox image (unseen52: unseen-sandbox:py312)")
    a = ap.parse_args()

    tasks = [json.loads(l) for l in open(a.tasks)]
    if a.task_id:
        want = set(a.task_id); tasks = [t for t in tasks if t["instance_id"] in want]
    if a.n:
        tasks = tasks[:a.n]

    results = []
    nres = 0
    for i, t in enumerate(tasks, 1):
        if a.mode == "gold":
            p = t["patch"]
        elif a.mode == "empty":
            p = ""
        else:
            p = open(a.patch).read()
        r = verify(t, p, a.snapshots, verbose=a.verbose, image=a.image)
        nres += bool(r["resolved"])
        results.append(r)
        flag = "RESOLVED" if r["resolved"] else "no       "
        print(f"[{i}/{len(tasks)}] {flag} {r['instance_id']:28s} "
              f"stage={r.get('stage')} apply={r.get('apply_pass')} "
              f"{r.get('seconds','?')}s  {str(r.get('error') or r.get('detail'))[:70]}")
        if a.out:
            with open(a.out, "a") as fh:
                fh.write(json.dumps(r) + "\n")

    print(f"\nmode={a.mode}  resolved {nres}/{len(results)}")
    if a.mode == "gold":
        print("EXPECTED: all resolved. Any 'no' is a defect in this verifier or an unsound task.")
    if a.mode == "empty":
        print("EXPECTED: none resolved. Any 'RESOLVED' means the test_patch passes WITHOUT the "
              "fix, i.e. the task is unsound or the verifier is not actually running the tests.")


if __name__ == "__main__":
    sys.exit(main())
