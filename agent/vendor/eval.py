#!/usr/bin/env python3
"""End-to-end local evaluation: Container A (agent) -> patch -> Container B (verify).

Mirrors Evaluator.evaluate_task. Reports the outcome taxonomy that the external 692-run
dataset uses, so our numbers are directly comparable to it:
  resolved / unresolved_nonempty_patch / empty_patch / session_timeout / tool_budget /
  turn_budget / patch_apply_error / model_error
"""
from __future__ import annotations
import argparse, json, os, sys, threading, time, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sandbox import Container, bootstrap
from tools import Tools, Budget, CLOCK
from verify import verify
import runner as R
import scorer_port as P

HERE = os.path.dirname(os.path.abspath(__file__))


def declared_tools(tree):
    """agent_runner._extract_declared_tools: every tool name declared anywhere in the agent hierarchy."""
    names, stack = set(), [tree]
    while stack:
        node = stack.pop()
        for t in node.get("tools") or []:
            if isinstance(t, str):
                names.add(t)
        if node.get("skills"):
            names |= {"run_skill_script", "load_skill_resource"}
        stack.extend(node.get("_sub") or [])
        stack.extend((node.get("_agent_tools") or {}).values())
    return names


def first_message(task, budget, tools, container, tree, graph_available=False):
    """The text of the first user turn, as agent_runner.build_agent_prompt builds it: problem statement, hints,
    the budget and environment rules, the instructions, and the workspace tree read from the container after setup."""
    rc, tree_out, _ = container.sh(P.build_workspace_tree_command())
    return P.build_agent_prompt(
        task.get("repo", ""), task["problem_statement"], task.get("hints_text") or "",
        time_minutes=budget.time_minutes, tool_calls=budget.tool_calls, turns=budget.turns,
        command_timeout_seconds=budget.command_timeout, max_stdout_chars=tools.max_stdout,
        max_file_lines=tools.max_file_lines, max_file_chars=tools.max_file_chars,
        workspace_tree=tree_out.strip() if rc == 0 else "", enable_sandbox_testing=True,
        declared_tools=declared_tools(tree), graph_available=graph_available)


def load_graph(graphs_dir, repo_short, base_commit):
    p = os.path.join(graphs_dir, f"{repo_short}_{base_commit}.json")
    if not os.path.exists(p):
        return None
    g = json.load(open(p))
    g["_ids"] = [n["id"] for n in g["nodes"]]
    g["_idset"] = set(g["_ids"])
    return g


def load_emb(emb_dir, repo_short, base_commit):
    p = os.path.join(emb_dir, f"{repo_short}_{base_commit}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return {k: z[k].tolist() for k in z.files}


SHORT = {"fastapi/fastapi": "fastapi", "Textualize/rich": "rich",
         "psf/requests": "requests", "encode/httpx": "httpx"}


def run_one(task, sub_root, snapshots, graphs, embs, backend_kind, base_url, model,
            eval_cfg, image=None, verbose=False, keep_trace=None,
            container=None, setup=None, verifier=None, clock=None):
    """container/setup/verifier: optional hooks (localeval/rebench.py runs SWE-rebench images through
    this same agent path). Defaults = the competition-faithful Container A / bootstrap / verify."""
    t0 = time.time()
    out = {"instance_id": task["instance_id"], "repo": task.get("repo", ""),
           "resolved": False, "outcome_class": None, "tool_calls": 0, "llm_turns": 0,
           "patch_bytes": 0, "agent_seconds": 0.0, "completion_tokens": None}
    # NOT os.environ["SWE_IMAGE"] = image. That is process-global, and --parallel runs tasks
    # in threads, so one task's image could be overwritten by another's before its container
    # was created -- which is how rich_3472 ended up failing on a newer CPython's
    # AttributeError wording rather than on anything the agent did.
    cfg, cfg_path = R.load_submission(sub_root)
    tree = R.resolve_tree(cfg, cfg_path)
    ev = (eval_cfg or {})
    # clock: "wall" (local wall time) or "grader" (the grader time model; tools.CLOCK). An explicit
    # argument wins; otherwise the task row's "_clock" decides, so a task file can carry its instrument.
    clock = clock or task.get("_clock") or "wall"
    out["clock"] = clock
    budget = Budget(tool_calls=ev.get("max_tool_calls", 24),
                    time_minutes=ev.get("max_time_minutes", 5),
                    turns=ev.get("max_turns", 40),
                    command_timeout=ev.get("timeout_seconds", 90),
                    clock=clock)

    short = SHORT.get(task.get("repo", ""), (task.get("repo") or "x/x").split("/")[-1])
    g = load_graph(graphs, short, task["base_commit"]) if graphs else None
    e = load_emb(embs, short, task["base_commit"]) if embs else None

    c = container() if container else Container("evalA", image=image)
    trace = []
    try:
        if setup:
            setup(c)
        else:
            bootstrap(c, os.path.join(snapshots, f"{task['instance_id']}.tgz"),
                      task.get("repo", ""), "baseline", reqs=task.get("frozen_reqs"))
            c.sh("cd /workspace && git tag -f _swegemma_baseline HEAD")
        tools = Tools(c, budget, graph=g, embeddings=e,
                      skills_root=os.path.join(sub_root, "skills"),
                      extra_protected=P.extract_test_files_from_patch(task.get("test_patch") or ""))
        if backend_kind == "gold":
            backend = R.GoldOracle(task)
        elif backend_kind == "script":
            backend = R.ScriptPolicy(task)
        else:
            backend = R.OpenAIBackend(base_url, model)
        state = {"problem_description": task["problem_statement"], "_contents": []}
        if (task.get("hints_text") or "").strip():
            state["hints"] = task["hints_text"].strip()
        # The scorer's first user message (agent_runner.build_agent_prompt): problem statement, hints, the budget
        # and environment rules, the five instructions and the workspace tree taken AFTER setup. Until 2026-11-04
        # this was two lines of text, so every earlier local run showed the model a different prompt.
        state["_contents"] = [{"role": "user", "content": first_message(task, budget, tools, c, tree, bool(g and e))}]
        out["harness_fidelity"] = P.FIDELITY
        budget.start()
        meter = R.Meter()
        CLOCK.fn = meter.clock_s if clock == "grader" else None
        out["stop_reason"] = R.drive(tree, tools, state, backend, sub_root, trace, meter=meter)
        budget.stop()
        CLOCK.fn = None
        out["agent_seconds"] = round(budget.elapsed, 1)
        out["wall_seconds"] = round(budget.wall_elapsed, 1)
        out["tool_calls"] = budget.used
        out["llm_turns"] = budget.turns_used
        if hasattr(backend, "completion_tokens"):
            out["completion_tokens"] = backend.completion_tokens
        # Critical path for the grader time model: equals the totals for a single agent; for a
        # ParallelAgent only the slowest branch counts (autoresearch.grader_s).
        out["crit_tokens"], out["crit_calls"] = meter.tokens, meter.calls
        out["crit_extra_s"], out["par_k"] = round(meter.extra_s, 1), meter.par_k

        patch = tools.submitted_patch
        if not tools.patch_submitted:
            # harness fallback: extract anyway (README 8.1.2)
            tools.submit_patch()
            patch = tools.submitted_patch
            out["unsubmitted_fallback"] = True
        out["patch_bytes"] = len(patch)
        out["edited_files"] = sorted(set(__import__("re").findall(r"^\+\+\+ b/(\S+)$", patch,
                                                                  __import__("re").M)))
    except Exception as ex:
        out["outcome_class"] = "model_error"
        out["error"] = f"{type(ex).__name__}: {ex}"
        patch = ""
    finally:
        CLOCK.fn = None   # this thread may run the next task
        if budget._final is None:
            budget.stop()
        c.stop()

    if keep_trace:
        with open(keep_trace, "a") as fh:
            fh.write(json.dumps({"instance_id": task["instance_id"], "trace": trace}) + "\n")

    if out["outcome_class"] == "model_error":
        return out
    if not patch.strip():
        out["outcome_class"] = ("session_timeout" if budget.remaining_seconds <= 0
                                else "tool_budget" if budget.remaining_calls <= 0
                                else "empty_patch_without_error")
        return out

    v = verifier(task, patch) if verifier else verify(task, patch, snapshots, verbose=verbose, image=image)
    out["resolved"] = v["resolved"]
    out["verify_error"] = v.get("error")
    if v.get("pytest_tail"):
        out["pytest_tail"] = v["pytest_tail"]
    out["required_source"] = v.get("required_source")
    if v["resolved"]:
        out["outcome_class"] = "resolved"
    elif v.get("error", "").startswith("Failed to apply agent patch"):
        out["outcome_class"] = "patch_apply_error"
    elif budget.remaining_seconds <= 0:
        out["outcome_class"] = "session_timeout"
    else:
        out["outcome_class"] = "unresolved_nonempty_patch"
    out["total_seconds"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--graphs", default=None)
    ap.add_argument("--embeddings", default=None)
    ap.add_argument("--backend", choices=["gold", "script", "openai"], default="script")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gemma-4-31b-it-qat-w4a16-ct")
    ap.add_argument("--task-id", action="append")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--env-map", default=os.path.join(HERE, "env_map.json"))
    ap.add_argument("--scoring-set", default=os.path.join(HERE, "scoring_set.json"),
                    help="EXP-012's valid set: gold passes AND empty fails. Restricting to it "
                         "is what makes a local rate mean anything, so it is the default.")
    ap.add_argument("--all-gold-passing", action="store_true",
                    help="use all 102 gold-passing tasks instead, including the 2 that resolve "
                         "with an EMPTY patch -- only for reproducing pre-EXP-012 numbers")
    ap.add_argument("--only-solved", action="store_true",
                    help="restrict to tasks whose gold patch is known to pass locally")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true", help="skip tasks already recorded in --out")
    ap.add_argument("--trace", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--parallel", type=int, default=1,
                    help="tasks in flight at once. Memory-bound, not CPU-bound: every task "
                         "holds a 4 GB container (the spec's cap, kept for fidelity), so on a "
                         "14 GB box 2-3 is the ceiling. Serial, a 3-arm x 100-task sweep is a "
                         "whole day; at 3 it is a few hours.")
    a = ap.parse_args()

    tasks = [json.loads(l) for l in open(a.tasks)]
    env_map = json.load(open(a.env_map)) if os.path.exists(a.env_map) else {}
    if a.only_solved:
        before = len(tasks)
        tasks = [t for t in tasks if t["instance_id"] in env_map]
        print(f"restricted to locally-verifiable tasks: {len(tasks)}/{before} "
              f"({before - len(tasks)} excluded as local blind spots)")
    # An explicit --task-id wins: naming a task is a debugging request, and silently
    # returning zero tasks for one of the two excluded ids is the kind of helpfulness that
    # wastes an hour.
    if not a.task_id and not a.all_gold_passing and a.scoring_set and os.path.exists(a.scoring_set):
        # EXP-012: a task counts only if GOLD passes and EMPTY fails. The 2 excluded here
        # (fastapi_14077, fastapi_14851) resolve with no patch at all because our wheel
        # resolver installs a starlette newer than the fix, so an agent gets credit for
        # nothing. Scoring them would flatter every arm equally and the headline most of all.
        valid = set(json.load(open(a.scoring_set)).get("scoring_set") or [])
        if valid:
            before = len(tasks)
            tasks = [t for t in tasks if t["instance_id"] in valid]
            print(f"restricted to the valid scoring set: {len(tasks)}/{before} "
                  f"(EXP-012; {before - len(tasks)} dropped as unmeasurable here)")
    if a.task_id:
        want = set(a.task_id); tasks = [t for t in tasks if t["instance_id"] in want]
    if a.n:
        tasks = tasks[:a.n]
    if a.resume and a.out and os.path.exists(a.out):
        # Long runs outlive sessions. Results and traces are append-only, so resuming is just
        # skipping what --out already holds; the finished rows are never re-run or rewritten.
        finished = {json.loads(l)["instance_id"] for l in open(a.out) if l.strip()}
        before = len(tasks)
        tasks = [t for t in tasks if t["instance_id"] not in finished]
        print(f"resume: {before - len(tasks)} already in {a.out}, {len(tasks)} to run")

    ec = os.path.join(a.submission, "eval_config.yaml")
    eval_cfg = (R.load_yaml(ec) or {}).get("evaluation") if os.path.exists(ec) else {}
    print(f"eval_config: {eval_cfg}")

    def one(t):
        return run_one(t, a.submission, a.snapshots, a.graphs, a.embeddings, a.backend,
                       a.base_url, a.model, eval_cfg,
                       image=env_map.get(t["instance_id"]), verbose=a.verbose,
                       keep_trace=a.trace)

    rows = []
    lock = threading.Lock()
    done = [0]

    def record(r):
        # One lock for the counter, the row list and the append-only file together: with
        # --parallel>1 two finishing tasks would otherwise interleave a line of JSONL and the
        # run would be unreadable exactly when it mattered most.
        with lock:
            done[0] += 1
            rows.append(r)
            print(f"[{done[0]}/{len(tasks)}] {'RESOLVED' if r['resolved'] else 'no      '} "
                  f"{r['instance_id']:28s} {r['outcome_class']:26s} "
                  f"calls={r['tool_calls']:3d} turns={r['llm_turns']:3d} "
                  f"patch={r['patch_bytes']:6d}B {r['agent_seconds']:6.1f}s "
                  f"{str(r.get('verify_error') or r.get('error') or '')[:48]}", flush=True)
            if a.out:
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(r) + "\n")

    if a.backend == "openai":
        # A dead server must stop the run, not score it. Job 393776's engines died mid-eval and
        # every later task "timed out" -- 5 retries x 300 s per call, 1,831 s median, 0 tokens --
        # and those rows read as the adapter failing. Rows already written stay valid; tasks in
        # flight when the server dies are not recorded, so `--resume` reruns exactly them.
        import urllib.request

        def watchdog():
            fails = 0
            while True:
                time.sleep(60)
                try:
                    urllib.request.urlopen(a.base_url.rstrip("/") + "/models", timeout=30).read()
                    fails = 0
                except Exception as e:
                    fails += 1
                    print(f"WATCHDOG: server probe failed ({fails}/3): {e}", flush=True)
                    if fails >= 3:
                        print("FATAL: model server unreachable for 3 minutes; aborting so no "
                              "outage is recorded as a result. Rerun with --resume.", flush=True)
                        os._exit(3)
        threading.Thread(target=watchdog, daemon=True).start()

    if a.parallel <= 1:
        for t in tasks:
            record(one(t))
    else:
        # Threads, not processes: every task is blocked on docker and on HTTP, so the GIL is
        # not the constraint. `run_one` already owns its container and its temp dirs, so the
        # only shared state is what `record` guards.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"running {a.parallel} tasks in flight", flush=True)
        with ThreadPoolExecutor(max_workers=a.parallel) as ex:
            futs = {ex.submit(one, t): t for t in tasks}
            for f in as_completed(futs):
                try:
                    record(f.result())
                except Exception as e:
                    iid = futs[f]["instance_id"]
                    record({"instance_id": iid, "resolved": False,
                            "outcome_class": "harness_error", "tool_calls": 0, "llm_turns": 0,
                            "patch_bytes": 0, "agent_seconds": 0.0,
                            "error": f"{type(e).__name__}: {e}"})

    n = len(rows); res = sum(r["resolved"] for r in rows)
    print(f"\nresolved {res}/{n} = {100*res/max(n,1):.1f}%   backend={a.backend}")
    print("outcome taxonomy:")
    for k, v in collections.Counter(r["outcome_class"] for r in rows).most_common():
        print(f"   {k:28s} {v:4d}  {100*v/n:5.1f}%")
    import statistics as st
    for lab, f in [("resolved", lambda r: r["resolved"]), ("not resolved", lambda r: not r["resolved"])]:
        g = [r for r in rows if f(r)]
        if g:
            print(f"   {lab:12s} median calls={st.median(r['tool_calls'] for r in g):.0f} "
                  f"turns={st.median(r['llm_turns'] for r in g):.0f} "
                  f"seconds={st.median(r['agent_seconds'] for r in g):.0f}")
    if a.backend == "gold":
        print("\nCONTROL: backend=gold writes the reference patch through the real tool path. "
              "Resolution here MUST match `verify.py --mode gold` on the same tasks. A gap "
              "means Container A, submit_patch extraction, or the A->B handoff is broken.")


if __name__ == "__main__":
    sys.exit(main())
