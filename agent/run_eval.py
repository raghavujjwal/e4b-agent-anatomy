"""End-to-end agent evaluation on our Docker-free sandbox, reusing the teammates' agent loop and tools.

  python agent/run_eval.py --backend gold   --task-id httpx_3672          # pipeline control, no model
  python agent/run_eval.py --backend openai --base-url http://127.0.0.1:8000/v1 --model <served-name> --task-id ...

`--backend gold` writes the reference patch THROUGH the real tool path (write_file / run_command /
submit_patch) and grades it; its resolution must equal `python -m grader.cli --mode gold` on the same tasks.
Run from the repository root. Needs: pyyaml, numpy.
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "agent" / "vendor"))          # their eval/runner/tools import `sandbox`, `verify`

import eval as E                                              # noqa: E402  (agent/vendor/eval.py)
import yaml                                                   # noqa: E402
from grader import config, container, verify as gverify       # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--tasks", default=str(config.TASKS_JSONL))
    ap.add_argument("--snapshots", default=str(config.SNAPSHOTS))
    ap.add_argument("--submission", default=str(REPO / "agent" / "submission_ref"))
    ap.add_argument("--graphs", default=None)
    ap.add_argument("--embeddings", default=None)
    ap.add_argument("--backend", choices=["gold", "script", "openai"], default="gold")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gemma-4-e4b-it-qat-w4a16-ct")
    ap.add_argument("--task-id", action="append")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", default=None, help="jsonl of results; finished tasks are skipped on resume")
    ap.add_argument("--trace", default=None, help="jsonl of agent traces")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="tasks run at once (the agent loop is I/O bound on the model; use at most 2 on a 2-CPU Colab VM)")
    a = ap.parse_args(argv)

    tasks = gverify.load_tasks(a.tasks)
    if a.task_id:
        tasks = [t for t in tasks if t["instance_id"] in set(a.task_id)]
    if a.n:
        tasks = tasks[:a.n]
    done = set()
    if a.out and Path(a.out).exists():
        done = {json.loads(l)["instance_id"] for l in open(a.out) if l.strip()}
    todo = [t for t in tasks if t["instance_id"] not in done]

    eval_cfg = (yaml.safe_load(open(Path(a.submission) / "eval_config.yaml")) or {}).get("evaluation", {})
    print(f"{len(tasks)} selected, {len(done)} done, {len(todo)} to run | backend={a.backend}", flush=True)

    lock = threading.Lock()
    state = {"done": 0, "resolved": 0}

    def one(task):
        # Each task writes its trace to a private file; it is appended to --trace under the lock, so concurrent workers
        # never interleave a long JSON line.
        tmp = None
        if a.trace:
            fd, tmp = tempfile.mkstemp(prefix="trace_", suffix=".jsonl")
            os.close(fd)
        try:
            out = E.run_one(task, a.submission, a.snapshots, a.graphs, a.embeddings, a.backend, a.base_url, a.model,
                            eval_cfg, verbose=a.verbose, keep_trace=tmp,
                            container=container.make_factory(),
                            setup=container.make_setup(task, a.snapshots),
                            verifier=container.make_verifier(a.snapshots))
            with lock:
                state["done"] += 1
                state["resolved"] += bool(out["resolved"])
                print(f"[{state['done']}/{len(todo)}] {'RESOLVED' if out['resolved'] else 'no':8s} {task['instance_id']:22s} "
                      f"{out.get('outcome_class')} calls={out.get('tool_calls')} turns={out.get('llm_turns')} "
                      f"{out.get('total_seconds', '?')}s {str(out.get('error') or out.get('verify_error') or '')[:80]}",
                      flush=True)
                if a.out:
                    with open(a.out, "a") as fh:
                        fh.write(json.dumps(out) + "\n")
                if tmp and os.path.getsize(tmp):
                    with open(a.trace, "a") as fh, open(tmp) as src:
                        fh.write(src.read())
            return out
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

    if a.workers > 1:
        with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(one, todo))
    else:
        for task in todo:
            one(task)
    print(f"\nresolved {state['resolved']}/{len(todo)}")
    if a.backend == "gold":
        print("EXPECTED: equals `python -m grader.cli --mode gold` on the same tasks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
