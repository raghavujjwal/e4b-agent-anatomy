"""S2.6: write the data card of a built training set.

  python -I -X utf8 train/s2_card.py --jobs D:/data/s2/pilot/jobs.jsonl --replayed D:/data/s2/pilot/replayed.jsonl \
      --sft D:/data/s2/pilot/sft.jsonl --out D:/data/s2/pilot/DATA_CARD.md [--title "S2 pilot"]

Everything in the card is computed from the three files, so it cannot drift from the data. The prose only states what the build does and does not do.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split import is_validation  # noqa: E402


def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def pct(n, d):
    return f"{100 * n / d:.0f}%" if d else "n/a"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--replayed", required=True)
    ap.add_argument("--sft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="S2 training set")
    ap.add_argument("--val-share", type=float, default=0.05, help="share of repositories held out for validation (train/split.py; the trainer's default is the same)")
    ap.add_argument("--own-runs", help="stats.json of train/build_replay_data.py (E4B's own turns): its tool mix is shown beside this set's")
    ap.add_argument("--condense-log", help="the log of train/s2_condense.py (its JSON summary adds the rejections to the funnel)")
    ap.add_argument("--render-log", help="the log of train/s2_render.py (what its own checks left out)")
    a = ap.parse_args(argv)
    jobs, rep, sft = read(a.jobs), read(a.replayed), read(a.sft)
    kept = [r for r in rep if "messages" in r]
    drops = Counter((r.get("dropped") or "?")[:48] for r in rep if "messages" not in r)
    lic = Counter(j["license"] for j in jobs if j["instance_id"] in {r["instance_id"] for r in sft})
    repos = Counter(o["repo"] for o in sft)
    toks = [o["n_tokens"] for o in sft]
    calls = Counter(o["n_calls"] for o in sft)
    tool_mix = Counter()
    for r in kept:
        if r["instance_id"] not in {o["instance_id"] for o in sft}:
            continue
        for m in r["messages"]:
            for tc in m.get("tool_calls") or []:
                tool_mix[tc["function"]["name"]] += 1
    syn = Counter(s for o in sft for s in o["synthesized"])
    think = sum(o["enable_thinking"] for o in sft)
    val_repos = {r for r in repos if is_validation(r, a.val_share)}
    val_n = sum(repos[r] for r in val_repos)
    cond_line = []
    if a.condense_log:
        txt = Path(a.condense_log).read_text(encoding="utf-8")
        summ = json.loads(txt[txt.index("{"):])
        rej = summ.get("rejected", {})
        cond_line = [f"- Condensing the selected instances (plan S1: about 2,545 issues in 1,095 repositories) wrote {summ['written']} jobs from {summ['repos']} repositories and rejected "
                     f"{sum(rej.values())}: " + (", ".join(f"{k} {v}" for k, v in sorted(rej.items(), key=lambda kv: -kv[1])) or "none") + "."]
    render_drops = {}
    if a.render_log:
        txt = Path(a.render_log).read_text(encoding="utf-8")
        render_drops = json.loads(txt[txt.index("{"):]).get("dropped_in_render", {})
    unlicensed = sum(v for k, v in lic.items() if k in (None, "", "None"))
    lic_note = []
    if unlicensed:
        lic_note = [f"- {unlicensed} samples come from repositories with no declared licence (`None` above); whether to train on them is a decision for the release "
                    "(render option `--drop-unlicensed` leaves them out)."]
    elif render_drops.get("repository has no declared licence"):
        lic_note = [f"- {render_drops['repository has no declared licence']} samples from repositories with no declared licence were left out of this set (render option `--drop-unlicensed`)."]
    own_line = []
    if a.own_runs:
        kb = json.load(open(a.own_runs, encoding="utf-8"))["kept_by_tool"]
        own = Counter({k: v for k, v in kb.items() if not k.startswith("holdout_")})
        names = ["read_file", "run_command", "edit_file", "submit_patch"]
        mine_total, own_total = sum(tool_mix.values()), sum(own.values())
        own_line = ["- Share of calls by tool, this set against E4B's own turns (runs of the harness before the fidelity fixes, as kept by `build_replay_data.py`): "
                    + "; ".join(f"{t} {pct(tool_mix[t], mine_total)} against {pct(own[t], own_total)}" for t in names)
                    + ". Every sample here contains an edit, so edits are over-represented; the first adapter has no turns of E4B's own to anchor the other step types "
                    "(plan S2.5 needs a run of the corrected harness first), and gates G3 and G4 are what check for drift."]
    lines = [f"# Data card: {a.title}", "",
             "Built by `train/s2_condense.py` (choose and condense), `train/s2_replay.py` (replay on real checkouts through the harness tool code) and "
             "`train/s2_render.py` (render and audit), for the E4B accuracy plan (`docs/PLAN_E4B_SFT.md`, step S2). Every number below is computed from the files.", "",
             "## Sources and licences",
             "- Trajectories: `nebius/SWE-rebench-OpenHands-Trajectories` (CC-BY-4.0; attribution: Nebius). The agent in them is Qwen3-Coder-480B-A35B-Instruct run in "
             "OpenHands 0.54.0; use of its outputs falls under that model's licence terms (Apache-2.0 as published on its model card; re-check before release).",
             "- Task data (problem statements, base commits, patches): `nebius/SWE-rebench` (CC-BY-4.0). Repository sources are fetched from GitHub at each task's base commit "
             "and are used only to produce tool observations; they are not copied into the training files except as the text the tools returned.",
             f"- Repository licences of the kept samples (counts are samples; {len(repos)} repositories): " + ", ".join(f"{k} {v}" for k, v in lic.most_common()) + ".",
             *lic_note, "",
             "## Funnel", *cond_line,
             f"- Condensed jobs: {len(jobs)}.",
             f"- Replayed on a real checkout and kept: {len(kept)} ({pct(len(kept), len(rep))} of {len(rep)} attempted).",
             "- Dropped in the replay: " + (", ".join(f"{k} {v}" for k, v in drops.most_common()) or "none") + ".",
             "- Left out by the render checks: " + (", ".join(f"{k} {v}" for k, v in sorted(render_drops.items(), key=lambda kv: -kv[1])) or "none (no render log given)") + ".",
             f"- Rendered and passed the audit: {len(sft)} ({pct(len(sft), len(kept))} of the replayed).", "",
             "## Samples",
             f"- Distinct repositories: {len(repos)}; most samples from one repository: {max(repos.values()) if repos else 0}.",
             f"- Tokens per sample: median {statistics.median(toks) if toks else 0:.0f}, max {max(toks) if toks else 0}, "
             f"loss tokens {pct(sum(o['n_loss_tokens'] for o in sft), sum(toks))} of all tokens.",
             f"- Tool calls per sample (counted by the harness, `submit_patch` included): " + ", ".join(f"{k}: {v}" for k, v in sorted(calls.items())) + ".",
             "- Calls by tool: " + ", ".join(f"{k} {v}" for k, v in tool_mix.most_common()) + ".", *own_line,
             f"- Thinking: {think} samples carry `<|think|>` in the system turn, {len(sft) - think} do not. The model's own turns have no thought channel in either (format T0).",
             f"- Validation split (`train/split.py`, {a.val_share:.0%} of the repositories chosen by a hash of the name, so it does not move when the set is rebuilt): "
             f"{len(val_repos)} repositories and {val_n} samples are held out, {len(sft) - val_n} samples train.", "",
             "## What is real and what is not",
             "- Real (produced by running the harness tool code on the checkout): the workspace tree, the search output, every `read_file` result, every `edit_file` result "
             "(strategy `exact`, and the resulting diff equals the teacher's change to those files), the `submit_patch` result.",
             f"- Not real: the verification output ({syn.get('verification_output', 0)} samples) is the teacher's recorded output with paths rewritten, because the repositories' "
             f"dependencies are not installed on the machine that replays; the search command ({syn.get('discovery_command', 0)} samples) was chosen by the builder from a term in "
             "the issue that occurs near the edit, because the teacher's own searches print no line numbers. The closing sentence is a fixed template.",
             "- Left out: hints (the prompt has none), tests and documentation changes, trajectories with `insert`, `undo_edit`, new modules or more than three source files, "
             "and samples whose verification command neither runs a test runner nor names an edited file (a snippet that checks how Python or a library behaves). "
             "Also absent: turns from E4B's own runs (plan S2.5), because the saved traces came from the harness before the fidelity fixes; they need a run with the corrected harness first.",
             "- Bias: only small, local, correctly grounded fixes in Python; the selection favours tasks where the issue names an identifier. Repositories whose tarball did not "
             "download within the time limit (`fetch:` in the funnel) or is over 150 MB (`repo_too_large`) are missing, so very large repositories are under-represented. Held-out evaluation repositories "
             "(fastapi, rich, requests, httpx, litestar, pydantic-settings, isort, urllib3, marshmallow, scrapy) are excluded by name.", "",
             "## Top repositories", ""]
    lines += [f"- {k}: {v}" for k, v in repos.most_common(10)]
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines[:22]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
