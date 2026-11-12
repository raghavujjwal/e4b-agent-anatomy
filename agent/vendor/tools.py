"""Container A: the agent-facing tools, mirroring HARNESS_README section 6 AND the scorer's own code.

Return shapes, truncation caps, error strings and budget-gating semantics follow swegemma 0.2.11 exactly
(see scorer_port.py for the source of each rule), because the agent's behaviour is shaped by them: a 150-line
read cap is why exploration is expensive, and submit_patch being free is why it should always be called last.
The six core tools (run_command, read_file, edit_file, write_file, submit_patch, get_status) are ported
function by function; the graph and skill tools are not part of the E4B submission and keep their old form.

History: before 2026-11-04 this file was written from the README and differed from the scorer in the edit
matcher, every error string, JSON escaping, read_file truncation and budget_warning (git tag pre-fidelity-203e80f).
"""
from __future__ import annotations
import json, os, posixpath, re, shlex, subprocess, threading, time

try:
    from . import scorer_port as P
except ImportError:                      # train/*.py put this directory on sys.path and import `tools` directly
    import scorer_port as P

MAX_STDOUT = 5000
MAX_FILE_LINES = 150
MAX_FILE_CHARS = 10000


# Grader clock (EXP-052). The harness's session cap is wall time on the GRADER (~25.5 tok/s decode); a
# local server is faster or slower depending on its GPU and load, so a wall-clock cap here cuts arms on a
# busy server early (E22/E23: 22-26 session timeouts vs 2-5 for the controls, on fewer tokens). With
# clock="grader" the session time is the grader time model (runner.Meter.clock_s), read through the
# calling thread's CLOCK.fn: one function per task thread, and per ParallelAgent branch.
CLOCK = threading.local()


class Budget:
    def __init__(self, tool_calls=24, time_minutes=5.0, turns=40, command_timeout=90, clock="wall"):
        self.tool_calls = tool_calls
        self.time_minutes = time_minutes      # as configured, so the prompt prints "5.5" and not 5.499999
        self.time_seconds = time_minutes * 60
        self.turns = turns
        self.command_timeout = command_timeout
        self.clock = clock
        self.used = 0
        self.turns_used = 0
        self.started = None
        self._final = None

    def start(self):
        self.started = time.time()

    def stop(self):
        """Freeze the session clock (the agent is done; patch extraction and grading are not timed)."""
        self._final = self.elapsed

    @property
    def wall_elapsed(self):
        return 0.0 if self.started is None else time.time() - self.started

    @property
    def elapsed(self):
        if self._final is not None:
            return self._final
        if self.started is None:
            return 0.0
        fn = getattr(CLOCK, "fn", None) if self.clock == "grader" else None
        return fn() if fn is not None else time.time() - self.started

    @property
    def remaining_seconds(self):
        return max(0.0, self.time_seconds - self.elapsed)

    @property
    def remaining_calls(self):
        return max(0, self.tool_calls - self.used)

    def exhausted(self):
        return self.remaining_calls <= 0 or self.remaining_seconds <= 0


def _ok(**kw):
    d = {"status": "ok"}; d.update(kw); return d


def _err(t, m, **details):
    return {"status": "error", "error_type": t, "error_message": m,
            **({"details": details} if details else {})}


def apply_replacement(text: str, old: str, new: str, allow_multiple: bool):
    """Compatibility wrapper for train/*.py: (new_text or None, occurrences, strategy). The matcher is the
    scorer's four-tier one (exact, flexible, regex, JSON-unescaped), not the old three-tier copy."""
    res = P.apply_replacement_core(text, old, new, allow_multiple)
    if res.error_message:
        return None, res.occurrences, res.strategy
    return res.new_content, res.occurrences, res.strategy


# Parameters the model may pass and the ones ADK treats as mandatory (FunctionTool._get_mandatory_args):
# unknown keys are dropped, a missing mandatory key returns ADK's message without running or counting the call.
SIGNATURES = {
    "run_command": (("command",), ("command",)),
    "read_file": (("filepath", "start_line", "end_line"), ("filepath",)),
    "edit_file": (("filepath", "old_string", "new_string", "allow_multiple"), ("filepath", "old_string", "new_string")),
    "write_file": (("filepath", "content"), ("filepath", "content")),
    "submit_patch": ((), ()),
    "get_status": ((), ()),
}


class Tools:
    """Binds the tool surface to one Container A plus one Budget."""

    def __init__(self, container, budget: Budget, graph=None, embeddings=None,
                 skills_root=None, baseline_ref="_swegemma_baseline", extra_protected=(),
                 max_stdout=MAX_STDOUT, max_file_lines=MAX_FILE_LINES, max_file_chars=MAX_FILE_CHARS):
        self.c = container
        self.b = budget
        self.graph = graph or {}
        self.emb = embeddings or {}
        self.skills_root = skills_root
        self.baseline_ref = baseline_ref
        self.extra_protected = tuple(extra_protected)   # task.test_files: paths the scorer strips from the patch
        self.max_stdout, self.max_file_lines, self.max_file_chars = max_stdout, max_file_lines, max_file_chars
        self.available = None        # names declared on the calling agent (set by the runner); None = no check
        self.submitted_patch = ""
        self.patch_submitted = False
        self.log = []

    # ---- plumbing -------------------------------------------------------------
    def _gate(self, counted):
        """context.check_budget: refuse when a budget is spent, else count the call. Free tools (submit_patch,
        get_status) skip the call and turn limits but still obey the clock."""
        err = P.budget_error(counted=counted, tool_calls_used=self.b.used, max_calls=self.b.tool_calls,
                             turns_used=self.b.turns_used, max_turns=self.b.turns,
                             time_minutes=self.b.time_minutes,
                             elapsed_seconds=self.b.elapsed if self.b.started is not None else None)
        if err is None and counted:
            self.b.used += 1
        return err

    def _wrap(self, name, fn, counts=True):
        err = self._gate(counts)
        if err is not None:
            self.log.append({"tool": name, "counted": counts, "seconds": 0.0, "status": "error"})
            return err
        t0 = time.time()
        try:
            res = fn()
        except Exception as e:          # swegemma's ToolErrorPlugin.on_tool_error_callback
            res = json.dumps({"status": "error", "error_type": type(e).__name__, "error": str(e)}, ensure_ascii=False)
        if isinstance(res, dict):       # graph and skill tools still build dicts
            res = json.dumps(res, ensure_ascii=False)
        if counts:
            res = P.attach_low_budget_warning(res, self.b.tool_calls, self.b.used)
        try:
            status = json.loads(res).get("status")
        except Exception:
            status = None
        self.log.append({"tool": name, "counted": counts, "seconds": round(time.time() - t0, 2), "status": status})
        return res

    def call(self, name, args: dict):
        args = args if isinstance(args, dict) else {}
        if self.available is not None and name not in self.available:
            return P.adk_unknown_tool_result(name, list(self.available))
        sig = SIGNATURES.get(name)
        if sig is not None:
            params, mandatory = sig
            args = {k: v for k, v in args.items() if k in params}
            missing = [m for m in mandatory if m not in args]
            if missing:
                return P.adk_missing_args_result(name, missing)
        if name == "run_command":
            return self._wrap(name, lambda: self.run_command(args["command"]))
        if name == "read_file":
            return self._wrap(name, lambda: self.read_file(args["filepath"], args.get("start_line"),
                                                           args.get("end_line")))
        if name == "edit_file":
            return self._wrap(name, lambda: self.edit_file(args["filepath"], args["old_string"], args["new_string"],
                                                           args.get("allow_multiple", False)))
        if name == "write_file":
            return self._wrap(name, lambda: self.write_file(args["filepath"], args["content"]))
        if name == "run_skill_script":
            return self._wrap(name, lambda: self.run_skill_script(args.get("skill_name", ""),
                                                                  args.get("file_path", ""),
                                                                  args.get("args")))
        if name == "get_code_neighbors":
            return self._wrap(name, lambda: self.get_code_neighbors(args.get("node", ""),
                                                                    args.get("edge_type"),
                                                                    int(args.get("max_neighbors", 50))))
        if name == "search_similar_code":
            return self._wrap(name, lambda: self.search_similar_code(args.get("query", ""),
                                                                     int(args.get("k", 10))))
        if name == "get_code_subgraph":
            return self._wrap(name, lambda: self.get_code_subgraph(args.get("nodes") or []))
        if name == "submit_patch":
            return self._wrap(name, self.submit_patch, counts=False)
        if name == "get_status":
            return self._wrap(name, self.get_status, counts=False)
        return json.dumps(_err("UnknownTool", f"No such tool: {name}"), ensure_ascii=False)

    # ---- file access ------------------------------------------------------------
    def _read_bytes(self, path):
        """The file as bytes. docker cp / the sandbox's own read, never newline-translated."""
        get_bytes = getattr(self.c, "get_bytes", None)
        data = get_bytes(path) if get_bytes else None
        if data is None and not get_bytes:
            text = self.c.get_text(path)
            data = None if text is None else text.encode("utf-8")
        if data is None:
            raise FileNotFoundError(P.docker_not_found_message(path))
        return data

    def _write_bytes(self, path, data: bytes):
        put_bytes = getattr(self.c, "put_bytes", None)
        if put_bytes:
            put_bytes(path, data)
        else:
            self.c.put_text(path, data.decode("utf-8"))

    # ---- execution ------------------------------------------------------------
    def run_command(self, command):
        cmd_timeout = self.b.command_timeout
        if self.b.time_minutes is not None:
            remaining = self.b.time_minutes * 60.0 if self.b.started is None else self.b.remaining_seconds
            effective_remaining = max(5, int(remaining))      # near the deadline a command still gets 5 s
            effective = min(cmd_timeout, effective_remaining) if cmd_timeout is not None else effective_remaining
        else:
            effective = cmd_timeout
        run = getattr(self.c, "exec_scorer", None) or self.c.sh
        try:
            rc, so, se = run(command, timeout=effective)
        except subprocess.TimeoutExpired:
            return P.error_response("TimeoutExceeded", f"Command timed out after {effective} seconds")
        so, se = so[:self.max_stdout], se[:self.max_stdout]
        if rc != 0:
            return P.error_response("CommandError", se if se else so,
                                    details={"stdout": so, "stderr": se, "exit_code": rc})
        return P.ok_response(stdout=so, stderr=se, exit_code=0)

    def submit_patch(self):
        self.c.sh("cd /workspace && git add -N .")
        rc, so, se = self.c.sh(f"cd /workspace && (git diff --binary {self.baseline_ref} "
                               f"2>/dev/null || git diff --binary HEAD)", timeout=120)
        if rc != 0:
            return P.error_response("GitDiffError", f"git diff failed with exit code {rc}: {se}")
        diff = so
        if diff.strip():
            diff = P.strip_protected_files_from_patch(diff, extra_protected_paths=self.extra_protected)
        self.submitted_patch = diff
        self.patch_submitted = True
        return P.ok_response(patch_size=len(diff), files_changed=diff.count("diff --git"))

    def get_status(self):
        d = {"status": "ok", "tool_calls_used": self.b.used, "patch_submitted": self.patch_submitted,
             "patch_size": len(self.submitted_patch) if self.submitted_patch else 0}
        if self.b.tool_calls is not None:
            d["tool_calls_remaining"] = max(0, self.b.tool_calls - self.b.used)
            d["max_tool_calls"] = self.b.tool_calls
        if self.b.time_minutes is not None:
            d["time_seconds_remaining"] = (self.b.time_minutes * 60.0 if self.b.started is None
                                           else self.b.remaining_seconds)
            d["max_time_minutes"] = self.b.time_minutes
            d["agent_elapsed_seconds"] = self.b.elapsed
        if self.b.turns is not None:
            d["max_turns"] = self.b.turns
        if self.b.command_timeout is not None:
            d["command_timeout_seconds"] = self.b.command_timeout
        return json.dumps(d, ensure_ascii=False)

    # ---- workspace ------------------------------------------------------------
    def read_file(self, filepath, start_line=None, end_line=None):
        try:
            start = P.coerce_optional_int(start_line, "start_line")
            end = P.coerce_optional_int(end_line, "end_line")
            dest = P.resolve_workspace_path(filepath, allow_root=False)
            raw = self._read_bytes(dest).decode("utf-8", errors="replace")
            content = raw.replace("\r\n", "\n").replace("\r", "\n")      # Path.read_text: universal newlines
            return P.read_file_response(filepath, content, start, end, self.max_file_lines, self.max_file_chars)
        except Exception as e:
            return P.error_response("FileReadError", str(e))

    def edit_file(self, filepath, old_string, new_string, allow_multiple=False):
        try:
            dest = P.resolve_workspace_path(filepath, allow_root=False)
            content = self._read_bytes(dest).decode("utf-8", errors="replace")      # newline=''; nothing folded
            if len(content) == 0:
                raise ValueError(f"File '{filepath}' is empty. Use write_file to populate an empty or newly "
                                 "created file.")
            res = P.apply_replacement(content, old_string, new_string, filepath=filepath,
                                      allow_multiple=P.coerce_allow_multiple(allow_multiple))
            self._write_bytes(dest, res.new_content.encode("utf-8"))
            diff_str, truncated = res.diff, False
            if self.max_stdout is not None and len(diff_str) > self.max_stdout:
                diff_str, truncated = diff_str[:self.max_stdout], True
            return P.ok_response(filepath=filepath, occurrences=res.occurrences, strategy=res.strategy,
                                 diff=diff_str, is_truncated=truncated)
        except Exception as e:
            return P.error_response("FileEditError", str(e))

    def write_file(self, filepath, content):
        try:
            dest = P.resolve_workspace_path(filepath, allow_root=False)
            parent = posixpath.dirname(dest)
            rc, _, se = self.c.sh(f"mkdir -p {shlex.quote(parent)}")
            if rc != 0:
                raise RuntimeError(f"Failed to create directory {parent} inside container: {se}")
            self._write_bytes(dest, content.encode("utf-8"))
            return P.ok_response(filepath=filepath, size=len(content))
        except Exception as e:
            return P.error_response("FileWriteError", str(e))

    def run_skill_script(self, skill_name, file_path, args):
        """args is a dict; the harness renders it as command-line flags."""
        if not self.skills_root:
            return _err("SkillError", "no skills are attached to this agent")
        local = os.path.join(self.skills_root, skill_name, file_path)
        if not os.path.exists(local):
            return _err("SkillError", f"skill script not found: {skill_name}/{file_path}")
        dest = f"/tmp/_skill_{skill_name}_{os.path.basename(file_path)}"
        self.c.put_text(dest, open(local).read())
        flags = []
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"issue": args}
        for k, v in (args or {}).items():
            flags += [f"--{k}", str(v)]
        cmd = f"cd /workspace && python3 {shlex.quote(dest)} " + " ".join(shlex.quote(f) for f in flags)
        to = min(self.b.command_timeout, max(5, int(self.b.remaining_seconds)))
        try:
            rc, so, se = self.c.sh(cmd, timeout=to)
        except subprocess.TimeoutExpired:
            return _err("TimeoutExceeded", f"skill script exceeded {to}s")
        if rc != 0:
            return _err("SkillError", f"exit {rc}", stdout=so[:MAX_STDOUT], stderr=se[:MAX_STDOUT])
        return _ok(stdout=so[:MAX_STDOUT], exit_code=0)

    # ---- graph ---------------------------------------------------------------
    def _resolve_node(self, node):
        ids = self.graph.get("_ids") or []
        if node in self.graph.get("_idset", set()):
            return node
        for i in ids:
            if i.endswith("." + node) or i.endswith("/" + node):
                return i
        low = node.lower()
        for i in ids:
            if i.lower() == low:
                return i
        for i in ids:
            if low in i.lower():
                return i
        return None

    def get_code_neighbors(self, node, edge_type=None, max_neighbors=50):
        if not self.graph:
            return _err("GraphUnavailable", "no code graph for this repository")
        n = self._resolve_node(node)
        if not n:
            return _err("NodeNotFound", f"symbol not in graph: {node}")
        out = []
        for e in self.graph.get("edges", []):
            if edge_type and (e.get("type") or "").lower() != edge_type.lower():
                continue
            if e["source"] == n:
                out.append(e["target"])
            elif e["target"] == n:
                out.append(e["source"])
        out = list(dict.fromkeys(out))[:max_neighbors]
        return _ok(node=n, neighbors=out, count=len(out))

    def search_similar_code(self, query, k=10):
        if not self.emb:
            return _err("EmbeddingsUnavailable", "no embeddings for this repository")
        key = self._resolve_node(query)
        if key is None or key not in self.emb:
            return _err("QueryUnresolved",
                        "query did not resolve to a known symbol name; this tool compares "
                        "node embeddings and cannot embed free-form natural language")
        import math
        v = self.emb[key]
        nv = math.sqrt(sum(x * x for x in v)) or 1.0
        scored = []
        for kk, vv in self.emb.items():
            if kk == key:
                continue
            d = sum(a * b for a, b in zip(v, vv))
            nn = math.sqrt(sum(x * x for x in vv)) or 1.0
            scored.append((d / (nv * nn), kk))
        scored.sort(reverse=True)
        txt = {nd["id"]: nd.get("text", "") for nd in self.graph.get("nodes", [])}
        return _ok(query=query, results=[{"node_name": kk, "code": txt.get(kk, "")[:600],
                                          "similarity": round(s, 4)} for s, kk in scored[:k]],
                   count=min(k, len(scored)))

    def get_code_subgraph(self, nodes):
        if not self.graph:
            return _err("GraphUnavailable", "no code graph for this repository")
        res = [self._resolve_node(n) for n in nodes]
        keep = {r for r in res if r}
        edges = [{"from": e["source"], "to": e["target"], "type": e.get("type")}
                 for e in self.graph.get("edges", [])
                 if e["source"] in keep and e["target"] in keep]
        return _ok(nodes=sorted(keep), edges=edges, node_count=len(keep), edge_count=len(edges))
