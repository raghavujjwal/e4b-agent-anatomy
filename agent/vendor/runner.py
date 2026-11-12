#!/usr/bin/env python3
"""Container A + the agent loop: a local stand-in for compile_submission + run_agent_sandbox.

Implements the parts of the ADK contract that this competition's submissions actually use:
  * agent.yaml / sub_agents/*.yaml with the sandboxed !include directive
  * LlmAgent and SequentialAgent
  * session state seeded with problem_description and hints (README 5.1)
  * {var} interpolation of session state into `instruction` (README 2.3)
  * output_key publishing an agent's final text back into session state
  * include_contents: none
  * the 3 continuation nudges and their termination rule (README 5.3)

Model backends are pluggable so the plumbing can be validated with no GPU:
  gold   -- an oracle that writes the reference patch. Proves Container A -> submit_patch ->
            Container B end to end; its resolution MUST match `verify.py --mode gold`.
  script -- a deterministic policy that exercises the real tool path
  openai -- an OpenAI-compatible endpoint (vLLM), for the real model
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml
from sandbox import Container, bootstrap
from tools import Tools, Budget, CLOCK
import scorer_port as P

MAX_NUDGES = 3


# ---------------- submission loading ----------------
class Loader(yaml.SafeLoader):
    pass


SUB_ROOT = None   # set by load_submission; !include may not resolve outside it


def _include(loader, node):
    rel = loader.construct_scalar(node)
    # The rule is "may not resolve OUTSIDE the submission root", not "contains no ..".
    # `sub_agents/coder.yaml` including `../prompts/coder.md` is legal and is what the
    # sample submission itself does; rejecting every `..` broke the whole tree.
    if rel.startswith("/") or "\0" in rel:
        raise ValueError(f"illegal !include path: {rel}")
    tgt = os.path.normpath(os.path.join(os.path.dirname(loader.name), rel))
    if SUB_ROOT and not os.path.abspath(tgt).startswith(os.path.abspath(SUB_ROOT) + os.sep):
        raise ValueError(f"!include escapes submission root: {rel}")
    if os.path.islink(tgt):
        raise ValueError(f"!include is a symlink: {rel}")
    if tgt.endswith((".md", ".txt")):
        return open(tgt, errors="replace").read()
    with open(tgt, errors="replace") as fh:
        return yaml.load(fh, Loader)


Loader.add_constructor("!include", _include)


def load_yaml(path):
    with open(path, errors="replace") as fh:
        return yaml.load(fh, Loader)


def load_submission(root):
    global SUB_ROOT
    SUB_ROOT = os.path.abspath(root)
    for n in ("agent.yaml", "agent.yml", "root_agent.yaml", "root_agent.yml"):
        p = os.path.join(root, n)
        if os.path.exists(p):
            return load_yaml(p), p
    raise FileNotFoundError("no root agent config")


def resolve_tree(cfg, cfg_path):
    """Expand sub_agents config_path references into nested dicts, depth-first."""
    cfg = dict(cfg)
    subs = []
    for sa in (cfg.get("sub_agents") or []):
        if isinstance(sa, dict) and sa.get("config_path"):
            p = os.path.join(os.path.dirname(cfg_path), sa["config_path"])
            subs.append(resolve_tree(load_yaml(p), p))
    cfg["_sub"] = subs
    # `tools: [{agent_tool: {config_path, skip_summarization}}]` (README 2.3): an agent the parent
    # CALLS like a tool, only when it decides to -- unlike sub_agents, which always run.
    ats = {}
    for t in (cfg.get("tools") or []):
        if isinstance(t, dict) and isinstance(t.get("agent_tool"), dict):
            p = os.path.join(os.path.dirname(cfg_path), t["agent_tool"]["config_path"])
            sc = resolve_tree(load_yaml(p), p)
            sc["_skip_summarization"] = bool(t["agent_tool"].get("skip_summarization", False))
            ats[sc.get("name", os.path.basename(p))] = sc
    cfg["_agent_tools"] = ats
    return cfg


class TemplateKeyError(KeyError):
    """A non-optional `{var}` had no session-state entry. ADK treats this as fatal."""


# Verbatim from google-adk 2.11.0 `_instructions_utils._TEMPLATE_VAR_PATTERN`.
_ADK_VAR = re.compile(r"(?<![\$\{\\]){+[^{}]*}+")
_ADK_PREFIXES = ("app:", "user:", "temp:")


def _valid_state_name(var: str) -> bool:
    parts = var.split(":")
    if len(parts) == 1:
        return var.isidentifier()
    if len(parts) == 2 and parts[0] + ":" in _ADK_PREFIXES:
        return parts[1].isidentifier()
    return False


def interpolate(instruction: str, state: dict):
    """ADK session-state templating, reproducing google-adk 2.11.0's measured semantics.

    This used to leave unknown keys verbatim and merely record them, on the theory that silent
    degradation was the thing worth seeing. **Q-040 measured the real ADK and that was wrong in
    the dangerous direction.** `inject_session_state` -> `_render_with_regex` *raises KeyError*
    on a non-optional missing variable, while rendering the instruction, before the request is
    ever built -- so the agent does not degrade, it never runs. A lenient local renderer would
    have hidden the single worst failure mode of our own architecture.

    Faithfully reproduced, because each rule changes which prompts are safe:
      * `{var?}`            -> empty string when missing, never raises
      * value is None       -> empty string
      * not a valid state name (`{"key": 1}`, `{a-b}`) -> left verbatim, no raise
      * `app:` / `user:` / `temp:` prefixes are valid state names
      * whitespace inside the braces is stripped: `{ var }` resolves
      * **`{{var}}` IS substituted** -- the greedy `{+ ... }+` swallows both braces, so
        doubling is not an escape. Escaping needs a preceding `\\` or `$`: `\\{var}`, `${var}`.
        (Guessed wrong while writing this; measured against google-adk 2.11.0 instead.)

    Returns `(text, missing)` where `missing` lists only the *optional* vars that were absent;
    a non-optional one raises, so it can never be reported as a mere note.
    """
    missing = []

    def sub(m):
        raw = m.group()
        var = raw.lstrip("{").rstrip("}").strip()
        optional = var.endswith("?")
        if optional:
            var = var[:-1]
        if not _valid_state_name(var):
            return raw
        if var in state:
            v = state[var]
            return "" if v is None else str(v)
        if optional:
            missing.append(var)
            return ""
        raise TemplateKeyError(f"Context variable not found: `{var}`")

    return _ADK_VAR.sub(sub, instruction or ""), missing


# ---------------- model backends ----------------
class GoldOracle:
    """Writes the reference patch via the real tool path, then submits. A pipeline control."""
    name = "gold"

    def __init__(self, task):
        self.task = task
        self.stage = 0

    def step(self, system, messages, tool_names):
        # A SequentialAgent runs the localizer step first, and that step has no write_file.
        # Without this guard the oracle burned all its stages there as UnknownTool errors and
        # the coder never wrote anything -- which is exactly what the control caught.
        if "write_file" not in tool_names:
            return {"text": "NONE", "calls": []}
        if self.stage == 0:
            self.stage = 1
            # A patch without a trailing newline makes both `git apply` and GNU `patch`
            # fail ("patch unexpectedly ends in middle of line"), and several reference
            # patches here have none -- fastapi_15661 among them. Container B's apply_patch
            # already normalises this; the agent side must too.
            pt = self.task["patch"]
            return {"text": "", "calls": [("write_file", {
                "filepath": "_gold.patch",
                "content": pt if pt.endswith("\n") else pt + "\n"})]}
        if self.stage == 1:
            self.stage = 2
            # Mirror the harness's resilient application. Plain `git apply` is not enough:
            # these reference patches write new files with an `--- a/<path>` header and an
            # `@@ -0,0` hunk instead of `--- /dev/null`, which git refuses outright. That is
            # why HARNESS_README 8.2 has four fallback passes.
            P = "/workspace/_gold.patch"
            chain = " || ".join([
                f"git apply --unsafe-paths -p1 {P}",
                f"git apply --unsafe-paths -p1 -3 {P}",
                f"git apply --unsafe-paths -p1 --ignore-space-change --ignore-whitespace {P}",
                f"git apply --unsafe-paths -p1 --recount {P}",
                f"git apply --unsafe-paths -p0 {P}",
                # --no-backup-if-mismatch is essential on the AGENT side: GNU patch otherwise
                # leaves <file>.orig behind, `git add -N .` sweeps it into the submitted patch,
                # and a 411-byte fix became a 34,957-byte diff. The harness's
                # .git/info/exclude list contains no *.orig or *.rej, so this is a real hazard
                # for any agent that shells out to `patch`.
                f"patch -p1 --batch --forward --no-backup-if-mismatch < {P}",
                f"patch -p0 --batch --forward --no-backup-if-mismatch < {P}",
            ])
            return {"text": "", "calls": [("run_command", {"command": f"cd /workspace && ({chain})"})]}
        if self.stage == 2:
            self.stage = 3
            return {"text": "", "calls": [("run_command", {
                "command": "cd /workspace && rm -f _gold.patch && "
                           "find . -name '*.orig' -o -name '*.rej' | xargs -r rm -f; true"})]}
        if self.stage == 3:
            self.stage = 4
            return {"text": "", "calls": [("submit_patch", {})]}
        return {"text": "done", "calls": []}


class ScriptPolicy:
    """Deterministic policy over the real tool surface: localize, read, then submit.
    Exercises run_skill_script and state interpolation without a model."""
    name = "script"

    def __init__(self, task):
        self.task = task
        self.stage = 0
        self.cands = []

    def step(self, system, messages, tool_names):
        if self.stage == 0:
            self.stage = 1
            if "run_skill_script" in tool_names:
                return {"text": "", "calls": [("run_skill_script", {
                    "skill_name": "localize", "file_path": "scripts/localize.py",
                    "args": {"issue": self.task["problem_statement"][:4000], "k": 15}})]}
            self.stage = 2
        if self.stage == 1:
            self.stage = 2
            last = messages[-1]["content"] if messages else ""
            for line in str(last).splitlines():
                p = line.split()
                if len(p) >= 4 and p[0].isdigit():
                    self.cands.append(p[2])
            return {"text": "\n".join(str(last).splitlines()[:20]), "calls": []}
        if self.stage == 2:
            self.stage = 3
            if self.cands:
                f = self.cands[0].rsplit(":", 1)[0]
                return {"text": "", "calls": [("read_file", {"filepath": f})]}
        if self.stage == 3:
            self.stage = 4
            return {"text": "", "calls": [("submit_patch", {})]}
        return {"text": "done", "calls": []}


# Gemma emits tool calls in a python-ish block, so a call the server's parser rejects comes back
# as ordinary assistant `content` with an EMPTY tool_calls list -- indistinguishable, without
# this, from the model simply talking. Anomaly A3 puts that rate at 31% and OUT-015's success
# criterion is "<5%", so it has to be counted rather than inferred.
_TOOL_ATTEMPT = re.compile(
    r"```\s*tool_code|<tool_call|default_api\.|print\s*\(\s*[A-Za-z_]+\s*\(|"
    r"\{\s*[\"']?(?:name|tool_name|function)[\"']?\s*:", re.I)


def looks_like_tool_attempt(text, tool_names):
    """True when assistant TEXT appears to be a tool call the server failed to parse.

    Deliberately two-pronged: either an unmistakable call syntax marker, or a declared tool
    name immediately followed by call-ish punctuation. Prose that merely mentions a tool
    ("I will read the file with read_file next") must NOT match, which is what the negative
    cases in `experiments/q004_malformed_detector.py` pin down.
    """
    if not text:
        return False
    if _TOOL_ATTEMPT.search(text):
        return True
    for t in tool_names or ():
        # `def read_file(` and `self.read_file(` are code the model is READING or quoting, not a
        # call it is attempting. The control caught both as false positives, and every false
        # positive inflates the very rate OUT-015 exists to drive down.
        if re.search(r"(?<!def )(?<!class )(?<!\.)\b" + re.escape(t) + r"\s*[({]", text):
            return True
    return False


class OpenAIBackend:
    """OpenAI-compatible chat completions with tool calling (vLLM)."""
    name = "openai"

    def __init__(self, base_url, model, gen=None, timeout=300, retries=5):
        self.base, self.model, self.gen, self.timeout = base_url.rstrip("/"), model, gen or {}, timeout
        # The harness binds LiteLlm(..., num_retries=5) (HARNESS_README:198). With no retries a
        # single transient 5xx costs whichever ARM happened to be running, which is a bias that
        # reads as a result -- so match the count rather than improvise one.
        self.retries = retries
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.calls = 0
        self.malformed = 0          # parsed as text but looks like an attempted call
        self.truncated = 0          # finish_reason == "length"
        self.http_retries = 0
        self.malformed_samples = [] # keep the first few verbatim: the point is to SEE them
        self.last_prompt_tokens = 0

    def summarize(self, prompt):
        """ADK's LlmEventSummarizer call: one user message, no tools. Its LlmRequest carries no generation config,
        so no temperature, top_p, top_k or token cap (vLLM falls back to the model's generation_config.json and
        nothing caps it but the context); but it goes through the same LiteLlm object as the agent's calls, and
        adk-submission attached the thinking settings and the seed to that object, so those are sent
        Returns (visible_text, reasoning, prompt_tokens)."""
        out = self.step(None, [{"role": "user", "content": prompt}], [], None, gen_defaults=False)
        return out.get("text") or "", out.get("reasoning") or "", self.last_prompt_tokens

    def step(self, system, messages, tool_names, tool_schemas=None, gen_defaults=True):
        import urllib.request, urllib.error
        # The generation config the scorer actually uses: the submission's values over the organizer defaults
        # (max_output_tokens 16384, thinking_budget 4096), then adk-submission's thinking bridge. The bridge's
        # fields (thinking, seed, reasoning_effort) live on the model object and go out with every call; the
        # sampling fields come from the request's own config, which the summarizer call does not have
        # (gen_defaults=False).
        merged = P.merge_generation_config(self.gen)
        cfg = merged if gen_defaults else {}
        think = P.thinking_request_fields(merged)
        def wire(m):
            out = {k: v for k, v in m.items() if not k.startswith("_")}
            if m.get("_parts"):          # another agent's reply as ADK hands it on: one user message made of several text parts (LiteLLM keeps the list)
                out["content"] = [{"type": "text", "text": t} for t in m["_parts"]]
            return out
        msgs = ([{"role": "system", "content": system}] if system else []) + [wire(m) for m in messages]
        # History in the scorer's wire format: a tool-call-only assistant turn has no `content` key, and thoughts go
        # back as reasoning_content AND reasoning (vLLM 0.19.1 reads only `reasoning`). include_thoughts: false
        # strips them (adk-submission install_litellm_reasoning_patch).
        for m in msgs:
            if m.get("role") != "assistant":
                continue
            if m.get("content") is None:
                m.pop("content", None)
            if think["include_thoughts"] is False:
                m.pop("reasoning_content", None)
                m.pop("reasoning", None)
            elif m.get("reasoning_content") is not None and m.get("reasoning") is None:
                m["reasoning"] = m["reasoning_content"]
        body = {"model": self.model, "messages": msgs}
        for k in ("max_output_tokens", "temperature", "top_p", "presence_penalty", "frequency_penalty",
                  "stop_sequences"):
            if cfg.get(k) is not None:
                body["max_completion_tokens" if k == "max_output_tokens" else
                     ("stop" if k == "stop_sequences" else k)] = cfg[k]
        if cfg.get("top_k") is not None:
            body["top_k"] = float(cfg["top_k"])          # ADK sends it through extra_body as a float
        if "seed" in think:
            body["seed"] = think["seed"]
        if "reasoning_effort" in think:
            body["reasoning_effort"] = think["reasoning_effort"]
        body.update(think["extra_body"])
        # thinking_config IS translated, the way adk-submission 0.2.13 does it (resolvers/generation.py,
        # apply_thinking_config_to_model): thinking_budget 0 -> enable_thinking false; a positive budget ->
        # true plus thinking_token_budget; nothing at all -> the organizer default (budget 4096). HARNESS_README
        # (section 2.4) documents this. Until 2026-11-04
        # this backend ignored thinking_config, on the strength of google-adk's own LiteLlm adapter not forwarding
        # it; the scorer does not use that adapter alone. The hidden scorer's package version is still not
        # verified, so the shipped bundle's thinking mode is an A/B (gate M1), not an assumption.
        #
        # A positive thinking_token_budget needs the SERVER started the way adk_submission.server.VllmServer
        # starts it for the gemma4 reasoning parser: `--reasoning-parser gemma4 --reasoning-config
        # '{"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}'`. Without the second flag
        # vLLM answers 400 to every request (handled below).
        body.setdefault("chat_template_kwargs", {"enable_thinking": True})
        # MEASUREMENT ONLY (EXP-035): LOCALEVAL_NOTHINK=1 turns thinking off, which the grader
        # cannot do by config. It measures what thinking buys per grader-second; if little, a
        # LoRA trained on these no-think completions can make the model skip it on the grader.
        if os.environ.get("LOCALEVAL_NOTHINK") == "1":
            body["chat_template_kwargs"] = {"enable_thinking": False}
        # TEACHER runs only (EXP-036): extra chat-template kwargs, e.g. '{"reasoning_effort": "low"}'
        # for Qwen3.8 -- its thoughts become Gemma's thinking targets, so they must stay short.
        if os.environ.get("LOCALEVAL_TEMPLATE_KWARGS"):
            body["chat_template_kwargs"].update(json.loads(os.environ["LOCALEVAL_TEMPLATE_KWARGS"]))
        # Exact token ids of the prompt and of the completion (vLLM extension; it changes
        # nothing about generation). This is the SFT data, by construction: the Gemma 4 template
        # DROPS past reasoning from history and opens the thought channel differently after a
        # user message and after a tool response, so re-rendering a conversation afterwards
        # cannot reproduce what the model actually saw and emitted (training/tokenize_sft.py
        # found the server's own rendering not prefix-stable at turn 4).
        body["return_token_ids"] = True
        if tool_schemas:
            body["tools"] = tool_schemas       # no tool_choice: the scorer's LiteLlm does not send one
        req = urllib.request.Request(f"{self.base}/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        last = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                # Carry the BODY. vLLM explains a 400 there (context length, bad tool schema,
                # malformed message order), and without it the trace recorded 185 identical
                # "HTTPError: HTTP Error 400: Bad Request" lines -- the same opaque-error
                # pattern that made the agent retry a broken skill call 24 times.
                try:
                    detail = e.read().decode("utf-8", "replace")[:1200]
                except Exception:
                    detail = "<body unreadable>"
                last = RuntimeError(f"HTTP {e.code}: {detail}")
                if e.code == 400 and "reasoning_config" in detail:
                    raise RuntimeError(
                        "the server refused thinking_token_budget: start vLLM with the scorer's reasoning flags, "
                        "--reasoning-parser gemma4 --reasoning-config "
                        "'{\"reasoning_start_str\": \"<|channel>\", \"reasoning_end_str\": \"<channel|>\"}' "
                        "(adk_submission.server.VllmServer adds the second one itself). Server said: " + detail)
                if e.code < 500 and e.code != 429:
                    raise last       # a 400 is our bug, not the server's weather -- fail loudly
            except Exception as e:
                last = e
            if attempt == self.retries:
                raise last
            self.http_retries += 1
            time.sleep(min(2 ** attempt, 16))
        self.calls += 1
        u = out.get("usage") or {}
        self.completion_tokens += u.get("completion_tokens", 0)
        self.prompt_tokens += u.get("prompt_tokens", 0)
        self.last_prompt_tokens = u.get("prompt_tokens", 0)
        ch = (out.get("choices") or [{}])[0]
        m = ch.get("message") or {}
        calls, raw_calls = [], []
        for i, t in enumerate(m.get("tool_calls") or []):
            fn = t.get("function") or {}
            try:
                a = json.loads(fn.get("arguments") or "{}")
            except Exception:
                a = {}
            calls.append((fn.get("name"), a))
            # Kept verbatim so the history we send back is what ADK's LiteLlm sends: an
            # assistant message carrying `tool_calls` (with ids), answered by role "tool".
            raw_calls.append({"type": "function", "id": t.get("id") or f"call_{self.calls}_{i}",
                              "function": {"name": fn.get("name"),
                                           "arguments": fn.get("arguments") or "{}"}})
        reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
        tok_rec = None
        if out.get("prompt_token_ids") is not None and ch.get("token_ids") is not None:
            tok_rec = {"prompt": out["prompt_token_ids"], "gen": ch["token_ids"],
                       "finish": ch.get("finish_reason")}
        text = m.get("content") or ""
        finish = ch.get("finish_reason") or ""
        if finish == "length":
            self.truncated += 1
        if not calls and looks_like_tool_attempt(text, tool_names):
            self.malformed += 1
            if len(self.malformed_samples) < 10:
                self.malformed_samples.append(text[:600])
        return {"text": text, "calls": calls, "raw_calls": raw_calls, "reasoning": reasoning,
                "finish_reason": finish, "tokens": tok_rec}


# Tool declarations as ADK builds them from the harness functions (HARNESS_README section 6):
# every parameter typed, parameters with defaults NOT required, Optional[int] nullable. The
# descriptions are verbatim from the host's "Model Selection, Budget, and Harness Rules" page
# (external/competition_pages/), which lists each tool's one-line docstring.
#
# This replaced placeholder schemas ("read_file tool", with start_line/end_line undeclared):
# the model was being shown a different tool block from the grader's, and that block is also
# the prompt every SFT example is trained under, so it has to match.
_STR, _INT, _BOOL = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}
_OPT_INT = {"type": "integer", "nullable": True}
TOOL_DECLS = {
    "run_command": ("Executes a shell command in `/bin/bash -c` inside `/workspace`.",
                    {"command": _STR}, ["command"]),
    "submit_patch": ("Stages untracked file intents (`git add -N .`) and captures `git diff HEAD` "
                     "from `/workspace`.", {}, []),
    "get_status": ("Returns live budget consumption and patch status.", {}, []),
    "read_file": ("Reads a file from `/workspace` with 1-indexed inclusive line slicing.",
                  {"filepath": _STR, "start_line": _OPT_INT, "end_line": _OPT_INT}, ["filepath"]),
    "edit_file": ("Replaces `old_string` with `new_string` in an existing non-empty file inside "
                  "`/workspace`.",
                  {"filepath": _STR, "old_string": _STR, "new_string": _STR,
                   "allow_multiple": _BOOL}, ["filepath", "old_string", "new_string"]),
    "write_file": ("Creates or overwrites a file at `/workspace/`, automatically creating parent "
                   "directories (`mkdir -p`).", {"filepath": _STR, "content": _STR},
                   ["filepath", "content"]),
    "get_code_neighbors": ("Finds incoming and outgoing neighbors of a symbol (`node`) in the "
                           "repository call/dependency graph.",
                           {"node": _STR, "edge_type": {"type": "string", "nullable": True},
                            "max_neighbors": _INT}, ["node"]),
    "search_similar_code": ("Finds top-`k` graph nodes with highest cosine similarity to `query` "
                            "in the pre-computed embeddings.",
                            {"query": _STR, "k": _INT}, ["query"]),
    "get_code_subgraph": ("Extracts the induced subgraph (all nodes and interconnecting edges) "
                          "for a list of symbols.",
                          {"nodes": {"type": "array", "items": _STR}}, ["nodes"]),
    "run_skill_script": ("Runs a script from a loaded skill.",
                         {"skill_name": _STR, "file_path": _STR, "args": {"type": "object"}},
                         ["skill_name", "file_path"]),
}
TOOL_SCHEMAS = {n: {k: v["type"] for k, v in d[1].items()} for n, d in TOOL_DECLS.items()}


def _scorer_tools():
    """Declarations captured from the scorer's own ADK/LiteLLM path (fixtures/scorer_tools.json)."""
    global _SCORER_TOOLS
    if _SCORER_TOOLS is None:
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "scorer_tools.json"),
                      encoding="utf-8") as fh:
                _SCORER_TOOLS = json.load(fh)["tools"]
        except OSError:
            _SCORER_TOOLS = {}
    return _SCORER_TOOLS


_SCORER_TOOLS = None


def schemas_for(names, agent_tools=None):
    out = []
    captured = _scorer_tools()
    for n in names:
        if n in captured and not (agent_tools and n in agent_tools):
            out.append(captured[n])       # byte-for-byte what the scorer sends for this tool
            continue
        if agent_tools and n in agent_tools:
            # google-adk AgentTool._get_declaration: no input_schema -> one string param `request`.
            out.append({"type": "function", "function": {
                "name": n, "description": agent_tools[n].get("description") or "",
                "parameters": {"type": "object", "properties": {"request": _STR},
                               "required": ["request"]}}})
            continue
        desc, props, req = TOOL_DECLS.get(n, (f"{n} tool", {}, []))
        out.append({"type": "function", "function": {
            "name": n, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}})
    return out


# ---------------- ADK events compaction (token threshold) ----------------
# The grader's App sets EventsCompactionConfig(token_threshold=14336, event_retention_size=5,
# compaction_interval=5, overlap_size=2) (HARNESS_README 7.2). Mirrors google-adk
# apps/compaction.py + apps/llm_event_summarizer.py + flows/llm_flows/contents.py:
#   * checked BEFORE every model call (CompactionRequestProcessor), against the latest observed
#     prompt_token_count;
#   * all events since the last compaction except the last 5 are summarized, the split moved
#     earlier so no retained tool response loses its call; the previous summary seeds the next;
#   * the summarizer sees ONLY text parts -- the issue, visible text and THOUGHTS. Tool calls and
#     tool results (all the code the agent read) are not in it;
#   * the summary re-enters the prompt as a MODEL message in place of the compacted events --
#     including the original user message, the issue itself.
# The sliding-window half (compaction_interval/overlap) counts INVOCATIONS, and one task is one
# invocation (plus nudges), so it is not modelled.
COMPACT_THRESHOLD = int(os.environ.get("LOCALEVAL_COMPACT_THRESHOLD", 14336))  # override = test only
COMPACT_RETAIN = 5
SUMMARY_PROMPT = ("The following is a conversation history between a user and an AI agent. Please "
                  "summarize the conversation, focusing on key information and decisions made, as "
                  "well as any unresolved questions or tasks. The summary should be concise and "
                  "capture the essence of the interaction.\n\n{conversation_history}")


def _events(messages):
    """Group messages into ADK events: consecutive tool responses of one agent form one event."""
    ev = []
    for m in messages:
        if m["role"] == "tool" and ev and ev[-1][0]["role"] == "tool" and ev[-1][0].get("_author") == m.get("_author"):
            ev[-1].append(m)
        else:
            ev.append([m])
    return ev


# ---------------- the session log and what each agent is shown of it ----------------
# google-adk keeps ONE list of events per session and every agent builds its prompt from it (flows/llm_flows/contents.py: _get_contents, _get_current_turn_contents,
# _present_other_agent_message). Here `state["_contents"]` is that list in chat form: each message an agent produced carries `_author` (its name); the user's messages
# (the scorer's first message, the nudges) and a compaction summary carry none.
#   * the user's events and the agent's OWN events are shown as they are;
#   * another agent's events become USER messages made of text parts, "For context:" first and then one part per piece of the event -- "[<agent>] said: <text>",
#     "[<agent>] called tool `<name>` with parameters: <args dict>", "[<agent>] `<name>` tool returned result: <response dict>" (Python repr; thoughts left out; an
#     event with nothing left is dropped). A later stage of a SequentialAgent sees the earlier one this way, and so does the first stage when a nudge starts the root
#     over (the scorer's outer loop sends the nudge as a new user message of the same session, and a SequentialAgent root always restarts at its first stage);
#   * include_contents: none starts at the latest event of the user or of another agent (the "current turn") and takes everything after it.
# The shapes were captured against the scorer's own code: tests/fixtures/scorer_requests_pipe_s2.json (stage after stage), _empty.json (a stage without text) and
# _nudge.json (the second pass). A single agent never sees another author, so for it the view is the log itself.
def _tool_names(log):
    return {c["id"]: c["function"]["name"] for m in log for c in (m.get("tool_calls") or [])}


def _repr_args(raw):
    try:
        v = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return v if isinstance(v, dict) else {}


def _repr_response(content):
    """The dict ADK holds as a function_response: the tool's result, wrapped as {'result': ...} when it is not a dict."""
    try:
        v = json.loads(content)
    except (TypeError, ValueError):
        return {"result": content}
    return v if isinstance(v, dict) else {"result": v}


def _visible(event):
    """ADK drops events with nothing in them (contents._contains_empty_content); a compaction summary always stays."""
    m = event[0]
    return bool(m.get("content") or m.get("tool_calls") or m.get("_summary")) if m["role"] in ("user", "assistant") else True


def _present_other(event, author, names):
    parts = []
    for m in event:
        if m["role"] == "assistant":
            if (m.get("content") or "").strip():
                parts.append(f"[{author}] said: {m['content']}")
            for c in (m.get("tool_calls") or []):
                parts.append(f"[{author}] called tool `{c['function']['name']}` with parameters: {_repr_args(c['function'].get('arguments'))}")
        elif m["role"] == "tool":
            parts.append(f"[{author}] `{names.get(m.get('tool_call_id'), '')}` tool returned result: {_repr_response(m.get('content'))}")
    if not parts:
        return None
    return {"role": "user", "content": "For context:\n" + "\n".join(parts), "_parts": ["For context:"] + parts}


def view(log, name, include_contents=None):
    """The messages the agent `name` is shown of the session log."""
    events = [e for e in _events(log) if _visible(e)]
    if include_contents == "none":
        for i in range(len(events) - 1, -1, -1):
            author = events[i][0].get("_author")
            if (events[i][0]["role"] == "user" and author is None) or (author is not None and author != name):
                events = events[i:]
                break
        else:
            return []
    names, out = _tool_names(log), []
    for e in events:
        author = e[0].get("_author")
        if author is None or author == name:
            out.extend(e)
        elif (m := _present_other(e, author, names)) is not None:
            out.append(m)
    return out


def traced(m):
    """A message as it is written to the trace: without the author tag (the event's `agent` says it for the conversation as a whole)."""
    return {k: v for k, v in m.items() if k != "_author"}


def maybe_compact(messages, backend, agent_name, trace):
    if getattr(backend, "last_prompt_tokens", 0) < COMPACT_THRESHOLD or not hasattr(backend, "summarize"):
        return messages
    seed = messages[0] if messages and messages[0].get("_summary") else None
    cand = _events(messages[1:] if seed else messages)
    if len(cand) <= COMPACT_RETAIN:
        return messages
    initial = len(cand) - COMPACT_RETAIN
    split, unmatched = 0, set()
    for i in range(len(cand) - 1, -1, -1):
        for m in cand[i]:
            if m["role"] == "tool":
                unmatched.add(m.get("tool_call_id"))
            for c in (m.get("tool_calls") or []):
                unmatched.discard(c["id"])
        if not unmatched and i <= initial:
            split = i
            break
    if split == 0:
        return messages
    lines = []
    if seed:
        lines.append(f"user: {seed['content']}")
    for e in cand[:split]:
        for m in e:
            if m["role"] == "user" and m.get("content"):
                lines.append(f"user: {m['content']}")
            elif m["role"] == "assistant":
                who = m.get("_author") or agent_name          # ADK's summarizer prefixes every line with the event's author
                if m.get("reasoning_content"):
                    lines.append(f"{who}: {m['reasoning_content']}")
                if m.get("content"):
                    lines.append(f"{who}: {m['content']}")
    t0 = time.time()
    try:
        text, reasoning, ptok = backend.summarize(SUMMARY_PROMPT.format(conversation_history="\n".join(lines)))
    except Exception as e:
        trace.append({"agent": agent_name, "event": "compaction_failed", "error": str(e)[:300]})
        return messages
    backend.last_prompt_tokens = ptok      # the compaction event is now the latest with usage
    summary = {"role": "assistant", "content": text or None, "_summary": True}
    if reasoning:
        summary["reasoning_content"] = reasoning
    kept = [m for e in cand[split:] for m in e]
    # The full pre-compaction history, so trajectory-level SFT (training/build_traj_sft.py) can
    # still train on the turns this compaction is about to drop from the message list.
    trace.append({"agent": agent_name, "event": "segment", "messages": [traced(m) for m in messages]})
    trace.append({"agent": agent_name, "event": "compaction", "events_compacted": split,
                  "events_kept": len(cand) - split, "summary_chars": len(text),
                  "seconds": round(time.time() - t0, 1), "summary_head": text[:300]})
    return [summary] + kept


# ---------------- grader-time meter ----------------
class Meter:
    """Critical-path cost of a run, for the grader time model (0.6 s/call + tokens / 25.5 tok/s).
    Sequential work adds; a ParallelAgent adds only its slowest branch, because the grader's vLLM
    batches concurrent requests (4x L4, TP=4: a batch-1 decode step is communication-bound)."""
    def __init__(self):
        self.tokens, self.calls, self.extra_s, self.par_k = 0, 0, 0.0, 1
        self.par_extra_s = 0.0   # clock only: the batch slowdown paid by absorbed parallel branches

    def grader_s(self):
        return 0.6 * self.calls + self.tokens / 25.5 + self.extra_s

    def clock_s(self, slowdown=1.0):
        """Grader session time so far (tools.Budget clock="grader"). `slowdown` scales this meter's
        own decode time: PAR_SLOWDOWN inside a parallel branch, where k streams share the GPU."""
        return 0.6 * self.calls + slowdown * self.tokens / 25.5 + self.extra_s + self.par_extra_s

    def absorb_slowest(self, branches):
        slow = max(branches, key=lambda m: m.grader_s())
        self.tokens += slow.tokens
        self.calls += slow.calls
        self.extra_s += slow.extra_s
        self.par_extra_s += slow.par_extra_s + (PAR_SLOWDOWN - 1) * slow.tokens / 25.5
        self.par_k = max(self.par_k, len(branches), *(m.par_k for m in branches))


# Decode slowdown of k concurrent streams on the grader (4x L4, TP=4), as in autoresearch.PAR_SLOWDOWN.
# Unmeasured on the real grader; 1.15 is the conservative guess the ship rule uses.
PAR_SLOWDOWN = 1.15


_COUNTERS = ("completion_tokens", "prompt_tokens", "calls", "malformed", "truncated", "http_retries")


def run_parallel(cfg, tools, state, backend, sub_root, trace, depth, meter):
    """google-adk 2.11 ParallelAgent: every sub-agent runs CONCURRENTLY in its own branch
    (`<parallel>.<sub>`). A branch sees the pre-fork history plus its own events only
    (contents._is_event_belongs_to_branch); an agent after the fork has no branch and sees all of
    them. asyncio.TaskGroup semantics: an exception in one branch cancels its siblings (here: the
    shared `_fatal` flag, checked at every turn)."""
    import copy, threading
    base = list(state.get("_contents", []))
    start = {a: getattr(backend, a) for a in _COUNTERS if hasattr(backend, a)}
    jobs = []
    for sub in cfg["_sub"]:
        bs = dict(state)
        bs["_contents"] = list(base)
        bs["_fatal_box"] = state.setdefault("_fatal_box", {})
        jobs.append((sub, bs, copy.copy(backend), Meter()))
    # Grader clock: every branch starts at the fork's session time and advances by its own cost at the
    # batch slowdown; after the join the parent's meter has absorbed the slowest branch.
    parent_clock = getattr(CLOCK, "fn", None)
    fork_s = parent_clock() if parent_clock is not None else None

    def go(job):
        sub, bs, bb, bm = job
        if fork_s is not None:
            CLOCK.fn = lambda: fork_s + bm.clock_s(PAR_SLOWDOWN)
        try:
            run_agent(sub, tools, bs, bb, sub_root, trace, depth + 1, meter=bm)
        except Exception as e:   # a crash in our own code is fatal too, as in the TaskGroup
            bs["_fatal_box"]["error"] = f"{sub.get('name')}: {type(e).__name__}: {e}"[:300]
    ths = [threading.Thread(target=go, args=(j,), daemon=True) for j in jobs]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    merged = list(base)
    for sub, bs, bb, bm in jobs:
        merged += bs.get("_contents", [])[len(base):]
        for k, v in bs.items():
            if not k.startswith("_"):
                state[k] = v
        state["_turn_had_tool_call"] = state.get("_turn_had_tool_call") or bs.get("_turn_had_tool_call")
        for k in ("_last_finish_reason", "_last_assistant_text"):
            if k in bs:
                state[k] = bs[k]
        for a, v0 in start.items():
            setattr(backend, a, getattr(backend, a) + getattr(bb, a) - v0)
    state["_contents"] = merged
    if meter is not None:
        meter.absorb_slowest([bm for *_, bm in jobs])
    trace.append({"agent": cfg.get("name"), "event": "parallel_join",
                  "branches": [{"agent": s.get("name"), "tokens": m.tokens, "calls": m.calls,
                                "grader_s": round(m.grader_s(), 1)} for s, _, _, m in jobs],
                  "fatal": state.get("_fatal_box", {}).get("error")})


# ---------------- the agent loop ----------------
def run_agent(cfg, tools: Tools, state: dict, backend, sub_root, trace: list, depth=0, meter=None):
    kind = cfg.get("agent_class", "LlmAgent")
    if state.get("_fatal_box", {}).get("error"):
        return
    if kind == "SequentialAgent":
        for sub in cfg["_sub"]:
            run_agent(sub, tools, state, backend, sub_root, trace, depth + 1, meter=meter)
            if tools.patch_submitted or state.get("_fatal_box", {}).get("error"):
                return
        return
    if kind == "ParallelAgent":
        run_parallel(cfg, tools, state, backend, sub_root, trace, depth, meter)
        return
    if kind == "LoopAgent":
        iters = int(cfg.get("max_iterations", 1))
        for _ in range(max(1, iters)):
            for sub in cfg["_sub"]:
                run_agent(sub, tools, state, backend, sub_root, trace, depth + 1, meter=meter)
                if tools.patch_submitted or state.get("_fatal_box", {}).get("error"):
                    return
        return

    name = cfg.get("name", f"agent{depth}")
    try:
        instr, missing = interpolate(cfg.get("instruction") or "", state)
    except TemplateKeyError as e:
        # What the real ADK does here: the instruction never renders, so this agent never
        # reaches the model. Recorded and returned rather than raised, so one poisoned task
        # does not abort a 100-task sweep -- but it scores as the zero it really is.
        trace.append({"agent": name, "event": "TEMPLATE_KEYERROR", "error": str(e)})
        return
    if missing:
        trace.append({"agent": name, "event": "OPTIONAL_VAR_ABSENT", "vars": missing})
    tool_names = [t for t in (cfg.get("tools") or []) if isinstance(t, str)]
    if cfg.get("skills"):
        tool_names = tool_names + ["run_skill_script", "load_skill_resource"]
    agent_tools = cfg.get("_agent_tools") or {}
    tool_names = tool_names + list(agent_tools)
    gen = cfg.get("generate_content_config") or {}
    # ADK appends its identity sentence to every agent's system instruction (flows/llm_flows/identity.py).
    system_text = P.adk_system_text(instr, name, cfg.get("description") or "")
    tools.available = list(tool_names)      # ADK answers an undeclared tool name with its own error text
    if hasattr(backend, "gen"):
        backend.gen = gen
    # `adapter: <name>` routes this agent's calls to the LoRA served as `<name>` (vLLM
    # --lora-modules name=path), as the grader's resolve_swegemma_adapter does; agents without
    # one use the base model. Set per agent, so a sub-agent's adapter never leaks to its parent.
    def bind_backend():   # re-applied after an agent_tool call: the sub-agent rebinds both
        if hasattr(backend, "gen"):
            backend.gen = gen
        if isinstance(backend, OpenAIBackend):
            if not hasattr(backend, "base_model"):
                backend.base_model = backend.model
            backend.model = cfg.get("adapter") or backend.base_model
    bind_backend()

    include = cfg.get("include_contents")      # "none" -> the current turn only (see view); anything else -> the whole session

    def emit(m):
        """Add a message this agent produced to the session log."""
        m["_author"] = name
        state["_contents"].append(m)

    final_text = ""
    ended_text = ""          # the text of the reply that ended this agent (ADK's final response); "" when the budget or a skipped summary ended it
    turn_had_tool_call = False
    turn_tokens = []
    while True:
        if state.get("_fatal_box", {}).get("error"):
            break
        if tools.b.exhausted() or tools.b.turns_used >= tools.b.turns:
            trace.append({"agent": name, "event": "budget_stop",
                          "calls": tools.b.used, "elapsed": round(tools.b.elapsed, 1)})
            break
        tools.b.turns_used += 1
        tok0 = getattr(backend, "completion_tokens", 0)
        # An AgentTool sub-session is a bare Runner(agent=...) with no App, hence NO events
        # compaction (google-adk 2.11 tools/agent_tool.py): its context only grows.
        if not state.get("_no_compact"):
            state["_contents"] = maybe_compact(state["_contents"], backend, name, trace)      # the session's events, as ADK's CompactionRequestProcessor sees them
        messages = view(state["_contents"], name, include)
        # google-adk lite_llm._append_fallback_user_content_if_missing: a request with no user
        # content gets this user message -- request-only.
        req = messages if any(m.get("role") in ("user", "tool") for m in messages) else \
            messages + [{"role": "user", "content": "Handle the requests as specified in the System Instruction."}]
        try:
            if isinstance(backend, OpenAIBackend):
                out = backend.step(system_text, req, tool_names, schemas_for(tool_names, agent_tools))
            else:
                out = backend.step(system_text, req, tool_names)
        except Exception as e:
            trace.append({"agent": name, "event": "model_error", "error": f"{type(e).__name__}: {e}"})
            if depth > 0 and "_fatal_box" in state:
                # Below the root a non-retryable model error RAISES out of the sub-agent / branch:
                # the AgentTool call fails, the ParallelAgent's TaskGroup cancels the siblings,
                # and the invocation ends. Only the harness's patch fallback survives.
                state["_fatal_box"]["error"] = f"{name}: {type(e).__name__}: {e}"[:300]
            break
        if meter is not None:
            meter.tokens += getattr(backend, "completion_tokens", 0) - tok0
        text, calls = out.get("text") or "", out.get("calls") or []
        if out.get("tokens"):
            # Delta-encoded against the previous turn's prompt: consecutive prompts share a long
            # prefix, so storing it whole would make a 25-turn trace ~4 MB.
            pr, prev = out["tokens"]["prompt"], (turn_tokens[-1]["_prompt"] if turn_tokens else [])
            k = 0
            while k < min(len(pr), len(prev)) and pr[k] == prev[k]:
                k += 1
            turn_tokens.append({"_prompt": pr, "prefix_len": k, "suffix": pr[k:],
                                "gen": out["tokens"]["gen"], "finish": out["tokens"]["finish"]})
        final_text = text or final_text
        # History in ADK's wire format (google/adk/models/lite_llm.py,
        # _content_to_message_param): the assistant turn keeps its tool_calls -- name AND
        # arguments -- plus reasoning_content, and each result returns as role "tool" with the
        # matching tool_call_id. This runner used to send `content` plus a bare call COUNT and
        # return results as user messages "[read_file result] ...", so the model never saw
        # which file, range or pattern it had asked for. Re-reading the same region and
        # re-running the same grep was the dominant failure of arm H; on that history it is
        # partly our artifact, not the model's.
        raw = out.get("raw_calls") or []
        if text or calls:
            am = {"role": "assistant", "content": text or None}
            if raw:
                am["tool_calls"] = raw
            if out.get("reasoning"):
                am["reasoning_content"] = out["reasoning"]
            emit(am)
        skip_summary = False
        for i, (cname, cargs) in enumerate(calls):
            if meter is not None and cname not in ("submit_patch", "get_status"):
                meter.calls += 1
            if cname in agent_tools:
                res = call_agent_tool(agent_tools[cname], (cargs or {}).get("request", ""),
                                      tools, state, backend, sub_root, trace, depth, meter=meter)
                bind_backend()
                skip_summary = skip_summary or agent_tools[cname]["_skip_summarization"]
            else:
                t_call = time.time()
                res = tools.call(cname, cargs)
                if meter is not None and cname == "run_skill_script":
                    # 0.6 s/call is the fitted average for ordinary commands; a skill script that
                    # runs tests (pick.py) costs its full wall time on the grader.
                    meter.extra_s += max(0.0, time.time() - t_call - 0.6)
            trace.append({"agent": name, "tool": cname, "args_keys": sorted((cargs or {}).keys()),
                          "result_head": res[:200]})
            if i < len(raw):
                emit({"role": "tool", "tool_call_id": raw[i]["id"], "content": res})
            else:   # scripted backends (gold/script) carry no ids; keep their old shape
                emit({"role": "user", "content": f"[{cname} result] {res[:6000]}"})
        if tools.patch_submitted and not state.get("_submit_traced"):
            # The scorer does NOT stop here: the function response goes back to the model, which answers with its
            # closing text-only message, and only that ends the session (agent_runner.py, "Agent submitted patch and
            # returned final completion message; terminating"). Until 2026-11-04 this loop broke at the submit, so a
            # local run was one model call shorter than a scored one and never contained the closing turn.
            trace.append({"agent": name, "event": "patch_submitted"})
            state["_submit_traced"] = True
        if skip_summary:
            # skip_summarization: ADK makes the function-response event this agent's final
            # response, so the turn ends here without the model reading the result.
            state["_last_finish_reason"] = ""
            state["_last_assistant_text"] = ""
            trace.append({"agent": name, "event": "final_response_skip_summarization"})
            break
        if calls:
            turn_had_tool_call = True
        else:
            # A text-only response IS this agent's final response (README 5.3). The agent step
            # ends here; the 3-nudge continuation mechanism belongs to the OUTER loop around
            # the ROOT agent, not inside each sub-agent.
            state["_last_finish_reason"] = (out.get("finish_reason") or "")
            state["_last_assistant_text"] = text
            ended_text = text
            trace.append({"agent": name, "event": "final_response", "chars": len(text)})
            break

    state["_turn_had_tool_call"] = state.get("_turn_had_tool_call") or turn_had_tool_call
    if cfg.get("output_key"):
        # llm_agent.py __maybe_save_output_to_state: only a FINAL response with a text part is stored; without one the key stays as it was (unset on a first
        # pass), so a required {key} raises in the next instruction, which is what the scorer does (Q-040) and why the pipeline arms use {findings?}.
        if ended_text.strip():
            state[cfg["output_key"]] = ended_text
        trace.append({"agent": name, "event": "output_key_set", "key": cfg["output_key"],
                      "chars": len(ended_text), "head": ended_text[:160]})
    # The full conversation, for SFT data (rejection-sampled self-distillation) and for
    # reading failures without guessing from 200-character result heads.
    trace.append({"agent": name, "event": "conversation", "system": system_text,
                  "messages": [traced(m) for m in view(state["_contents"], name, include)],
                  "turn_tokens": [{k: v for k, v in t.items() if k != "_prompt"} for t in turn_tokens]})
    return final_text


def call_agent_tool(sub_cfg, request, tools, state, backend, sub_root, trace, depth, meter=None):
    """google-adk AgentTool.run_async: the sub-agent runs in a FRESH session whose state is a copy
    of the caller's, with `request` as its only user message; its final text is the tool result
    and its state changes flow back. The call itself counts as one tool call, and every tool the
    sub-agent uses draws on the same per-task budget (the harness's budget plugin is global).
    The fresh session is a bare Runner(agent=...): no App, so no events compaction."""
    if tools.b.exhausted():
        return json.dumps({"status": "error", "error_type": "BudgetExhausted",
                           "error_message": "No tool calls or time remaining."})
    tools.b.used += 1
    sub_state = {k: v for k, v in state.items() if not k.startswith("_")}
    sub_state["_contents"] = [{"role": "user", "content": request or "(empty request)"}]
    sub_state["_no_compact"] = True
    sub_state["_fatal_box"] = state.setdefault("_fatal_box", {})
    trace.append({"agent": sub_cfg.get("name"), "event": "agent_tool_call", "request_chars": len(request or "")})
    text = run_agent(sub_cfg, tools, sub_state, backend, sub_root, trace, depth + 1, meter=meter) or ""
    for k, v in sub_state.items():
        if not k.startswith("_"):
            state[k] = v
    return text if text.strip() else "(the agent returned no text)"


def drive(tree, tools, state, backend, sub_root, trace, meter=None):
    """Outer loop (README 5.3): re-invoke the ROOT agent while the budget allows, nudging when
    a turn ended without submit_patch. Three CONSECUTIVE nudge-only turns terminate it; any
    turn containing a valid tool call resets the counter."""
    consecutive_nudges = 0
    while True:
        state["_turn_had_tool_call"] = False
        run_agent(tree, tools, state, backend, sub_root, trace, meter=meter)
        if tools.patch_submitted:
            return "submitted"
        if state.get("_fatal_box", {}).get("error"):
            # The invocation raised (sub-agent/branch model error): the session ends and only the
            # harness's patch fallback runs.
            trace.append({"event": "fatal", "error": state["_fatal_box"]["error"]})
            return "fatal"
        if tools.b.exhausted() or tools.b.turns_used >= tools.b.turns:
            return "budget"
        if state.get("_turn_had_tool_call"):
            consecutive_nudges = 0
        else:
            consecutive_nudges += 1
            if consecutive_nudges > MAX_NUDGES:
                trace.append({"event": "max_nudges"})
                return "max_nudges"
        fr = (state.get("_last_finish_reason") or "").upper()
        last = state.get("_last_assistant_text") or ""
        if "<|tool_call" in last and "<|tool_call|>" not in last[-30:]:
            msg = ("Your previous response reached the token limit before the tool call finished "
                   "closing. Do NOT repeat your prior reasoning in thought-emit your next tool "
                   "call immediately, and if calling edit_file or write_file, split the change "
                   "into smaller incremental edits.")
        elif "LENGTH" in fr or "MAX_TOKENS" in fr:
            msg = ("Your previous response reached the token limit while thinking before a tool "
                   "call was completed. Do NOT repeat your analysis in thought-keep reasoning "
                   "under a few sentences and emit your next tool call immediately, or call "
                   "submit_patch when you have completed and verified your changes.")
        else:
            msg = ("Please continue your work using the available tools, or call submit_patch "
                   "when you have completed and verified your changes.")
        # The scorer's outer loop sends the nudge as a new USER message of the same session (agent_runner.py), so it joins the log for every agent to see.
        state["_contents"].append({"role": "user", "content": msg})
        trace.append({"event": "nudge", "n": consecutive_nudges})
