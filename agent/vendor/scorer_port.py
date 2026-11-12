"""Behavioral port of the scorer's agent-facing code, so the local harness shows the model what the scorer shows it.

Written 2026-11-04 from a read of the organizers' wheelhouse (not copied; the wheels carry no license text):
  swegemma 0.2.11        tools/{base,workspace,execution}.py, context.py (check_budget), edit.py,
                         harness/agent_runner.py (build_agent_prompt, build_workspace_tree_command),
                         harness/verification.py (strip_protected_files_from_patch), models/task.py
  adk-eval-core 0.1.0    editing/edit.py (the four-tier matcher)
  google-adk 1.36.1      flows/llm_flows/identity.py, functions.py (_get_tool), tools/function_tool.py

Why this exists: the earlier local copy was written from the README and differed from the scorer in the
edit matcher (3 tiers, CRLF folding), every error string, JSON escaping (ensure_ascii), read_file truncation,
budget_warning on free tools, get_status rounding, patch stripping, and the whole first user message. Numbers
measured on it are labeled pre-fidelity (git tag pre-fidelity-203e80f).

tests/test_scorer_port.py checks each function here against fixtures recorded from the scorer path;
mine/diff_scorer_port.py checks them against the real modules, with random inputs, inside the scorer venv.
Stdlib only on purpose: it has to import on Colab, in WSL and in the scorer venv alike.
"""
from __future__ import annotations

import dataclasses
import difflib
import json
import posixpath
import re
from typing import Any, Iterable, Sequence

# Stamped into every eval result. Results with a different (or missing) stamp are not comparable: see R1 in
# docs/PLAN_E4B_SFT.md. "scorer-port-1" = tools, prompt, thinking bridge and wire format ported 2026-11-04.
FIDELITY = "scorer-port-1"

# ------------------------------------------------------------------------------------------ envelopes


def ok_response(**kwargs: Any) -> str:
    data: dict[str, Any] = {"status": "ok"}
    data.update(kwargs)
    return json.dumps(data, ensure_ascii=False)


def error_response(error_type: str, error_message: str, details: dict[str, Any] | None = None) -> str:
    data: dict[str, Any] = {"status": "error", "error_type": error_type, "error_message": error_message}
    if details is not None:
        data["details"] = details
    return json.dumps(data, ensure_ascii=False)


def attach_low_budget_warning(raw: str, max_calls: int | None, used_calls: int | None) -> str:
    """base._attach_low_budget_warning: only when the call budget is >= 20 and <= 10 calls remain."""
    if max_calls is None or used_calls is None or max_calls < 20:
        return raw
    remaining = max(0, max_calls - used_calls)
    if remaining > 10:
        return raw
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict) and "budget_warning" not in payload:
            payload["budget_warning"] = (
                f"Only {remaining} tool call(s) remaining ({used_calls}/{max_calls} used). "
                "Finalize your edits and call submit_patch soon."
            )
            return json.dumps(payload, ensure_ascii=False)
    except Exception:
        pass
    return raw


def budget_error(*, counted: bool, tool_calls_used: int, max_calls: int | None, turns_used: int,
                 max_turns: int | None, time_minutes: float | None, elapsed_seconds: float | None) -> str | None:
    """context.check_budget without the cost/token budgets (the competition sets neither). Returns the error
    JSON string, or None when the call may proceed. The caller increments tool_calls_used afterwards."""
    if counted and max_calls is not None and tool_calls_used >= max_calls:
        return json.dumps({"status": "error", "error_type": "BudgetExceeded",
                           "error_message": f"Tool call budget exhausted ({max_calls} calls)"}, ensure_ascii=False)
    if counted and max_turns is not None and turns_used >= max_turns:
        return json.dumps({"status": "error", "error_type": "BudgetExceeded",
                           "error_message": f"Turn budget exhausted ({max_turns} turns)"}, ensure_ascii=False)
    if time_minutes is not None:
        if time_minutes <= 0 or (elapsed_seconds is not None and elapsed_seconds / 60.0 > time_minutes):
            return json.dumps({"status": "error", "error_type": "TimeoutExceeded",
                               "error_message": f"Session time budget exhausted ({time_minutes} minutes)"},
                              ensure_ascii=False)
    return None


# ------------------------------------------------------------------------------------- the edit matcher


@dataclasses.dataclass
class EditResult:
    new_content: str
    occurrences: int
    strategy: str
    error_message: str | None = None


def make_diff(old_content: str, new_content: str, filepath: str = "file") -> str:
    return "".join(difflib.unified_diff(old_content.splitlines(keepends=True),
                                        new_content.splitlines(keepends=True),
                                        fromfile=f"a/{filepath}", tofile=f"b/{filepath}"))


def apply_indentation(lines: list[str], target_indent: str) -> list[str]:
    if not lines:
        return []
    ref_indent = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
    return [target_indent + l[ref_indent:] if l.strip() else "" for l in lines]


_MULTI = ("Expected 1 occurrence but found {n}. "
          "If you intended to replace multiple occurrences, set 'allow_multiple' to true.")


def _exact(content: str, old: str, new: str, allow_multiple: bool) -> EditResult | None:
    n = content.count(old)
    if n == 0:
        return None
    if not allow_multiple and n > 1:
        return EditResult(content, n, "exact", _MULTI.format(n=n))
    return EditResult(content.replace(old, new), n, "exact")


def _flexible(content: str, old: str, new: str, allow_multiple: bool) -> EditResult | None:
    cur = content.splitlines(keepends=True)
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    if not old_lines:
        return None

    def scan(norm) -> list[int]:
        want = [norm(l) for l in old_lines]
        found: list[int] = []
        i = 0
        while i <= len(cur) - len(old_lines):
            if [norm(l) for l in cur[i:i + len(old_lines)]] == want:
                found.append(i)
                i += len(old_lines)
            else:
                i += 1
        return found

    matches = scan(lambda l: l.rstrip())
    if not matches:
        matches = scan(lambda l: l.strip())
    if not matches:
        return None
    if not allow_multiple and len(matches) > 1:
        return EditResult(content, len(matches), "flexible", _MULTI.format(n=len(matches)))

    line_sep = "\r\n" if "\r\n" in content else "\n"
    old_ref = min((len(l) - len(l.lstrip()) for l in old_lines if l.strip()), default=0)
    new_ref = min((len(l) - len(l.lstrip()) for l in new_lines if l.strip()), default=0)
    delta = max(0, new_ref - old_ref)

    for idx in reversed(matches):          # bottom to top so the earlier indices stay valid
        if new == "":
            cur[idx:idx + len(old_lines)] = []
            continue
        window = cur[idx:idx + len(old_lines)]
        non_empty = [l for l in window if l.strip()]
        if non_empty:
            line = min(non_empty, key=lambda l: len(l) - len(l.lstrip()))
        else:
            line = cur[idx]
        indent = re.match(r"^([ \t]*)", line).group(1)
        target = indent + " " * delta
        text = line_sep.join(apply_indentation(new_lines, target))
        last = cur[idx + len(old_lines) - 1]
        if last.endswith("\r\n"):
            if not text.endswith("\r\n"):
                text = text.rstrip("\r\n") + "\r\n"
        elif last.endswith("\n"):
            if not text.endswith("\n"):
                text += "\n"
        elif text.endswith(("\r\n", "\n")):
            text = text.rstrip("\r\n")
        cur[idx:idx + len(old_lines)] = text.splitlines(keepends=True)
    return EditResult("".join(cur), len(matches), "flexible")


_TOKEN = re.compile(r"==|!=|<=|>=|->|:=|\+=|-=|\*\*|//|[():\[\]{},><=+\-*/]|\w+|\S+")


def _regex(content: str, old: str, new: str, allow_multiple: bool) -> EditResult | None:
    tokens = _TOKEN.findall(old)
    if not tokens:
        return None
    parts: list[str] = []
    for i, tok in enumerate(tokens):
        if i > 0:
            parts.append(r"\s+" if re.fullmatch(r"\w+", tokens[i - 1]) and re.fullmatch(r"\w+", tok) else r"\s*")
        parts.append(re.escape(tok))
    pattern = f"^([ \t]*){''.join(parts)}[ \t]*(?=\\r?\\n|$)"
    try:
        found = list(re.finditer(pattern, content, flags=re.MULTILINE))
    except re.error:
        return None
    if not found:
        return None
    if not allow_multiple and len(found) > 1:
        return EditResult(content, len(found), "regex", _MULTI.format(n=len(found)))
    new_lines = new.splitlines()
    line_sep = "\r\n" if "\r\n" in content else "\n"

    def repl(m: re.Match) -> str:
        return line_sep.join(apply_indentation(new_lines, m.group(1) or ""))

    out = re.sub(pattern, repl, content, count=0 if allow_multiple else 1, flags=re.MULTILINE)
    return EditResult(out, len(found), "regex")


_UNICODE_ONLY = re.compile(r"(?<!\\)\\u([0-9a-fA-F]{4})")
_INLINE_JSON = re.compile(r'\\(u[0-9a-fA-F]{4}|["\\])')
_FULL_JSON = re.compile(r'\\(u[0-9a-fA-F]{4}|["\\/bfnrt])')
_FULL_MAP = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


def _unescape_unicode_only(text: str) -> str:
    return _UNICODE_ONLY.sub(lambda m: chr(int(m.group(1), 16)), text)


def _unescape_inline_json(text: str) -> str:
    def repl(m: re.Match) -> str:
        seq = m.group(1)
        return chr(int(seq[1:], 16)) if seq.startswith("u") else seq
    return _INLINE_JSON.sub(repl, text)


def _unescape_full_json(text: str) -> str:
    def repl(m: re.Match) -> str:
        seq = m.group(1)
        return chr(int(seq[1:], 16)) if seq.startswith("u") else _FULL_MAP.get(seq, seq)
    return _FULL_JSON.sub(repl, text)


def _unescaped(content: str, old: str, new: str, allow_multiple: bool) -> EditResult | None:
    if "\\" not in old:
        return None
    transforms = (
        _unescape_unicode_only,
        _unescape_inline_json,
        lambda s: _unescape_inline_json(_unescape_inline_json(s)),
        _unescape_full_json,
        lambda s: _unescape_full_json(_unescape_full_json(s)),
    )
    seen = {old}
    for tf in transforms:
        cand_old = tf(old)
        if not cand_old or cand_old in seen:
            continue
        seen.add(cand_old)
        cand_new = tf(new)
        if cand_old == cand_new:
            continue
        for tier in (_exact, _flexible):
            res = tier(content, cand_old, cand_new, allow_multiple)
            if res is not None:
                if res.error_message is None:
                    res.strategy = "unescaped"
                return res
    return None


def apply_replacement_core(content: str, old: str, new: str, allow_multiple: bool = False) -> EditResult:
    """adk_eval_core.editing.apply_replacement: exact, flexible, regex, then JSON-unescaped."""
    if not old:
        return EditResult(content, 0, "none", "old_string cannot be empty for existing files.")
    if old == new:
        return EditResult(content, 1, "exact", "No changes to apply. old_string and new_string are identical.")
    for tier in (_exact, _flexible, _regex, _unescaped):
        res = tier(content, old, new, allow_multiple)
        if res:
            return res
    return EditResult(content, 0, "none",
                      "Failed to replace: old_string not found. "
                      "Ensure you're not escaping content incorrectly and check whitespace, indentation, and context.")


@dataclasses.dataclass
class AppliedEdit:
    new_content: str
    diff: str
    occurrences: int = 1
    strategy: str = "exact"


def apply_replacement(content: str, old: str, new: str, filepath: str = "file",
                      allow_multiple: bool = False) -> AppliedEdit:
    """swegemma.edit.apply_replacement: the core result turned into the exceptions edit_file reports."""
    if not old:
        raise ValueError("old_string cannot be empty.")
    res = apply_replacement_core(content, old, new, allow_multiple=allow_multiple)
    if res.error_message:
        if res.occurrences > 1:
            if res.strategy == "flexible":
                raise ValueError(f"Target string flexibly matches {res.occurrences} locations in {filepath}. "
                                 "Please provide more surrounding context.")
            raise ValueError(f"Target string occurs {res.occurrences} times in {filepath}. "
                             "Please provide more surrounding context.")
        raise ValueError(res.error_message)
    return AppliedEdit(res.new_content, make_diff(content, res.new_content, filepath), res.occurrences, res.strategy)


# ------------------------------------------------------------------------------------------- workspace


def resolve_workspace_path(filepath: str, allow_root: bool = True) -> str:
    """workspace._resolve_workspace_path for a Linux sandbox whose workspace is /workspace."""
    ws = "/workspace"
    norm = filepath
    if norm == "/workspace":
        norm = ""
    elif norm.startswith("/workspace/"):
        norm = norm[len("/workspace/"):]
    if norm.startswith("/"):
        raise ValueError(f"Path traversal detected: '{filepath}' escapes workspace root.")
    resolved = posixpath.normpath(posixpath.join(ws, norm))
    if not (resolved == ws or resolved.startswith(ws + "/")):
        raise ValueError(f"Path traversal detected: '{filepath}' escapes workspace root.")
    if not allow_root and resolved == ws:
        raise ValueError(f"Invalid file path '{filepath}': refers to workspace root directory, not a file.")
    return resolved


def coerce_optional_int(val: Any, param_name: str) -> int | None:
    if val is None:
        return None
    if isinstance(val, bool):
        raise TypeError(f"{param_name} must be an integer, got boolean {val}")
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        s = val.strip()
        if not s or s.lower() in ("none", "null", "nil"):
            return None
        try:
            return int(s)
        except ValueError:
            try:
                f = float(s)
                if f.is_integer():
                    return int(f)
            except ValueError:
                pass
            raise ValueError(f"Invalid integer for {param_name}: {val!r}") from None
    raise ValueError(f"Invalid integer for {param_name}: {val!r}")


def read_file_response(filepath: str, content: str, start_line: Any, end_line: Any,
                       max_lines: int | None = 150, max_chars: int | None = 10000) -> str:
    """The part of workspace.read_file after the file has been copied out and decoded (universal newlines)."""
    coerced_start = coerce_optional_int(start_line, "start_line")
    coerced_end = coerce_optional_int(end_line, "end_line")
    lines = content.splitlines()
    total = len(lines)
    if total == 0:
        return ok_response(filepath=filepath, content="", start_line=1, end_line=0, total_lines=0, is_truncated=False)
    start = coerced_start if coerced_start is not None else 1
    end = coerced_end if coerced_end is not None else total
    start = max(start, 1)
    end = min(end, total)
    if start > end:
        return error_response(error_type="ValueError",
                              error_message=f"start_line ({start}) cannot be greater than end_line ({end})",
                              details={})
    truncated = False
    if max_lines is not None and (end - start + 1) > max_lines:
        end = start + max_lines - 1
        truncated = True
    snippet = "\n".join(lines[start - 1:end])
    if max_chars is not None and len(snippet) > max_chars:
        snippet = snippet[:max_chars]          # end_line is deliberately not recomputed, as in the scorer
        truncated = True
    return ok_response(filepath=filepath, content=snippet, start_line=start, end_line=end,
                       total_lines=total, is_truncated=truncated)


def coerce_allow_multiple(v: Any) -> bool:
    return v.strip().lower() in ("true", "1", "yes") if isinstance(v, str) else bool(v)


def docker_not_found_message(path: str, cid: str = "0" * 64, api: str = "1.51") -> str:
    """str() of the docker-py NotFound that container.get_archive raises for a missing path. The API version and
    container id vary per run, so this is an approximation of the text, flagged as such in the tests."""
    from urllib.parse import quote
    return (f"404 Client Error for http+docker://localhost/v{api}/containers/{cid}/archive?path={quote(path, safe='')}: "
            f'Not Found ("Could not find the file {path} in container {cid}")')


# ----------------------------------------------------------------------------------- patch extraction


def _unquote_git_path(raw: str) -> str:
    s = raw.strip()
    if not s:
        return ""
    if s.startswith('"'):
        end = 1
        while end < len(s):
            if s[end] == '"' and s[end - 1] != "\\":
                break
            end += 1
        s = s[1:end].encode("utf-8").decode("unicode_escape")
    else:
        s = s.split("\t", 1)[0].strip()
    if s.startswith(("a/", "b/")):
        s = s[2:]
    return s.strip()


def extract_test_files_from_patch(patch: str) -> list[str]:
    """models.task.extract_test_files_from_patch: the paths a test patch touches (Task.test_files). The scorer
    strips these from the agent's patch in addition to the name-based protected set."""
    if not patch or not patch.strip():
        return []
    files: list[str] = []
    seen: set[str] = set()
    in_hunk = False
    rem_old = rem_new = 0
    deleted = False

    def add(p: str) -> None:
        if p and p != "/dev/null" and p not in seen:
            seen.add(p)
            files.append(p)

    for line in patch.splitlines():
        if line.startswith("diff --git "):
            in_hunk = False
            rem_old = rem_new = 0
            deleted = False
            continue
        if line.startswith("@@ "):
            in_hunk = True
            m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if m:
                rem_old = int(m.group(2)) if m.group(2) is not None else 1
                rem_new = int(m.group(4)) if m.group(4) is not None else 1
            continue
        if line.startswith(("--- a/", "--- /dev/null")):
            in_hunk = False
            continue
        if in_hunk:
            if line.startswith(" "):
                rem_old -= 1
                rem_new -= 1
            elif line.startswith("-"):
                rem_old -= 1
            elif line.startswith("+"):
                rem_new -= 1
            if rem_old <= 0 and rem_new <= 0:
                in_hunk = False
            continue
        if line.startswith("diff --git "):
            deleted = False
            continue
        if line.startswith("--- "):
            continue
        if line.startswith("deleted file mode"):
            deleted = True
        elif line.startswith("rename to "):
            add(_unquote_git_path(line[len("rename to "):]))
        elif line.startswith("+++ ") and not deleted:
            add(_unquote_git_path(line[4:]))
    return files


_PROTECTED_NAMES = {"conftest.py", "pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg", ".pytest.ini",
                    "sitecustomize.py", "usercustomize.py", "_swegemma_stubs.py"}


def is_protected_path(rel_path: str) -> bool:
    from pathlib import PurePosixPath
    p = PurePosixPath(rel_path)
    name = p.name
    if name in _PROTECTED_NAMES or name.endswith(".pth"):
        return True
    if name.startswith("test_") and name.endswith(".py"):
        return True
    if name.endswith("_test.py"):
        return True
    parts = {part.lower() for part in p.parts[:-1]}
    return bool({"tests", "test", "testing"} & parts and name.endswith(".py"))


def strip_protected_files_from_patch(patch_text: str, extra_protected_paths: Sequence[str] = ()) -> str:
    """verification.strip_protected_files_from_patch: drop diff sections that touch tests or runner config."""
    if not patch_text or not patch_text.strip():
        return ""
    extra = {p.strip() for p in extra_protected_paths if p and p.strip()}

    def protected(path_str: str) -> bool:
        if not path_str or path_str == "/dev/null":
            return False
        return path_str in extra or is_protected_path(path_str)

    sections: list[tuple[list[str], list[str]]] = []
    cur_lines: list[str] = []
    cur_paths: list[str] = []
    in_hunk = False
    rem_old = rem_new = 0
    saw_hunk = False
    has_git_header = False

    def flush() -> None:
        nonlocal cur_lines, cur_paths, in_hunk, rem_old, rem_new, saw_hunk, has_git_header
        if cur_lines:
            sections.append((cur_lines, cur_paths))
        cur_lines, cur_paths = [], []
        in_hunk = False
        rem_old = rem_new = 0
        saw_hunk = False
        has_git_header = False

    for raw_line in patch_text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        if line.startswith("diff --git "):
            flush()
            has_git_header = True
            cur_lines.append(raw_line)
            m = re.match(r'^diff --git ("?a/.*?"?) ("?b/.*?"?)$', line)
            if m:
                for grp in (m.group(1), m.group(2)):
                    p = _unquote_git_path(grp)
                    if p:
                        cur_paths.append(p)
            continue
        if (not in_hunk and line.startswith(("--- a/", '--- "a/', "--- /dev/null"))
                and cur_lines and (saw_hunk or not has_git_header)):
            flush()
        if line.startswith("@@ "):
            in_hunk = True
            saw_hunk = True
            m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if m:
                rem_old = int(m.group(2)) if m.group(2) is not None else 1
                rem_new = int(m.group(4)) if m.group(4) is not None else 1
                if rem_old <= 0 and rem_new <= 0:
                    in_hunk = False
            cur_lines.append(raw_line)
            continue
        if in_hunk:
            if line.startswith((" ", "+", "-", "\\")):
                cur_lines.append(raw_line)
                if line.startswith(" "):
                    rem_old -= 1
                    rem_new -= 1
                elif line.startswith("-"):
                    rem_old -= 1
                elif line.startswith("+"):
                    rem_new -= 1
                if rem_old <= 0 and rem_new <= 0:
                    in_hunk = False
                continue
            in_hunk = False
        cur_lines.append(raw_line)
        if not saw_hunk:
            if line.startswith(("--- ", "+++ ")):
                p = _unquote_git_path(line[4:])
                if p:
                    cur_paths.append(p)
            else:
                for prefix in ("rename from ", "rename to ", "copy from ", "copy to "):
                    if line.startswith(prefix):
                        p = _unquote_git_path(line[len(prefix):])
                        if p:
                            cur_paths.append(p)
                        break
    flush()
    kept: list[str] = []
    for sec_lines, sec_paths in sections:
        if any(protected(p) for p in sec_paths):
            continue
        kept.extend(sec_lines)
    result = "".join(kept)
    return result if result.strip() else ""


# ------------------------------------------------------------------------------------ the first prompt

PRUNED_TOP_DIRS = ("docs", "docs_src", "tests", "test_data", "examples", "benchmarks", "assets", "imgs",
                   "questions", "site", "g3doc", "google3")


def build_workspace_tree_command(max_lines: int = 200, workspace_dir: str = "/workspace") -> str:
    prune = " ".join(f'-not -path "./{d}/*"' for d in PRUNED_TOP_DIRS)
    return (f"cd {workspace_dir} && find . -mindepth 1 -maxdepth 3 "
            '-not -path "./.*" -not -name ".*" '
            '-not -name "*.pyc" -not -name "__pycache__" -not -path "*/__pycache__/*" '
            '-not -path "./*.egg-info*" '
            f"{prune} "
            f"| awk -F/ '{{print NF, $0}}' | sort -n -k1,1 -k2 | head -{max_lines} "
            "| cut -d' ' -f2- | sort")


def build_agent_prompt(repo: str, problem_statement: str, hints_text: str = "", *,
                       time_minutes: float | None = None, tool_calls: int | None = None,
                       turns: int | None = None, cost_usd: float | None = None,
                       command_timeout_seconds: int | None = None, max_stdout_chars: int | None = None,
                       max_file_lines: int | None = None, max_file_chars: int | None = None,
                       workspace_tree: str = "", enable_sandbox_testing: bool = True,
                       declared_tools: Iterable[str] | None = None, graph_available: bool = False) -> str:
    """harness.agent_runner.build_agent_prompt with the config objects flattened into keyword arguments.
    `graph_available` replaces the scorer's check for graph/embedding files on disk."""
    declared = set(declared_tools) if declared_tools is not None else None

    def has_tool(name: str) -> bool:
        return declared is None or name in declared

    budget_lines = []
    if time_minutes is not None:
        budget_lines.append(f"- Time allowance: {time_minutes} minutes")
    if tool_calls is not None:
        budget_lines.append(f"- Tool calls allowance: {tool_calls} calls")
    if turns is not None:
        budget_lines.append(f"- Max loop iterations: {turns} turns")
    if cost_usd is not None:
        budget_lines.append(f"- Cost budget: ${cost_usd:.2f} USD")

    harness_lines = []
    if command_timeout_seconds is not None:
        harness_lines.append(f"- Single command timeout: {command_timeout_seconds} seconds "
                             "(commands exceeding this fail without ending the session)")
    if max_stdout_chars is not None:
        harness_lines.append(f"- Command output limit: {max_stdout_chars} characters")
    if max_file_lines is not None and has_tool("read_file"):
        harness_lines.append(f"- File view limit: {max_file_lines} lines per read_file call")
    if max_file_chars is not None and has_tool("read_file"):
        harness_lines.append(f"- File character limit: {max_file_chars} characters per read_file call")
    harness_lines.append("- Environment is offline (no network/PyPI access). All repository and test dependencies are "
                         "ALREADY pre-installed. Do NOT attempt to run pip install or download packages.")

    parts = [f"You are evaluating a software engineering task for repository {repo}.\n\n"
             f"Problem Statement:\n{problem_statement}\n"]
    hints = hints_text or ""
    if isinstance(hints, str) and hints.strip():
        parts.append(f"## Hints:\n{hints.strip()}\n")
    if budget_lines:
        parts.append("## Task Budget (Session terminates when any budget is exhausted)\n"
                     + "\n".join(budget_lines) + "\n")
    if harness_lines:
        parts.append("## Execution Environment Rules\n" + "\n".join(harness_lines) + "\n")

    instr = [
        "## Instructions:",
        "0. All source code is under `/workspace`. Do NOT search outside `/workspace` (e.g. `/usr/`, `/wheels/`, "
        "system site-packages). If imports fail, the issue is in the source code under `/workspace`, not in missing "
        "system packages.",
        "1. Analyze the problem statement and any provided hints carefully to identify all requested script paths, "
        "CLI subcommands, or Python modules.",
        "2. Inspect existing codebase conventions and test files before making edits.",
    ]
    if not enable_sandbox_testing:
        instr.append("3. The `pytest` and `unittest` test frameworks are intentionally disabled in this sandbox to "
                     "avoid test sweeps and pre-existing environment failure loops. Do NOT import `unittest` or run "
                     "test discovery. To verify your implementation, run targeted inline assertions via "
                     '`python3 -c "..."` without unittest.')
    else:
        instr.append("3. Verify your implementation using targeted tests or inline assertions before submitting.")
    instr.extend([
        "4. Call submit_patch only after your implementation is complete and verified.",
        "5. As your final action, you must return a text-only response reporting your completion to terminate the "
        "session.",
    ])
    parts.append("\n".join(instr))

    if graph_available:
        lines = []
        if has_tool("search_similar_code"):
            lines.append("- `search_similar_code(query)`: Find functions/classes with similar precomputed embeddings "
                         "given a symbol name (e.g. `pkg.module.Class.method`, `Class.method`, or `func_name`).")
        if has_tool("get_code_neighbors"):
            lines.append("- `get_code_neighbors(node)`: Find callers and callees of a symbol node (e.g. "
                         "`pkg.module.Class.method` or `Class.method`).")
        if has_tool("get_code_subgraph"):
            lines.append("- `get_code_subgraph(nodes)`: Get the induced call subgraph connecting a list of symbol "
                         "nodes.")
        if lines:
            parts.append("## Code Intelligence Tools\n"
                         "This repository has pre-built code graph and embedding data indexed by dotted symbol path "
                         "(e.g. `pkg.module.Class.method`). Use these tools for fast, targeted navigation:\n"
                         + "\n".join(lines) + "\n")
    if workspace_tree:
        parts.append("## Workspace Layout\n"
                     "The repository is located at `/workspace`. Here is the directory tree (up to 3 levels):\n"
                     "```\n" + workspace_tree + "\n```\n")
    return "\n".join(parts)


# --------------------------------------------------------------------------------- what ADK adds around it


def adk_system_text(instruction: str, name: str, description: str = "") -> str:
    """The system message vLLM receives: the agent instruction, then ADK's identity sentence, joined by a blank
    line (flows/llm_flows/identity.py + LlmRequest.append_instructions)."""
    si = f'You are an agent. Your internal name is "{name}".'
    if description:
        si += f' The description about you is "{description}".'
    return instruction + "\n\n" + si


def adk_unknown_tool_result(name: str, available: Sequence[str]) -> str:
    """ADK raises ValueError for an undeclared tool name and swegemma's ToolErrorPlugin returns it as a result."""
    msg = (f"Tool '{name}' not found.\nAvailable tools: {', '.join(available)}\n\nPossible causes:\n"
           "  1. LLM hallucinated the function name - review agent instruction clarity\n"
           "  2. Tool not registered - verify agent.tools list\n  3. Name mismatch - check for typos\n\n"
           "Suggested fixes:\n  - Review agent instruction to ensure tool usage is clear\n"
           "  - Verify tool is included in agent.tools list\n  - Check for typos in function name")
    return json.dumps({"status": "error", "error_type": "ValueError", "error": msg}, ensure_ascii=False)


def adk_missing_args_result(tool: str, missing: Sequence[str]) -> str:
    """FunctionTool.run_async when a mandatory parameter is absent from the model's call."""
    err = (f"Invoking `{tool}()` failed as the following mandatory input parameters are not present:\n"
           + "\n".join(missing)
           + "\nYou could retry calling this tool, but it is IMPORTANT for you to provide all the mandatory parameters.")
    return json.dumps({"error": err}, ensure_ascii=False)


ORGANIZER_GENERATION_DEFAULTS = {"max_output_tokens": 16384, "thinking_config": {"thinking_budget": 4096}}


def merge_generation_config(cfg: dict | None) -> dict:
    """swegemma.config.build_submission_limits defaults + GenerationConstraints.validate_config step 5: the
    submission's values win, thinking_config is merged key by key. The same dict is the starting point when the
    YAML has no generate_content_config at all."""
    merged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in ORGANIZER_GENERATION_DEFAULTS.items()}
    default_tc = merged.get("thinking_config")
    cfg = {k: v for k, v in (cfg or {}).items() if v is not None}
    merged.update(cfg)
    if isinstance(default_tc, dict) and isinstance(cfg.get("thinking_config"), dict):
        merged["thinking_config"] = {**default_tc, **cfg["thinking_config"]}
    return merged


def thinking_request_fields(gen_config: dict | None) -> dict:
    """adk-submission 0.2.13 resolvers/generation.apply_thinking_config_to_model.

    Input is the MERGED generation config (see merge_generation_config). Output: the request-body fields the
    scorer's LiteLlm sends beyond the sampling ones: {"extra_body": {...}, "reasoning_effort": str?, "seed": int?}
    plus "include_thoughts" (bool or None), which decides whether earlier thoughts are sent back in the history
    (False strips reasoning from assistant messages). An empty extra_body means the server default applies."""
    gen_config = gen_config or {}
    seed = gen_config.get("seed")
    thinking_cfg = gen_config.get("thinking_config")
    has_valid_seed = seed is not None and not isinstance(seed, bool)
    enable_thinking = None
    include_thoughts = None
    level = None
    budget = None
    if thinking_cfg is not None:
        include_thoughts = thinking_cfg.get("include_thoughts")
        raw_level = thinking_cfg.get("thinking_level")
        budget = thinking_cfg.get("thinking_budget")
        if raw_level is not None:
            level = str(getattr(raw_level, "value", raw_level)).lower()
        has_budget = budget is not None and not isinstance(budget, bool)
        if level == "none" or (has_budget and int(budget) <= 0):
            enable_thinking = False
        elif level in {"minimal", "low", "medium", "high"} or (has_budget and int(budget) > 0):
            enable_thinking = True
        elif include_thoughts is False:
            enable_thinking = False
        elif include_thoughts is True:
            enable_thinking = True
    out: dict[str, Any] = {"extra_body": {}, "include_thoughts": include_thoughts if isinstance(include_thoughts, bool) else None}
    if has_valid_seed:
        out["seed"] = int(seed)
    if enable_thinking is not None:
        out["extra_body"]["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        if enable_thinking and budget is not None and not isinstance(budget, bool) and int(budget) > 0:
            out["extra_body"]["thinking_token_budget"] = int(budget)
        if enable_thinking and level in {"low", "medium", "high"}:
            out["reasoning_effort"] = level
    return out
