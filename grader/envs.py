"""Python environments for grading, built with `uv`.

Design (docs/GRADER_DESIGN.md section 4B):
  * A TEMPLATE venv is built once per signature = (interpreter, requirement names, wheelhouse). It holds
    the repo's declared and test dependencies, installed per package so one unavailable package never
    aborts the rest (the failure mode of the official single-command install).
  * Every run gets its OWN COPY of the template, so the editable install of the workspace under test
    can never be seen by another run.
Fidelity comes from the dataset wheelhouse first, then an optional extra wheelhouse (packages the
dataset omits, e.g. h11), then PyPI as a recorded last resort. The provenance of every package is kept.
"""
from __future__ import annotations
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import config

ENV_VERSION = 2

# Dockerfile.public: build backends and test tooling baked into the image
TOOLING = ["pytest", "pytest-timeout==2.1.0", "typer", "setuptools", "wheel", "poetry-core",
           "hatchling", "flit-core", "pdm-backend", "editables"]

# From localeval/testdeps.py on `main`: runners and clients repositories use without declaring them.
COMMON_TEST_DEPS = ["httpx", "pytest", "pytest-timeout", "anyio", "trio", "flask", "sqlalchemy",
                    "itsdangerous", "pyyaml", "jinja2", "python-multipart", "email-validator",
                    "coverage", "requests", "urllib3", "certifi", "idna", "charset-normalizer",
                    "typing-extensions", "orjson", "ujson",
                    # found missing by the 65-task gold control (fastapi tutorials / requests fixtures)
                    "inline-snapshot", "dirty-equals", "pytest-httpbin", "trustme"]


def load_official_setup():
    """Import the competition's sandbox/setup.py as a module (it only defines functions at import)."""
    spec = importlib.util.spec_from_file_location("official_setup", str(config.SETUP_PY))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_setup = None


def official():
    global _setup
    if _setup is None:
        _setup = load_official_setup()
    return _setup


def uv_bin() -> str:
    for cand in (shutil.which("uv"), str(Path.home() / ".local/bin/uv")):
        if cand and Path(cand).exists():
            return cand
    raise RuntimeError("uv not found. Install it with: pip install uv")


def _run(cmd: list[str], timeout=1800, env=None) -> tuple[int, str]:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env or os.environ.copy())
    return r.returncode, (r.stdout + r.stderr)


def python_for(task_id: str) -> str:
    try:
        m = json.loads(Path(config.ENV_MAP).read_text())
    except (OSError, ValueError):
        m = {}
    return m.get(task_id, config.DEFAULT_PYTHON)


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


TEST_GROUP = re.compile(r"^(all|standard|test|tests|testing|dev|develop|development)$", re.I)


_REQ = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*(\([^)]*\)|[<>=!~][^;]*)?\s*(?:;(.*))?$")


def _parse_req(line: str):
    """(name, specifier, extras list) from a requirement line, or None. Needs no third-party package
    (system python on Colab/WSL has no `packaging`). URL / VCS lines and `extra ==` markers are dropped;
    other environment markers are ignored, so a requirement meant for another platform is still tried
    and, being best-effort, simply skipped if it cannot install."""
    line = line.split(" #")[0].strip()
    if not line or line.startswith(("-", "git+", "http")) or "@" in line.split(";")[0]:
        return None
    m = _REQ.match(line)
    if not m:
        return None
    name, ex, spec, marker = m.groups()
    if marker and "extra" in marker:
        return None
    spec = (spec or "").strip().strip("()").replace(" ", "")
    extras = [x.strip() for x in ex.strip("[]").split(",") if x.strip()] if ex else []
    return name, spec, extras


# The four repositories of the public dataset. Their environments were validated with the
# discovery rules above; the wider discovery below applies to every other repository (mined tasks, mine/README.md)
# so that the public results cannot change.
PUBLIC_REPOS = {"fastapi", "rich", "requests", "httpx"}


def envs_norm_repo(task: dict) -> str:
    return norm(task.get("repo", "").split("/")[-1])


def asof_date(task: dict) -> str | None:
    """Mined tasks: the first day of the month after the PR was merged (dependency versions of that time)."""
    m = re.match(r"(\d{4})-(\d{2})", task.get("merged_at") or "")
    if not m or envs_norm_repo(task) in PUBLIC_REPOS:
        return None
    y, mo = int(m.group(1)), int(m.group(2))
    return f"{y + (mo == 12)}-{1 if mo == 12 else mo + 1:02d}-01"


REQ_FILE = re.compile(r"(test|dev)", re.I)


def _group_requirements(groups: dict, wanted: list[str]) -> list[str]:
    """Flatten PEP 735 [dependency-groups] entries for the named groups, following {include-group = ...}."""
    out, seen = [], set()

    def walk(name):
        if name in seen or name not in groups:
            return
        seen.add(name)
        for item in groups[name]:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict) and "include-group" in item:
                walk(item["include-group"])

    for w in wanted:
        walk(w)
    return out


def _extended_specs(ws: Path, add_extra) -> None:
    """Test dependencies declared in places the base discovery does not read (non-public repositories only):
    PEP 735 dependency groups (test-like groups, else `dev`), requirements files whose path mentions test or dev,
    and the build backend's own requirements (the editable install runs with --no-build-isolation)."""
    import tomllib
    py = ws / "pyproject.toml"
    if py.exists():
        try:
            data = tomllib.loads(py.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        groups = data.get("dependency-groups", {})
        wanted = [g for g in groups if re.match(r"^(test|tests|testing|dev|develop)$", g, re.I)]
        optional = data.get("project", {}).get("optional-dependencies", {})
        own = norm(data.get("project", {}).get("name", ""))
        for d in _group_requirements(groups, wanted):
            r = _parse_req(d)
            if r and norm(r[0]) == own:      # `pkg[extra, ...]` pointing back at the project: take those extras
                for ex in r[2]:
                    for dd in optional.get(ex, []):
                        add_extra(dd)
            else:
                add_extra(d)
        for d in data.get("build-system", {}).get("requires", []):
            add_extra(d)
        # optional-dependency groups other than docs / lint / typing (frameworks test their integrations: redis, ...)
        for group, deps in data.get("project", {}).get("optional-dependencies", {}).items():
            if not re.search(r"(doc|lint|mypy|typing|type-check|pre-commit|build|release)", group, re.I):
                for d in deps:
                    add_extra(d)
    for rf in sorted(ws.glob("**/*requirements*.txt")):
        rel = rf.relative_to(ws).as_posix()
        if REQ_FILE.search(rel) and not {".git", "docker", "docs", "doc"} & set(rf.relative_to(ws).parts):
            for line in rf.read_text(encoding="utf-8", errors="replace").splitlines():
                add_extra(line)
    for rf in sorted(ws.glob("requirements/*.txt")):
        if REQ_FILE.search(rf.name):
            for line in rf.read_text(encoding="utf-8", errors="replace").splitlines():
                add_extra(line)


def requirement_specs(ws: Path, repo: str) -> tuple[list[str], list[str]]:
    """(runtime, extras) requirement strings, version constraints KEPT.

    The official setup.py strips constraints and installs the highest wheel of each name; for fastapi
    that selects starlette 1.x, which a fastapi pinned `starlette<0.49` cannot import (docs/GRADER_DESIGN.md
    fidelity ledger). We honour what the repository declares instead.
      runtime = [project].dependencies
      extras  = test-like optional groups, requirements*test*.txt, and COMMON_TEST_DEPS
    """
    import tomllib
    own = norm(repo.split("/")[-1]) if repo else ""
    runtime: list[str] = []
    extras: list[str] = []

    def add(lst, line):
        r = _parse_req(line)
        if r and norm(r[0]) != own:
            name, spec, ex = r
            lst.append(name + (f"[{','.join(ex)}]" if ex else "") + spec)

    py = ws / "pyproject.toml"
    if py.exists():
        try:
            proj = tomllib.loads(py.read_text(encoding="utf-8")).get("project", {})
        except Exception:
            proj = {}
        for d in proj.get("dependencies", []):
            add(runtime, d)
        for group, deps in proj.get("optional-dependencies", {}).items():
            if TEST_GROUP.match(group):
                for d in deps:
                    add(extras, d)
    for rf in sorted(ws.glob("**/requirements*.txt")):
        if re.search(r"test", rf.name, re.I) and ".git" not in rf.parts and "docker" not in rf.parts:
            for line in rf.read_text(encoding="utf-8", errors="replace").splitlines():
                add(extras, line)
    if own not in PUBLIC_REPOS:
        _extended_specs(ws, lambda line: add(extras, line))
    for n in COMMON_TEST_DEPS:
        extras.append(n)
    runtime += env_pins().get(own, [])
    seen, ex_out = set(), []
    for e in extras:
        k = norm(re.split(r"[\[<>=!~ ]", e, 1)[0])
        if k and k not in seen and k not in {norm(re.split(r"[\[<>=!~ ]", r, 1)[0]) for r in runtime}:
            seen.add(k)
            ex_out.append(e)
    return runtime, ex_out


def env_pins() -> dict:
    """Per-repository version pins, with the reason, from grader/env_pins.json (key = repo short name)."""
    try:
        data = json.loads((config.HERE / "env_pins.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {norm(k): v["pins"] for k, v in data.items() if not k.startswith("_")}


def _wheelhouse_fingerprint() -> str:
    h = hashlib.sha1()
    for d in (config.WHEELS, config.EXTRA_WHEELS):
        if Path(d).is_dir():
            h.update("\n".join(sorted(os.listdir(d))).encode())
    return h.hexdigest()[:8]


def signature(py: str, names: list[str]) -> str:
    raw = json.dumps([ENV_VERSION, py, sorted(norm(n) for n in names), _wheelhouse_fingerprint(),
                      config.ALLOW_PYPI])
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def site_packages(venv: Path) -> Path:
    found = sorted((venv / "lib").glob("python*/site-packages"))
    if not found:
        raise RuntimeError(f"no site-packages in {venv}")
    return found[0]


def _install_shims(venv: Path, py: str):
    """imp.py / telnetlib.py: stdlib modules removed in 3.12 / 3.13 that old test suites still import."""
    sp = site_packages(venv)
    for name in ("imp.py", "telnetlib.py"):
        src = config.SHIMS / name
        if src.exists() and tuple(map(int, py.split("."))) >= (3, 12):
            shutil.copy(src, sp / name)


def build_template(runtime: list[str], extras: list[str], py: str, sig: str,
                   pypi_first: bool = False, exclude_newer: str | None = None) -> tuple[Path, dict]:
    """Build (or reuse) the template venv for a signature. Returns (path, provenance log).

    1. tooling; 2. the repository's runtime dependencies resolved TOGETHER with their constraints;
    3. every extra on its own (one unavailable package never aborts the rest), constrained to the
    versions step 2 chose so an extra can never upgrade e.g. starlette past the repo's pin.
    Sources tried per install: dataset wheels -> extra wheelhouse -> PyPI (recorded).
    `pypi_first` (mined repositories, see PUBLIC_REPOS) tries PyPI before the wheelhouse: the wheelhouse was built for the
    four public repositories and holds older versions (pydantic) than 2025-26 code of other repositories needs.
    `exclude_newer` (YYYY-MM-DD) hides PyPI releases after that date, so a task gets the dependency versions of its own time.
    """
    tdir = config.ROOT / "envs" / sig
    logf = tdir.with_suffix(".json")
    tdir.parent.mkdir(parents=True, exist_ok=True)
    lock = open(tdir.parent / f"{sig}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        if (tdir / ".complete").exists() and logf.exists():
            return tdir, json.loads(logf.read_text())
        if tdir.exists():
            shutil.rmtree(tdir)
        uv = uv_bin()
        log: dict = {"python": py, "sources": {}, "missing": [], "runtime": runtime}
        rc, out = _run([uv, "venv", "--python", py, str(tdir)])
        if rc != 0:
            raise RuntimeError(f"uv venv failed: {out[-400:]}")
        pyexe = str(tdir / "bin" / "python")
        rc, out = _run([uv, "pip", "install", "--python", pyexe] + TOOLING)
        log["tooling"] = "ok" if rc == 0 else f"rc={rc} {out[-300:]}"
        _install_shims(tdir, py)
        find = []
        for d in (config.WHEELS, config.EXTRA_WHEELS):
            if Path(d).is_dir():
                find += ["--find-links", str(d)]

        def install(reqs: list[str], constraint: Path | None = None) -> tuple[bool, str]:
            base = [uv, "pip", "install", "--python", pyexe]
            cons = ["-c", str(constraint)] if constraint else []
            dated = ["--exclude-newer", exclude_newer] if exclude_newer else []
            if pypi_first and config.ALLOW_PYPI:
                rc, out = _run(base + dated + cons + reqs)
                if rc == 0:
                    return True, "pypi"
            rc, out = _run(base + ["--no-index", "--no-build-isolation"] + find + cons + reqs)
            if rc == 0:
                return True, "wheels"
            if config.ALLOW_PYPI and not pypi_first:
                rc, out = _run(base + dated + cons + reqs)
                if rc == 0:
                    return True, "pypi"
            return False, out[-300:]

        if runtime:
            ok, src = install(runtime)
            log["runtime_source"] = src if ok else "FAILED: " + src
            if not ok:                                   # fall back to one at a time
                for r in runtime:
                    ok1, s1 = install([r])
                    log["sources"][r] = s1 if ok1 else "missing"
                    if not ok1:
                        log["missing"].append(r)
        rc, frozen = _run([uv, "pip", "freeze", "--python", pyexe])
        cons = tdir / "constraints.txt"
        cons.write_text("\n".join(l for l in frozen.splitlines() if "==" in l) + "\n")
        for n in extras:
            ok, src = install([n], cons)
            if ok:
                log["sources"][n] = src
            else:
                log["missing"].append(n)
        rc, out = _run([uv, "pip", "freeze", "--python", pyexe])
        log["frozen"] = out.strip().splitlines() if rc == 0 else []
        logf.write_text(json.dumps(log, indent=1))
        (tdir / ".complete").write_text("ok")
        return tdir, log
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def make_run_venv(template: Path, dest: Path) -> Path:
    """A private copy of the template. `cp -a` keeps the interpreter symlink and file modes."""
    r = subprocess.run(["cp", "-a", str(template), str(dest)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"venv copy failed: {r.stderr[:300]}")
    return dest


def editable_install(venv: Path, ws: Path) -> tuple[bool, str]:
    """Install the workspace under test in editable mode, no dependencies (README 4.2 step 4)."""
    # The snapshot has no git history, so setuptools-scm / hatch-vcs cannot derive a version; give them one.
    # Projects that do not use them ignore the variable.
    env = os.environ.copy()
    env.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION", "0.0.0+grader")
    rc, out = _run([uv_bin(), "pip", "install", "--python", str(venv / "bin" / "python"), "--no-deps",
                    "--no-build-isolation", "-e", str(ws)], env=env)
    return rc == 0, out[-300:]
