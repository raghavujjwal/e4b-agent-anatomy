"""Capture the exact chat-completion requests the scorer's ADK + LiteLLM path sends to vLLM.

Runs the scorer's own code (swegemma 0.2.11, adk_submission 0.2.13, google-adk 1.36.1, litellm) against a
mock OpenAI-compatible server that replays a scripted list of model replies. Nothing here talks to a real
model. Tool execution goes to a fake sandbox that maps /workspace to a temp directory.

Run with the scorer venv (swegemma, adk-submission, adk-eval-core and google-adk installed from the organizers'
wheelhouse; see docs/MEASUREMENTS.md, "Scorer fidelity audit"):

    D:\\envs\\scorer\\Scripts\\python.exe -I mine\\capture_scorer_requests.py ^
        --submission agent\\submission_e4b_v2rec --script mine\\fidelity_scripts\\script_basic.json --out capture.json

The two scripts in mine/fidelity_scripts are the ones behind tests/fixtures/scorer_requests_v2rec.json and
scorer_envelopes.json. The workspace files, tree and diff below are what those fixtures assume; change them together.
"""
import argparse
import asyncio
import json
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--submission', required=True)
ap.add_argument('--script', required=True, help='JSON list of scripted model replies')
ap.add_argument('--out', required=True)
ap.add_argument('--time-minutes', type=float, default=5.5)
ap.add_argument('--tool-calls', type=int, default=24)
ap.add_argument('--turns', type=int, default=40)
ap.add_argument('--cmd-timeout', type=int, default=90)
ap.add_argument('--port', type=int, default=18765)
ap.add_argument('--outer-loop', action='store_true',
                help="also run the scorer's continuation loop (swegemma agent_runner.py, the `while True` around runner.run_async): "
                     "a pass that ends without a submitted patch is followed by the plain nudge as a new user message of the same session, "
                     "at most 3 in a row. Without the flag there is ONE pass, as the fixtures of the flat arms were recorded.")
args = ap.parse_args()

import swegemma  # noqa: F401  (sets litellm.drop_params = True, as the scorer does)
from adk_eval_core.plugins import ModelRetryPlugin
from adk_eval_core.sandbox import ExecutionResult
from adk_submission import ModelRegistry, compile_submission
from google.adk.agents.run_config import RunConfig
from google.adk.apps.app import App
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from swegemma.budget import EvaluationBudget, HarnessLimits
from swegemma.config import EvalConfig
from swegemma.context import SwegemmaContext
from swegemma.harness.agent_runner import (
    ToolErrorPlugin,
    _extract_declared_tools,
    build_agent_prompt,
    build_workspace_tree_command,
    resolve_swegemma_adapter,
)
from swegemma.models import Task

SCRIPT = json.loads(Path(args.script).read_text(encoding='utf-8'))
# the nudge of swegemma 0.2.11 agent_runner.py for an ordinary stop (the two token-limit variants are not needed here)
NUDGE = ('Please continue your work using the available tools, or call submit_patch when '
         'you have completed and verified your changes.')
CAPTURED = []
_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a, **k):
        pass

    def do_POST(self):
        n = int(self.headers.get('content-length', 0))
        body = json.loads(self.rfile.read(n))
        with _lock:
            CAPTURED.append({'path': self.path, 'body': body})
            idx = len(CAPTURED) - 1
            item = SCRIPT[idx] if idx < len(SCRIPT) else {'content': 'done'}
        calls = item.get('calls') or []
        msg = {'role': 'assistant', 'content': item.get('content')}
        if calls:
            msg['tool_calls'] = [
                {
                    'id': f'chatcmpl-tool-{idx}-{i}',
                    'type': 'function',
                    'function': {'name': c['name'], 'arguments': json.dumps(c.get('args', {}))},
                }
                for i, c in enumerate(calls)
            ]
        if item.get('reasoning'):
            msg['reasoning'] = item['reasoning']
            msg['reasoning_content'] = item['reasoning']
        resp = {
            'id': f'chatcmpl-{idx}',
            'object': 'chat.completion',
            'created': 0,
            'model': body.get('model', 'm'),
            'choices': [
                {'index': 0, 'message': msg, 'finish_reason': 'tool_calls' if calls else 'stop'}
            ],
            'usage': {'prompt_tokens': 100 + idx, 'completion_tokens': 20, 'total_tokens': 120 + idx},
        }
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()

WS_FILES = {
    'pkg/__init__.py': '',
    'pkg/mod.py': 'def foo():\n    return 1\n\n\ndef bar():\n    return 2\n',
    'README.md': '# demo\n',
    'pkg/uni.py': 'NAME = "café → ok"\nVALUE = 1\nVALUE2 = 1\n',
    'pkg/empty.py': '',
}
TREE = './README.md\n./pkg\n./pkg/__init__.py\n./pkg/mod.py'
DIFF = 'diff --git a/pkg/mod.py b/pkg/mod.py\n--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1,2 +1,2 @@\n def foo():\n-    return 1\n+    return 2\n'


class FakeDocker:
    """Maps /workspace to a temp dir; just enough of the sandbox manager interface for the tools."""

    def __init__(self, root: Path):
        self.root = root
        self.cmds = []
        for rel, text in WS_FILES.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding='utf-8', newline='')

    def _map(self, src):
        ws = Path('/workspace').resolve()
        return self.root / Path(str(src).rstrip('/\\')).relative_to(ws)

    def copy_from(self, cid, src, local):
        real = self._map(src)
        if not real.exists():
            raise FileNotFoundError(f'no such file: {src}')
        shutil.copyfile(real, local)

    def copy_to(self, cid, local, dest_dir):
        d = self._map(dest_dir)
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, d / Path(local).name)

    def exec(self, cid, command, timeout=None):
        self.cmds.append(command)
        if 'find . -mindepth 1' in command:
            return ExecutionResult(status='ok', stdout=TREE, exit_code=0)
        if 'git diff' in command:
            return ExecutionResult(status='ok', stdout=DIFF, exit_code=0)
        if command.startswith('cd /workspace && git add'):
            return ExecutionResult(status='ok', exit_code=0)
        if command.startswith('grep'):
            return ExecutionResult(status='ok', stdout='pkg/mod.py:1:def foo():\n', exit_code=0)
        if command.startswith('false'):
            return ExecutionResult(status='error', stdout='', stderr='boom\n', exit_code=1)
        return ExecutionResult(status='ok', stdout='ok\n', stderr='', exit_code=0)

    def get_sandbox(self, *a, **k):
        return None


async def main():
    tmp = Path(tempfile.mkdtemp(prefix='fid_'))
    docker = FakeDocker(tmp / 'ws')
    endpoint = f'http://127.0.0.1:{args.port}/v1'
    models = ModelRegistry()
    alias = 'gemma-4-e4b-it-qat-w4a16-ct'
    models.register(alias, LiteLlm(model=f'openai/{alias}', api_base=endpoint, api_key='EMPTY', num_retries=5))
    budget = EvaluationBudget(time_minutes=args.time_minutes, tool_calls=args.tool_calls, turns=args.turns)
    harness = HarnessLimits(command_timeout_seconds=args.cmd_timeout, max_stdout_chars=5000,
                            max_file_lines=150, max_file_chars=10000)
    config = EvalConfig(tasks_path=tmp / 'tasks.jsonl', snapshots_dir=tmp / 'snap', results_dir=tmp / 'res',
                        submission_dir=Path(args.submission), models=models, budget=budget, harness=harness)
    task = Task(instance_id='demo__repo-1', repo='demo/repo', base_commit='abc123',
                problem_statement='foo() should return 2, not 1.', hints_text='')
    ctx = SwegemmaContext(docker_manager=docker, container_id='c1', problem_statement=task.problem_statement,
                          hints_text='', task=task, repo=task.repo, graph_dir=config.graph_dir,
                          embeddings_dir=config.embeddings_dir, budget=config.budget, harness=config.harness)
    tools = ctx.create_tools()
    agent = compile_submission(
        submission_dir=config.submission_dir, tool_registry=tools, model_registry=config.models,
        code_executor=None, script_timeout=args.cmd_timeout, limits=config.limits,
        generation_constraints=config.generation_constraints, adapter_manifest=config.adapter_manifest,
        adapter_resolver_fn=lambda base, info: resolve_swegemma_adapter(base, info, config.models))
    tree_cmd = build_workspace_tree_command()
    tree = docker.exec('c1', tree_cmd).stdout.strip()
    prompt = build_agent_prompt(task=task, config=config, workspace_tree=tree, enable_sandbox_testing=True,
                                declared_tools=_extract_declared_tools(agent))

    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name='swegemma_eval', user_id='eval_user',
                                                   state={'problem_description': task.problem_statement})
    app = App(name='swegemma_eval', root_agent=agent, plugins=[ModelRetryPlugin(), ToolErrorPlugin()])
    runner = Runner(app=app, session_service=session_service)
    ctx.start_agent_session()
    msg = genai_types.Content(role='user', parts=[genai_types.Part(text=prompt)])
    events = 0
    passes = 0
    consecutive_nudges = 0
    current = msg
    while True:
        passes += 1
        turn_has_tool_call = False
        async for ev in runner.run_async(user_id='eval_user', session_id=session.id, new_message=current,
                                         run_config=RunConfig(max_llm_calls=args.turns)):
            ctx.handle_adk_event(ev, agent_name=agent.name)
            events += 1
            parts = (ev.content.parts if ev.content else None) or []
            turn_has_tool_call = turn_has_tool_call or any(getattr(p, 'function_call', None) for p in parts)
        # agent_runner.py: stop at a submitted patch; a pass with a tool call resets the count; the third nudge in a row ends the loop
        if not args.outer_loop or ctx.patch_submitted:
            break
        if turn_has_tool_call:
            consecutive_nudges = 0
        if consecutive_nudges >= 3:
            break
        consecutive_nudges += 1
        current = genai_types.Content(role='user', parts=[genai_types.Part(text=NUDGE)])

    gcc = getattr(agent, 'generate_content_config', None)
    model = getattr(agent, 'model', None)
    out = {
        'versions': {m: __import__('importlib.metadata', fromlist=['version']).version(m)
                     for m in ('google-adk', 'litellm', 'adk-submission', 'swegemma', 'adk-eval-core', 'google-genai')},
        'config': {'time_minutes': args.time_minutes, 'tool_calls': args.tool_calls, 'turns': args.turns,
                   'cmd_timeout': args.cmd_timeout},
        'agent_instruction': getattr(agent, 'instruction', None),
        'generate_content_config': json.loads(gcc.model_dump_json(exclude_none=True)) if gcc is not None else None,
        'model_additional_args': getattr(model, '_additional_args', None),
        'workspace_tree_cmd': tree_cmd,
        'initial_prompt': prompt,
        'events': events,
        'passes': passes,
        'tool_calls_used': ctx.tool_calls_used,
        'llm_calls_used': ctx.llm_calls_used,
        'sandbox_commands': docker.cmds,
        'requests': CAPTURED,
    }
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    print('captured', len(CAPTURED), 'requests ->', args.out, '| events', events, '| passes', passes)
    shutil.rmtree(tmp, ignore_errors=True)


asyncio.run(main())
server.shutdown()
