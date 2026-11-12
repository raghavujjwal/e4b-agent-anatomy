"""train/format_probe.py: the classes of an unreadable call and the report, against a fake server."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
import format_probe as F  # noqa: E402

Q = '<|"|>'
OK = f'<|tool_call>call:edit_file{{filepath:{Q}a.py{Q},new_string:{Q}y{Q},old_string:{Q}x{Q}}}<tool_call|>'
KWARGS = '<|tool_call>call:edit_file{filepath="a.py",new_string="y",old_string="x"}<tool_call|>'
TRIPLE = f'<|tool_call>call:edit_file{{filepath:{Q}a.py{Q},new_string:"""y""",old_string:{Q}x{Q}}}<tool_call|>'


@pytest.mark.parametrize("text, cls", [
    (OK, "ok"),
    ("thinking, no call", "no_call"),
    (OK + OK, "several_calls"),
    (OK.replace("<tool_call|>", ""), "unclosed"),
    (KWARGS, "python_kwargs"),
    (TRIPLE, "unreadable"),
    (f'<|tool_call>call:edit_file{{filepath:{Q}a.py{Q},old_string:{Q}x{Q}}}<tool_call|>', "missing_argument"),
    (f'<|tool_call>call:launch_rocket{{filepath:{Q}a.py{Q}}}<tool_call|>', "undeclared"),
])
def test_a_generation_is_classed_by_how_its_call_is_written(text, cls):
    assert F.classify(text)[0] == cls


CANNED = {}                                    # model -> list of texts handed out in turn


class Fake(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        texts = CANNED[body["model"]]
        with LOCK:
            text = texts[Fake.n.setdefault(body["model"], 0) % len(texts)]
            Fake.n[body["model"]] += 1
        out = json.dumps({"choices": [{"text": text, "finish_reason": "stop"}], "usage": {"completion_tokens": len(text) // 4}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


LOCK = threading.Lock()
Fake.n = {}


def test_the_probe_reports_each_model_on_the_same_prefixes(tmp_path, capsys):
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    CANNED["base"] = [KWARGS, OK, TRIPLE, OK]                   # 2 ok, 1 python_kwargs, 1 unreadable
    CANNED["arm"] = [OK]
    prefixes = tmp_path / "check_rejected.jsonl"
    prefixes.write_text("".join(json.dumps({"run": "r", "instance_id": f"i{k}", "turn": 3, "tool": "edit_file", "prompt_ids": [1, 2, 3], "gen_ids": [4]}) + "\n" for k in range(4)),
                        encoding="utf-8")
    out = tmp_path / "out"
    try:
        assert F.main(["--rejected", str(prefixes), "--base-url", f"http://127.0.0.1:{srv.server_port}", "--model", "base", "--model", "arm", "--samples", "2",
                       "--workers", "1", "--out", str(out)]) == 0
    finally:
        srv.shutdown()
    s = json.loads((out / "probe_summary.json").read_text(encoding="utf-8"))
    assert s["base"]["n"] == 8 and s["base"]["prefixes"] == 4 and s["base"]["ok_%"] == 50.0 and s["base"]["rejected_%"] == 25.0
    assert s["base"]["classes"] == {"ok": 4, "python_kwargs": 2, "unreadable": 2}
    assert s["base"]["prefixes_with_a_rejected_%"] == 50.0, "the two kwargs generations fall on two prefixes of four"
    assert s["arm"]["ok_%"] == 100.0 and s["arm"]["rejected_%"] == 0.0 and s["arm"]["request_errors"] == 0
    rows = [json.loads(l) for l in open(out / "base_probe.jsonl", encoding="utf-8")]
    assert len(rows) == 8 and {r["prefix"] for r in rows} == {f"r|i{k}|3" for k in range(4)}
    assert "arm: rejected-style generations 0.0% (base 25.0%)" in capsys.readouterr().out
