"""mine/rank_files_probe.py: identifier tokens and the BM25 ranking, without any repo."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mine"))
import rank_files_probe as rp  # noqa: E402


def test_tokens_split_identifiers_and_keep_the_whole_name():
    t = rp.tokens("fix the OAuth2PasswordBearer and get_openapi_path")
    assert "oauth2passwordbearer" in t and "password" in t and "bearer" in t
    assert "openapi" in t and "get_openapi_path" in t or "path" in t
    assert "the" not in t and "and" not in t


def test_bm25_puts_the_matching_document_first():
    docs = [rp.tokens("def render_table(): pass"), rp.tokens("class Router: pass"), rp.tokens("def parse_args(): pass")]
    bm = rp.Bm25(docs)
    order = rp.ranked(["a.py", "b.py", "c.py"], bm.scores(rp.tokens("render table is broken")))
    assert order[0] == "a.py"
