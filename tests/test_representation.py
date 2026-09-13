"""Comment-first representation invariants (needs the model tokenizer;
all-MiniLM-L6-v2 is cached locally after the first build/eval run)."""

import pytest

tw = pytest.importorskip("transformers")

from reviewerbrain.embeddings.model import MODEL_NAME  # noqa: E402
from reviewerbrain.retrieval import representation as repr_mod  # noqa: E402

MAX_LEN = 256


@pytest.fixture(scope="module")
def tok():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME, device="cpu",
                                    local_files_only=True)
        return model.tokenizer
    except OSError:
        pytest.skip("all-MiniLM-L6-v2 not cached locally "
                    "(run scripts/retrieval/build_indexes.py once)")


def make_record(hunk_lines=400):
    hunk = "@@ -1,3 +1,400 @@\n" + "\n".join(
        f"+line {i} of the new implementation" for i in range(hunk_lines))
    return {"reviewer": "thockin", "repo": "kubernetes/kubernetes",
            "pr_number": 1, "pr_title": "t", "pr_description": "d",
            "file": "pkg/apis/core/types.go", "diff_hunk": hunk,
            "review_comment": "Fix the off-by-one in the loop bound.",
            "is_code_related": True, "led_to_code_change": True,
            "follow_up_patch": "", "_line": 100}


def test_comment_is_token_prefix_and_in_window(tok):
    r = make_record()
    text, info = repr_mod.doc_text(r, tok, MAX_LEN)
    assert r["review_comment"] in text
    p_ids = tok.encode(repr_mod.DOC_SCAFFOLD + r["review_comment"],
                       add_special_tokens=False)
    t_ids = tok.encode(text, add_special_tokens=False)
    assert list(t_ids[:len(p_ids)]) == list(p_ids)
    assert not info["comment_overflows_window"]
    assert len(p_ids) + 2 <= MAX_LEN


def test_diff_is_trimmed_to_window(tok):
    r = make_record(hunk_lines=400)
    text, info = repr_mod.doc_text(r, tok, MAX_LEN)
    assert info["lines_total"] == 400
    # ~220 diff tokens at ~9-10 tokens/line -> low-20s lines kept
    assert info["lines_kept"] < 40
    assert info["total_tokens"] <= MAX_LEN


def test_anchor_matches_comment_line(tok):
    r = make_record()  # _line=100 -> new-file line 100 -> body index 99
    text, info = repr_mod.doc_text(r, tok, MAX_LEN)
    assert info["anchor_source"] == "matched_line"
    assert "line 99 of the new implementation" in text


def test_query_has_no_comment(tok):
    r = make_record()
    q, info = repr_mod.query_text(r, tok, MAX_LEN)
    assert r["review_comment"] not in q
    assert q.startswith("File:\n")
    assert info["total_tokens"] <= MAX_LEN
