"""Unit tests for the local-review-generation inference package.

Covered without any LLM server or model download:
  - prompt loading/rendering (baseline has no reviewer info; RAG carries
    verbatim retrieved documents and anti-copy instruction)
  - deterministic held-out query selection + verified clean-record join
  - llm endpoint policy (loopback-only, scheme/credential checks)
  - output path validation (slug components, containment round-trip)

The retrieval module is integration-tested by the harness dry-run against
the real indexes (documented in local_llm_baseline.md); a mock-free unit
test would require duplicating the frozen pipeline.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reviewerbrain.inference import (llm_client, outputs, prompts,  # noqa: E402
                                     heldout_queries)

# Non-slug components are exactly what the slug check exists to reject:
# separators, dot-only, and empty components each make an escaping path
# impossible to express, so rejecting all of them is the property tested.
BAD_TAGS = ["a/b", "a\\b", ".", "", "-lead", "tag.with.dots"]


def make_record(**over):
    rec = {
        "reviewer": "thockin", "repo": "kubernetes/kubernetes",
        "pr_number": 42, "pr_title": "Add feature", "pr_description": "",
        "file": "pkg/foo.go",
        "diff_hunk": "@@ -1,2 +1,3 @@\n context\n-old\n+new\n+added",
        "review_comment": "Why is this not validated?",
        "is_code_related": True, "led_to_code_change": True,
        "follow_up_patch": "", "_line": 2,
    }
    rec.update(over)
    return rec


class TestPrompts:
    def test_baseline_template_has_no_reviewer_info(self):
        t = prompts.load_template("baseline_v1")
        assert t["version"] == "baseline_v1"
        assert not t.get("requires_reviewer")
        blob = t["system"] + t["user_template"]
        assert "thockin" not in blob and "ezyang" not in blob
        assert "Historical" not in blob and "{retrieved_examples}" not in blob

    def test_rag_template_requires_reviewer_and_examples(self):
        t = prompts.load_template("rag_v1")
        assert t["version"] == "rag_v1"
        assert t.get("requires_reviewer")
        assert "{retrieved_examples}" in t["user_template"]
        assert "{reviewer}" in t["system"]

    def test_render_baseline_mentions_no_reviewer(self):
        t = prompts.load_template("baseline_v1")
        msgs = prompts.build_messages(t, make_record(), "thockin")
        assert msgs[0]["role"] == "system"
        blob = msgs[0]["content"] + msgs[1]["content"]
        assert "thockin" not in blob
        assert "```diff" in msgs[1]["content"]
        assert "pkg/foo.go" in msgs[1]["content"]

    def test_render_rag_carries_verbatim_documents(self):
        t = prompts.load_template("rag_v1")
        doc = ("Reviewer comment:\nWatch the error path.\n\nFile:\n"
               "pkg/bar.go\n\nRelevant diff:\n+ x := 1")
        retrieved = [{"doc_id": "thockin:5", "cos": 0.77, "file": "pkg/bar.go",
                      "pr_number": 9, "comment": "Watch the error path.",
                      "document": doc}]
        msgs = prompts.build_messages(t, make_record(), "thockin", retrieved)
        blob = msgs[0]["content"] + msgs[1]["content"]
        assert "thockin" in blob
        assert doc in blob                      # verbatim validated document
        assert "NEVER copy" in msgs[0]["content"]

    def test_render_rag_without_hits_uses_note(self):
        t = prompts.load_template("rag_v1")
        msgs = prompts.build_messages(t, make_record(), "thockin", [])
        assert "No historical examples passed" in msgs[1]["content"]

    def test_pr_context_capped(self):
        t = prompts.load_template("baseline_v1")
        rec = make_record(pr_description="x" * 5000)
        msgs = prompts.build_messages(t, rec, "thockin")
        assert "description truncated" in msgs[1]["content"]
        assert len(msgs[1]["content"]) < 8000


class TestQueries:
    def test_select_queries_deterministic_and_evenly_spaced(self):
        held = [{"reviewer": "t", "query_id": f"t:{i}"} for i in range(20)]
        a = heldout_queries.select_queries(held, "t", 5)
        b = heldout_queries.select_queries(held, "t", 5)
        assert a == b
        assert len(a) == 5
        assert [r["query_id"] for r in a] == ["t:0", "t:4", "t:8", "t:12",
                                              "t:16"]

    def test_load_clean_join_verified(self, tmp_path, monkeypatch):
        # build a tiny fake clean file and matching held-out record
        rec = make_record()
        clean_dir = tmp_path / "processed"
        clean_dir.mkdir()
        clean_path = clean_dir / "thockin_clean.jsonl"
        clean_path.write_text(
            json.dumps(make_record(pr_number=1)) + "\n"
            + json.dumps(rec) + "\n", encoding="utf-8")
        monkeypatch.setattr(heldout_queries.paths, "CLEAN_FILES",
                            {"thockin": clean_path})
        held = {"query_id": "thockin:1", "reviewer": "thockin",
                "query_pr_number": 42, "query_review_comment":
                    rec["review_comment"], "file": rec["file"]}
        assert heldout_queries.load_clean_for_heldout(held)["pr_number"] == 42
        bad = dict(held, query_pr_number=43)
        with pytest.raises(ValueError):
            heldout_queries.load_clean_for_heldout(bad)


class TestLLMPolicy:
    def test_rejects_non_loopback_host(self):
        with pytest.raises(llm_client.LLMError):
            llm_client.validate_endpoint("http://example.com:11434")

    def test_rejects_private_and_non_http_schemes(self):
        with pytest.raises(llm_client.LLMError):
            llm_client.validate_endpoint("http://192.168.1.5:11434")
        with pytest.raises(llm_client.LLMError):
            llm_client.validate_endpoint("ftp://127.0.0.1:11434")

    def test_accepts_localhost_and_loopback_ip(self):
        # 127.0.0.1 is a literal; localhost must resolve to loopback here
        assert llm_client.validate_endpoint("http://127.0.0.1:11434")[
            0] in ("http", "https")
        scheme, host, port = llm_client.validate_endpoint("http://localhost:11434")
        assert host == "localhost" and port == 11434


class TestOutputs:
    def test_rejects_bad_tags(self):
        for bad in BAD_TAGS:
            with pytest.raises(SystemExit):
                outputs.resolve_run_dir(bad)
        run_dir = outputs.resolve_run_dir("ok-tag_1")
        assert run_dir == outputs.OUT_ROOT / "ok-tag_1"

    def test_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(outputs, "OUT_ROOT", tmp_path)
        run_dir = outputs.resolve_run_dir("rt")
        with outputs.open_run_file(run_dir, "rag", "generations.jsonl") as f:
            f.write(json.dumps({"a": 1}) + "\n")
        assert outputs.read_generations(run_dir, "rag") == [{"a": 1}]
