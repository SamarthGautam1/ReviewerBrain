"""Four-mode pipeline, adapter registry and backend tests — no model ever.

The real 7B model is never loaded here: retrieval is a fake retriever,
generation is MockBackend, and the HFBackend is only checked to the
extent that importing backends.py stays model-free (hermetic subprocess).
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reviewerbrain.inference import adapters, backends, pipeline as pipe  # noqa: E402

REVIEWER = "thockin"


def make_record(**over):
    rec = {
        "reviewer": REVIEWER, "repo": "kubernetes/kubernetes",
        "pr_number": 42, "pr_title": "Add feature", "pr_description": "",
        "file": "pkg/foo.go",
        "diff_hunk": "@@ -1,2 +1,3 @@\n context\n-old\n+new\n+added",
        "review_comment": "Why is this not validated?",
        "is_code_related": True, "led_to_code_change": True,
        "follow_up_patch": "",
    }
    rec.update(over)
    return rec


def fake_retriever(calls):
    def retrieve(rec):
        calls.append(rec["pr_number"])
        return [{"doc_id": "thockin:5", "cos": 0.77, "file": "pkg/bar.go",
                 "pr_number": 9, "comment": "Watch the error path.",
                 "document": "Reviewer comment:\nWatch the error path."}]
    return retrieve


@pytest.fixture
def with_adapter(tmp_path):
    d = tmp_path / REVIEWER
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "r": 16}), encoding="utf-8")
    return tmp_path


class TestAdapterRegistry:
    def test_find_and_has(self, with_adapter):
        found = adapters.find_adapters(with_adapter)
        assert found == {REVIEWER: with_adapter / REVIEWER}
        assert adapters.has_adapter(REVIEWER, with_adapter)
        assert not adapters.has_adapter("ezyang", with_adapter)

    def test_marker_required(self, tmp_path):
        (tmp_path / "thockin").mkdir()          # no adapter_config.json
        assert adapters.find_adapters(tmp_path) == {}
        assert not adapters.has_adapter("thockin", tmp_path)
        with pytest.raises(adapters.AdapterNotAvailable):
            adapters.adapter_dir("thockin", tmp_path)

    def test_slug_rejection(self, tmp_path):
        assert not adapters.has_adapter("../evil", tmp_path)
        assert not adapters.has_adapter("a/b", tmp_path)
        with pytest.raises(adapters.AdapterNotAvailable):
            adapters.adapter_dir("", tmp_path)

    def test_meta_roundtrip(self, with_adapter):
        meta = adapters.adapter_meta(REVIEWER, with_adapter)
        assert meta["peft_type"] == "LORA"

    def test_error_mentions_kaggle_workflow(self, tmp_path):
        with pytest.raises(adapters.AdapterNotAvailable, match="kaggle"):
            adapters.adapter_dir(REVIEWER, tmp_path)


class TestModeSemantics:
    def test_mode_validation(self):
        for m in ("base", "base_rag", "lora", "lora_rag"):
            pipe.check_mode(m)
        for bad in ("", "rag", "lora+rag", "BASE"):
            with pytest.raises(ValueError):
                pipe.check_mode(bad)

    def test_flags(self):
        assert [pipe.mode_uses_rag(m) for m in pipe.MODES] == \
            [False, True, False, True]
        assert [pipe.mode_uses_adapter(m) for m in pipe.MODES] == \
            [False, False, True, True]

    def test_templates_are_distinct_by_design(self):
        t = pipe.MODE_TO_TEMPLATE
        assert t["base"] == "baseline_v1"
        assert t["base_rag"] == t["lora_rag"] == "rag_v1"
        assert t["lora"] == "sft_v1"


class TestRunReview:
    def test_base_mode_no_rag_no_identity(self):
        calls = []
        rec = make_record()
        out = pipe.run_review("base", REVIEWER, rec, backends.MockBackend(),
                              retriever=fake_retriever(calls))
        assert calls == []                      # retrieval never invoked
        blob = out["messages"][0]["content"] + out["messages"][1]["content"]
        assert REVIEWER not in blob
        assert out["generated_comment"] and out["meta"]["mock"] is True

    def test_base_rag_retrieves_and_carries_documents(self):
        calls = []
        rec = make_record()
        out = pipe.run_review("base_rag", REVIEWER, rec,
                              backends.MockBackend(),
                              retriever=fake_retriever(calls))
        assert calls == [42]
        doc = out["retrieved"][0]["document"]
        blob = out["messages"][0]["content"] + out["messages"][1]["content"]
        assert doc in blob                      # verbatim evidence in prompt
        assert REVIEWER in out["messages"][0]["content"]

    def test_lora_requires_adapter(self):
        with pytest.raises(adapters.AdapterNotAvailable):
            pipe.run_review("lora", REVIEWER, make_record(),
                            backends.MockBackend(), retriever=fake_retriever([]))

    def test_lora_mode_uses_sft_template_no_retrieval(self, with_adapter):
        calls = []
        out = pipe.run_review("lora", REVIEWER, make_record(),
                              backends.MockBackend(),
                              retriever=fake_retriever(calls),
                              adapters_root=with_adapter)
        assert calls == []                      # lora mode: no retrieval
        assert out["prompt_template"] == "sft_v1"
        assert REVIEWER in out["messages"][0]["content"]
        assert out["uses_adapter"] and not out["uses_rag"]
        assert "retrieved_examples" not in out["messages"][1]["content"]

    def test_lora_rag_combines_adapter_and_rag(self, with_adapter):
        calls = []
        out = pipe.run_review("lora_rag", REVIEWER, make_record(),
                              backends.MockBackend(),
                              retriever=fake_retriever(calls),
                              adapters_root=with_adapter)
        assert calls == [42]
        assert out["prompt_template"] == "rag_v1"
        assert out["uses_adapter"] and out["uses_rag"]

    def test_record_shape(self, with_adapter):
        out = pipe.run_review("lora_rag", REVIEWER, make_record(),
                              backends.MockBackend(),
                              retriever=fake_retriever([]),
                              adapters_root=with_adapter)
        assert set(out) == {"reviewer", "mode", "prompt_template", "uses_rag",
                            "uses_adapter", "messages", "retrieved",
                            "generated_comment", "meta"}
        # generation prompt: system + user only (no assistant turn)
        roles = [m["role"] for m in out["messages"]]
        assert roles == ["system", "user"]

    def test_reviewer_mismatch_rejected(self, with_adapter):
        with pytest.raises(ValueError):
            pipe.run_review("base", REVIEWER, make_record(reviewer="ezyang"),
                            backends.MockBackend())


class TestBackends:
    def test_mock_deterministic_and_varying(self):
        a = backends.MockBackend().generate(
            [{"role": "system", "content": "s"},
             {"role": "user", "content": "File: pkg/foo.go\n```diff\n+x\n```"}])
        b = backends.MockBackend().generate(
            [{"role": "system", "content": "s"},
             {"role": "user", "content": "File: pkg/foo.go\n```diff\n+x\n```"}])
        c = backends.MockBackend().generate(
            [{"role": "system", "content": "s"},
             {"role": "user", "content": "File: pkg/bar.go\n```diff\n+y\n```"}])
        assert a[0] == b[0]                     # same input -> same review
        assert a[0] != c[0]                     # different input varies
        assert a[1]["mock"] is True

    def test_ollama_endpoint_policy_enforced(self):
        be = backends.OllamaBackend("qwen2.5-coder:7b",
                                    endpoint="http://example.com:11434")
        with pytest.raises(Exception):          # LLMError from llm_client
            be.generate([{"role": "user", "content": "hi"}])

    def test_backends_import_is_model_free(self):
        root = Path(__file__).resolve().parents[1]
        heavy = repr({"torch", "transformers", "peft"})
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(root / 'src')!r})\n"
            "from reviewerbrain.inference import backends, pipeline, adapters\n"
            f"assert not {heavy} & set(sys.modules), 'heavy module imported'\n"
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=60)
        assert r.returncode == 0, r.stderr

    def test_hf_backend_rejects_missing_adapter_without_loading(self, tmp_path):
        """Adapter validation happens before any weight loading."""
        with pytest.raises(adapters.AdapterNotAvailable):
            backends.HFBackend("Qwen/Qwen2.5-Coder-7B-Instruct",
                               reviewer_adapter=REVIEWER,
                               adapters_root=tmp_path)
