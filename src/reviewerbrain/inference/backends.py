"""Generation backends for the four-mode pipeline.

A backend turns chat messages into (text, meta). Three implementations:

  MockBackend   — deterministic canned review, zero model loading. For
                  local UI development and tests on machines that must
                  never run an LLM.
  OllamaBackend — local Ollama server (loopback-only policy enforced by
                  reviewerbrain.inference.llm_client). Serves either the
                  plain base/instruct model (modes base, base_rag) or a
                  model loaded with a reviewer adapter (modes lora,
                  lora_rag — adapter wiring is the server's Modelfile).
  HFBackend     — transformers + PEFT in-process generation. Requires the
                  base weights and a GPU (Kaggle or a machine that
                  intentionally loaded them); imports torch/transformers/
                  peft lazily so importing this module stays model-free.
"""
import hashlib

from reviewerbrain.inference import adapters


class MockBackend:
    """Deterministic mock generator — NEVER a real model.

    Picks one of a few reviewer-shaped phrasings by hashing the user
    message, so the same diff always yields the same mock review while
    different diffs vary. Output is clearly marked as mock in meta.
    """
    name = "mock"
    requires_model = False

    _TEMPLATES = [
        ("This looks mostly fine, but {n}: the error path here is silently "
         "dropped. Please propagate or log it."),
        ("Small nit — {n}: consider validating the inputs before use; the "
         "caller may pass an empty value."),
        ("{n}: this changes behavior for existing callers. Is there a test "
         "covering the transition? If not, please add one."),
        ("We should not grow the API surface here ({n}). Can this reuse the "
         "existing helper instead?"),
        ("Concurrency concern at {n}: the state is read without holding the "
         "lock — the race window is small but real."),
    ]

    def generate(self, messages):
        user = messages[1]["content"] if len(messages) > 1 else ""
        file_line = next((l for l in user.splitlines()
                          if l.startswith("File:")), "File: (unknown)")
        path = file_line[len("File:"):].strip() or "(unknown)"
        idx = int(hashlib.sha256(user.encode("utf-8")).hexdigest(), 16) \
            % len(self._TEMPLATES)
        text = self._TEMPLATES[idx].format(n=path)
        return text, {"backend": "mock", "mock": True}


class OllamaBackend:
    """Local Ollama chat server via the validated loopback-only client."""
    name = "ollama"
    requires_model = True   # requires a server, not a local model load

    def __init__(self, model, endpoint=None, options=None, timeout_s=None):
        from reviewerbrain.inference import llm_client
        self.model = model
        self.endpoint = endpoint or llm_client.DEFAULT_ENDPOINT
        self.options = options or {}
        self.timeout_s = timeout_s or llm_client.DEFAULT_TIMEOUT_S

    def generate(self, messages):
        from reviewerbrain.inference import llm_client
        return llm_client.chat(self.endpoint, self.model, messages,
                               self.options, timeout_s=self.timeout_s)


class HFBackend:
    """In-process transformers+PEFT generation. GPU + weights machine only.

    reviewer_adapter: reviewer name whose adapter is loaded from the
    registry (None for base modes). Constructing this object downloads /
    loads several GB of weights — it must never be instantiated on the
    local development machine or in tests.
    """
    name = "hf"
    requires_model = True

    def __init__(self, base_model, reviewer_adapter=None, adapters_root=None,
                 max_new_tokens=512, temperature=0.0, device_map="auto"):
        if reviewer_adapter is not None:
            adapters.adapter_dir(reviewer_adapter, adapters_root)  # validate
        self._args = dict(base_model=base_model,
                          reviewer_adapter=reviewer_adapter,
                          adapters_root=adapters_root,
                          max_new_tokens=max_new_tokens,
                          temperature=temperature, device_map=device_map)
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch                     # noqa: F401 — lazy, GPU machine only
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self._args["base_model"])
        model = AutoModelForCausalLM.from_pretrained(
            self._args["base_model"], torch_dtype=torch.float16,
            device_map=self._args["device_map"])
        if self._args["reviewer_adapter"] is not None:
            from peft import PeftModel
            d = adapters.adapter_dir(self._args["reviewer_adapter"],
                                     self._args["adapters_root"])
            model = PeftModel.from_pretrained(model, str(d))
        self._model = (model.eval(), tok)

    def generate(self, messages):
        import torch
        self._load()
        model, tok = self._model
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        do_sample = self._args["temperature"] > 0
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=self._args["max_new_tokens"],
                do_sample=do_sample,
                temperature=(self._args["temperature"]
                             if do_sample else None))
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True)
        if not text.strip():
            raise RuntimeError("empty HF completion")
        return text.strip(), {"backend": "hf",
                              "adapter": self._args["reviewer_adapter"]}
