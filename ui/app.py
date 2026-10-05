"""ReviewerBrain demonstration UI (Streamlit).

Lightweight seminar demo of the four-mode personalized review pipeline:

    reviewer + diff → [frozen RAG retrieval] → [reviewer LoRA adapter]
    → generated review → displayed with the retrieved evidence

The model backend is configurable; the default is the deterministic
MockBackend so the demonstration runs on ANY machine (including the
development machine, which must never load the 7B model). Switch to the
Ollama backend to use a local server (loopback only); adapters come from
the Kaggle workflow (kaggle/README.md) — modes lora / lora_rag require
them and explain themselves when missing.

Run:  streamlit run ui/app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

from reviewerbrain.inference import adapters, backends  # noqa: E402
from reviewerbrain.inference import pipeline as pipe  # noqa: E402
from reviewerbrain.inference import prompts  # noqa: E402

REVIEWERS = ("thockin", "ezyang")
REVIEWER_INFO = {
    "thockin": ("Tim Hockin — kubernetes/kubernetes",
                "Networking/API conventions: API surface, validation, "
                "backward compatibility, test coverage."),
    "ezyang": ("Edward Yang — pytorch/pytorch",
                "PyTorch internals: dispatch, autograd semantics, dtype "
                "handling, subtle correctness edge cases."),
}
DIFF_MAX_CHARS = 6000
DESC_MAX_CHARS = 1000

st.set_page_config(page_title="ReviewerBrain", page_icon="🧠", layout="wide")
st.title("🧠 ReviewerBrain — reviewer-personalized code review")
st.caption("RAG over the reviewer's own history (frozen pipeline) + "
           "reviewer-specific LoRA (trained separately on Kaggle)")


def build_record(reviewer, file_path, diff, pr_title, pr_description):
    """Assemble a cleaned-schema-shaped record from UI inputs.

    pr_number is 0 (a new diff has none); the retrieval leakage guard
    drops candidates from held-out PRs and re-checks candidates against
    this PR, exactly as in serving.
    """
    return {
        "reviewer": reviewer, "repo": "", "pr_number": 0,
        "pr_title": pr_title, "pr_description": pr_description,
        "file": file_path, "diff_hunk": diff, "review_comment": "",
        "is_code_related": True, "led_to_code_change": False,
        "follow_up_patch": "",
    }


with st.sidebar:
    st.header("Configuration")
    reviewer = st.selectbox("Reviewer", REVIEWERS, key="reviewer",
                            format_func=lambda r: REVIEWER_INFO[r][0])
    st.caption(REVIEWER_INFO[reviewer][1])

    mode = st.radio("Mode", pipe.MODES, key="mode",
                    help="base: generic · base_rag: + retrieved history · "
                         "lora: reviewer adapter · lora_rag: adapter + "
                         "retrieved history")
    template_name = pipe.MODE_TO_TEMPLATE[mode]
    st.caption(f"Prompt template: `{template_name}` (versioned)")

    backend_name = st.radio("Model backend",
                            ["mock (no model — local demo)", "ollama "
                             "(local server, loopback)"],
                            key="backend", index=0)
    if backend_name.startswith("ollama"):
        model_name = st.text_input("Ollama model", "qwen2.5-coder:7b",
                                   key="ollama_model")
        endpoint = st.text_input("Ollama endpoint", "http://127.0.0.1:11434",
                                 key="ollama_endpoint")
        note = ("Modes lora / lora_rag need the Ollama model to carry the "
                "reviewer adapter (Modelfile ADAPTER).")
        st.caption(note)
        backend = backends.OllamaBackend(model_name, endpoint=endpoint)
    else:
        backend = backends.MockBackend()

    needs_adapter = pipe.mode_uses_adapter(mode)
    has_adapter = adapters.has_adapter(reviewer)
    st.divider()
    st.subheader("Adapter status")
    if has_adapter:
        st.success(f"Adapter present for {reviewer}")
        with st.expander("adapter_config.json"):
            st.json(adapters.adapter_meta(reviewer))
    else:
        st.warning("No trained adapter. Modes lora / lora_rag need one — "
                   "train via the Kaggle workflow (kaggle/README.md), then "
                   "unpack to adapters/<reviewer>/.")

left, right = st.columns([3, 2])

with left:
    st.subheader("Code change under review")
    file_path = st.text_input("File path", "pkg/apis/core/types.go",
                              key="file_path")
    uploaded = st.file_uploader("…or upload a diff/patch",
                                type=["diff", "patch", "txt"])
    default_diff = "@@ -118,7 +118,9 @@\n \tfunc (v *Validator) Validate() error {\n-\tif v.Name == \"\" {\n+\tif v.Name == \"\" && !v.Optional {\n \t\treturn fmt.Errorf(\"name required\")\n \t}\n+\t// TODO: validate spec fields\n+\tv.retryCount = 0\n \treturn nil\n }"
    if uploaded is not None:
        diff = uploaded.read().decode("utf-8", errors="replace")
    else:
        diff = st.text_area("Diff hunk", default_diff, height=260,
                            key="diff_hunk")
    pr_title = st.text_input("PR title (optional)",
                             "Add optional name to Validator", key="pr_title")
    pr_description = st.text_area("PR description (optional, capped)",
                                  height=80, key="pr_description")

rec = build_record(reviewer, file_path, diff, pr_title, pr_description)

right_col = right.container()
with right_col:
    st.subheader("Run")

    if st.button("1 — Run retrieval (evidence preview)",
                 disabled=not pipe.mode_uses_rag(mode),
                 help=("Enabled for the RAG modes. The frozen pipeline "
                       "retrieves top-3 reviewer-specific examples, "
                       "cosine gate ≥ 0.5.")):
        try:
            with st.spinner("Retrieving from the frozen per-reviewer "
                            "index…"):
                retrieved = pipe.default_retriever()(rec)
            st.session_state["preview_retrieved"] = retrieved
        except Exception as e:                          # noqa: BLE001
            st.session_state["preview_retrieved"] = None
            st.error(f"Retrieval unavailable: {e}")

    if st.button("2 — Generate review", type="primary"):
        try:
            with st.spinner("Generating…"):
                result = pipe.run_review(mode, reviewer, rec, backend)
            st.session_state["result"] = result
        except adapters.AdapterNotAvailable as e:
            st.session_state.pop("result", None)   # drop the stale display
            st.error(str(e))
        except Exception as e:                          # noqa: BLE001
            st.session_state.pop("result", None)
            st.error(f"Generation failed: {e}")

result = st.session_state.get("result")
preview = st.session_state.get("preview_retrieved")

st.subheader("Generated review")
if result:
    st.success(f"`{result['mode']}` · template `{result['prompt_template']}` "
               f"· backend `{result['meta'].get('backend', '?')}`")
    st.markdown(result["generated_comment"].replace("\n", "  \n"))
    if result["meta"].get("mock"):
        st.caption("⚠️ MockBackend output — deterministic canned text for "
                   "local demonstration; not model-generated.")
    shown = result["retrieved"]
elif preview is not None:
    shown = preview
else:
    shown = None

st.subheader("Retrieved historical evidence")
if shown is None:
    st.caption("No retrieval yet — run step 1 (RAG modes) or step 2 with a "
               "RAG mode.")
elif not shown:
    st.info("No historical examples passed the cosine gate (≥ 0.5) for "
            "this change — the prompt relies on general judgment "
            "(same rule as serving).")
else:
    for i, r in enumerate(shown, 1):
        with st.expander(f"Example {i} — cosine {r['cos']:.3f} · "
                         f"PR#{r['pr_number']} · {r['file']}"):
            st.markdown(r["document"].replace("\n", "  \n"))

with st.expander("Prompt actually sent (audit)"):
    if result:
        for m in result["messages"]:
            st.markdown(f"**{m['role']}**")
            st.code(m["content"], language=None)
    else:
        st.caption("Generate a review first.")

st.divider()
st.caption("RAG examples are inference-time evidence over the reviewer's "
           "history; LoRA adapters are trained separately on the same "
           "reviewer's data (never on retrieved examples). follow_up_patch "
           "is excluded everywhere by design.")
