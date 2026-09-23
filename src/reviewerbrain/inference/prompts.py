"""Versioned prompt loading and rendering for review generation.

Templates live in configs/prompts/<name>.yaml. Rendering only substitutes
placeholders; it never edits the validated representation or datasets.
"""
from pathlib import Path

import yaml

from reviewerbrain import paths
from reviewerbrain.inference import heldout_queries as q

PROMPT_DIR = paths.ROOT / "configs" / "prompts"

NO_EXAMPLES_NOTE = ("(No historical examples passed the similarity gate "
                    "for this change — rely on your general review "
                    "judgment.)")


def load_template(name, prompt_dir=None):
    """Load a versioned prompt template, e.g. name='baseline_v1'."""
    d = Path(prompt_dir) if prompt_dir else PROMPT_DIR
    with open(d / f"{name}.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def rendered_examples(retrieved):
    """Render the retrieved-examples block from `retrieve()` output.

    Each example is the validated index document inserted verbatim, so the
    prompt shows exactly what the retrieval stage stores — no re-derivation.
    """
    if not retrieved:
        return NO_EXAMPLES_NOTE
    blocks = []
    for i, r in enumerate(retrieved, 1):
        blocks.append(f"[Example {i}] (cosine {r['cos']:.3f}, "
                      f"PR#{r['pr_number']}, file: {r['file']})\n"
                      f"{r['document']}")
    return "\n\n---\n\n".join(blocks)


def build_messages(template, rec, reviewer, retrieved=None):
    """Render the template into chat messages for one cleaned record.

    `retrieved` is required for RAG templates and ignored by baseline
    templates; the split is driven by the template's own `version`.
    """
    fmt = {}
    if "{reviewer}" in template["user_template"] or \
            "{reviewer}" in template["system"]:
        fmt["reviewer"] = reviewer
    fmt["file_path"] = rec["file"] or "(unknown path)"
    fmt["pr_context"] = q.pr_context(rec)
    fmt["code_change"] = q.cap_diff(rec["diff_hunk"])
    if "{retrieved_examples}" in template["user_template"]:
        fmt["retrieved_examples"] = rendered_examples(retrieved or [])
    system = template["system"].format(**fmt)
    user = template["user_template"].format(**fmt)
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]
