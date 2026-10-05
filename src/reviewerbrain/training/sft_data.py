"""Reviewer-specific SFT dataset construction with mandatory leakage guards.

Objective (frozen): given the reviewer identity and a code change, generate
the kind of review comment that reviewer would write. The supervised target
is the historical `review_comment`; the model input is the same shape the
inference harness serves (file path, PR context, capped diff hunk).

Data rules (all enforced here, asserted before any byte is written):
  1. PR-level splitting. The frozen deterministic held-out rule
     (reviewerbrain.evaluation.split, every 5th sorted PR) defines the
     protected evaluation PRs; NO record from a held-out PR enters train
     or validation — this is the same guard the serving retrieval applies,
     so the SFT set, the RAG serving set and the held-out evaluation can
     never overlap.
  2. Validation carve-out is also PR-level: from the remaining sorted PRs,
     every val_period-th (position % val_period == val_position) becomes
     the training-time validation monitor. No RNG anywhere.
  3. `follow_up_patch` is never read into any example text — it post-dates
     the review and would leak future information.
  4. `led_to_code_change` / `is_code_related` are labels/metadata, never
     model features; they are not rendered into any example.
  5. Retrieved RAG examples are inference-time evidence and are NOT part
     of SFT data (no RAG-example leakage into the training set).
  6. The assistant target is the cleaned `review_comment` verbatim.

Message format is the Hugging Face chat convention
([{"role": "system"|"user"|"assistant", "content": ...}]) rendered from the
versioned template configs/prompts/sft_v1.yaml — the SAME template `mode:
lora` serves, so training and inference prompts cannot drift.

Pure Python: importing or running this module never loads a model.
"""
import json
from pathlib import Path

import yaml

from reviewerbrain import paths
from reviewerbrain.evaluation.split import held_out_prs
from reviewerbrain.inference import heldout_queries as q
from reviewerbrain.inference import prompts as pr

# Reuse the inference harness caps verbatim so the training distribution
# matches what the serving prompt can contain (heldout_queries constants).
DIFF_MAX_CHARS = q.QUERY_DIFF_MAX_CHARS
DESC_MAX_CHARS = q.DESC_MAX_CHARS

DEFAULT_VAL_PERIOD = 10
DEFAULT_VAL_POSITION = 9


def load_clean_records(clean_file):
    """Load one cleaned JSONL file in file order (deterministic)."""
    rows = []
    with open(clean_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def split_prs(records, val_period=DEFAULT_VAL_PERIOD,
              val_position=DEFAULT_VAL_POSITION):
    """Deterministic PR-level three-way split for one reviewer's records.

    Returns (held_prs, val_prs, train_prs) as sets of pr_number:
      held  — the frozen held-out evaluation PRs (NEVER trained on)
      val   — every val_period-th of the remaining sorted PRs
      train — the rest
    The same PR can appear in exactly one set; no RNG is involved.
    """
    if not 0 <= val_position < val_period:
        raise ValueError(f"val_position must be in [0, {val_period}): "
                         f"got {val_position}")
    held = held_out_prs(records)[_single_reviewer(records)]
    remaining = sorted({r["pr_number"] for r in records
                        if r["pr_number"] not in held})
    val = {p for i, p in enumerate(remaining)
           if i % val_period == val_position}
    train = set(remaining) - val
    return held, val, train


def _single_reviewer(records):
    names = {r["reviewer"] for r in records}
    if len(names) != 1:
        raise ValueError(f"records must be one reviewer's, got {names}")
    return names.pop()


def build_example(rec, reviewer, template=None, diff_max_chars=DIFF_MAX_CHARS):
    """One chat-format SFT example from a cleaned record.

    Returns {"messages": [...], "meta": {...}}. The reviewer name is the
    only reviewer-specific content (system message); the user content is
    the file path, optional PR context and the character-capped diff —
    the same fields the inference harness serves. follow_up_patch,
    led_to_code_change and is_code_related are never read.
    """
    if rec["reviewer"] != reviewer:
        raise ValueError(f"record reviewer {rec['reviewer']!r} != "
                         f"requested reviewer {reviewer!r}")
    template = template or pr.load_template("sft_v1")
    fmt = {
        "reviewer": reviewer,
        "file_path": rec["file"] or "(unknown path)",
        "pr_context": q.pr_context(rec),
        "code_change": q.cap_diff(rec["diff_hunk"], diff_max_chars),
    }
    system = template["system"].format(**fmt)
    user = template["user_template"].format(**fmt)
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": rec["review_comment"]},
        ],
        "meta": {
            "reviewer": reviewer,
            "pr_number": rec["pr_number"],
            "file": rec["file"],
        },
    }


def build_reviewer_dataset(records, reviewer, template=None,
                           val_period=DEFAULT_VAL_PERIOD,
                           val_position=DEFAULT_VAL_POSITION,
                           diff_max_chars=DIFF_MAX_CHARS):
    """Full leakage-guarded SFT split for one reviewer.

    Returns (train_examples, val_examples, stats). Raises if any guard
    trips; the guards re-verify the written rows would be clean rather
    than trusting the split bookkeeping.
    """
    held, val, train = split_prs(records, val_period, val_position)

    train_ex, val_ex = [], []
    n_held_records = 0
    for rec in records:
        p = rec["pr_number"]
        if p in held:
            n_held_records += 1
            continue          # frozen held-out PR: never trained on
        ex = build_example(rec, reviewer, template, diff_max_chars)
        if p in val:
            val_ex.append(ex)
        elif p in train:
            train_ex.append(ex)
        else:
            raise AssertionError(f"PR {p} mapped to no split — "
                                 "split bookkeeping is inconsistent")

    # Guard 3: a record's own follow_up_patch (the only plausible future-
    # information leak path — the builder formats nothing else) must not
    # appear in its example. Structural field-name checks run later in
    # verify_examples() on the exact objects written.
    fu_by_pr = {}
    for r in records:
        fu = (r.get("follow_up_patch") or "").strip()
        if fu:
            fu_by_pr.setdefault(r["pr_number"], []).append(fu)
    for p, fus in fu_by_pr.items():
        pool = val_ex if p in val else train_ex
        for ex in pool:
            if ex["meta"]["pr_number"] != p:
                continue
            blob = json.dumps(ex["messages"], ensure_ascii=False)
            if any(fu in blob for fu in fus):
                raise AssertionError(
                    f"PR {p}: follow_up_patch text reached its SFT example")
    stats = {
        "reviewer": reviewer,
        "records": len(records),
        "prs_total": len({r["pr_number"] for r in records}),
        "held_out_prs": len(held),
        "held_out_records": n_held_records,
        "train_prs": len(train),
        "val_prs": len(val),
        "train_examples": len(train_ex),
        "val_examples": len(val_ex),
        "val_period": val_period,
        "val_position": val_position,
        "prompt_template": (template or pr.load_template("sft_v1"))["version"],
        "diff_max_chars": diff_max_chars,
    }
    return train_ex, val_ex, stats


def verify_examples(examples, source_records):
    """Post-build verification applied to the actual example objects.

    Checks, per example:
      - exactly system/user/assistant roles in order
      - assistant content equals some cleaned review_comment verbatim
        (target fidelity)
      - reviewer identity present in the system message
      - no follow_up_patch text and no label fields rendered anywhere
    Raises AssertionError on any violation. Also used by tests.
    """
    targets = {r["review_comment"] for r in source_records}
    for ex in examples:
        roles = [m["role"] for m in ex["messages"]]
        assert roles == ["system", "user", "assistant"], roles
        blob = json.dumps(ex, ensure_ascii=False)
        for forbidden in ("follow_up_patch", "led_to_code_change",
                          "is_code_related"):
            assert forbidden not in blob, f"{forbidden} rendered into example"
        assert ex["messages"][2]["content"] in targets
        assert ex["meta"]["reviewer"] in ex["messages"][0]["content"]
    return True


def write_reviewer_files(out_dir, reviewer, train_ex, val_ex, stats,
                         source_records):
    """Write the reviewer's SFT JSONL files + stats after verification.

    Files: <out_dir>/<reviewer>_sft_train.jsonl, _sft_val.jsonl. Rows carry
    messages + meta only. Verification runs on the exact objects written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    verify_examples(train_ex + val_ex, source_records)
    for split, rows in (("train", train_ex), ("val", val_ex)):
        p = out_dir / f"{reviewer}_sft_{split}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    stats_path = out_dir / f"{reviewer}_sft_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats_path


def load_template_checked(name):
    """Load a prompt template for SFT use; the template must be
    reviewer-conditioned (the supervised objective is reviewer-specific)."""
    tpl = pr.load_template(name)
    if not tpl.get("requires_reviewer"):
        raise ValueError(f"SFT template must be reviewer-conditioned: {name}")
    if "{reviewer}" not in tpl["system"] and "{reviewer}" not in tpl["user_template"]:
        raise ValueError(f"SFT template carries no reviewer placeholder: {name}")
    return tpl


def load_training_sft_config(config_path=None):
    """SFT section of the training config with defaults filled in."""
    with open(config_path or paths.TRAINING_CONFIG_PATH, "r",
              encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sft = dict(cfg.get("sft") or {})
    sft.setdefault("prompt_template", "sft_v1")
    sft.setdefault("val_period", DEFAULT_VAL_PERIOD)
    sft.setdefault("val_position", DEFAULT_VAL_POSITION)
    sft.setdefault("diff_max_chars", DIFF_MAX_CHARS)
    return sft
