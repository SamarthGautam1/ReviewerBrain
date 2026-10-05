"""SFT dataset construction: split isolation, leakage prevention, format.

No model is loaded anywhere in this file — the SFT builder is pure
Python. Fake cleaned corpora follow the 11-field schema from
reviewerbrain.data.schema; PR numbers are chosen so the frozen held-out
rule (every 5th sorted PR, position % 5 == 4) actually holds PRs out.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reviewerbrain import paths  # noqa: E402
from reviewerbrain.data.schema import CLEAN_RECORD_FIELDS  # noqa: E402
from reviewerbrain.evaluation.split import held_out_prs  # noqa: E402
from reviewerbrain.training import sft_data  # noqa: E402
from reviewerbrain.training.sft_data import load_training_sft_config  # noqa: E402,E501
from reviewerbrain.training.config import (TrainingConfigError,  # noqa: E402
                                           load_training_config,
                                           validate_training_config)

REVIEWER = "thockin"


def make_record(pr_number, **over):
    rec = {
        "reviewer": REVIEWER, "repo": "kubernetes/kubernetes",
        "pr_number": pr_number, "pr_title": f"PR {pr_number}",
        "pr_description": "What this PR does.",
        "file": "pkg/foo.go",
        "diff_hunk": "@@ -1,2 +1,3 @@\n context\n-old\n+new\n+added",
        "review_comment": f"Comment on PR {pr_number}: check the error path.",
        "is_code_related": True, "led_to_code_change": True,
        "follow_up_patch": "",
    }
    rec.update(over)
    return rec


def fake_corpus(n_prs=9):
    """One record per PR; PR numbers 1..n_prs so the frozen rule holds out
    PR 5 (position 4 of the sorted unique list)."""
    return [make_record(p) for p in range(1, n_prs + 1)]


class TestSplitPrs:
    def test_pr_level_and_deterministic(self):
        recs = fake_corpus(9)
        held, val, train = sft_data.split_prs(recs)
        assert held == {5}                       # frozen rule: 5th sorted PR
        assert held_out_prs(recs)[REVIEWER] == held
        # every 10th remaining PR (position 9 of 1..9 minus {5} = 8 PRs): none
        assert val == set()
        assert train == {1, 2, 3, 4, 6, 7, 8, 9}
        assert sft_data.split_prs(recs) == (held, val, train)  # no RNG

    def test_val_carveout_is_pr_level(self):
        recs = fake_corpus(30)                   # held: {5,10,15,20,25,30}
        held, val, train = sft_data.split_prs(recs)
        remaining = sorted(set(range(1, 31)) - held)
        expected_val = {p for i, p in enumerate(remaining)
                        if i % 10 == 9}
        assert val == expected_val and len(val) == 2
        assert not (val & held) and not (train & held) and not (val & train)
        assert train | val | held == set(range(1, 31))  # nothing lost

    def test_rejects_bad_val_position(self):
        with pytest.raises(ValueError):
            sft_data.split_prs(fake_corpus(3), val_period=10, val_position=10)

    def test_requires_single_reviewer(self):
        recs = fake_corpus(3) + [make_record(1, reviewer="ezyang")]
        with pytest.raises(ValueError):
            sft_data.split_prs(recs)


class TestBuildExample:
    def test_roles_target_and_identity(self):
        rec = make_record(42)
        ex = sft_data.build_example(rec, REVIEWER)
        roles = [m["role"] for m in ex["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert rec["review_comment"] == ex["messages"][2]["content"]
        assert REVIEWER in ex["messages"][0]["content"]
        assert "pkg/foo.go" in ex["messages"][1]["content"]
        assert ex["meta"]["pr_number"] == 42

    def test_follow_up_patch_never_rendered(self):
        fu = ("diff --git a/pkg/foo.go b/pkg/foo.go\n"
              "+FUTURE_PATCH_MARKER_12345")
        rec = make_record(7, follow_up_patch=fu)
        ex = sft_data.build_example(rec, REVIEWER)
        assert "FUTURE_PATCH_MARKER_12345" not in json.dumps(ex)
        assert "follow_up_patch" not in json.dumps(ex)

    def test_labels_not_features(self):
        rec = make_record(7)
        blob = json.dumps(sft_data.build_example(rec, REVIEWER))
        for forbidden in ("led_to_code_change", "is_code_related",
                          "True", "False"):
            assert forbidden not in blob

    def test_wrong_reviewer_rejected(self):
        with pytest.raises(ValueError):
            sft_data.build_example(make_record(7, reviewer="ezyang"),
                                   REVIEWER)

    def test_diff_cap_applied(self):
        rec = make_record(7, diff_hunk="x" * 9000)
        ex = sft_data.build_example(rec, REVIEWER, diff_max_chars=6000)
        assert "diff truncated" in ex["messages"][1]["content"]


class TestBuildReviewerDataset:
    def test_held_out_prs_fully_excluded(self):
        recs = fake_corpus(9)                    # frozen held-out: {5}
        # extra sibling comment on the held-out PR must also be excluded
        recs.append(make_record(5, file="pkg/bar.go",
                                review_comment="Sibling on held-out PR"))
        train_ex, val_ex, stats = sft_data.build_reviewer_dataset(recs,
                                                                  REVIEWER)
        held_prs = {ex["meta"]["pr_number"] for ex in train_ex + val_ex}
        assert 5 not in held_prs
        assert stats["held_out_prs"] == 1
        assert stats["held_out_records"] == 2
        assert stats["train_examples"] == 8

    def test_verify_examples_passes_on_valid(self):
        recs = fake_corpus(9)
        train_ex, val_ex, _ = sft_data.build_reviewer_dataset(recs, REVIEWER)
        assert sft_data.verify_examples(train_ex + val_ex, recs)

    def test_verify_examples_catches_target_drift(self):
        recs = fake_corpus(9)
        train_ex, val_ex, _ = sft_data.build_reviewer_dataset(recs, REVIEWER)
        broken = json.loads(json.dumps(train_ex[0]))
        broken["messages"][2]["content"] += " (edited)"
        with pytest.raises(AssertionError):
            sft_data.verify_examples([broken], recs)

    def test_future_info_guard_trips(self, monkeypatch):
        """If a follow_up_patch text were rendered, the builder must raise."""
        marker = "FUTURE_PATCH_MARKER_987654321"
        recs = fake_corpus(9)
        recs[0]["follow_up_patch"] = marker
        real_build = sft_data.build_example

        def leaky_build(rec, reviewer, template=None, diff_max_chars=6000):
            ex = real_build(rec, reviewer, template, diff_max_chars)
            if marker in rec["diff_hunk"]:
                ex["messages"][1]["content"] += marker   # simulated leak
            return ex

        monkeypatch.setattr(sft_data, "build_example", leaky_build)
        recs[0]["diff_hunk"] += marker
        with pytest.raises(AssertionError, match="follow_up_patch"):
            sft_data.build_reviewer_dataset(recs, REVIEWER)

    def test_schema_fields_preserved_in_corpus(self):
        for rec in fake_corpus(3):
            assert set(rec.keys()) == set(CLEAN_RECORD_FIELDS)


class TestWriteAndEndToEnd:
    def test_write_roundtrip(self, tmp_path):
        recs = fake_corpus(9)
        train_ex, val_ex, stats = sft_data.build_reviewer_dataset(recs,
                                                                  REVIEWER)
        stats_path = sft_data.write_reviewer_files(tmp_path, REVIEWER,
                                                   train_ex, val_ex, stats,
                                                   recs)
        assert stats_path.exists()
        train_rows = [json.loads(l) for l in
                      (tmp_path / f"{REVIEWER}_sft_train.jsonl")
                      .read_text(encoding="utf-8").splitlines() if l]
        assert train_rows == train_ex
        assert set(train_rows[0].keys()) == {"messages", "meta"}
        assert json.loads(stats_path.read_text(encoding="utf-8")) == stats

    def test_real_corpus_end_to_end(self):
        """Integration against the real cleaned corpora + frozen split
        (skipped when data/processed is not regenerated locally)."""
        if not all(p.exists() for p in paths.CLEAN_FILES.values()):
            pytest.skip("cleaned datasets not regenerated locally")
        for name in ("thockin", "ezyang"):
            recs = sft_data.load_clean_records(paths.CLEAN_FILES[name])
            train_ex, val_ex, stats = sft_data.build_reviewer_dataset(
                recs, name)
            assert stats["records"] == (1995 if name == "thockin" else 669)
            assert (stats["train_examples"] + stats["val_examples"]
                    + stats["held_out_records"]) == stats["records"]
            held = held_out_prs(recs)[name]
            used = {ex["meta"]["pr_number"] for ex in train_ex + val_ex}
            assert not (used & held), "held-out PR reached SFT data"
            sft_data.verify_examples(train_ex + val_ex, recs)


class TestTrainingSftConfig:
    def test_defaults_from_repo_config(self):
        sft = load_training_sft_config()
        assert sft["prompt_template"] == "sft_v1"
        assert 0 <= sft["val_position"] < sft["val_period"]
