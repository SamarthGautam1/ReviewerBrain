"""Cleaned-record schema for ReviewerBrain.

Every cleaned training/evaluation example carries exactly these fields
(established in the approved cleaning step; follow_up_patch is metadata
only and is never model input or indexed text).
"""

CLEAN_RECORD_FIELDS = (
    "reviewer",
    "repo",
    "pr_number",
    "pr_title",
    "pr_description",
    "file",
    "diff_hunk",
    "review_comment",
    "is_code_related",
    "led_to_code_change",
    "follow_up_patch",
)


def validate_clean_record(rec):
    """True if the record carries exactly the cleaned schema fields."""
    return set(rec.keys()) == set(CLEAN_RECORD_FIELDS)
