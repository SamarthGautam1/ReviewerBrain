"""Central path resolution for ReviewerBrain.

Every script/module resolves locations through this module so the
repository stays portable across machines. No absolute paths anywhere
else in the code base.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = ROOT / "indexes"
CHROMA_V2_DIR = INDEX_DIR / "chroma" / "v2"
EVALUATIONS_DIR = ROOT / "evaluations"
HELDOUT_DIR = EVALUATIONS_DIR / "heldout"
REPORTS_DIR = EVALUATIONS_DIR / "reports"
CONFIG_PATH = ROOT / "configs" / "rag" / "default.yaml"
DOCS_DIR = ROOT / "docs"
ARCHIVE_DIR = ROOT / "experiments" / "archive"

RAW_FILES = {
    "thockin": RAW_DIR / "thockin_kubernetes_training.jsonl",
    "ezyang": RAW_DIR / "ezyang_pytorch_training.jsonl",
}
CLEAN_FILES = {
    "thockin": PROCESSED_DATA_DIR / "thockin_clean.jsonl",
    "ezyang": PROCESSED_DATA_DIR / "ezyang_clean.jsonl",
}
COMBINED_CLEAN_FILE = PROCESSED_DATA_DIR / "combined_clean.jsonl"
HELDOUT_FILE = HELDOUT_DIR / "eval_heldout_queries.jsonl"
