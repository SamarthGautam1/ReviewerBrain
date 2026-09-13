"""Frozen-configuration loader for ReviewerBrain.

The YAML file under configs/rag/ documents the validated RAG configuration.
Scripts keep their own constants (identical values, unchanged behavior);
tests assert the two stay in sync.
"""
from pathlib import Path

from reviewerbrain import paths


def load_config(path=None):
    import yaml
    p = Path(path) if path else paths.CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
