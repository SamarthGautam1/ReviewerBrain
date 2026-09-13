"""Frozen-configuration consistency: config file must match code constants."""

from reviewerbrain.config import load_config
from reviewerbrain.embeddings.model import EMBEDDING_DIM, MODEL_NAME
from reviewerbrain.evaluation import split
from reviewerbrain.retrieval import representation


def test_config_matches_frozen_values():
    cfg = load_config()
    assert cfg["embedding"]["model"] == MODEL_NAME
    assert cfg["embedding"]["dimensions"] == EMBEDDING_DIM
    assert cfg["embedding"]["max_seq_tokens"] == 256

    assert cfg["representation"]["min_diff_tokens"] == representation.MIN_DIFF_TOKENS
    assert cfg["representation"]["safety_margin"] == representation.SAFETY
    assert cfg["representation"]["min_diff_tokens"] == 24

    assert cfg["retrieval"]["top_k"] == 3
    assert cfg["retrieval"]["gate"]["metric"] == "cosine"
    assert cfg["retrieval"]["gate"]["min"] == 0.5

    assert cfg["evaluation"]["split"]["period"] == split.PERIOD
    assert cfg["evaluation"]["split"]["held_position"] == split.HELD_POSITION
    assert cfg["evaluation"]["thresholds"]["cosine"] == 0.5
    assert cfg["evaluation"]["thresholds"]["fidelity"] == 0.25


def test_config_loads_from_repo_root():
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert "embedding" in cfg and "retrieval" in cfg
