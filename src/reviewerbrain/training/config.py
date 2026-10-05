"""Training-configuration loading and validation.

The YAML file under configs/training/ is the single source of truth for
the LoRA/QLoRA runs; scripts/training/train_lora.py and the Kaggle
notebook both load it through this module. Validation is strict and pure
(no model imports) so it can run anywhere, including tests.
"""
from pathlib import Path

import yaml

from reviewerbrain import paths

REVIEWERS = ("thockin", "ezyang")
QUANT_TYPES = ("nf4", "fp4")
SCHEDULERS = ("linear", "cosine", "cosine_with_restarts", "constant")
SAVE_STRATEGIES = ("no", "epoch", "steps")


class TrainingConfigError(ValueError):
    pass


def load_training_config(path=None):
    p = Path(path) if path else paths.TRAINING_CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_training_config(cfg)
    return cfg


def _require(condition, message):
    if not condition:
        raise TrainingConfigError(message)


def validate_training_config(cfg):
    """Raise TrainingConfigError unless every section is well-formed."""
    _require(isinstance(cfg, dict), "config must be a mapping")
    _require(isinstance(cfg.get("base_model"), str)
             and cfg["base_model"].strip(),
             "base_model must be a non-empty model id")
    reviewers = cfg.get("reviewers")
    _require(isinstance(reviewers, list) and reviewers
             and all(r in REVIEWERS for r in reviewers),
             f"reviewers must be a non-empty subset of {REVIEWERS}")

    quant = cfg.get("quantization") or {}
    _require(isinstance(quant.get("load_in_4bit"), bool),
             "quantization.load_in_4bit must be a bool")
    _require(quant.get("bnb_4bit_quant_type") in QUANT_TYPES,
             f"bnb_4bit_quant_type must be one of {QUANT_TYPES}")
    _require(isinstance(quant.get("bnb_4bit_use_double_quant"), bool),
             "bnb_4bit_use_double_quant must be a bool")
    _require(quant.get("bnb_4bit_compute_dtype") in ("float16", "bfloat16"),
             "bnb_4bit_compute_dtype must be float16 or bfloat16")

    lora = cfg.get("lora") or {}
    _require(lora.get("task_type") == "CAUSAL_LM",
             "lora.task_type must be CAUSAL_LM")
    _require(isinstance(lora.get("r"), int) and lora["r"] >= 1,
             "lora.r must be an int >= 1")
    _require(isinstance(lora.get("lora_alpha"), int) and lora["lora_alpha"] >= 1,
             "lora.lora_alpha must be an int >= 1")
    drop = lora.get("lora_dropout")
    _require(isinstance(drop, (int, float)) and 0 <= drop < 1,
             "lora.lora_dropout must be in [0, 1)")
    targets = lora.get("target_modules")
    _require(isinstance(targets, list) and targets
             and all(isinstance(t, str) and t for t in targets),
             "lora.target_modules must be a non-empty list of module names")
    _require(lora.get("bias") in ("none", "all", "lora_only"),
             "lora.bias must be none|all|lora_only")

    t = cfg.get("training") or {}
    _require(isinstance(t.get("output_root"), str)
             and t["output_root"].strip(),
             "training.output_root must be a non-empty path")
    _require(isinstance(t.get("learning_rate"), (int, float))
             and 0 < t["learning_rate"] <= 1,
             "training.learning_rate must be in (0, 1]")
    _require(isinstance(t.get("num_train_epochs"), (int, float))
             and t["num_train_epochs"] >= 1,
             "training.num_train_epochs must be >= 1")
    for key in ("per_device_train_batch_size", "gradient_accumulation_steps"):
        _require(isinstance(t.get(key), int) and t[key] >= 1,
                 f"training.{key} must be an int >= 1")
    _require(isinstance(t.get("gradient_checkpointing"), bool),
             "training.gradient_checkpointing must be a bool")
    _require(t.get("lr_scheduler_type") in SCHEDULERS,
             f"training.lr_scheduler_type must be one of {SCHEDULERS}")
    wr = t.get("warmup_ratio")
    _require(isinstance(wr, (int, float)) and 0 <= wr < 1,
             "training.warmup_ratio must be in [0, 1)")
    wd = t.get("weight_decay")
    _require(isinstance(wd, (int, float)) and wd >= 0,
             "training.weight_decay must be >= 0")
    mgn = t.get("max_grad_norm")
    _require(isinstance(mgn, (int, float)) and mgn > 0,
             "training.max_grad_norm must be > 0")
    _require(isinstance(t.get("fp16"), bool)
             and isinstance(t.get("bf16"), bool),
             "training.fp16/bf16 must be bools")
    _require(not (t.get("fp16") and t.get("bf16")),
             "training.fp16 and bf16 are mutually exclusive")
    _require(isinstance(t.get("logging_steps"), int)
             and t["logging_steps"] >= 1,
             "training.logging_steps must be an int >= 1")
    _require(t.get("save_strategy") in SAVE_STRATEGIES,
             f"training.save_strategy must be one of {SAVE_STRATEGIES}")
    _require(isinstance(t.get("seed"), int), "training.seed must be an int")
    msl = t.get("max_seq_length")
    _require(isinstance(msl, int) and 128 <= msl <= 32768,
             "training.max_seq_length must be an int in [128, 32768]")

    s = cfg.get("sft") or {}
    tpl = s.get("prompt_template")
    _require(isinstance(tpl, str) and tpl.strip(),
             "sft.prompt_template must be a non-empty template name")
    vp, vpos = s.get("val_period"), s.get("val_position")
    _require(isinstance(vp, int) and vp >= 1,
             "sft.val_period must be an int >= 1")
    _require(isinstance(vpos, int) and 0 <= vpos < vp,
             "sft.val_position must be in [0, val_period)")
    for key in ("diff_max_chars", "desc_max_chars"):
        _require(isinstance(s.get(key), int) and s[key] >= 256,
                 f"sft.{key} must be an int >= 256")
    _require(isinstance(s.get("sft_dir"), str) and s["sft_dir"].strip(),
             "sft.sft_dir must be a non-empty path")
    return cfg
