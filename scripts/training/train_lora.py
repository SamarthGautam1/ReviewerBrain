"""LoRA/QLoRA training entry point for reviewer-specific adapters (Kaggle GPU).

DESIGNED TO RUN ON KAGGLE GPU — do NOT run the training path on a machine
without a GPU; this repository's development machine has neither the
hardware nor the weights. Everything heavy (torch, transformers, peft)
is imported lazily inside the training path, so importing this module,
validating configuration, building/tokenizing the dataset (--prepare-only)
and running the tests never touches a model.

Per trained reviewer the script writes one PEFT adapter directory:

  <out-root>/<reviewer>/
      adapter_config.json, adapter_model.safetensors   (LoRA, not merged)
      tokenizer files
      training_report.json                             (reproducibility)

The report records: base model, full training config + its SHA-256, seed,
package versions, git commit, dataset provenance/stats and the loss
history — everything needed to audit or rerun the experiment.

Data provenance is the frozen SFT pipeline (reviewerbrain.training.sft_data):
frozen held-out PRs excluded, follow_up_patch excluded, deterministic
PR-level validation carve-out, sft_v1 prompt template (train/serve match).

Examples (Kaggle notebook runs these; see kaggle/README.md):
  python scripts/training/train_lora.py --prepare-only
  python scripts/training/train_lora.py --reviewer thockin
  python scripts/training/train_lora.py --reviewer thockin --smoke
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from reviewerbrain import paths  # noqa: E402
from reviewerbrain.training import sft_data  # noqa: E402
from reviewerbrain.training.config import load_training_config  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default=None,
                   help="training config YAML (default: repo config)")
    p.add_argument("--reviewer", required=True,
                   choices=["thockin", "ezyang"],
                   help="which reviewer adapter to train (one per run; "
                        "the Kaggle notebook loops over the config list)")
    p.add_argument("--sft-dir", default=None,
                   help="output dir for built SFT data (default: sft.sft_dir "
                        "from the config, relative to the repo root)")
    p.add_argument("--out-root", default=None,
                   help="adapter output root (default: training.output_root "
                        "from the config)")
    p.add_argument("--prepare-only", action="store_true",
                   help="build + tokenize nothing heavier than text: build "
                        "the SFT split, print stats, exit (no torch)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny run: 16 examples, 4 optimizer steps — verifies "
                        "the full path on a GPU before a real run")
    p.add_argument("--max-examples", type=int, default=None,
                   help="cap training examples (debug)")
    return p.parse_args(argv)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenize_example(messages, tok, max_seq_length):
    """Chat-tokenize one SFT example with assistant-only loss masking.

    Works with any tokenizer exposing apply_chat_template (Qwen2.5 does).
    Labels are -100 everywhere except the assistant response tokens (which
    keep the template's closing <|im_end|> so the model learns to stop).
    Sequences longer than max_seq_length are LEFT-truncated (the assistant
    answer and the prompt tail are kept); the truncation is visible in the
    returned flags, never silent.
    """
    full_ids = tok.apply_chat_template(messages, tokenize=True,
                                       add_special_tokens=False)
    prompt_ids = tok.apply_chat_template(messages[:-1], tokenize=True,
                                         add_special_tokens=False,
                                         add_generation_prompt=True)
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template is not prefix-consistent; "
                         "assistant-only masking would be unsafe")
    labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids):])
    truncated = False
    if len(full_ids) > max_seq_length:
        cut = len(full_ids) - max_seq_length
        full_ids = full_ids[cut:]
        labels = labels[cut:]
        truncated = True
    return {"input_ids": full_ids, "labels": labels,
            "attention_mask": [1] * len(full_ids)}, {
        "n_tokens": len(full_ids), "loss_tokens": sum(1 for l in labels
                                                      if l != -100),
        "truncated": truncated}


class _TorchDataset:
    """Minimal torch Dataset over pre-tokenized rows (no `datasets` dep)."""

    def __init__(self, rows, pad_id):
        import torch

        self.rows = rows
        self.pad_id = pad_id
        self._torch = torch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {"input_ids": self._torch.tensor(r["input_ids"],
                                                dtype=self._torch.long),
                "labels": self._torch.tensor(r["labels"],
                                             dtype=self._torch.long),
                "attention_mask": self._torch.tensor(r["attention_mask"],
                                                     dtype=self._torch.long)}


class Collator:
    """Right-pad to the longest row; pad labels with -100."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        batch = {}
        for key, pad_value in (("input_ids", self.pad_id),
                               ("labels", -100),
                               ("attention_mask", 0)):
            batch[key] = torch.tensor(
                [f[key] + [pad_value] * (max_len - len(f[key]))
                 for f in features], dtype=torch.long)
        return batch


def build_tokenized_split(records, reviewer, tok, sft_cfg, training_cfg):
    """Build + tokenize the leakage-guarded SFT split for one reviewer."""
    template = sft_data.load_template_checked(sft_cfg["prompt_template"])
    train_ex, val_ex, stats = sft_data.build_reviewer_dataset(
        records, reviewer, template=template,
        val_period=sft_cfg["val_period"],
        val_position=sft_cfg["val_position"],
        diff_max_chars=sft_cfg["diff_max_chars"])
    rows, flags = [], {"truncated": 0}
    for ex in train_ex + val_ex:
        row, flag = tokenize_example(ex["messages"], tok,
                                     training_cfg["max_seq_length"])
        rows.append(row)
        flags["truncated"] += int(flag["truncated"])
    n_train = len(train_ex)
    return rows[:n_train], rows[n_train:], stats, flags


def write_report(out_dir, reviewer, cfg, config_sha, args, stats, flags,
                 trainer_state, versions):
    report = {
        "kind": "reviewerbrain_lora_training_report",
        "reviewer": reviewer,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_model": cfg["base_model"],
        "adapter": str(out_dir),
        "merged": False,
        "config_sha256": config_sha,
        "config": cfg,
        "cli_args": {k: str(v) for k, v in vars(args).items()},
        "seed": cfg["training"]["seed"],
        "dataset": stats,
        "tokenization_flags": flags,
        "loss_history": [h for h in trainer_state
                         if "loss" in h or "eval_loss" in h],
        "package_versions": versions,
        "git_commit": _git_commit(),
    }
    with open(out_dir / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return None


def train(args, cfg):     # lazy heavy imports — Kaggle GPU path only
    import torch
    import transformers
    import peft
    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments)

    reviewer = args.reviewer
    quant, lora, t = cfg["quantization"], cfg["lora"], cfg["training"]
    sft_cfg = cfg["sft"]

    records = sft_data.load_clean_records(paths.CLEAN_FILES[reviewer])
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train_rows, val_rows, stats, flags = build_tokenized_split(
        records, reviewer, tok, sft_cfg, t)
    if args.max_examples:
        train_rows = train_rows[:args.max_examples]
    if args.smoke:
        train_rows = train_rows[:16]
    print(f"[{reviewer}] train rows={len(train_rows)} "
          f"val rows={len(val_rows)} truncated={flags['truncated']}",
          flush=True)

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=quant["load_in_4bit"],
        bnb_4bit_quant_type=quant["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, quant["bnb_4bit_compute_dtype"]))
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=quant_cfg,
        torch_dtype=getattr(torch, quant["bnb_4bit_compute_dtype"]),
        device_map="auto")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        task_type=lora["task_type"], r=lora["r"],
        lora_alpha=lora["lora_alpha"], lora_dropout=lora["lora_dropout"],
        target_modules=lora["target_modules"], bias=lora["bias"]))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=str(paths.ROOT / "checkpoints" / reviewer),
        learning_rate=t["learning_rate"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        gradient_checkpointing=t["gradient_checkpointing"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],
        fp16=t["fp16"], bf16=t["bf16"],
        logging_steps=t["logging_steps"],
        save_strategy=t["save_strategy"],
        eval_strategy="epoch" if val_rows else "no",
        per_device_eval_batch_size=t["per_device_train_batch_size"],
        seed=t["seed"], data_seed=t["seed"],
        report_to=[], optim="paged_adamw_8bit")
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=_TorchDataset(train_rows, tok.pad_token_id),
        eval_dataset=(_TorchDataset(val_rows, tok.pad_token_id)
                      if val_rows else None),
        data_collator=Collator(tok.pad_token_id))
    trainer.train()

    out_root = Path(args.out_root) if args.out_root else Path(t["output_root"])
    out_dir = out_root / reviewer
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)          # adapter only — never merged
    tok.save_pretrained(out_dir)
    versions = {"python": sys.version.split()[0], "torch": torch.__version__,
                "transformers": transformers.__version__,
                "peft": peft.__version__}
    write_report(out_dir, reviewer, cfg, _config_sha(args), args, stats,
                 flags, trainer.state.log_history, versions)
    print(f"[{reviewer}] adapter + training_report.json -> {out_dir}",
          flush=True)


def _config_sha(args):
    from reviewerbrain.training.config import _config_path
    return sha256_file(_config_path(args.config))


def main(argv=None):
    args = parse_args(argv)
    cfg = load_training_config(args.config)
    if args.prepare_only:
        reviewer = args.reviewer
        sft_cfg = cfg["sft"]
        records = sft_data.load_clean_records(paths.CLEAN_FILES[reviewer])
        template = sft_data.load_template_checked(sft_cfg["prompt_template"])
        _, _, stats = sft_data.build_reviewer_dataset(
            records, reviewer, template=template,
            val_period=sft_cfg["val_period"],
            val_position=sft_cfg["val_position"],
            diff_max_chars=sft_cfg["diff_max_chars"])
        print(json.dumps(stats, indent=2))
        return 0
    train(args, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
