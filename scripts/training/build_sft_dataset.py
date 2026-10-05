"""Build reviewer-specific SFT datasets from the cleaned corpora.

Reads data/processed/<reviewer>_clean.jsonl (the approved cleaned
datasets — never modified), applies the frozen PR-level held-out rule and
a deterministic PR-level validation carve-out, renders chat-format
examples from the versioned sft_v1 template, runs every leakage guard,
and writes:

  data/processed/sft/<reviewer>_sft_train.jsonl
  data/processed/sft/<reviewer>_sft_val.jsonl
  data/processed/sft/<reviewer>_sft_stats.json

Outputs are generated artifacts (data/processed/ is gitignored). The run
is deterministic: file-order records, sorted PR splits, no RNG.

Leakage rules (enforced, see reviewerbrain.training.sft_data):
  held-out PRs never trained on; follow_up_patch never rendered;
  led_to_code_change / is_code_related never rendered; RAG examples are
  not training data.

Examples
  python scripts/training/build_sft_dataset.py
  python scripts/training/build_sft_dataset.py --reviewer thockin
  python scripts/training/build_sft_dataset.py --val-period 10 --val-position 9
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from reviewerbrain import paths  # noqa: E402
from reviewerbrain.training import sft_data  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--reviewer", choices=["thockin", "ezyang", "both"],
                   default="both")
    p.add_argument("--config", default=None,
                   help="training config YAML (default: "
                        "configs/training/lora_qwen25coder7b.yaml)")
    p.add_argument("--val-period", type=int, default=None,
                   help="override sft.val_period from the config")
    p.add_argument("--val-position", type=int, default=None,
                   help="override sft.val_position from the config")
    p.add_argument("--out-dir", default=None,
                   help="output dir (default: <sft.sft_dir> from config, "
                        "i.e. data/processed/sft)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sft = sft_data.load_training_sft_config(args.config)
    val_period = args.val_period if args.val_period is not None \
        else sft["val_period"]
    val_position = args.val_position if args.val_position is not None \
        else sft["val_position"]
    out_dir = Path(args.out_dir) if args.out_dir \
        else ROOT / sft["sft_dir"]
    template = sft_data.load_template_checked(sft["prompt_template"])
    reviewers = (["thockin", "ezyang"] if args.reviewer == "both"
                 else [args.reviewer])

    all_stats = {}
    for name in reviewers:
        records = sft_data.load_clean_records(paths.CLEAN_FILES[name])
        train_ex, val_ex, stats = sft_data.build_reviewer_dataset(
            records, name, template=template,
            val_period=val_period, val_position=val_position,
            diff_max_chars=sft["diff_max_chars"])
        sft_data.verify_examples(train_ex + val_ex, records)
        stats_path = sft_data.write_reviewer_files(
            out_dir, name, train_ex, val_ex, stats, records)
        all_stats[name] = stats
        print(f"[{name}] train={stats['train_examples']} "
              f"val={stats['val_examples']} "
              f"(held-out PRs excluded: {stats['held_out_prs']} "
              f"covering {stats['held_out_records']} records) "
              f"-> {stats_path}", flush=True)

    summary_path = out_dir / "stats.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"sft_config": sft, "reviewers": all_stats}, f, indent=2)
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
