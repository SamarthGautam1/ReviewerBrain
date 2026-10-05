"""Training-side tests that never touch a model or GPU.

Covered:
  - assistant-only loss masking (tokenize_example) against a stub chat
    tokenizer — masking arithmetic, prefix-consistency guard, left
    truncation behavior
  - training-config validation (repo config valid; mutations rejected)
  - the --prepare-only data path of scripts/training/train_lora.py
    (pure Python; no torch import)
"""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"
                       / "training"))
import train_lora  # noqa: E402

from reviewerbrain import paths  # noqa: E402
from reviewerbrain.training import sft_data  # noqa: E402
from reviewerbrain.training.config import (TrainingConfigError,  # noqa: E402
                                           load_training_config,
                                           validate_training_config)


# ---------------------------------------------------------------- stub tokenizer
def _ids(text):
    return [3 + (ord(c) * 7 % 250) for c in text]


class StubChatTokenizer:
    """Deterministic prefix-consistent stand-in for Qwen2.5's chat template.

    full(messages)   = T("<sys>..." + "<usr>..." + "<asst>...<|im_end|>")
    prompt(messages) = full(messages[:-1]) + "<asst>" generation marker
    """

    IM_END = "<|im_end|>"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=True,
                            add_special_tokens=False,
                            add_generation_prompt=False):
        self.calls.append((len(messages), add_generation_prompt))
        text = ""
        for m in messages:
            text += f"<|im_start|>{m['role']}\n{m['content']}{self.IM_END}\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return _ids(text) if tokenize else text

    pad_token_id = 0


def make_messages():
    return [
        {"role": "system", "content": "You are reviewer thockin."},
        {"role": "user", "content": "File: pkg/foo.go\n```diff\n+new\n```"},
        {"role": "assistant", "content": "Check the error path."},
    ]


class TestTokenizeExample:
    def test_masking_covers_assistant_only(self):
        tok = StubChatTokenizer()
        row, flag = train_lora.tokenize_example(make_messages(), tok, 2048)
        assert not flag["truncated"]
        assert len(row["attention_mask"]) == len(row["input_ids"])
        kept = [v for v, l in zip(row["input_ids"], row["labels"])
                if l != -100]
        answer_ids = _ids("Check the error path."
                          + StubChatTokenizer.IM_END + "\n")
        assert kept == answer_ids
        n_masked = sum(1 for l in row["labels"] if l == -100)
        assert n_masked == len(row["input_ids"]) - len(answer_ids)

    def test_loss_tokens_counted(self):
        tok = StubChatTokenizer()
        _, flag = train_lora.tokenize_example(make_messages(), tok, 2048)
        assert flag["loss_tokens"] == len(
            _ids("Check the error path." + StubChatTokenizer.IM_END + "\n"))

    def test_left_truncation_keeps_answer(self):
        tok = StubChatTokenizer()
        msgs = make_messages()
        row, flag = train_lora.tokenize_example(msgs, tok, 2048)
        full_len = len(row["input_ids"])
        small = full_len - 5                     # drop 5 leading tokens
        row2, flag2 = train_lora.tokenize_example(msgs, tok, small)
        assert flag2["truncated"]
        assert len(row2["input_ids"]) == small
        # the tail (answer included) survives; input_ids match the tail
        assert row2["input_ids"] == row["input_ids"][-small:]
        kept2 = [v for v, l in zip(row2["input_ids"], row2["labels"])
                 if l != -100]
        assert kept2                              # answer still trained on

    def test_prefix_inconsistency_rejected(self):
        class Inconsistent(StubChatTokenizer):
            def apply_chat_template(self, messages, tokenize=True,
                                    add_special_tokens=False,
                                    add_generation_prompt=False):
                if add_generation_prompt:
                    return _ids("<INCONSISTENT>")
                return super().apply_chat_template(
                    messages, tokenize=tokenize,
                    add_special_tokens=add_special_tokens,
                    add_generation_prompt=add_generation_prompt)

        with pytest.raises(ValueError, match="prefix"):
            train_lora.tokenize_example(make_messages(), Inconsistent(), 2048)


class TestCollatorAndDataset:
    def test_collator_pads_and_masks(self):
        torch = pytest.importorskip("torch")
        rows = [
            {"input_ids": [1, 2, 3], "labels": [-100, 5, 6],
             "attention_mask": [1, 1, 1]},
            {"input_ids": [4, 5], "labels": [-100, 6],
             "attention_mask": [1, 1]},
        ]
        batch = train_lora.Collator(pad_id=0)(rows)
        assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]
        assert batch["labels"].tolist() == [[-100, 5, 6], [-100, 6, -100]]
        assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
        assert batch["input_ids"].dtype == torch.long

    def test_dataset_len_and_item(self):
        pytest.importorskip("torch")
        rows = [{"input_ids": [1, 2], "labels": [-100, 2],
                 "attention_mask": [1, 1]}]
        ds = train_lora._TorchDataset(rows, pad_id=0)
        assert len(ds) == 1
        item = ds[0]
        assert item["input_ids"].tolist() == [1, 2]


# ---------------------------------------------------------------- config tests
def mutate(base, path, value):
    cfg = copy.deepcopy(base)
    node = cfg
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return cfg


@pytest.fixture(scope="module")
def repo_cfg():
    return load_training_config()


class TestTrainingConfigValidation:
    def test_repo_config_is_valid(self, repo_cfg):
        validate_training_config(repo_cfg)   # must not raise
        assert repo_cfg["base_model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
        assert set(repo_cfg["reviewers"]) <= {"thockin", "ezyang"}
        assert repo_cfg["quantization"]["load_in_4bit"] is True
        assert not (repo_cfg["training"]["fp16"]
                    and repo_cfg["training"]["bf16"])

    @pytest.mark.parametrize("path,value", [
        (("base_model",), ""),
        (("reviewers",), []),
        (("reviewers",), ["liggitt"]),
        (("quantization", "load_in_4bit"), "yes"),
        (("quantization", "bnb_4bit_quant_type"), "int8"),
        (("lora", "r"), 0),
        (("lora", "r"), "16"),
        (("lora", "lora_dropout"), 1.0),
        (("lora", "target_modules"), []),
        (("lora", "bias"), "everything"),
        (("lora", "task_type"), "SEQ_CLS"),
        (("training", "learning_rate"), 0),
        (("training", "num_train_epochs"), 0),
        (("training", "per_device_train_batch_size"), 0),
        (("training", "gradient_accumulation_steps"), 0),
        (("training", "lr_scheduler_type"), "polynomial"),
        (("training", "warmup_ratio"), 1.0),
        (("training", "max_grad_norm"), 0),
        (("training", "fp16"), True),   # + bf16 True below -> exclusive
        (("training", "save_strategy"), "daily"),
        (("training", "optim"), "sgd"),
        (("training", "max_seq_length"), 64),
        (("sft", "val_position"), 10),
        (("sft", "diff_max_chars"), 10),
        (("sft", "prompt_template"), ""),
    ])
    def test_invalid_mutations_rejected(self, repo_cfg, path, value):
        cfg = mutate(repo_cfg, path, value)
        if path == ("training", "fp16"):
            cfg = mutate(cfg, ("training", "bf16"), True)
        with pytest.raises(TrainingConfigError):
            validate_training_config(cfg)

    def test_fp16_bf16_mutually_exclusive(self, repo_cfg):
        cfg = mutate(repo_cfg, ("training", "bf16"), True)
        with pytest.raises(TrainingConfigError, match="exclusive"):
            validate_training_config(cfg)

    def test_val_position_bounds(self, repo_cfg):
        cfg = mutate(repo_cfg, ("sft", "val_position"),
                     repo_cfg["sft"]["val_period"])
        with pytest.raises(TrainingConfigError):
            validate_training_config(cfg)


# ------------------------------------------------------- prepare-only path
def fake_clean_files(tmp_path, monkeypatch, n_prs=9):
    for name in ("thockin", "ezyang"):
        p = tmp_path / f"{name}_clean.jsonl"
        rows = []
        for pr_number in range(1, n_prs + 1):
            rows.append({
                "reviewer": name, "repo": "x/y", "pr_number": pr_number,
                "pr_title": f"PR {pr_number}", "pr_description": "",
                "file": "a/b.py",
                "diff_hunk": "@@ -1 +1,2 @@\n c\n+n",
                "review_comment": f"Note {name} {pr_number}.",
                "is_code_related": True, "led_to_code_change": False,
                "follow_up_patch": "",
            })
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                     encoding="utf-8")
        monkeypatch.setitem(paths.CLEAN_FILES, name, p)


class TestPrepareOnly:
    def test_prepare_only_prints_stats(self, tmp_path, monkeypatch, capsys):
        fake_clean_files(tmp_path, monkeypatch)
        rc = train_lora.main(["--prepare-only", "--reviewer", "ezyang"])
        out = capsys.readouterr().out
        assert rc == 0
        stats = json.loads(out)
        assert stats["reviewer"] == "ezyang"
        assert stats["train_examples"] + stats["val_examples"] \
            + stats["held_out_records"] == 9

    def test_module_import_is_model_free(self):
        """Importing train_lora in a clean interpreter must not import
        torch/transformers/peft (hermetic subprocess check)."""
        root = Path(__file__).resolve().parents[1]
        heavy = repr({"torch", "transformers", "peft"})
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(root / 'src')!r})\n"
            f"sys.path.insert(0, {str(root / 'scripts' / 'training')!r})\n"
            "import train_lora\n"
            f"assert not {heavy} & set(sys.modules), 'heavy module imported'\n"
        )
        import subprocess
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=60)
        assert r.returncode == 0, r.stderr
