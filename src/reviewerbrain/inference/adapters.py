"""Reviewer-specific LoRA adapter registry (filesystem, no model loading).

Adapters produced by the Kaggle workflow (kaggle/README.md) are unpacked
under `<repo>/adapters/<reviewer>/` — a PEFT adapter directory
(`adapter_config.json` + `adapter_model.safetensors`, never merged into
the base). This module only DISCOVERS and validates that layout; loading
weights happens exclusively in backends.HFBackend on a machine that has
the base model and a GPU.

`adapters/` is gitignored: binaries never enter git.
"""
import json
from pathlib import Path

from reviewerbrain import paths

ADAPTER_MARKER = "adapter_config.json"
_SLUG = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")


class AdapterNotAvailable(RuntimeError):
    pass


def _safe_slug(name):
    return bool(name) and set(name) <= _SLUG and not name.startswith("-")


def adapters_root(root=None):
    return Path(root) if root else paths.ADAPTERS_DIR


def find_adapters(root=None):
    """{reviewer: adapter_dir} for every adapter directory present.

    A directory counts as an adapter only if it contains the PEFT marker
    file. Reviewer names must be slugs (defense against a hand-crafted or
    downloaded layout escaping the root)."""
    base = adapters_root(root)
    found = {}
    if not base.is_dir():
        return found
    for child in sorted(base.iterdir()):
        if child.is_dir() and _safe_slug(child.name) \
                and (child / ADAPTER_MARKER).is_file():
            found[child.name] = child
    return found


def has_adapter(reviewer, root=None):
    if not _safe_slug(reviewer):
        return False
    return (adapters_root(root) / reviewer / ADAPTER_MARKER).is_file()


def adapter_dir(reviewer, root=None):
    """Adapter directory for one reviewer; AdapterNotAvailable otherwise."""
    if not _safe_slug(reviewer):
        raise AdapterNotAvailable(f"bad reviewer slug: {reviewer!r}")
    d = adapters_root(root) / reviewer
    if not (d / ADAPTER_MARKER).is_file():
        raise AdapterNotAvailable(
            f"no trained adapter for reviewer {reviewer!r} under "
            f"{adapters_root(root)} — train it via the Kaggle workflow "
            f"(kaggle/README.md) and unpack it there")
    return d


def adapter_meta(reviewer, root=None):
    """Parsed adapter_config.json (metadata only — no weights)."""
    path = adapter_dir(reviewer, root) / ADAPTER_MARKER
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
