"""Validated writers for inference-run artifacts.

Every artifact path is constructed from the fixed run root
(evaluations/inference/<tag>/<mode>/...), the components are restricted to
safe slugs, and the joined path is normalized and verified to stay inside
the run root before any file is opened. Nothing outside
evaluations/inference is ever written.
"""
import json
import os
from pathlib import Path

from reviewerbrain import paths

OUT_ROOT = paths.EVALUATIONS_DIR / "inference"
_SLUG = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
# Literal allowlist of artifact names — a caller-supplied path component can
# never reach the filesystem, so no traversal is expressible at all.
_ALLOWED_ARTIFACTS = frozenset({
    "generations.jsonl", "manifest.json", "metrics.json",
    "metrics_summary.md", "qualitative_examples.md",
})


def _safe_component(name):
    """Slug component: [A-Za-z0-9_-], must start alphanumeric — excludes
    separators, dot/parent segments, and option-like leading dashes."""
    return (bool(name) and not name.startswith("-")
            and set(name) <= _SLUG)


def resolve_run_dir(tag):
    """evaluations/inference/<tag>, containment-verified."""
    if not _safe_component(tag):
        raise SystemExit(f"run tag must be a slug [A-Za-z0-9-_]: {tag!r}")
    run_dir = OUT_ROOT / tag
    if not _contained(OUT_ROOT, run_dir):
        raise SystemExit(f"run dir escapes {OUT_ROOT}: {run_dir}")
    return run_dir


def _contained(base, target):
    b = os.path.normpath(str(base))
    t = os.path.normpath(str(target))
    try:
        return os.path.commonpath([b, t]) == b and t != b
    except ValueError:
        # different drives (Windows) can never be contained
        return False


def open_run_file(run_dir, mode, filename):
    """Open a run artifact for writing: <run_dir>/<mode>/<filename>.
    The filename must be a known artifact name (literal allowlist), the
    mode component is slug-validated, and the joined path is verified to
    stay inside the run dir and OUT_ROOT."""
    if filename not in _ALLOWED_ARTIFACTS:
        raise SystemExit(f"filename must be one of "
                         f"{sorted(_ALLOWED_ARTIFACTS)}: got {filename!r}")
    if mode != "." and not _safe_component(mode):
        raise SystemExit(f"bad mode component: {mode!r}")
    p = Path(run_dir).resolve() / mode / filename
    if not _contained(run_dir, p) or not _contained(OUT_ROOT, p):
        raise SystemExit(f"artifact path escapes allowed roots: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p.open("w", encoding="utf-8")


def write_manifest(run_dir, manifest):
    with open_run_file(run_dir, ".", "manifest.json") as f:
        json.dump(manifest, f, indent=2)
    return run_dir / "manifest.json"


def read_generations(run_dir, mode):
    """Read <run_dir>/<mode>/generations.jsonl (containment-verified)."""
    run_dir = Path(run_dir).resolve()
    if not _safe_component(mode):
        raise SystemExit(f"bad mode component: {mode!r}")
    p = run_dir / mode / "generations.jsonl"
    if not _contained(OUT_ROOT, p):
        raise SystemExit(f"generations path escapes {OUT_ROOT}: {p}")
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out
