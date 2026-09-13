"""Approved amendments to the cleaned datasets (run AFTER clean_dataset.py).

Removes exactly two things, nothing else:
  1. examples with an empty/whitespace diff_hunk
  2. the ezyang example (file=test/dynamo/test_export.py.bak,
     review_comment="delete me")

Writes NO files itself; emits one JSONL section per reviewer to stdout
(plus a ===SUMMARY=== section). Rebuild data/processed/*_clean.jsonl from
the THOCKIN/EZYANG sections and regenerate combined_clean.jsonl by
concatenating the two files.
"""
import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))

from reviewerbrain import paths

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BAK_EXAMPLE = ("test/dynamo/test_export.py.bak", "delete me")

for name, path in paths.CLEAN_FILES.items():
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    kept = [r for r in rows
            if (r.get("diff_hunk") or "").strip()
            and (r.get("file"), r.get("review_comment")) != BAK_EXAMPLE]
    print(f"==={name.upper()}===")
    for r in kept:
        print(json.dumps(r, ensure_ascii=False))
    print("===SUMMARY===")
    print(f"{name}: kept {len(kept)} of {len(rows)} "
          f"(removed {len(rows) - len(kept)})")
