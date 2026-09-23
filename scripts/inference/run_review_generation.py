"""Run local-LLM review generation over held-out queries (pilot harness).

Modes
  baseline : generic code-review prompt (configs/prompts/baseline_v1.yaml)
             — no reviewer-specific information anywhere in the prompt.
  rag      : RAG-conditioned prompt (configs/prompts/rag_v1.yaml) — the
             current change plus the top-3 historical examples retrieved
             from the validated per-reviewer ChromaDB index (frozen
             pipeline, gate cosine >= 0.5), inserted verbatim.

Queries
  Held-out queries come from evaluations/heldout/eval_heldout_queries.jsonl.
  Sampling is deterministic (no RNG): evenly-spaced entries of the
  per-reviewer held-out list, so baseline and RAG runs can be pointed at
  exactly the same queries. Every query is joined back to its cleaned
  record and the join is verified before use.

Inference
  A local Ollama server (loopback only — see
  reviewerbrain.inference.llm_client). Greedy decoding (temperature 0) for
  reproducibility.

Outputs (generated artifacts, gitignored; all writes go through the
validated writers in reviewerbrain.inference.outputs)
  evaluations/inference/<tag>/<mode>/generations.jsonl
  evaluations/inference/<tag>/manifest.json

Examples
  python scripts/inference/run_review_generation.py --tag pilot_local \
      --mode baseline
  python scripts/inference/run_review_generation.py --tag pilot_local \
      --mode rag
  python scripts/inference/run_review_generation.py --reviewer thockin \
      --queries 3 --dry-run --tag smoke

The validated RAG pipeline is imported, never modified.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from reviewerbrain.inference import llm_client, outputs  # noqa: E402
from reviewerbrain.inference import prompts as pr  # noqa: E402
from reviewerbrain.inference import heldout_queries as q  # noqa: E402
from reviewerbrain.inference import rag_retrieval as retr  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROMPT_TEMPLATES = {"baseline": "baseline_v1", "rag": "rag_v1"}
DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_ENDPOINT = llm_client.DEFAULT_ENDPOINT
DEFAULT_NUM_CTX = 8192
DEFAULT_NUM_PREDICT = 512
SEED = 42


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--reviewer", choices=["thockin", "ezyang", "both"],
                   default="both")
    p.add_argument("--mode", choices=["baseline", "rag", "both"],
                   default="both")
    p.add_argument("--queries", type=int, default=10,
                   help="deterministic evenly-spaced sample size per reviewer")
    p.add_argument("--query-ids", default=None,
                   help="comma-separated explicit query_ids (overrides "
                        "--queries), e.g. thockin:71,ezyang:5")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    p.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout-s", type=int, default=llm_client.DEFAULT_TIMEOUT_S)
    p.add_argument("--tag", required=True,
                   help="run name (slug); outputs go to "
                        "evaluations/inference/<tag>/")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing run directory")
    p.add_argument("--dry-run", action="store_true",
                   help="build prompts, skip the LLM call (prompt audit)")
    p.add_argument("--print", action="store_true",
                   help="print each prompt and generation to stdout")
    return p.parse_args(argv)


def pick_query_ids(heldout, args):
    """Resolve which held-out records to run, deterministically."""
    reviewers = (["thockin", "ezyang"] if args.reviewer == "both"
                 else [args.reviewer])
    if args.query_ids:
        wanted = {}
        for qid in [s.strip() for s in args.query_ids.split(",") if s.strip()]:
            name = qid.split(":")[0]
            wanted.setdefault(name, []).append(qid)
        by_id = {r["query_id"]: r for r in heldout}
        out = []
        for name in reviewers:
            for qid in wanted.get(name, []):
                if qid not in by_id:
                    raise SystemExit(f"query_id {qid} not in held-out artifact")
                out.append(by_id[qid])
        return reviewers, out
    out = []
    for name in reviewers:
        out.extend(q.select_queries(heldout, name, args.queries))
    return reviewers, out


def template_sha256(name):
    h = hashlib.sha256()
    path = pr.PROMPT_DIR / f"{name}.yaml"
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    args = parse_args(argv)
    reviewers, held = pick_query_ids(q.load_heldout(), args)
    modes = (["baseline", "rag"] if args.mode == "both" else [args.mode])

    run_dir = outputs.resolve_run_dir(args.tag)
    if run_dir.exists() and not args.force:
        raise SystemExit(f"run dir {run_dir} exists (use --force to overwrite)")
    run_dir.mkdir(parents=True, exist_ok=True)

    templates = {m: pr.load_template(PROMPT_TEMPLATES[m]) for m in modes}
    options = {"temperature": args.temperature, "num_ctx": args.num_ctx,
               "num_predict": args.num_predict, "seed": SEED}

    outputs.write_manifest(run_dir, {
        "tag": args.tag,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "endpoint": args.endpoint,
        "options": options,
        "sample": {"rule": "evenly-spaced over per-reviewer held-out list "
                           "(artifact order); no RNG",
                   "queries_per_reviewer": args.queries,
                   "explicit_query_ids": args.query_ids,
                   "reviewers": reviewers,
                   "query_ids": [r["query_id"] for r in held]},
        "modes": modes,
        "prompt_templates": {
            m: {"name": PROMPT_TEMPLATES[m],
                "sha256": template_sha256(PROMPT_TEMPLATES[m])}
            for m in modes},
        "python": sys.version.split()[0],
        "dry_run": args.dry_run,
    })

    for mode in modes:
        n_done = 0
        with outputs.open_run_file(run_dir, mode, "generations.jsonl") as out:
            for held_rec in held:
                reviewer = held_rec["reviewer"]
                rec = q.load_clean_for_heldout(held_rec)
                retrieved = None
                if mode == "rag":
                    retrieved = retr.retrieve(rec)
                messages = pr.build_messages(templates[mode], rec, reviewer,
                                             retrieved)
                record = {
                    "query_id": held_rec["query_id"],
                    "reviewer": reviewer,
                    "mode": mode,
                    "prompt_template": PROMPT_TEMPLATES[mode],
                    "model": args.model,
                    "file": held_rec["file"],
                    "pr_number": held_rec["query_pr_number"],
                    "ground_truth_comment": held_rec["query_review_comment"],
                    "messages": messages,
                    "retrieved": retrieved,
                    "options": options,
                }
                if args.dry_run:
                    record["generated_comment"] = None
                    record["meta"] = {"dry_run": True}
                else:
                    try:
                        content, meta = llm_client.chat(
                            args.endpoint, args.model, messages, options,
                            timeout_s=args.timeout_s)
                    except llm_client.LLMError as e:
                        record["generated_comment"] = None
                        record["error"] = str(e)
                        print(f"[{mode}] {held_rec['query_id']}: ERROR {e}",
                              flush=True)
                    else:
                        record["generated_comment"] = content
                        record["meta"] = meta
                        n_done += 1
                        print(f"[{mode}] {held_rec['query_id']} "
                              f"({meta['latency_s']}s, "
                              f"{meta.get('eval_count')} out-tokens)",
                              flush=True)
                if args.print:
                    print("=== PROMPT ===")
                    for m in messages:
                        print(f"--- {m['role']} ---\n{m['content']}")
                    print("=== GENERATED ===")
                    print(record.get("generated_comment"))
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{mode}] wrote {len(held)} records "
              f"({n_done} generated) -> {run_dir / mode}", flush=True)
    retr.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
