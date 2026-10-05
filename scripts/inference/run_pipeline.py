"""Run the four-mode ReviewerBrain pipeline over held-out queries.

Modes (separate by design — see reviewerbrain.inference.pipeline):
  base      generic baseline prompt            (validated pilot mode)
  base_rag  RAG-conditioned prompt             (validated pilot mode)
  lora      reviewer adapter, no retrieval     (needs trained adapter)
  lora_rag  reviewer adapter + RAG retrieval   (needs trained adapter)

Backends
  mock    deterministic canned reviews, zero model loading (default —
          the local development machine never runs an LLM)
  ollama  local Ollama server (loopback-only; see llm_client). Modes
          lora/lora_rag require the server-side model to carry the
          reviewer adapter (Ollama Modelfile ADAPTER).

Queries: the frozen held-out artifact, deterministic evenly-spaced sample
per reviewer (same scheme as run_review_generation.py).

Outputs (generated, gitignored) under evaluations/inference/<tag>/ via the
validated writers; per-mode generations.jsonl records mirror the pilot
harness shape, so scripts/inference/evaluate_generations.py works
unchanged on them.

Examples
  python scripts/inference/run_pipeline.py --tag demo_mock --backend mock --mode all
  python scripts/inference/run_pipeline.py --tag pilot2 --backend ollama --mode base --queries 10
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from reviewerbrain.inference import adapters, backends, outputs  # noqa: E402
from reviewerbrain.inference import pipeline as pipe  # noqa: E402
from reviewerbrain.inference import heldout_queries as q  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_MOCK = "MockBackend"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", default="all",
                   help="base|base_rag|lora|lora_rag|all (default: all)")
    p.add_argument("--reviewer", choices=["thockin", "ezyang", "both"],
                   default="both")
    p.add_argument("--queries", type=int, default=10)
    p.add_argument("--query-ids", default=None,
                   help="comma-separated query_ids (overrides --queries)")
    p.add_argument("--backend", choices=["mock", "ollama"], default="mock")
    p.add_argument("--model", default="qwen2.5-coder:7b",
                   help="model name for the ollama backend")
    p.add_argument("--endpoint", default=None,
                   help="ollama endpoint (loopback only)")
    p.add_argument("--tag", required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="build prompts, skip generation (no backend call)")
    p.add_argument("--print", action="store_true")
    return p.parse_args(argv)


def resolve_modes(spec):
    if spec == "all":
        return list(pipe.MODES)
    if spec not in pipe.MODES:
        raise SystemExit(f"unknown mode {spec!r}; expected one of "
                         f"{list(pipe.MODES)} or 'all'")
    return [spec]


def pick_queries(args):
    heldout = q.load_heldout()
    reviewers = (["thockin", "ezyang"] if args.reviewer == "both"
                 else [args.reviewer])
    if args.query_ids:
        by_id = {r["query_id"]: r for r in heldout}
        out = []
        for qid in [s.strip() for s in args.query_ids.split(",") if s.strip()]:
            if qid not in by_id:
                raise SystemExit(f"query_id {qid} not in held-out artifact")
            out.append(by_id[qid])
        return reviewers, out
    out = []
    for name in reviewers:
        out.extend(q.select_queries(heldout, name, args.queries))
    return reviewers, out


def build_backend(args):
    if args.backend == "mock":
        return backends.MockBackend()
    return backends.OllamaBackend(args.model, endpoint=args.endpoint)


def main(argv=None):
    args = parse_args(argv)
    modes = resolve_modes(args.mode)
    for m in modes:
        pipe.check_mode(m)
    reviewers, held = pick_queries(args)

    adapter_state = {r: adapters.has_adapter(r) for r in reviewers}
    unavailable = [(m, r) for m in modes for r in reviewers
                   if pipe.mode_uses_adapter(m) and not adapter_state[r]]
    if unavailable and args.backend == "ollama":
        # real generation with a missing adapter is a hard error unless
        # the user excluded those modes up front
        missing = ", ".join(f"{m}/{r}" for m, r in unavailable)
        raise SystemExit(f"missing adapters for: {missing} "
                         f"(train via kaggle/README.md, or drop those modes)")

    run_dir = outputs.resolve_run_dir(args.tag)
    if run_dir.exists() and not args.force:
        raise SystemExit(f"run dir {run_dir} exists (use --force)")
    run_dir.mkdir(parents=True, exist_ok=True)

    backend = build_backend(args)
    outputs.write_manifest(run_dir, {
        "tag": args.tag,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "harness": "run_pipeline.py",
        "backend": args.backend,
        "model": (args.model if args.backend == "ollama"
                  else backends.MockBackend.name),
        "endpoint": args.endpoint,
        "dry_run": args.dry_run,
        "modes": modes,
        "mode_templates": {m: pipe.MODE_TO_TEMPLATE[m] for m in modes},
        "adapters_available": adapter_state,
        "sample": {"queries_per_reviewer": args.queries,
                   "explicit_query_ids": args.query_ids,
                   "reviewers": reviewers,
                   "query_ids": [r["query_id"] for r in held]},
        "python": sys.version.split()[0],
    })

    for mode in modes:
        n_done = 0
        with outputs.open_run_file(run_dir, mode, "generations.jsonl") as out:
            for held_rec in held:
                reviewer = held_rec["reviewer"]
                rec = q.load_clean_for_heldout(held_rec)
                record = {
                    "query_id": held_rec["query_id"],
                    "reviewer": reviewer,
                    "mode": mode,
                    "prompt_template": pipe.MODE_TO_TEMPLATE[mode],
                    "model": (args.model if args.backend == "ollama"
                              else DEFAULT_MOCK),
                    "file": held_rec["file"],
                    "pr_number": held_rec["query_pr_number"],
                    "ground_truth_comment": held_rec["query_review_comment"],
                }
                if pipe.mode_uses_adapter(mode) and not adapter_state[reviewer]:
                    record.update({"messages": None, "retrieved": None,
                                   "generated_comment": None,
                                   "error": "adapter not available",
                                   "meta": {"backend": args.backend}})
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue
                try:
                    result = pipe.run_review(mode, reviewer, rec, backend)
                    record.update(result)
                    record["options"] = None
                    n_done += 1
                    print(f"[{mode}] {held_rec['query_id']} "
                          f"({record['meta'].get('backend')} backend)",
                          flush=True)
                except Exception as e:                      # noqa: BLE001
                    record.update({"messages": None, "retrieved": None,
                                   "generated_comment": None,
                                   "error": str(e),
                                   "meta": {"backend": args.backend}})
                    print(f"[{mode}] {held_rec['query_id']}: ERROR {e}",
                          flush=True)
                if args.print:
                    print("=== GENERATED ===")
                    print(record.get("generated_comment"))
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{mode}] wrote {len(held)} records -> "
              f"{run_dir / mode}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
