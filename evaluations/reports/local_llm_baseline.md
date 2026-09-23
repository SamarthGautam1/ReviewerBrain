# Local LLM Baseline + RAG Inference — Pilot Report

Phase: **local review generation** (before any fine-tuning).
Date: 2026-09-20. Machine-local, no cloud APIs, no data egress.

Scope guardrails honored: the validated RAG pipeline (representation,
embeddings, indexes, split) was only **imported, never modified**; raw and
processed datasets are read-only; no LoRA/QLoRA work was started; nothing
was committed or pushed.

---

## 1. Hardware audit

| Component | Value |
|---|---|
| CPU | Intel Core i5-13420H (8C/12T, 13th gen) |
| RAM | 15.7 GB total (~3.5 GB free at audit time) |
| GPU | NVIDIA GeForce RTX 4050 Laptop, **6 GB VRAM** (5.9 GB free), compute 8.9 |
| GPU driver / CUDA | 581.29, CUDA 13.0 (driver-level) |
| PyTorch | 2.14.0 **+cpu** — CUDA unavailable in Python |
| Python | 3.13.7 (system install) |
| Disk | C: 53 GB free, D: 95.5 GB free (project on D:) |
| Ollama | **not installed** at audit time |

Implications: the only usable GPU path for LLM inference is a runtime with
its own CUDA backend (Ollama/llama.cpp); the Python stack stays CPU-only.
6 GB VRAM caps dense model size at roughly 7B/Q4.

## 2. Local model audit and selection

Ollama was absent (checked PATH, `Program Files`, `%LOCALAPPDATA%`,
`~/.ollama`, running processes). It was installed (0.34.2, per-user,
silent) and one model pulled — this is the "explicitly necessary"
download, because the pilot requires an actual local model.

**Primary model: `qwen2.5-coder:7b`** (Q4_K_M, 4.7 GB, Apache-2.0).

Rationale against the hardware constraints:

- **VRAM**: 4.7 GB weights fit the 6 GB card with 8k context (~0.5 GB KV
  + buffers); verified running fully on GPU (nvidia-smi: ~4 GB used).
- **Context**: 32k native; runs pinned at 8192 — pilot prompts peaked at
  2.8k tokens, so ample headroom.
- **Code understanding**: strongest open ~7B code model family at this
  quantization (code-heavy training); suited to reading diffs.
- **Instruction following**: reliable short-answer format compliance.
- **Speed**: ~12-45 tok/s GPU-offloaded; a 20-query mode finishes in
  ~1.5-2 min.
- **Suitability**: same family later serves LoRA experiments (Qwen2.5
  has clean HF/PEFT support).

Fallback (documented, not needed): `qwen2.5-coder:3b` (~1.9 GB) if VRAM
pressure or throughput becomes a problem. Larger families (Phi-4 14B,
GPT-OSS-20B, Qwen3-Coder-30B-A3B) do not fit 6 GB VRAM.

## 3. Baseline prompt (baseline_v1)

File: `configs/prompts/baseline_v1.yaml` (versioned; SHA-256 recorded in
every run manifest). Contains **no** reviewer-specific information: no
reviewer name, no persona, no historical examples. A single system
instruction ("experienced software engineer… write the review comment you
would post… be concise and technical… do not praise") plus a user block:

```
File: {file_path}
{pr_context}                       ← PR title + first 1000 chars of description
Code change (diff hunk):
```diff
{code_change}                      ← full hunk, capped at 6000 chars
```
Respond with only the text of your review comment — …
```

Inputs: file path, diff hunk, optional PR title/description. Output: one
review comment. Kept deliberately simple — its job is to be the generic
LLM floor.

## 4. RAG prompt (rag_v1)

File: `configs/prompts/rag_v1.yaml`. The core task instruction is
**identical** to baseline_v1; two things are added:

1. **Retrieved evidence**: the top-3 historical examples, inserted
   **verbatim as stored** in the validated per-reviewer ChromaDB index
   (comment-first documents — no re-derivation), each labeled
   `[Example i] (cosine x.xxx, PR#, file:)`.
2. **Reviewer instruction**: system prompt names the reviewer, frames the
   examples as *evidence of focus and phrasing*, and explicitly forbids
   copying: "NEVER copy a historical comment: the current change is
   different, and a copied comment would be wrong."

No hand-written persona descriptions were added — they would inject
unvalidated assumptions; the examples are the only style evidence.

## 5. Inference harness

New package `src/reviewerbrain/inference/` (thin clients of the frozen
pipeline) + scripts under `scripts/inference/`:

- `run_review_generation.py` — CLI: `--reviewer thockin|ezyang|both`,
  `--mode baseline|rag|both`, `--queries N` (deterministic evenly-spaced
  held-out sample; no RNG), `--query-ids` (explicit), `--dry-run`
  (prompt audit without LLM), `--print`, `--tag`. Writes
  `evaluations/inference/<tag>/<mode>/generations.jsonl` +
  `manifest.json` (model, options, query ids, prompt SHAs, timestamps —
  everything needed to reproduce a run).
- `evaluate_generations.py` — corpus BLEU-4, sentence BLEU-4 (add-1),
  ROUGE-L F1, optional BERTScore, output-length stats, and
  `qualitative_examples.md` (current code / ground truth / baseline /
  RAG / retrieved examples side by side).
- Package modules: `heldout_queries.py` (held-out selection + **verified** join
  back to the clean record — reviewer, PR, comment, file must all match),
  `rag_retrieval.py`, `prompts.py`, `llm_client.py`, `outputs.py`.

Two correctness guards worth calling out:

- **Leakage guard**: the persistent indexes contain every cleaned example
  *including* held-out PRs (the validated eval excluded held-out rows at
  query time). The harness replicates that exclusion: candidates from
  held-out PRs (frozen split rule) are dropped before the cosine gate,
  and each result is asserted not to come from the query's PR. A dry-run
  without this guard retrieved the query's own document at cos 0.965 —
  the guard is load-bearing.
- **Egress guard**: the LLM client only accepts http(s) endpoints whose
  host is `localhost`/loopback (resolved and re-checked per request);
  output paths are slug-validated and containment-checked.

## 6. Controlled pilot

Deterministic sample (identical for both modes): 10 thockin + 10 ezyang
held-out queries, evenly spaced over each reviewer's held-out list
(`thockin:71,150,195,608,782,903,962,1289,1524,1750`,
`ezyang:47,89,159,173,223,274,353,446,488,554`).

Run params: `qwen2.5-coder:7b`, temperature 0 (greedy), seed 42,
num_ctx 8192, num_predict 512, Ollama 127.0.0.1:11434. Outputs in
`evaluations/inference/pilot_local/` (manifest, two generation files,
metrics, qualitative examples). 40/40 generations, 0 errors.

Retrieval behavior under the frozen gate (cos ≥ 0.5, top-3):
19/20 queries got 3 examples, 1 query (`thockin:1750`) got **0** — the
gate genuinely fires; the prompt falls back to a "no examples passed the
gate" note. Mean top-3 cosine 0.644; 5/20 examples were same-file.
Prompt size: 670 → 1292 mean tokens (baseline → RAG); latency 4.2 s →
5.0 s mean.

## 7. Automatic metrics (with interpretation caveat)

Ground truth is **one** comment per query; many valid comments exist per
diff. These numbers are orientation only — **they do not measure reviewer
replication**.

| metric | baseline | rag |
|---|---|---|
| corpus BLEU-4 | 0.0069 | 0.0199 |
| mean sentence BLEU-4 (add-1) | 0.0356 | 0.0408 |
| mean ROUGE-L F1 | 0.1007 | 0.0602 |
| mean BERTScore F1 (roberta-large) | 0.835 | 0.825 |
| avg generated words | 36.0 | 30.1 |
| avg ground-truth words | 29.2 | 29.2 |

Reading: all values are near the "any two English sentences" floor
(BERTScore ~0.83 with no discriminative power; ROUGE/BLEU near zero).
Length, not content, dominates the lexical differences. The only notable
direction is corpus BLEU-4 tripling under RAG (more 4-gram overlap with
the reviewer's actual phrasing), consistent with the qualitative register
shift below — but n=20, so no significance claim is made.

## 8. Qualitative analysis

Side-by-side blocks for the first 3 queries per reviewer are in
`evaluations/inference/pilot_local/qualitative_examples.md`; all 40 pairs
were inspected. Summary per reviewer:

### thockin (terse, challenging questions)

| query | ground truth | baseline | rag |
|---|---|---|---|
| :71 | "testcase to prove that it works when both are set?" | long prose about a missing validation check | "Could make this a tweak func?" |
| :608 | run each case with the gate on and off | unsupported claim that expectation should be `false` | "Also add a case for when `…ZeroValue` is `false`." |
| :782 | "IPAddresses… is there any address that is not an IP?" | praise-ish restatement | "Do we need all these fields? Can we simplify this?" |

- **Relevant findings (RAG)**: :608 is a substantive near-hit — "add the
  false case" is the same point as "run each case with the gate on and
  off". :71, :782 match the interrogative register (6-10 words, question
  mark) though not the exact substance.
- **Baseline**: fluent but generic; noticed the validation-ordering issue
  on :150 (GT also questions the reordering) but packaged as lecture.
- **Hallucinations (baseline)**: :608 (asserts the expected value is
  `false`), :1524 (claims a field "is missing a type" when it has one).
- **Form leakage (RAG)**: :903, :1289, :1524, :1750 emit ```go/```diff
  blocks — echoing the diff form of retrieved examples (and, on :1750,
  even with zero examples passing the gate) instead of a comment.

### ezyang (design soundness probing, asks for examples)

| query | ground truth | baseline | rag |
|---|---|---|---|
| :446 | "needs a comment explaining what the string keys/values mean" | generic implementation warning | "What does `RawDataExportMap` represent?" |
| :159 | "I'm not sure this transformation is sound…" | verbose correctness lecture | "…the logic is incomplete… only checks for `PadPacked`…" |
| :89 | "Is there a way to make this code Python agnostic?" | **false** claim "This header is empty" | "What is the purpose of including `python_stub.h`?" |

- **Relevant findings (RAG)**: :446 is a genuine near-hit (asking what
  the map is ≈ asking for it to be documented). :159 engages with the
  actual transformation logic; :89 asks a question about the same
  include-region of code.
- **Copied/template-like behavior**: :47 outputs an import block that
  mirrors the retrieved example's diff; :223 echoes example code; :488
  produces a TODO-form comment shaped like an example. ~25% of RAG
  outputs show some form copying — the anti-copy instruction reduced but
  did not eliminate it.
- **Baseline misses**: :488 "Paging @apaszke" and :353 "Thanks… :)" are
  social/process comments no code-focused prompt can predict.

### Cross-cutting observations

1. **The register shift is the clearest RAG effect.** Baseline writes
   30-60-word imperative prose ("Add a check…", "Consider using…"); RAG
   writes 5-15-word questions/challenges ("Could make this a tweak
   func?", "Why are you adding a new condition for `IntegerTensor`?"),
   which is exactly the ground-truth style of both reviewers. This comes
   from the examples, not the instruction (identical task text in both
   prompts).
2. **Retrieved examples are demonstrably used**: register shift across
   20/20 queries, form leakage (diff/code echoes), two substantive
   near-hits (:608 thockin, :446 ezyang), and BLEU-4 tripling.
3. **Content accuracy is the weak spot for both modes**: both
   hallucinate specific claims occasionally, and comments requiring
   PR-wide context (design questions spanning files) are out of reach
   with a single hunk.
4. **Substance vs style**: RAG currently buys *style* more than
   *substance*; exact ground-truth concerns are hit roughly, not
   precisely.

## 9. Answers to the phase questions

1. **Does the generic model produce usable code reviews?**
   Partially. Fluent, technically plausible, mostly well-targeted at the
   hunk; but generic in register (verbose lecture prose), occasionally
   hallucinated, and stylistically unlike either reviewer. Usable as a
   review *assistant*, not yet as a reviewer *simulation*.
2. **Does RAG provide useful reviewer-specific context?**
   Yes, as style/register conditioning: the terse question form, reviewer
   vocabulary, and two substantive near-hits. It does not yet reliably
   recover the reviewers' specific *concerns* on new diffs.
3. **Are the retrieved examples actually being used by the model?**
   Yes — provably: identical task instructions, only the example block
   differs, and outputs shift register 20/20, imitate example forms
   (diff/code echoes), and overlap reviewer phrasing (BLEU-4 ×3).
4. **Which model should be frozen for the next LoRA experiment?**
   `qwen2.5-coder:7b` (Q4_K_M serving now; LoRA on the fp16/base
   Qwen2.5-Coder-7B-Instruct weights, matching the family already
   validated for serving). It fits the 6 GB VRAM envelope for inference
   and is well supported by PEFT tooling.
5. **What prompt changes are necessary before fine-tuning?**
   - Enforce output form: forbid code/diff blocks in the response
     (~25% of RAG outputs leaked example form; also 1 baseline prose
     over-structure). Suggested: "Reply with plain prose only — never
     code, diffs, or fenced blocks."
   - Add few-shot-consistent output normalization so fine-tuning targets
     match serving inputs (the fine-tune will replace the example block,
     but the task instruction should stay frozen to keep comparability).
   - Consider adding "if you have no specific concern, ask the single
     most useful clarifying question" — the ground truth is dominated by
     questions; this makes the greedy-decode failure mode softer.
   - Keep `baseline_v1`/`rag_v1` instruction text fixed from here on so
     pre/post-fine-tune comparisons stay attributable.
6. **What problems remain?**
   - Single-hunk input: design-level comments (a large share of both
     reviewers' real comments) need PR/file context beyond the hunk.
   - Social/process comments ("Paging @x", "Thanks") are unpredictable
     from code and will cap any automatic metric.
   - Metrics: single-reference BLEU/ROUGE/BERTScore cannot see quality;
     an LLM-judge or pairwise human preference protocol is needed before
     claiming reviewer replication.
   - Output-form control needs the prompt fix above (or post-filtering).
   - n=20 pilot: all differences are directional, not significant.
   - 6 GB VRAM constrains context (~8k) and model size; a machine with
     12-16 GB would widen model choice for LoRA.

## 10. Reproduction

```bash
# one-time: Ollama 0.34.2 installed (per-user); model pulled
ollama pull qwen2.5-coder:7b        # 4.7 GB, Q4_K_M

# from the repo root, server running on 127.0.0.1:11434:
python scripts/inference/run_review_generation.py --tag pilot_local --queries 10 \
    --mode both --force
python scripts/inference/evaluate_generations.py \
    --run-dir evaluations/inference/pilot_local --bertscore
python -m pytest tests/ -q           # 26 passed (incl. inference units)
```

Artifacts: `evaluations/inference/pilot_local/{manifest.json,
baseline/generations.jsonl, rag/generations.jsonl, metrics.json,
metrics_summary.md, qualitative_examples.md}` (gitignored, regenerable).

STOP — fine-tuning (LoRA/QLoRA) intentionally not started.
