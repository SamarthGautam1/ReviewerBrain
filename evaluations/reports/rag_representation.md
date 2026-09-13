# Step 4 — Representation v2 audit + index rebuild

## Exact truncation rule (TASK A)

- model window: 256 wordpiece tokens (all-MiniLM-L6-v2); 2 reserved for [CLS]/[SEP], 2 safety.
- document = `Reviewer comment:\n<comment>\n\nFile:\n<path>\n\nRelevant diff:\n<trimmed diff>` — the comment comes FIRST, so it is always inside the window; the diff absorbs all cuts.
- diff token budget = window − comment − path − scaffold − 4; floored at 24 tokens. If the floor pushes the total past the window the model truncates the diff tail only (comment still fully embedded). Docs whose comment ALONE cannot fit are flagged `comment_overflows_window` (counted below, none deleted).
- diff trimming: parse the `@@ -a,b +c,d @@` header, assign each body line its new-file line number, anchor at the line matching the comment's `line` field (nearest match if out of range; midpoint of +/- lines when `line` is null), then grow a contiguous window outward (nearest line first) until the budget is reached. No elision markers; the @@ header itself is dropped. A single line longer than the whole budget is hard-truncated proportionally and flagged.
- query format (eval/production): `File:\n<path>\n\nRelevant diff:\n<trimmed diff>` — same trim rule, no comment block.
- follow_up_patch is never used in any representation.

## Enrichment join (read-only, originals untouched)

- records: 2664; join misses (no original fresh-comment match): 0

## TASK B — token audit of the v2 representation

### thockin (1995 docs)

- final doc tokens: p50=234 p90=252 p99=441 max=2440; docs over 256: 60 (all flagged comment-overflow: False)
- final doc chars: p50=688 max=7320
- comment fully inside window: 1941 (97.3%) — flagged overflow: 54
- comment `line` field available for anchoring: 1995 (100.0%)
- anchor source: {'matched_line': 1677, 'nearest_line': 318}
- docs whose diff was trimmed: 1123 (56.3%); kept/total hunk-line ratio: p10=0.09 p50=0.70
- no @@ header: 0; multi @@: 0; oversized anchor line: 34
- no-header docs were skipped from trimming stats where applicable

Comment-overflow examples (comment alone exceeds the window — flagged, kept):
- [thockin:34] PR#137050 `staging/src/k8s.io/api/coordination/v1alpha1/types.go` comment_tokens=251
- [thockin:37] PR#137050 `pkg/apis/coordination/validation/validation.go` comment_tokens=357
- [thockin:155] PR#138205 `staging/src/k8s.io/code-generator/cmd/validation-gen/validators/item.go` comment_tokens=482
- [thockin:209] PR#137271 `staging/src/k8s.io/api/scheduling/v1alpha2/types.go` comment_tokens=296
- [thockin:213] PR#137271 `staging/src/k8s.io/api/scheduling/v1alpha2/types.go` comment_tokens=486

Longest inputs:
- [thockin:286] 2440 tok / 7320 chars — PR#130724 `pkg/registry/core/replicationcontroller/strategy_test.go`
- [thockin:1540] 1297 tok / 3380 chars — PR#115204 `staging/src/k8s.io/cloud-provider/controllers/service/controller_test.go`
- [thockin:274] 1165 tok / 3072 chars — PR#134564 `pkg/apis/scheduling/validation/validation.go`
- [thockin:1062] 892 tok / 3518 chars — PR#125488 `staging/src/k8s.io/api/resource/v1alpha3/types.go`
- [thockin:1799] 822 tok / 2203 chars — PR#111661 `pkg/proxy/healthcheck/healthcheck_test.go`

Most substantially trimmed examples (lowest kept/total):
- [thockin:1062] PR#125488 `staging/src/k8s.io/api/resource/v1alpha3/types.go` kept 1/366 hunk lines (0%)
- [thockin:314] PR#136284 `staging/src/k8s.io/code-generator/cmd/validation-gen/output_tests/tags/shadow/fields/zz_generated.validations.go` kept 1/273 hunk lines (0%)
- [thockin:1013] PR#125488 `staging/src/k8s.io/api/resource/v1alpha3/types.go` kept 2/467 hunk lines (0%)
- [thockin:1071] PR#125488 `staging/src/k8s.io/api/resource/v1alpha3/types.go` kept 1/220 hunk lines (0%)
- [thockin:1016] PR#125488 `staging/src/k8s.io/api/resource/v1alpha3/types.go` kept 3/512 hunk lines (1%)

### ezyang (669 docs)

- final doc tokens: p50=222 p90=252 p99=254 max=565; docs over 256: 5 (all flagged comment-overflow: True)
- final doc chars: p50=631 max=2025
- comment fully inside window: 664 (99.3%) — flagged overflow: 5
- comment `line` field available for anchoring: 669 (100.0%)
- anchor source: {'matched_line': 544, 'nearest_line': 124, 'changed_midpoint': 1}
- docs whose diff was trimmed: 310 (46.3%); kept/total hunk-line ratio: p10=0.16 p50=1.00
- no @@ header: 0; multi @@: 0; oversized anchor line: 3
- no-header docs were skipped from trimming stats where applicable

Comment-overflow examples (comment alone exceeds the window — flagged, kept):
- [ezyang:284] PR#7506 `torch/onnx/symbolic.py` comment_tokens=289
- [ezyang:418] PR#7059 `test/test_cpp_extensions.py` comment_tokens=314
- [ezyang:465] PR#5767 `test/run_test.py` comment_tokens=524
- [ezyang:511] PR#5127 `torch/csrc/autograd/variable.h` comment_tokens=276
- [ezyang:545] PR#4999 `torch/_utils.py` comment_tokens=333

Longest inputs:
- [ezyang:465] 565 tok / 2025 chars — PR#5767 `test/run_test.py`
- [ezyang:545] 374 tok / 1352 chars — PR#4999 `torch/_utils.py`
- [ezyang:418] 358 tok / 1296 chars — PR#7059 `test/test_cpp_extensions.py`
- [ezyang:284] 323 tok / 959 chars — PR#7506 `torch/onnx/symbolic.py`
- [ezyang:511] 315 tok / 1198 chars — PR#5127 `torch/csrc/autograd/variable.h`

Most substantially trimmed examples (lowest kept/total):
- [ezyang:511] PR#5127 `torch/csrc/autograd/variable.h` kept 1/156 hunk lines (1%)
- [ezyang:284] PR#7506 `torch/onnx/symbolic.py` kept 1/69 hunk lines (1%)
- [ezyang:582] PR#4161 `aten/src/ATen/CPUApplyUtils.h` kept 4/223 hunk lines (2%)
- [ezyang:418] PR#7059 `test/test_cpp_extensions.py` kept 1/52 hunk lines (2%)
- [ezyang:621] PR#3634 `torch/csrc/jit/interpreter.cpp` kept 1/34 hunk lines (3%)

## Explicit verification

- full review-comment substring present in every doc text: PASS (checked all 2664 docs by exact string match)
- the comment's tokens are an exact token-prefix of the encoded doc (guarantees nothing before the comment can displace it): PASS
- every non-overflow doc has its comment block (incl. [CLS]) ending within the 256-token window: PASS
- docs with total_tokens > 256: 65 — all are budget-floored or comment-overflow cases where the MODEL truncates the diff tail only; the comment (first block) is unaffected.

## TASK C — chroma_v2 index rebuild

- thockin: 1995 docs embedded in 83.7s (42.0 ms/doc)
- ezyang: 669 docs embedded in 24.8s (37.1 ms/doc)
- collection `thockin`: 1995 docs (2.8s incl. add)
- collection `ezyang`: 669 docs (0.8s incl. add)
- collection `combined`: 2664 docs (3.4s incl. add)

