# ReviewerBrain — Step 3 RAG Report

## Index build

- model: all-MiniLM-L6-v2 loaded on CPU in 6.9s, telemetry disabled
- indexed thockin: 1995 docs, embedding took 106.5s (53.4 ms/doc)
- indexed ezyang: 669 docs, embedding took 35.1s (52.4 ms/doc)
- chroma collection `thockin`: 1995 docs
- chroma collection `ezyang`: 669 docs
- chroma collection `combined`: 2664 docs
- embedding dimensions: 384
- document format: diff_hunk + newline + review_comment (follow_up_patch NOT indexed)
- combined index metadata includes reviewer identity; total corpus: 2664 docs

## Metric experiment — cosine vs quantum fidelity

5 queries sampled deterministically (no RNG) from combined_clean.jsonl: 3 thockin at 10%/50%/90%, 2 ezyang at 30%/70% of each reviewer's block. Query = that example's diff_hunk; its own document is excluded from both metrics. Identical embeddings for both metrics; only the ranking metric differs. Candidate universe for both: 2663 docs (full corpus minus self). Gate: score >= 0.5 kept. Scores shown to 6 decimals.

### Query 1 — from thockin:199 (PR#137772 `staging/src/k8s.io/code-generator/cmd/validation-gen/lint_test.go`, thockin)

```diff
@@ -210,24 +211,58 @@ func TestRuleStability(t *testing.T) {
 		},
 		{
 			name:     "alpha context, alpha tag",
-			comments: []string{"+k8s:alpha=+k8s:validateTrue"}, // Alpha context, Alpha tag
+			comments: []string{"+k8s:alpha=+k8s:validateTrueAlpha"}, // Alpha context, Alpha tag
 			wantMsg:  "",
 		},
 		{
 			name:     "stable context, alpha tag",
-			comments: []string{"+k8s:validateTrue"}, // Stable context, Alpha tag
-			wantMsg:  `tag "k8s:validateTrue" with stability level "Alpha" cannot be used in Stable validation`,
+			comments: []string{"+k8s:validateTrueAlpha"}, // Stable context, Alpha tag
+			wantMsg:  `tag "k8s:validateTrueAlpha" with stability level "Alpha" cannot be used in Stable validation`,
 		},
 		{
 			name:     "beta context, alpha tag",
-			comments: []string{"+k8s:beta=+k8s:validateTrue"}, // Beta context, Alpha tag
-			wantMsg:  `tag "k8s:validateTrue" w
… (truncated)
```

**Cosine top-3** (universe 2663 = 2663; dropped by gate: 2620):
  1. [thockin:197] PR#137772 `staging/src/k8s.io/code-generator/cmd/validation-gen/lint_test.go` (thockin) cos=0.868387
  2. [thockin:413] PR#134302 `staging/src/k8s.io/code-generator/cmd/validation-gen/validators/limits.go` (thockin) cos=0.599632
  3. [thockin:414] PR#134302 `staging/src/k8s.io/code-generator/cmd/validation-gen/validators/limits.go` (thockin) cos=0.599632

**Fidelity |<a|b>|² top-3** (universe 2663; dropped by gate: 2662):
  1. [thockin:197] PR#137772 `staging/src/k8s.io/code-generator/cmd/validation-gen/lint_test.go` (thockin) fid=0.754097 (signed cos +0.868387)

- top-3 set overlap: 1/3 ['thockin:197']
- ranking changed: YES

### Query 2 — from thockin:997 (PR#125488 `staging/src/k8s.io/api/resource/v1alpha3/types.go`, thockin)

```diff
@@ -0,0 +1,998 @@
+/*
+Copyright 2022 The Kubernetes Authors.
+
+Licensed under the Apache License, Version 2.0 (the "License");
+you may not use this file except in compliance with the License.
+You may obtain a copy of the License at
+
+    http://www.apache.org/licenses/LICENSE-2.0
+
+Unless required by applicable law or agreed to in writing, software
+distributed under the License is distributed on an "AS IS" BASIS,
+WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
+See the License for the specific language governing permissions and
+limitations under the License.
+*/
+
+package v1alpha3
+
+import (
+	v1 "k8s.io/api/core/v1"
+	"k8s.io/apimachinery/pkg/api/resource"
+	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
+	"k8s.io/apimachinery/pkg/runtime"
+	"k8s.io/apimachinery/pkg/types"
+	"k8s.io/apimachinery/pkg/util/validation"
+)
+
+const (
+	// Finalizer is the 
… (truncated)
```

**Cosine top-3** (universe 2663 = 2663; dropped by gate: 2286):
  1. [thockin:977] PR#125488 `staging/src/k8s.io/api/resource/v1alpha3/types.go` (thockin) cos=1.000000
  2. [thockin:978] PR#125488 `staging/src/k8s.io/api/resource/v1alpha3/types.go` (thockin) cos=1.000000
  3. [thockin:985] PR#125488 `staging/src/k8s.io/api/resource/v1alpha3/types.go` (thockin) cos=1.000000

**Fidelity |<a|b>|² top-3** (universe 2663; dropped by gate: 2330):
  1. [thockin:1010] PR#125488 `staging/src/k8s.io/api/resource/v1alpha3/types.go` (thockin) fid=1.000000 (signed cos +1.000000)
  2. [thockin:1011] PR#125488 `staging/src/k8s.io/api/resource/v1alpha3/types.go` (thockin) fid=1.000000 (signed cos +1.000000)
  3. [thockin:1009] PR#125488 `staging/src/k8s.io/api/resource/v1alpha3/types.go` (thockin) fid=1.000000 (signed cos +1.000000)

- top-3 set overlap: 0/3 []
- ranking changed: YES

### Query 3 — from thockin:1795 (PR#116232 `test/e2e/network/pod_lifecycle.go`, thockin)

```diff
@@ -0,0 +1,276 @@
+/*
+Copyright 2023 The Kubernetes Authors.
+
+Licensed under the Apache License, Version 2.0 (the "License");
+you may not use this file except in compliance with the License.
+You may obtain a copy of the License at
+
+    http://www.apache.org/licenses/LICENSE-2.0
+
+Unless required by applicable law or agreed to in writing, software
+distributed under the License is distributed on an "AS IS" BASIS,
+WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
+See the License for the specific language governing permissions and
+limitations under the License.
+*/
+
+package network
+
+import (
+	"context"
+	"fmt"
+	"net"
+	"strconv"
+	"strings"
+	"time"
+
+	"github.com/onsi/ginkgo/v2"
+	v1 "k8s.io/api/core/v1"
+	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
+	"k8s.io/apimachinery/pkg/util/intstr"
+	"k8s.io/apimachinery/pkg/util/wait"
+	clientset "k8s.io/c
… (truncated)
```

**Cosine top-3** (universe 2663 = 2663; dropped by gate: 2283):
  1. [thockin:1787] PR#116232 `test/e2e/network/pod_lifecycle.go` (thockin) cos=1.000000
  2. [thockin:1788] PR#116232 `test/e2e/network/pod_lifecycle.go` (thockin) cos=1.000000
  3. [thockin:1789] PR#116232 `test/e2e/network/pod_lifecycle.go` (thockin) cos=1.000000

**Fidelity |<a|b>|² top-3** (universe 2663; dropped by gate: 2330):
  1. [thockin:1791] PR#116232 `test/e2e/network/pod_lifecycle.go` (thockin) fid=1.000000 (signed cos +1.000000)
  2. [thockin:1794] PR#116232 `test/e2e/network/pod_lifecycle.go` (thockin) fid=1.000000 (signed cos +1.000000)
  3. [thockin:1793] PR#116232 `test/e2e/network/pod_lifecycle.go` (thockin) fid=1.000000 (signed cos +1.000000)

- top-3 set overlap: 0/3 []
- ranking changed: YES

### Query 4 — from ezyang:200 (PR#5537 `torch/_torch_docs.py`, ezyang)

```diff
@@ -5906,3 +5911,328 @@
     )
 
 """)
+
+add_docstr(torch.fft,
+           r"""
+fft(input, signal_ndim, normalized=False) -> Tensor
+
+Complex-to-complex Discrete Fourier Transform
+
+This method computes the complex-to-complex discrete Fourier transform.
+Ignoring the batch dimension, it computes the following expression:
+
+.. math::
+    X[\omega_1, \dots, \omega_d] =
+        \frac{1}{\prod_{i=1}^d N_i} \sum_{n_1=0}^{N_1} \dots \sum_{n_d=0}^{N_d} x[n_1, \dots, n_d]
+         e^{-j\ 2 \pi \sum_{i=0}^d \frac{\omega_i n_i}{N_i}},
+
+where :math:`d`=:attr:`signal_ndim` is number of dimensions for the
+signal, and :math:`N_i` is the size of signal dimension :math:`i`.
+
+This method supports 1D, 2D and 3D complex-to-complex transforms, indicated
+by :attr:`signal_ndim`. :attr:`input` must be a tensor with last dimension
+of size 2, representing the real and imaginary components of compl
… (truncated)
```

**Cosine top-3** (universe 2663 = 2663; dropped by gate: 2652):
  1. [ezyang:201] PR#5537 `torch/_torch_docs.py` (ezyang) cos=1.000000
  2. [ezyang:199] PR#5537 `torch/_torch_docs.py` (ezyang) cos=0.997822
  3. [ezyang:204] PR#5537 `test/test_autograd.py` (ezyang) cos=0.556737

**Fidelity |<a|b>|² top-3** (universe 2663; dropped by gate: 2661):
  1. [ezyang:201] PR#5537 `torch/_torch_docs.py` (ezyang) fid=1.000000 (signed cos +1.000000)
  2. [ezyang:199] PR#5537 `torch/_torch_docs.py` (ezyang) fid=0.995649 (signed cos +0.997822)

- top-3 set overlap: 2/3 ['ezyang:199', 'ezyang:201']
- ranking changed: YES

### Query 5 — from ezyang:468 (PR#5575 `aten/src/ATen/native/TensorShape.cpp`, ezyang)

```diff
@@ -121,6 +121,93 @@ Tensor repeat(const Tensor& self, IntList repeats) {
   return result;
 }
 
+static std::vector<int64_t> infer_size(IntList shape, int64_t numel) {
+  auto res = shape.vec();
+  int64_t newsize = 1;
+  auto infer_dim = at::optional<int64_t>();
```

**Cosine top-3** (universe 2663 = 2663; dropped by gate: 2644):
  1. [ezyang:469] PR#5575 `aten/src/ATen/native/TensorShape.cpp` (ezyang) cos=0.883635
  2. [ezyang:467] PR#5575 `aten/src/ATen/native/TensorShape.cpp` (ezyang) cos=0.862380
  3. [ezyang:332] PR#8666 `aten/src/ATen/native/TensorShape.cpp` (ezyang) cos=0.677889

**Fidelity |<a|b>|² top-3** (universe 2663; dropped by gate: 2661):
  1. [ezyang:469] PR#5575 `aten/src/ATen/native/TensorShape.cpp` (ezyang) fid=0.780809 (signed cos +0.883634)
  2. [ezyang:467] PR#5575 `aten/src/ATen/native/TensorShape.cpp` (ezyang) fid=0.743699 (signed cos +0.862380)

- top-3 set overlap: 2/3 ['ezyang:467', 'ezyang:469']
- ranking changed: YES

## Per-reviewer retrieval demos (cosine metric)

3 handcrafted synthetic mini-diffs per reviewer, run against that reviewer's own index. These are new diffs, so no self-exclusion applies; the 0.5 gate is active.

### T1: new optional field added to a core API struct — index `thockin` (top-10 fetched, dropped by gate: 0)
```diff
@@ -512,6 +512,11 @@ type ServiceSpec struct {
 	// +optional
 	SessionAffinityConfig *SessionAffinityConfig `json:"sessionAffinityConfig,omitempty"`
+
+	// TrafficDistribution offers a hint for how traffic should be distributed
+	// among the endpoints of this service.
+	// +optional
+	TrafficDistribution *string `json:"trafficDistribution,omitempty" protobuf:"bytes,16,opt,name=trafficDistribution"`
```
- 1. PR#123487 `staging/src/k8s.io/api/core/v1/types.go` cos=0.775279
  > On names.  Sigh.  Let's think about how it is consumed.  You're a user reading a blog post, and the author says "You should set `trafficDistribution: PreferClose`".  You look at your own pods and find no such field.  This is USUALLY why we set default values, like "Default".

We talked before abou…
- 2. PR#124572 `staging/src/k8s.io/api/core/v1/types.go` cos=0.600770
  > need to update-codegen
- 3. PR#115433 `staging/src/k8s.io/api/discovery/v1/types.go` cos=0.537459
  > Same comments as above

### T2: validation logic added for an existing field — index `thockin` (top-10 fetched, dropped by gate: 0)
```diff
@@ -204,8 +204,13 @@ func validateVolumes(vols []core.Volume, fldPath *field.Path) field.ErrorList {
 		}
+		if vol.VolumeSource.EmptyDir != nil && vol.VolumeSource.EmptyDir.SizeLimit != nil {
+			if vol.VolumeSource.EmptyDir.SizeLimit.Sign() < 0 {
+				allErrs = append(allErrs, field.Invalid(idxPath.Child("emptyDir").Child("sizeLimit"),
+					vol.VolumeSource.EmptyDir.SizeLimit, "must be greater than 0"))
+			}
+		}
```
- 1. PR#137050 `pkg/apis/core/validation/validation.go` cos=0.751914
  > You don't want to use `Required()` on list elements.  Your format check should handle zero-length.
- 2. PR#137050 `pkg/apis/core/validation/validation.go` cos=0.751914
  > s/should/must/
- 3. PR#111401 `pkg/apis/core/validation/validation.go` cos=0.749053
  > Do we care if there is a dup name across init/ephemeral/real containers?

### T3: build script binary list change — index `thockin` (top-10 fetched, dropped by gate: 0)
```diff
@@ -45,6 +45,9 @@ readonly KUBE_TEST_BINARIES=(
 )
+readonly KUBE_TEST_SERVER_BINARIES=(
+  kube-apiserver
+  kube-controller-manager
+)
```
- 1. PR#137349 `staging/src/k8s.io/code-generator/kube_codegen.sh` cos=0.610771
  > I hate to pick, because it doesn't matter THAT much, but the correct invocation of printf is in update-codegen.sh:

```
    validation-gen \
        -v "${KUBE_VERBOSE}" \
        --go-header-file "${BOILERPLATE_FILENAME}" \
        --output-file "${output_file}" \
        $(printf -- " --rea…
- 2. PR#115243 `hack/lib/init.sh` cos=0.603548
  > missing EOL
- 3. PR#131755 `go.mod` cos=0.595183
  > this is wrong

### E1: dynamo variable-tracking change — index `ezyang` (top-10 fetched, dropped by gate: 10)
```diff
@@ -712,6 +712,9 @@ class VariableTracker:
     def var_getattr(self, obj, name):
+        if name.startswith("_"):
+            raise Unsupported("accessing private attribute")
         guards = self.guards
```
**No results passed the 0.5 gate.** Best candidate: [ezyang:28] PR#89032 cos=0.460485

### E2: C++ tensor concatenation size check — index `ezyang` (top-10 fetched, dropped by gate: 3)
```diff
@@ -3584,6 +3584,11 @@ void THTensor_(catArray)(THTensor *result, THTensor **inputs, int numInputs, int dimension)
+  int64_t outNumel = 0;
+  for (int i = 0; i < numInputs; i++) {
+    outNumel += THTensor_(nElement)(inputs[i]);
+  }
+  THArgCheck(outNumel >= 0, 2, "invalid concatenation size");
```
- 1. PR#8559 `aten/src/TH/generic/THTensor.cpp` cos=0.581796
  > Did you want to write this code to also support negative strides? I worked out the math and wrote this helper function: https://github.com/ezyang/pytorch/blob/c10/caffe2/c10_prototype/c10/Utils.h#L30
- 2. PR#8559 `aten/src/THC/THCTensor.cpp` cos=0.563081
  > I cannot wait for THTensor and THCTensor to be the same thing and then we don't need to copy paste the code.
- 3. PR#8666 `aten/src/TH/generic/THTensorMath.cpp` cos=0.545366
  > Mention, I suppose, this is a non-owning reference.

### E3: autograd test addition — index `ezyang` (top-10 fetched, dropped by gate: 0)
```diff
@@ -914,6 +914,9 @@ class TestAutograd(TestCase):
     def test_autograd_simple(self):
+        x = torch.randn(3, requires_grad=True)
+        y = x * 2
+        self.assertTrue(y.requires_grad)
```
- 1. PR#5362 `test/test_nn.py` cos=0.670702
  > Note to self; this will intersect with #5192. (Ailing, you don't have to do anything)
- 2. PR#5362 `test/test_nn.py` cos=0.670702
  > This is just a nit, but... why not just avoid converting at all if `dtype == torch.HalfTensor` is true?
- 3. PR#5176 `test/test_jit.py` cos=0.578298
  > If you have:

```
while False:
    third = ...
    ...
st = second + third
```

aren't you in trouble if you follow Python scoping? I am seriously skeptical initializing `third` to an undefined tensor here is a good idea.

## Latency (CPU, this machine)

- query embedding (MiniLM, single text): avg 54.1 ms over 11 queries
- chroma cosine query (top-10 fetch): avg 4.9 ms over 11 queries
- fidelity full-scan scoring (numpy, 2664×384): avg 0.22 ms over 5 queries

## Aggregates

- cosine vs fidelity top-3 set overlap per experiment query: [1, 0, 0, 2, 2] → mean 1.0/3
- experiment queries where the ordered top-3 changed: 5/5
- gate drops in experiment (universe 2663 per query): cosine 12485 across 5 queries; fidelity 12644
- experiment result slots filled: cosine 15/15, fidelity 11/15
- chroma cosine vs numpy cosine max abs difference: 6.56e-07

## Supplementary analysis — why rankings "changed"

Signed-cosine scan over the full corpus for each experiment query (loaded from
the persisted index, same embeddings):

| query | docs with cos<0 | most negative cos | max \|neg\| |
|---|---|---|---|
| thockin:199 | 16/2663 | -0.0748 | 0.075 |
| thockin:997 | 210/2663 | -0.1169 | 0.117 |
| thockin:1795 | 100/2663 | -0.1012 | 0.101 |
| ezyang:200 | 379/2663 | -0.1274 | 0.127 |
| ezyang:468 | 791/2663 | -0.1401 | 0.140 |

Findings:

1. The sign-fold risk of |<a|b>|² (a negative cosine scoring high fidelity)
   never fired: the most negative cosine anywhere is -0.14, far below the
   positive top-3 scores. Every fidelity pick had positive signed cosine.
2. For every query, the ORDER of shared results was identical between the
   two metrics — fidelity = cos² is monotone on positive cosines, so it
   cannot reorder distinct positive scores.
3. All 5 "ranking changed" verdicts reduce to two mechanisms:
   - Queries 2 and 3: top-8 cosines are EXACTLY 1.0 in float32 — the tie
     group members are bit-identical embeddings, so top-3 membership within
     the tie group is arbitrary for both metrics (overlap 0/3 is tie
     shuffling, not a metric difference).
   - Queries 1, 4, 5: the fidelity gate (score 0.5 ⇔ |cos| >= 0.7071) is
     strictly stricter than the cosine gate (cos >= 0.5), so fidelity's
     3rd slot is dropped where cosine keeps it.

## Supplementary analysis — embedding truncation (material finding)

all-MiniLM-L6-v2 truncates at 256 wordpiece tokens. Measured over the
indexed documents (hunk + newline + comment):

| index | docs | tokens p50 / p90 / max | >256 tokens | hunk alone >=256 (comment fully cut off) |
|---|---|---|---|---|
| thockin | 1995 | 304 / 1600 / 15062 | 1136 (56.9%) | 1001 (50.2%) |
| ezyang | 669 | 232 / 1055 / 6661 | 314 (46.9%) | 281 (42.0%) |
| combined | 2664 | 284 / 1486 / 15062 | 1450 (54.4%) | 1282 (48.1%) |

For ~48% of the corpus the review comment contributes NOTHING to the
embedding — retrieval is effectively hunk-prefix-only there, and comments
on the same large hunk become indistinguishable (the exact-1.0 tie groups
above). Retrieval quality did not visibly collapse because the hunk is the
right retrieval key for this pipeline (queries are hunks), but the document
format should be revisited before scaling this up: options are (a) truncate
the hunk first and always keep the comment inside the window, (b) put the
comment before the hunk, or (c) move to a longer-context embedder. Decision
deferred to the user.
