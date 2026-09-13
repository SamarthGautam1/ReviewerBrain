# Step 4 — Held-out RAG evaluation (v2 representation)

## TASK D — split definition

- deterministic: per reviewer, PRs sorted by pr_number, every 5th PR (position % 5 == 4) held out. No RNG; reruns reproduce the split exactly.
- thockin: 67 held-out PRs → 358 queries; index 1637 docs
- ezyang: 39 held-out PRs → 138 queries; index 531 docs
- total queries: 496; query embedding time 18.2s; index rows exclude every held-out PR (same-PR leakage impossible by construction; verified: 0 leaks in top-5 across all queries/modes)
- relevance proxies (objective): same-file hit = retrieved file == query file; same-dir hit = same parent directory. Semantic relevance is assessed qualitatively (see evaluations/reports).
- gated hit@k counts a miss when nothing passes the gate.

## TASK E — cosine vs fidelity (equivalent thresholds)

Gates: cosine >= 0.5 vs fidelity >= 0.25 (equivalent admission for positive cosines). 'no thresh' rows use pure ranking. Mean-score rows report each metric's own score over its own top-k.

### mode: per_reviewer — 496 queries

| metric | cosine | fidelity |
|---|---|---|
| same-file hit@1 (no thresh) | 24.6% | 24.6% |
| same-dir hit@1 (no thresh) | 37.9% | 37.9% |
| mean score of top-1 (no thresh) | 0.7023 | 0.5017 |
| same-file hit@1 (gated) | 24.6% | 24.6% |
| same-file hit@3 (no thresh) | 34.7% | 34.7% |
| same-dir hit@3 (no thresh) | 50.4% | 50.4% |
| mean score of top-3 (no thresh) | 0.6742 | 0.4624 |
| same-file hit@3 (gated) | 34.5% | 34.5% |
| same-file hit@5 (no thresh) | 40.1% | 40.1% |
| same-dir hit@5 (no thresh) | 56.9% | 56.9% |
| mean score of top-5 (no thresh) | 0.6577 | 0.4404 |
| same-file hit@5 (gated) | 39.9% | 39.9% |
| zero-candidate queries (gated) | 9 | 9 |

- top-k set overlap cosine∩fidelity (no thresh): @1=1.00/1, @3=3.00/3, @5=5.00/5
- mean Spearman rank correlation cos vs fid over full candidate lists: 0.9999 (1.0 = identical ordering)
- queries where a fidelity top-5 member has negative cosine (sign-fold fired): 0

### mode: combined — 496 queries

| metric | cosine | fidelity |
|---|---|---|
| same-file hit@1 (no thresh) | 24.6% | 24.6% |
| same-dir hit@1 (no thresh) | 37.9% | 37.9% |
| mean score of top-1 (no thresh) | 0.7023 | 0.5017 |
| same-file hit@1 (gated) | 24.6% | 24.6% |
| same-file hit@3 (no thresh) | 34.7% | 34.7% |
| same-dir hit@3 (no thresh) | 50.4% | 50.4% |
| mean score of top-3 (no thresh) | 0.6742 | 0.4624 |
| same-file hit@3 (gated) | 34.5% | 34.5% |
| same-file hit@5 (no thresh) | 40.1% | 40.1% |
| same-dir hit@5 (no thresh) | 56.9% | 56.9% |
| mean score of top-5 (no thresh) | 0.6577 | 0.4404 |
| same-file hit@5 (gated) | 39.9% | 39.9% |
| zero-candidate queries (gated) | 9 | 9 |

- top-k set overlap cosine∩fidelity (no thresh): @1=1.00/1, @3=3.00/3, @5=5.00/5
- mean Spearman rank correlation cos vs fid over full candidate lists: 0.9947 (1.0 = identical ordering)
- queries where a fidelity top-5 member has negative cosine (sign-fold fired): 0
- queries where top-1 differs between the per_reviewer and combined index: 0

## TASK F — qualitative inspection (held-out queries)

10 evenly-spaced held-out queries per reviewer, top-3 by cosine over the combined held-out index (no threshold). Flags: [DIR] same directory, [BOILER] license/boilerplate head in a hunk, [WEAK] cos < 0.3, [LEAK] same PR (expected 0).

### thockin — 10 held-out queries

#### thockin:71 — PR#139422 `test/declarative_validation/batch/cronjob/declarative_validation_test.go` (comment_id 3376420627)
- **Query review:** testcase to prove that it works when both are set?
```diff
 				field.Required(field.NewPath("spec", "schedule"), "").MarkAlpha(),
 			},
 		},
+		"jobTemplate.spec.maxFailedIndexes set without backoffLimitPerIndex": {
```
- 1. thockin:101 PR#136589 `pkg/registry/scheduling/workload/declarative_validation_test.go` cos=0.5804 
  > This could just be:

`field.Invalid(field.NewPath("spec.podGroupTemplates[0].priority"), nil, "").WithOrigin("maximum")`

The origin ensures that the right error was triggered without embedding the exact error string into each test case.
- 2. thockin:212 PR#137271 `pkg/registry/scheduling/podgroup/declarative_validation_test.go` cos=0.5757 
  > bounds testing of a valid key?
- 3. thockin:306 PR#135164 `pkg/registry/scheduling/workload/declarative_validation_test.go` cos=0.5724 [BOILER]
  > could make it a tweak func.  I won't repeat this any more, I think you see the pattern.  Using a literal func as a tweak should be reserved for very special cases.  Prefer instead to build up a library of tweaks.

#### thockin:150 — PR#138205 `staging/src/k8s.io/code-generator/cmd/validation-gen/output_tests/tags/levels/structs/zz_generated.validations.go` (comment_id 3125287270)
- **Query review:** What is causing this reordering?
```diff
 				return nil
 			}
 			// call field-attached validations
-			errs = append(errs, validate.Minimum(ctx, op, fldPath, obj, oldObj, 5)...)
 			errs = append(errs, validate.Minimum(ctx, op, fldPath, obj, oldObj, 10).MarkAlpha()...)
+			errs = append(errs, validate.Minimum(ctx, op, fldPath, obj, oldObj, 5)...)
```
- 1. thockin:314 PR#136284 `staging/src/k8s.io/code-generator/cmd/validation-gen/output_tests/tags/shadow/fields/zz_generated.validations.go` cos=0.6704 [BOILER]
  > Something is fishy -- how does this end up being the first thing emitted? 

Edit: I investigated, and it is correct -- union is a type-validator, which comes before field validators.  I am leaving this comment as evidence that we thought about this :)

We could probably emit bett…
- 2. thockin:415 PR#134302 `pkg/apis/resource/v1/zz_generated.validations.go` cos=0.6444 
  > Are we planning to do `+k8s:subfield(name)=+k8s:format=k8s-long-name` or whatever in 1.35, too?
- 3. thockin:377 PR#135123 `staging/src/k8s.io/code-generator/cmd/validation-gen/validation.go` cos=0.6407 
  > linewrap?

#### thockin:195 — PR#136976 `pkg/registry/scheduling/podgroup/declarative_validation_test.go` (comment_id 2906601078)
- **Query review:** Corresponding to the other comments about alpha - I see it is just immutable that is still marked alpha, but I am not sure why.  I think it is not needed
```diff
+		oldObj       scheduling.PodGroup
+		updateObj    scheduling.PodGroup
+		expectedErrs field.ErrorList
+	}{
+		"valid update": {
+			oldObj:    mkValidPodGroup(setResourceVersion("1")),
+			updateObj: mkValidPodGroup(setResourceVersion("1")),
+		},
+		"invalid update empty podGroupTemplateRef": {
+			oldObj:    mkValidPodGroup(setResourceVersion("1")),
+			updateObj: mkValidPodGroup(setResourceVersion("1"), setEmptyPodGroupTemplateRef()),
+			expectedErrs: field.ErrorList{
+				field.Invalid(field.NewPath("spec", "podGroupTemplateRef"), nil, "field is immutable").WithOrigin("immutable").MarkA… (trimmed)
```
- 1. thockin:103 PR#136589 `pkg/registry/scheduling/podgroup/strategy.go` cos=0.8340 [DIR,BOILER]
  > same note about in-use check
- 2. thockin:91 PR#136589 `pkg/registry/scheduling/podgroup/declarative_validation_test.go` cos=0.8196 [DIR,BOILER]
  > also test nil input
- 3. thockin:86 PR#136589 `pkg/apis/scheduling/validation/validation.go` cos=0.7967 [BOILER]
  > We should not need manual validation for this anymore, right?

#### thockin:608 — PR#130621 `pkg/api/pod/util_test.go` (comment_id 2002157924)
- **Query review:** This test should probably run each case with the gate set to on and off and assert the expected result.  As it is, there's no longer any coverage for the gate being disbled?
```diff
 					},
 				},
 			},
-			expectAllowPodLifecycleSleepActionZeroValue: false,
+			expectAllowPodLifecycleSleepActionZeroValue: true,
```
- 1. thockin:1656 PR#110477 `pkg/registry/core/pod/strategy_test.go` cos=0.7048 
  > Also add 1 case for matching false when explicit false and 1 when context is nil
- 2. thockin:267 PR#134564 `pkg/apis/core/validation/validation_test.go` cos=0.6961 
  > and for too long?
- 3. thockin:754 PR#128407 `pkg/apis/core/v1/defaults_test.go` cos=0.6718 
  > hard to read :)

#### thockin:782 — PR#128240 `pkg/apis/resource/types.go` (comment_id 1825100991)
- **Query review:** Most places say "IPAddresses" or just "IPs" - is there any sort of address we expect to put here that is not an IP?
```diff
+}
+
+// NetworkDeviceData provides network-related details for the allocated device.
+// This information may be filled by drivers or other components to configure
+// or identify the device within a network context.
+type NetworkDeviceData struct {
+	// InterfaceName specifies the name of the network interface associated with
+	// the allocated device. This might be the name of a physical or virtual
+	// network interface.
+	//
+	// +optional
+	InterfaceName string
+
+	// Addresses lists the network addresses assigned to the device's network interface.
+	// This can include both IPv4 and IPv… (trimmed)
```
- 1. thockin:1684 PR#115075 `pkg/apis/networking/types.go` cos=0.6748 
  > Do we really need UID?
- 2. thockin:1402 PR#116516 `pkg/apis/networking/types.go` cos=0.6599 
  > Comment on format?  E.g. "This can be in any valid IPv6 syntax."
- 3. thockin:1912 PR#109090 `pkg/apis/networking/types.go` cos=0.6568 
  > """
...is 4 (16 IPs).
"""

#### thockin:903 — PR#116429 `pkg/apis/core/types.go` (comment_id 1131800085)
- **Query review:** Why do we need this in Ephemeral?  It's not actually usable here, right?
```diff
 	// +featureGate=InPlacePodVerticalScaling
 	// +optional
 	ResizePolicy []ContainerResizePolicy
+	// Restart policy for the container.
+	// +featureGate=SidecarContainers
+	// +optional
+	RestartPolicy *RestartPolicy
```
- 1. thockin:1204 PR#102884 `staging/src/k8s.io/api/core/v1/types.go` cos=0.6674 
  > ResourceResizePolicy ?  It is for a single resource.
- 2. thockin:1213 PR#102884 `pkg/apis/core/v1/defaults.go` cos=0.6380 
  > I'm confused - don't you want to loop over the defined resources in the pod, rather than the policies?  This loop doesn't make sense to me.
- 3. thockin:1221 PR#102884 `pkg/apis/core/types.go` cos=0.6336 [DIR]
  > the small renaming we discussed will be a followup?

#### thockin:962 — PR#109706 `staging/src/k8s.io/cloud-provider/controllers/service/controller_test.go` (comment_id 931631137)
- **Query review:** I dislike that this re-creates "s0", "s1", etc with args that have to match the above.  Can we just reference them directly?
```diff
+			stateChanges: []stateChanges{
+				{
+					nodes: []*v1.Node{node1, node2, node3},
+				},
+			},
+			expectedUpdateCalls: []fakecloud.UpdateBalancerCall{},
+		},
+		{
+			desc:         "1 new node gets added",
+			initialState: []*v1.Node{node1, node2},
+			stateChanges: []stateChanges{
+				{
+					nodes: []*v1.Node{node1, node2, node3},
+				},
+			},
+			expectedUpdateCalls: []fakecloud.UpdateBalancerCall{
+				{Service: newETPLocalService("s0", "777", v1.ServiceTypeLoadBalancer), Hosts: []*v1.Node{node1, node2, node3}},
```
- 1. thockin:1368 PR#121091 `staging/src/k8s.io/cloud-provider/controllers/service/controller_test.go` cos=0.8095 [DIR]
  > Comment why this last is duplicated and why we don't see service1 with all 3 ?
- 2. thockin:1639 PR#116277 `staging/src/k8s.io/cloud-provider/controllers/service/controller_test.go` cos=0.6867 [DIR]
  > For later cleanup -= I think this is testing an impossible case.
- 3. thockin:1367 PR#121091 `staging/src/k8s.io/cloud-provider/controllers/service/controller_test.go` cos=0.6688 [DIR]
  > comment on what objects is?

#### thockin:1289 — PR#113245 `pkg/api/service/warnings.go` (comment_id 1218646053)
- **Query review:** This is more contentious - can you move it to a different PR?  In this case, what they are doing is VALID and makes sense, but has risk.  We should discuss with sig-net whether to warn (annoying) or not.
```diff
 		warnings = append(warnings, getWarningsForIP(field.NewPath("spec").Child("loadBalancerIP"), service.Spec.LoadBalancerIP)...)
 	}
 
+	// warning on duplicate ports with different protocols
+	if len(service.Spec.Ports) > 1 {
+		warnings = append(warnings, getDupPortsWarningsForService(service.Spec.Ports)...)
+	}
+
 	for i, cidr := range service.Spec.LoadBalancerSourceRanges {
 		warnings = append(warnings, getWarningsForCIDR(field.NewPath("spec").Child("loadBalancerSourceRanges").Index(i), cidr)...)
 	}
 
 	return warnings
 }
 
+// warning if there are duplicate ports no matter if the protoco… (trimmed)
```
- 1. thockin:356 PR#97058 `pkg/apis/networking/validation/validation.go` cos=0.6490 
  > ```
"may not be specified when `port` is not specified"
```
- 2. thockin:354 PR#97058 `pkg/apis/networking/validation/validation.go` cos=0.6464 
  > validation errors already include the field name (that is what `port.Path.Child("endPort")` is for) - don't repeat it.
- 3. thockin:355 PR#97058 `pkg/apis/networking/validation/validation.go` cos=0.6451 
  > ```
"may not be specified when `port` is non-numeric"
```

#### thockin:1524 — PR#118895 `pkg/apis/core/types.go` (comment_id 1258429762)
- **Query review:** I think this is TOO specific.  kube-proxy is not a given on every installation (there are at least 3 other full implementations).  I think we should say something more abstract like:

"""
IPMode specifies how the load-balancer IP behaves, and may only be specified when the ip field is specified.  Setting this to "VIP" indicates that traffic is delivered to the node with the destination set to t…
```diff
 	// +optional
 	Hostname string
 
+	// IPMode specifies how the load-balancer IP behaves.
```
- 1. thockin:1352 PR#121912 `staging/src/k8s.io/apiserver/pkg/cel/library/ip.go` cos=0.5181 [BOILER]
  > "...when the string used to create the IP Address was..." ?
- 2. thockin:480 PR#132214 `pkg/api/service/warnings.go` cos=0.5142 
  > Since these are user-facing I will pick at the words:

"The" is not needed.

"is not supported" is not correct - "is ignored"

"when the service is set as headless" -> "for headless services"
- 3. thockin:1912 PR#109090 `pkg/apis/networking/types.go` cos=0.5055 
  > """
...is 4 (16 IPs).
"""

#### thockin:1750 — PR#114930 `pkg/registry/batch/job/strategy.go` (comment_id 1119133413)
- **Query review:** just to be sure I follow - existing Jobs, which have the "naked" label also have a selector which selects that label, so there's no risk of abandonment here, right?  New instances of Job will have the new label and the new selector, IIUC?
```diff
 	if obj.Spec.Selector.MatchLabels == nil {
 		obj.Spec.Selector.MatchLabels = make(map[string]string)
 	}
-	if _, found := obj.Spec.Selector.MatchLabels["controller-uid"]; !found {
-		obj.Spec.Selector.MatchLabels["controller-uid"] = string(obj.ObjectMeta.UID)
+
+	if _, found := obj.Spec.Selector.MatchLabels[batch.ControllerUidLabel]; !found {
```
- 1. thockin:1482 PR#115100 `pkg/controller/lookup_cache.go` cos=0.4930 
  > Do we need any of the ` MatchingCache` type?  It seems not, by grepping.  I think you can kill off the type and the methods.
- 2. thockin:1326 PR#111023 `pkg/registry/resource/podscheduling/strategy.go` cos=0.4918 [BOILER]
  > PodScheduling?
- 3. thockin:1252 PR#122541 `pkg/registry/core/service/strategy.go` cos=0.4859 
  > Only used in the test - define it there?

### ezyang — 10 held-out queries

#### ezyang:47 — PR#64104 `tools/codegen/dest/lazy_ir.py` (comment_id 698677100)
- **Query review:** ~~A thought: in the old code structure in XLA, we generate functions which take Value/ValueList as arguments and then call into them. I'm not sure this is really necessary: couldn't we directly pass Tensors into the generated IR construction code, and just have the IR construction convert the Tensor into a Value? Then you wouldn't need `process_ir_types` below.~~ Actually you still need it for the…
```diff
+                                 gets_generated_out_inplace_wrapper)
+from tools.codegen.api.types import (BaseTy, BaseCppType, BaseCType, OptionalCType,
+                                     Binding, ConstRefCType, NamedCType,
+                                     CppSignature, CppSignatureGroup,
+                                     Expr, MutRefCType, kernel_signature,
+                                     DispatcherSignature)
+import tools.codegen.api.meta as meta
+import tools.codegen.api.cpp as cpp
+import tools.codegen.api.structured as structured
+from tools.codegen.api.translate impor… (trimmed)
```
- 1. ezyang:69 PR#89595 `torchgen/executorch/api/types/signatures.py` cos=0.5872 
  > don't call this cpp. Call it executorch.
- 2. ezyang:72 PR#89595 `torchgen/executorch/api/cpp.py` cos=0.5818 
  > You don't actually want to implement SymInt directly right, this should devolve to int.
- 3. ezyang:66 PR#89595 `torchgen/executorch/api/cpp.py` cos=0.5654 
  > this is worth a comment

#### ezyang:89 — PR#7677 `torch/csrc/autograd/anomaly_mode.h` (comment_id 193277201)
- **Query review:** Is there a way to make this code Python agnostic? It's nearly so, except for the store/print stack functionality. We're trying to expunge all of the `NO_PYTHON` bits from the codebase.
```diff
+#pragma once
+
+#include "torch/csrc/utils/python_stub.h"
```
- 1. ezyang:594 PR#3866 `torch/csrc/jit/interpreter.cpp` cos=0.7534 
  > Second time you included it!
- 2. ezyang:381 PR#7970 `torch/csrc/autograd/python_engine.cpp` cos=0.6505 [DIR]
  > Avoiding the static initializer this way seems legit.
- 3. ezyang:112 PR#3291 `torch/autograd/variable.py` cos=0.6112 
  > What was the reason for this?

#### ezyang:159 — PR#4695 `torch/csrc/jit/passes/onnx/peephole.cpp` (comment_id 163433042)
- **Query review:** I'm not sure this transformation is sound. Can you give an example trace where it's necessary to walk across inputs to successfully eliminate the packing?
```diff
   }
 }
 
+void eliminatePackedSequence(std::shared_ptr<Graph>& graph) {
+  for (auto it = graph->begin(); it != graph->end(); ++it) {
+    auto* n = *it;
+
+    // for any PadPacked, if we can trace the PackedSequence back to
```
- 1. ezyang:534 PR#4982 `torch/csrc/jit/ir.cpp` cos=0.6195 
  > This will need updating with the blocks PR right?
- 2. ezyang:606 PR#3705 `torch/csrc/jit/passes/graph_fuser.cpp` cos=0.5970 
  > This looks semantically different, so I just wanted to make sure this was intended.
- 3. ezyang:521 PR#4626 `torch/csrc/jit/export.cpp` cos=0.5908 
  > I think this comment needs to be elaborated, because we had a pretty long discussion about what is going on here, and it needs to be clear that this code is responsible for eliminating the correspondence between `state_dict` and the parameters of an exported ONNX model, so basica…

#### ezyang:173 — PR#4695 `torch/nn/utils/rnn.py` (comment_id 166442704)
- **Query review:** Here is where the BC code if a user feeds us a `list` as `batch_sizes` should live; we should generate a Variable LongTensor on the fly in that case.
```diff
         padding_value (float, optional): values for padded elements.
 
     Returns:
-        Tuple of Variable containing the padded sequence, and a list of lengths
-        of each sequence in the batch.
+        Tuple of Variable containing the padded sequence, and Variable
+        containing the list of lengths of each sequence in the batch.
+
     """
```
- 1. ezyang:645 PR#3117 `torch/autograd/_functions/utils.py` cos=0.6392 
  > Heh :)
- 2. ezyang:602 PR#3489 `torch/onnx/__init__.py` cos=0.5555 
  > It should be OK to unconditionally set `attrs["outputs"]`.
- 3. ezyang:286 PR#7506 `torch/onnx/symbolic.py` cos=0.5469 
  > Hoist the append here to top level.

#### ezyang:223 — PR#5856 `test/test_torch.py` (comment_id 176803733)
- **Query review:** Do the tests need to be this big?
```diff
+
+        # contiguous case
+        _test_real((100,), 1)
+        _test_real((100, 100), 1)
+        _test_real((100, 100), 2)
+        _test_real((90, 80, 60), 2)
+        _test_real((50, 80, 100), 3)
+        _test_real((30, 50, 50, 40), 3)
+
+        _test_complex((100, 2), 1)
+        _test_complex((100, 100, 2), 1)
+        _test_complex((100, 100, 2), 2)
+        _test_complex((90, 80, 60, 2), 2)
+        _test_complex((50, 80, 100, 2), 3)
+        _test_complex((30, 50, 50, 40, 2), 3)
```
- 1. ezyang:279 PR#7394 `torch/onnx/symbolic.py` cos=0.5490 
  > The diff looks fairly repetitive, is there really no duplication going on here?
- 2. ezyang:530 PR#4982 `test/test_jit.py` cos=0.5398 [DIR]
  > Comment here docing params?
- 3. ezyang:312 PR#8641 `test/test_jit.py` cos=0.5289 [DIR]
  > oh? What happened here?

#### ezyang:274 — PR#6856 `torch/nn/functional.py` (comment_id 183469746)
- **Query review:** Isn't this the wrong place to put the kwarg? It should go at the end.
```diff
         target: Tensor of the same shape as input
         weight (Tensor, optional): a manual rescaling weight
                 if provided it's repeated to match input tensor shape
+        pos_weight (Tensor, optional): a weight of positive examples.
```
- 1. ezyang:101 PR#3211 `torch/_utils.py` cos=0.6698 
  > I understand this is a helper function, so the naming/docs are not as important (i.e., "read the source code if you actually want to know what it does"). However, for future reference when writing docstrings, I'd recommend adding: (1) the precise types of all the arguments (e.g.,…
- 2. ezyang:649 PR#3084 `torch/autograd/_functions/basic_ops.py` cos=0.6457 
  > This is not what we want long term and it should be documented accordingly (with this code, if you add a constant n to a 100 x 100 tensor, ONNX will serialize a size 100 1-dim tensor; not great.)  In hopefully not too long, we'll serialize this as a scalar getting rid of this goo…
- 3. ezyang:118 PR#3291 `torch/csrc/jit/ir.cpp` cos=0.6362 
  > Nice!

#### ezyang:353 — PR#8337 `aten/src/TH/THTensor.hpp` (comment_id 194591700)
- **Query review:** Thanks for the comment, much appreciated :)
```diff
     inline T * unsafe_data() const {
       return storage->unsafe_data<T>() + storageOffset;
     }
+
+    // NOTE: this returns the "old" TH dimension view where no dimensions represents an empty tensor.
+    // There will be a dim() function that gives the new view that supports 0-sized dimensions.
```
- 1. ezyang:309 PR#8883 `aten/src/TH/THTensor.cpp` cos=0.6718 [DIR]
  > Is this comment still valid? No mention of `view_size`.
- 2. ezyang:345 PR#8468 `aten/src/TH/generic/THTensor.h` cos=0.6609 
  > Oh, here's the comment.
- 3. ezyang:346 PR#8468 `aten/src/THS/generic/THSTensor.hpp` cos=0.6072 
  > Yep, I'm on it :)

#### ezyang:446 — PR#6016 `torch/csrc/jit/export.h` (comment_id 177451361)
- **Query review:** This needs a comment explaining what the string keys and the string values mean. And also more generally what this is used for.
```diff
 
 namespace torch { namespace jit {
 
-std::string ExportGraph(const std::shared_ptr<Graph>& graph,
-                        const std::vector<at::Tensor> & initializers,
-                        int64_t onnx_opset_version);
+using RawDataExportMap = std::unordered_map<std::string, std::string>;
```
- 1. ezyang:254 PR#6392 `torch/csrc/jit/export.cpp` cos=0.8190 [DIR]
  > What does `export_raw_ir` mean? (You say it in your PR description, but not in the code ;)
- 2. ezyang:425 PR#6924 `torch/csrc/jit/export.cpp` cos=0.7248 [DIR]
  > I'm not sure what's going on here.
- 3. ezyang:470 PR#5654 `torch/csrc/jit/export.cpp` cos=0.7057 [DIR]
  > Should this go somewhere more well-known to encourage more use of error reporting?

#### ezyang:488 — PR#4786 `torch/csrc/autograd/python_function.cpp` (comment_id 166468451)
- **Query review:** Paging @apaszke
```diff
   // tracing_state->in_eval_subgraph (it's always false, because they are never part of backward
   // subgraphs AND we don't even materialize the forward function).
   if (!passes_state_transparently) {
+    // TODO: sgross and ezyang don't know if this is right
```
- 1. ezyang:420 PR#6873 `torch/csrc/autograd/function.cpp` cos=0.6498 [DIR]
  > Yay comments!!!
- 2. ezyang:530 PR#4982 `test/test_jit.py` cos=0.6132 
  > Comment here docing params?
- 3. ezyang:539 PR#4982 `torch/csrc/jit/graph_executor.cpp` cos=0.6056 
  > interpreter

#### ezyang:554 — PR#4883 `tools/autograd/gen_variable_type.py` (comment_id 164461939)
- **Query review:** Can someone explain to me where we are getting these names from? Why aren't we using names consistent with the public facing PyTorch API?
```diff
             return '_opt'
         elif dynamic_type == 'IndexTensor':
             return '_long'
+        elif dynamic_type == 'IntegerTensor':
```
- 1. ezyang:307 PR#7869 `tools/autograd/gen_variable_factories.py` cos=0.6110 [DIR]
  > I'm skeptical. Why are you "creating a variable right after"?
- 2. ezyang:102 PR#1573 `torch/autograd/variable.py` cos=0.5963 
  > So, the reason that this is specific to Byte and Long is that these are the only "indexing" tensor types?
- 3. ezyang:569 PR#4487 `tools/autograd/gen_variable_type.py` cos=0.5226 [DIR]
  > Nice!

