# ReviewerBrain — Step 1 Dataset Audit

## Reviewer: thockin
File: `data/thockin_kubernetes_training.jsonl`

- (a) PRs total: **839**
- (b) review_comments total (flattened): **3405**
- (c) is_code_related=True: **3393** (99.6%)
- (d) led_to_code_change=True: **2864** (84.1%)
- (e) replies (in_reply_to_id set): **1305** vs fresh top-level: **2100** (38.3% replies)
- (f) docs/config files: **56** of 3405 = **1.6%** (<=20%, ok)
- (g) body length chars — min=2  max=12813  median=96  avg=219; empty bodies: 0
- (h) non-empty follow_up_patch: **2852** (83.8%)
- duplicates on (pr_number, path, body): 46 extra rows in 29 duplicate groups
- comment created_at range: 2018-02-23T00:19:38Z → 2026-07-23T15:50:03Z
- comments per PR: min=0 max=223 avg=4.1
- null/missing fields: {'in_reply_to_id': 2100, 'follow_up_patch': 541}

### (f) Full extension breakdown

| ext | count | % | class |
|---|---|---|---|
| .go | 3118 | 91.6% | source/other |
| .sh | 200 | 5.9% | source/other |
| .yaml | 38 | 1.1% | docs/config |
| .mod | 12 | 0.4% | source/other |
| <no-ext> | 9 | 0.3% | source/other |
| .md | 8 | 0.2% | docs/config |
| .proto | 5 | 0.1% | docs/config |
| .list | 4 | 0.1% | source/other |
| .in | 2 | 0.1% | source/other |
| .sum | 2 | 0.1% | docs/config |
| .json | 2 | 0.1% | docs/config |
| .py | 2 | 0.1% | source/other |
| .work | 1 | 0.0% | source/other |
| .pb | 1 | 0.0% | source/other |
| .conf | 1 | 0.0% | docs/config |

### Filtering funnel (Step 2 candidate filters, applied in order)

| stage | kept | dropped |
|---|---|---|
| len>=10 | 3273 | -132 |
| not-suggestion | 3266 | -7 |
| fresh-only | 2012 | -1254 |
| source-only | 1984 | -28 |
| deduped | 1962 | -22 |

### (j) Token estimate

- After basic filters: **1962 examples**, ~**1,114,336 tokens** total (~568 tokens/example avg)
- Unfiltered reference: ~1,888,929 tokens

### (i) 5 random full examples (fresh, non-empty hunk+body)

#### Example 1 — PR #96120 “KEP 2258: add node log query”
- path: `staging/src/k8s.io/api/node/v1alpha1/types.go`  line: 165  is_code_related: True  led_to_code_change: False

**diff_hunk:**
```diff
@@ -114,3 +114,66 @@ type RuntimeClassList struct {
 	// Items is a list of schema objects.
 	Items []RuntimeClass `json:"items" protobuf:"bytes,2,rep,name=items"`
 }
+
+// +k8s:conversion-gen:explicit-from=net/url.Values
+// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
+
+// NodeLogOptions is the query options for a Node's logs REST call.
+type NodeLogOptions struct {
+	metav1.TypeMeta `json:",inline"`
+
+	// SinceTime is an RFC3339 timestamp from which to show logs. /If this value precedes the time a log file or service
+	// was created, only logs since the log file or service was created will be returned. If this value is in the
+	// future, no logs will be returned.
+	// +optional
+	SinceTime *metav1.Time `json:"sinceTime,omitempty" protobuf:"bytes,1,opt,name=sinceTime"`
+
+	// UntilTime is an RFC3339 timestamp until which to show logs.
+	// +optional
+	UntilTime *metav1.Time `json:"untilTime,omitempty" protobuf:"bytes,2,opt,name=untilTime"`
+
+	// TailLines if set, return up to this many lines (not more than 100k) from the end of the log. Only applies to
+	// service logs.
+	// +optional
+	TailLines *int64 `json:"tailLines,omitempty" protobuf:"varint,3,opt,name=tailLines"`
+
+	// Path is used to retrieve the specified path within the node's /var/logs/ folder. The 'journal' value will allow
+	// querying the journal on supported operating systems.
+	// +optional
+	Path string `json:"path,omitempty" protobuf:"bytes,4,opt,name=path"`
+
+	// Format is used to display Linux journal logs in an alternate format (short, cat, json, short-unix). This only
+	// applies to Linux service logs.
+	// +optional
+	Format string `json:"format,omitempty" protobuf:"bytes,5,opt,name=format"`
+
+	// Pattern filters log entries by the provided regex pattern. Only applies to service logs.
+	// +optional
+	Pattern string `json:"pattern,omitempty" protobuf:"bytes,6,opt,name=pattern"`
+
+	// PatternCaseSensitive is used to apply case sensitivity or not to the pattern.
+	// +optional
+	PatternCaseSensitive bool `json:"patternCaseSensitive,omitempty" protobuf:"varint,7,opt,name=patternCaseSensitive"`
+
+	// Boot show messages from a specific boot. Allowed values are [-100, 0] and passing invalid boot offset will fail
+	// retrieving logs. Only applies to Linux service logs.
+	// +optional
+	Boot *int64 `json:"boot,omitempty" protobuf:"varint,8,opt,name=boot"`
+
+	// Services are the specified service(s) to return log entries from. Only applies to Linux journal or WinEvent
+	// Application provider logs. If a native service log is not found an attempt will be made to get logs from
+	// /var/service/service.log or /var/log/service/service.log or /var/log/service*INFO or
```

**body (target output):**
```
I would not document the whole heuristic here but say "the server implements a heuristic, for example..."
```

#### Example 2 — PR #132558 “KEP-4762: Allows setting any FQDN as the pod's hostname”
- path: `staging/src/k8s.io/api/core/v1/types.go`  line: 4209  is_code_related: True  led_to_code_change: True

**diff_hunk:**
```diff
@@ -4206,6 +4206,16 @@ type PodSpec struct {
 	// +featureGate=PodLevelResources
 	// +optional
 	Resources *ResourceRequirements `json:"resources,omitempty" protobuf:"bytes,40,opt,name=resources"`
+	// hostnameOverride specifies an explicit override for the Pod's hostname.
```

**body (target output):**
```
"...for the Pod's hostname as perceived by the pod."

I want to be clear that this has nothing to do with DNS.  Maybe even say that explicitly?
```

#### Example 3 — PR #132626 “KEP-3721: Support for env files”
- path: `staging/src/k8s.io/api/core/v1/types.go`  line: 2521  is_code_related: True  led_to_code_change: True

**diff_hunk:**
```diff
@@ -2386,6 +2386,35 @@ type EnvVarSource struct {
 	// Selects a key of a secret in the pod's namespace
 	// +optional
 	SecretKeyRef *SecretKeySelector `json:"secretKeyRef,omitempty" protobuf:"bytes,4,opt,name=secretKeyRef"`
+	// FileKeyRef selects a key of the env file.
+	// Requires the EnvFiles feature gate to be enabled.
+	//
+	// +featureGate=EnvFiles
+	// +optional
+	FileKeyRef *FileKeySelector `json:"fileKeyRef,omitempty" protobuf:"bytes,5,opt,name=fileKeyRef"`
+}
+
+// FileKeySelector selects a key of the env file.
+// +structType=atomic
+type FileKeySelector struct {
+	// The name of the volume mount containing the env file.
+	VolumeName string `json:"volumeName" protobuf:"bytes,1,opt,name=volumeName"`
+	// The path within the volume from which to select the file.
+	// May be specified as either an absolute path or relative to the volume.
+	Path string `json:"path" protobuf:"bytes,2,opt,name=path"`
+	// The key within the env file. An invalid key will prevent the pod from starting.
+	// During Alpha stage of the EnvFiles feature gate, the key size is limited to 128 characters.
+	Key string `json:"key" protobuf:"bytes,3,opt,name=key"`
```

**body (target output):**
```
Is there any other validation?  Can it be ANY characters?
```

#### Example 4 — PR #123385 “Allow almost all printable ASCII characters in environment variables”
- path: `staging/src/k8s.io/apimachinery/pkg/util/validation/validation.go`  line: 443  is_code_related: True  led_to_code_change: True

**diff_hunk:**
```diff
@@ -431,6 +435,26 @@ func IsEnvVarName(value string) []string {
 	return errs
 }
 
+// IsRelaxedEnvVarName Tests if a string is a valid environment variable name that is strictly validated.
+func IsRelaxedEnvVarName(value string) []string {
+	var errs []string
+
+	if len(value) == 0 {
+		errs = append(errs, RegexError(RelaxedEnvVarNameFmtErrMsg, RelaxedEnvVarNameFmt, "my.env-name", "MY_ENV.NAME", "MyEnvName1"))
```

**body (target output):**
```
EmptyError()
```

#### Example 5 — PR #59286 “Delete stale UDP conntrack entries that use hostPort”
- path: `pkg/kubelet/network/hostport/hostport_manager.go`  line: 168  is_code_related: True  led_to_code_change: True

**diff_hunk:**
```diff
@@ -150,6 +165,21 @@ func (hm *hostportManager) Add(id string, podPortMapping *PodPortMapping, natInt
 		// clean up opened host port if encounter any error
 		return utilerrors.NewAggregate([]error{err, hm.closeHostports(hostportMappings)})
 	}
+	isIpv6 := conntrack.IsIPv6(podPortMapping.IP)
```

**body (target output):**
```
For future work - we should move the IsIPv6() stuff to a more generic place.
```

---

## Reviewer: ezyang
File: `data/ezyang_pytorch_training.jsonl`

- (a) PRs total: **499**
- (b) review_comments total (flattened): **1244**
- (c) is_code_related=True: **1234** (99.2%)
- (d) led_to_code_change=True: **928** (74.6%)
- (e) replies (in_reply_to_id set): **529** vs fresh top-level: **715** (42.5% replies)
- (f) docs/config files: **60** of 1244 = **4.8%** (<=20%, ok)
- (g) body length chars — min=1  max=2991  median=98  avg=162; empty bodies: 0
- (h) non-empty follow_up_patch: **919** (73.9%)
- duplicates on (pr_number, path, body): 4 extra rows in 3 duplicate groups
- comment created_at range: 2017-04-27T00:06:23Z → 2025-01-27T18:57:19Z
- comments per PR: min=0 max=30 avg=2.5
- null/missing fields: {'in_reply_to_id': 715, 'follow_up_patch': 316}

### (f) Full extension breakdown

| ext | count | % | class |
|---|---|---|---|
| .py | 439 | 35.3% | source/other |
| .cpp | 397 | 31.9% | source/other |
| .h | 228 | 18.3% | source/other |
| .cu | 33 | 2.7% | source/other |
| .sh | 30 | 2.4% | source/other |
| .txt | 27 | 2.2% | docs/config |
| .yaml | 17 | 1.4% | docs/config |
| .expect | 12 | 1.0% | source/other |
| .cmake | 10 | 0.8% | source/other |
| .md | 8 | 0.6% | docs/config |
| .patch | 7 | 0.6% | source/other |
| .cuh | 7 | 0.6% | source/other |
| .rst | 7 | 0.6% | docs/config |
| .cwrap | 4 | 0.3% | source/other |
| <no-ext> | 3 | 0.2% | source/other |
| .c | 3 | 0.2% | source/other |
| .hpp | 3 | 0.2% | source/other |
| .in | 2 | 0.2% | source/other |
| .hip | 2 | 0.2% | source/other |
| .cc | 2 | 0.2% | source/other |
| .bak | 1 | 0.1% | source/other |
| .json | 1 | 0.1% | docs/config |
| .options | 1 | 0.1% | source/other |

### Filtering funnel (Step 2 candidate filters, applied in order)

| stage | kept | dropped |
|---|---|---|
| len>=10 | 1192 | -52 |
| not-suggestion | 1192 | -0 |
| fresh-only | 674 | -518 |
| source-only | 643 | -31 |
| deduped | 639 | -4 |

### (j) Token estimate

- After basic filters: **639 examples**, ~**215,224 tokens** total (~337 tokens/example avg)
- Unfiltered reference: ~533,832 tokens

### (i) 5 random full examples (fresh, non-empty hunk+body)

#### Example 1 — PR #8641 “Allow autograd to work even when the shape of values cannot be determined”
- path: `torch/csrc/jit/graph_executor.cpp`  line: 415  is_code_related: True  led_to_code_change: True

**diff_hunk:**
```diff
@@ -409,70 +412,117 @@ struct GraphExecutorImpl {
     return false;
   }
 
-
-  // remove ReplaceIfUndef(v, replacement) nodes that consume inputs with 'v' if
-  // the input is defined, and 'replacement' if it is not.
-  // Note: this is a very limited pass. It looks at undefined inputs,
-  // and cleans up ReplaceIfUndef nodes inserted by autodiff.
+  // propagate undefined information through a gradient graph
```

**body (target output):**
```
Is this our first dataflow analysis! :)
```

#### Example 2 — PR #8300 “[build] remove the use of NO_CUDA”
- path: `.jenkins/pytorch/build-asan.sh`  line: 20  is_code_related: True  led_to_code_change: False

**diff_hunk:**
```diff
@@ -17,5 +17,5 @@ export ASAN_OPTIONS=detect_leaks=0:symbolize=1
 # TODO: Make the ASAN flags a more unified env var
 CC="clang" CXX="clang++" LDSHARED="clang --shared" \
   CFLAGS="-fsanitize=address -shared-libasan" \
-  NO_CUDA=1 DEBUG=1 \
+  USE_CUDA=0 DEBUG=1 \
```

**body (target output):**
```
Yeah, I just want to point out that passing `NO_CUDA=1` as an env var to the setup.py script is considered part of the externally facing interface, so you'll probably want to keep at least one of these to make sure the BC-layer (which will check `NO_CUDA` envvar if `USE_CUDA` is not set) is working.
```

#### Example 3 — PR #8666 “Some 0-sized dimension support, port catArray away from resizeLegacy.”
- path: `aten/src/TH/generic/THTensorMath.cpp`  line: 3594  is_code_related: True  led_to_code_change: True

**diff_hunk:**
```diff
@@ -3584,62 +3584,59 @@ inline void THTensor_(check_shape_except_dim)(THTensor *first, THTensor *second,
 
 void THTensor_(catArray)(THTensor *result, THTensor **inputs, int numInputs, int dimension)
 {
-  // Find a non-empty tensor to record nDims
-  int allEmpty = 1;
-  int nDims = 0;
-  THTensor *notEmptyTensor;
+  // previously, size [0] tensors were the only possible empty tensors; thus, it wasn't possible
+  // to cat empty tensors unless all the other tensors were 1-dimensional, so we allowed these tensors
+  // to be "skipped".  We maintain this behavior for backwards compatibility, but only for this specific
+  // size (i.e. other empty sizes are not skipped).
+  // FIXME: warn if this is the cacse
+  bool allSkipped= true;
+  int64_t nDims = 0;
+  THTensor *notSkippedTensor;
```

**body (target output):**
```
Mention, I suppose, this is a non-owning reference.
```

#### Example 4 — PR #106431 “Fix guarding issues w/ numpy”
- path: `torch/_dynamo/variables/builder.py`  line: 995  is_code_related: True  led_to_code_change: True

**diff_hunk:**
```diff
@@ -964,13 +971,13 @@ def wrap_tensor(self, value: torch.Tensor):
     def wrap_numpy_ndarray(self, value):
         assert isinstance(value, np.ndarray)
 
-        source = self.get_source()
+        source = NumpyTensorSource(self.get_source())
         tensor_value = torch.as_tensor(value)
-
+        tensor_vt = VariableBuilder(self.tx, source)(tensor_value)
         proxy = self.tx.output.root_tracer.create_graph_input(
             re.sub(r"[^a-zA-Z0-9]+", "_", self.name), type(tensor_value)
         )
-        options = {"source": source}
+        options = {"source": source, "guards": tensor_vt.guards}
         numpy_ndarray_variable = wrap_fx_proxy_cls(
```

**body (target output):**
```
It appears these are the substantive changes
```

#### Example 5 — PR #4982 “Initial GraphExecutor Implementation.”
- path: `torch/csrc/jit/interpreter.cpp`  line: 243  is_code_related: True  led_to_code_change: True

**diff_hunk:**
```diff
@@ -218,6 +230,31 @@ Operation getOperation(jit::Node *node) {
     return [](const list_of_retainable & inputs, list_of_retainable & outputs) {
       outputs.push_back(toRetainableSteal(at::Tensor()));
     };
+  IR_ELSEIF(ReplaceIfUndef)
+    return [](const list_of_retainable & inputs, list_of_retainable & outputs) {
+      auto result = inputs[0];
+      //TODO: refcounting stuff here is ugly but TensorTemporary is not
+      //present. Consider whether we
+      // 1. expose tensor temporary here
+      // 2. keep as is
+      // 3. remove all of this retainable stuff anyway since the new
+      // execution paths do not need handle types.
+      // Note that list_of_retainable is painful because it is yet another
+      // list of pointers that require needless copies.
```

**body (target output):**
```
OK, it's true, a stack machine would help here.
```

---

