# ReviewerBrain — Step 2 Cleaning Report

## Extension classification applied (decision #4)

Excluded as documentation/configuration: .adoc, .cfg, .conf, .csv, .htm, .html, .ini, .json, .md, .rst, .svg, .tex, .toml, .tsv, .txt, .xml, .yaml, .yml

Kept as source/build (includes judgment calls to review): `.sh`, `.cmake`, `.expect` (explicitly requested), plus `.proto`, `.mod`, `.sum`, `.lock`, `.patch`, `.bak`, `.cwrap`, `.options`, `.in`, `.list`, `.go`, `.py`, `.cpp`, `.h`, `.hpp`, `.cc`, `.c`, `.cu`, `.cuh`, `.hip`, and any other unlisted extension.

### thockin — extensions seen pre-filter (class: count)

- `.go`: 3118 — kept source/build
- `.sh`: 200 — kept source/build
- `.yaml`: 38 — EXCLUDED docs/config
- `.mod`: 12 — kept source/build
- `.md`: 8 — EXCLUDED docs/config
- `<no-ext>`: 8 — kept source/build
- `.proto`: 5 — kept source/build
- `.list`: 4 — kept source/build
- `.in`: 2 — kept source/build
- `.sum`: 2 — kept source/build
- `.json`: 2 — EXCLUDED docs/config
- `.py`: 2 — kept source/build
- `.gitignore`: 1 — kept source/build
- `.work`: 1 — kept source/build
- `.pb`: 1 — kept source/build
- `.conf`: 1 — EXCLUDED docs/config
- `<no-ext>` paths kept as source: ['build/build-image/Dockerfile', 'build/root/Makefile', 'build/server-image/kubectl/Dockerfile', 'cmd/gotemplate/OWNERS', 'file', 'vendor/OWNERS']

### ezyang — extensions seen pre-filter (class: count)

- `.py`: 439 — kept source/build
- `.cpp`: 397 — kept source/build
- `.h`: 228 — kept source/build
- `.cu`: 33 — kept source/build
- `.sh`: 30 — kept source/build
- `.txt`: 27 — EXCLUDED docs/config
- `.yaml`: 17 — EXCLUDED docs/config
- `.expect`: 12 — kept source/build
- `.cmake`: 10 — kept source/build
- `.md`: 8 — EXCLUDED docs/config
- `.patch`: 7 — kept source/build
- `.cuh`: 7 — kept source/build
- `.rst`: 7 — EXCLUDED docs/config
- `.cwrap`: 4 — kept source/build
- `.c`: 3 — kept source/build
- `.hpp`: 3 — kept source/build
- `.in`: 2 — kept source/build
- `<no-ext>`: 2 — kept source/build
- `.hip`: 2 — kept source/build
- `.cc`: 2 — kept source/build
- `.bak`: 1 — kept source/build
- `.gitmodules`: 1 — kept source/build
- `.json`: 1 — EXCLUDED docs/config
- `.options`: 1 — kept source/build
- `<no-ext>` paths kept as source: ['docker/caffe2/ubuntu-16.04-cuda8-cudnn7-all-options/Dockerfile', 'docs/Makefile']

## Reviewer: thockin (funnel, decision order)

- total PRs: 839 (of which 340 contribute ≥1 final example)
- total comments: 3405
- replies removed: 1305
- body < 5 chars removed: 39
- pure suggestion blocks removed: 6
- non-source files removed: 26
- duplicates removed: 27
- **final examples: 2002** (58.8% of original comments)

Post-filter profile:

- led_to_code_change=True: 1754 (87.6%)
- non-empty follow_up_patch: 1747 (87.3%)
- is_code_related=True: 2002 (100.0%)
- empty diff_hunk (kept, flagged as weak): 7
- empty pr_description: 0

Extension breakdown after filtering:

| ext | count | % |
|---|---|---|
| .go | 1930 | 96.4% |
| .sh | 58 | 2.9% |
| .mod | 5 | 0.2% |
| .list | 4 | 0.2% |
| .sum | 2 | 0.1% |
| <no-ext> | 2 | 0.1% |
| .proto | 1 | 0.0% |

## Reviewer: ezyang (funnel, decision order)

- total PRs: 499 (of which 199 contribute ≥1 final example)
- total comments: 1244
- replies removed: 529
- body < 5 chars removed: 10
- pure suggestion blocks removed: 0
- non-source files removed: 31
- duplicates removed: 4
- **final examples: 670** (53.9% of original comments)

Post-filter profile:

- led_to_code_change=True: 503 (75.1%)
- non-empty follow_up_patch: 503 (75.1%)
- is_code_related=True: 665 (99.3%)
- empty diff_hunk (kept, flagged as weak): 0
- empty pr_description: 30

Extension breakdown after filtering:

| ext | count | % |
|---|---|---|
| .py | 250 | 37.3% |
| .cpp | 235 | 35.1% |
| .h | 106 | 15.8% |
| .cu | 23 | 3.4% |
| .sh | 19 | 2.8% |
| .expect | 9 | 1.3% |
| .cmake | 7 | 1.0% |
| .patch | 6 | 0.9% |
| .hpp | 3 | 0.4% |
| .hip | 2 | 0.3% |
| .cuh | 2 | 0.3% |
| .cc | 2 | 0.3% |
| .bak | 1 | 0.1% |
| .in | 1 | 0.1% |
| .c | 1 | 0.1% |
| .gitmodules | 1 | 0.1% |
| .cwrap | 1 | 0.1% |
| .options | 1 | 0.1% |

## Combined

- thockin: 2002 examples (58.8% retained)
- ezyang: 670 examples (53.9% retained)
- combined_clean.jsonl: **2672 examples** (57.5% of all 4649 original comments)
- combined led_to_code_change=True: 84.5%
- combined non-empty follow_up_patch: 84.2%

