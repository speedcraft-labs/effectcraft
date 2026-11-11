# Failed experiment: inactive post-sync WGSL cleanup

Starting fork-point-to-tip GUI short-export comparison on synced main: 20 alternating pairs, fork/upstream 1.0648 (95% paired bootstrap CI 1.0433–1.0775), decoded hashes identical.

Tried removing inactive WGSL branches in `common.wgsl` and `kernels.wgsl` (source changes from retained GPU commits including `7d200932` and `949f8bef`). On `d08a240a`, 10 paired GUI exports against upstream remained slower: ratio 1.0347, 95% CI 1.0023–1.0552; all decoded hashes identical. It recovers about 3 percentage points but remains a regression, below the ~5% typical-payload bar, so do not land.

Reproduce with `/jobs/work/speedcraft-labs/keep/249/post-sync/run-export-shader-deadcode-candidate.sh`; data is in `keep/249/post-sync/export-shader-candidate-pairs.tsv`.
