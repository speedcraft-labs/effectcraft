# Inconclusive experiment: isolate the 949f8bef solid sampler

Variant: retain `949f8bef`'s solid-sampling WGSL while removing the other inactive sampling branches, on top of shader cleanup commit `d08a240a`. Two alternating GUI exports against upstream completed; both decoded hashes matched. Ratios were 1.044 and 1.012 (n=2; exploratory only). The 10-pair split was stopped while queued behind shared benchmark/GPU locks; this sample is too small to attribute the residual slowdown to `949f8bef`.

Reproduce the variant build from `0a894c84` and use `/jobs/work/speedcraft-labs/keep/249/post-sync/run-export-solid-isolation.sh`. Raw samples: `keep/249/post-sync/export-solid-isolation-candidate-pairs.tsv`.
