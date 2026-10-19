# A100 large-graph throughput and memory

Exact 50 neighbors; complete E/F/virial; neural/force FP32 with TF32 off; FP64 virial reduction.
Frozen inference measurements. Original checkpoint/seed/model terms are unchanged.
Memory is PyTorch reserved bytes including the CHORUS graph pool, shown in decimal GB. Cold compilation, reference generation and profiling are outside timing.

| Record | ms | Reserved GB | Original reference | Status |
|---|---:|---:|---|---|
| capacity_forcecsr_v6/final/n32768/default.json | — | — | not passed | FAIL |
| capacity_forcecsr_v6/final/n32768/none_forcecsr_tiles_65536.json | — | — | not passed | FAIL |
| capacity_forcecsr_v6/final/n32768/persistent_distance_destination_forcecsr_tiles_65536.json | — | — | not passed | FAIL |
| capacity_forcecsr_v6/final/n65536/default.json | — | — | not passed | OOM |
| capacity_forcecsr_v6/final/n65536/none_forcecsr_tiles_65536.json | — | — | not passed | OOM |
| capacity_forcecsr_v6/final/n65536/persistent_distance_destination_forcecsr_tiles_65536.json | — | — | not passed | FAIL |
| capacity_public_v7/persistent/n100000/default.json | — | — | not passed | OOM |
| capacity_public_v7/persistent/n100000/dpa4_full.json | — | — | not passed | OOM |
| capacity_public_v7/persistent/n100000/dpa4_lower.json | — | — | not passed | OOM |
| capacity_public_v7/persistent/n100000/public_destination_tiles_131072.json | 872.166 | 26.523 | full size | PASS |
| capacity_public_v7/persistent/n100000/public_source_tiles_131072.json | 1018.621 | 26.523 | full size | PASS |
| capacity_public_v7/persistent/n131072/default.json | — | — | not passed | OOM |
| capacity_public_v7/persistent/n131072/public_destination_tiles_131072.json | 1150.643 | 34.664 | full size | PASS |
| capacity_public_v7/persistent/n131072/public_source_tiles_131072.json | 1342.112 | 34.664 | full size | PASS |
| capacity_v4/final/n32768/default.json | 202.490 | 29.993 | full size | PASS |
| capacity_v4/final/n32768/dpa4_full.json | 414.278 | 41.054 | 1024 atoms only | PASS |
| capacity_v4/final/n32768/dpa4_lower.json | 409.709 | 40.943 | 1024 atoms only | PASS |
| capacity_v4/final/n32768/jvp_tiles_131072.json | — | — | not passed | FAIL |
| capacity_v4/final/n32768/jvp_tiles_65536.json | — | — | not passed | FAIL |
| capacity_v4/final/n65536/default.json | — | — | not passed | OOM |
| capacity_v4/final/n65536/dpa4_full.json | — | — | not passed | OOM |
| capacity_v4/final/n65536/dpa4_lower.json | — | — | not passed | OOM |
| capacity_v4/final/n65536/jvp_tiles_131072.json | — | — | not passed | FAIL |
| capacity_v4/final/n65536/jvp_tiles_65536.json | — | — | not passed | FAIL |
| capacity_v4/persistent/n32768/default.json | 233.654 | 31.896 | full size | PASS |
| capacity_v4/persistent/n32768/dpa4_full.json | 409.585 | 41.054 | 1024 atoms only | PASS |
| capacity_v4/persistent/n32768/dpa4_lower.json | 405.371 | 40.943 | 1024 atoms only | PASS |
| capacity_v4/persistent/n32768/jvp_tiles_131072.json | 276.297 | 10.215 | full size | PASS |
| capacity_v4/persistent/n32768/jvp_tiles_65536.json | 281.533 | 8.739 | full size | PASS |
| destination_v3_n10000/default.json | 72.130 | 9.651 | full size | PASS |
| destination_v3_n10000/jvp_tiles_131072.json | 83.548 | 5.067 | full size | PASS |
| destination_v3_n10000/jvp_tiles_262144.json | 83.055 | 7.288 | full size | PASS |
| destination_v3_n10000/jvp_tiles_32768.json | 88.397 | 3.360 | full size | PASS |
| destination_v3_n10000/jvp_tiles_65536.json | 85.651 | 3.959 | full size | PASS |
| destination_v3_n100000/default.json | — | — | not passed | OOM |
| destination_v3_n100000/jvp_tiles_131072.json | 871.540 | 26.523 | full size | PASS |
| destination_v3_n100000/jvp_tiles_65536.json | 907.671 | 26.523 | full size | PASS |
| destination_v3_n32768/default.json | 233.270 | 31.896 | full size | PASS |
| destination_v3_n32768/jvp_tiles_131072.json | 276.561 | 10.215 | full size | PASS |
| destination_v3_n32768/jvp_tiles_262144.json | 274.314 | 12.715 | full size | PASS |
| destination_v3_n32768/jvp_tiles_32768.json | 292.925 | 8.739 | full size | PASS |
| destination_v3_n32768/jvp_tiles_65536.json | 281.593 | 8.739 | full size | PASS |
| destination_v3_n65536/default.json | — | — | not passed | OOM |
| destination_v3_n65536/jvp_tiles_131072.json | 555.796 | 17.555 | full size | PASS |
| destination_v3_n65536/jvp_tiles_65536.json | 572.924 | 17.555 | full size | PASS |
| final_precision_v5_trained_32768/final_default_n32768.json | — | — | not passed | FAIL |
| final_precision_v5_trained_32768/final_jvp_n32768.json | — | — | not passed | FAIL |
| final_precision_v5_trained_32768/final_jvp_tiles_n32768.json | — | — | not passed | FAIL |
| final_precision_v5_trained_32768/final_tiles_n32768.json | — | — | not passed | FAIL |
| inline_v5/persistent/n10000/default.json | 71.680 | 9.651 | full size | PASS |
| inline_v5/persistent/n10000/dpa4_full.json | 128.736 | 12.593 | full size | PASS |
| inline_v5/persistent/n10000/dpa4_lower.json | 125.437 | 12.560 | full size | PASS |
| inline_v5/persistent/n10000/persistent_distance_destination_tiles_65536.json | 85.634 | 3.959 | full size | PASS |
| inline_v5/persistent/n10000/persistent_distance_inline_forward_tiles_65536.json | 128.082 | 3.731 | full size | PASS |
| inline_v5/persistent/n10000/persistent_distance_inline_tiles_65536.json | 188.415 | 2.993 | full size | PASS |
| inline_v5/persistent/n32768/default.json | 237.268 | 31.896 | full size | PASS |
| inline_v5/persistent/n32768/dpa4_full.json | 416.543 | 41.054 | 1024 atoms only | PASS |
| inline_v5/persistent/n32768/dpa4_lower.json | 410.403 | 40.943 | 1024 atoms only | PASS |
| inline_v5/persistent/n32768/persistent_distance_destination_tiles_65536.json | 284.902 | 8.739 | full size | PASS |
| inline_v5/persistent/n32768/persistent_distance_inline_forward_tiles_65536.json | 422.861 | 8.741 | full size | PASS |
| inline_v5/persistent/n32768/persistent_distance_inline_tiles_65536.json | 622.139 | 8.741 | full size | PASS |
| tile_v2_n10000/default.json | 72.006 | 9.651 | full size | PASS |
| tile_v2_n10000/jvp_tiles_131072.json | 96.991 | 5.067 | full size | PASS |
| tile_v2_n10000/jvp_tiles_262144.json | 95.995 | 7.288 | full size | PASS |
| tile_v2_n10000/jvp_tiles_32768.json | 101.462 | 3.360 | full size | PASS |
| tile_v2_n10000/jvp_tiles_65536.json | 98.546 | 3.959 | full size | PASS |
| tile_v2_n32768/default.json | 232.986 | 31.896 | full size | PASS |
| tile_v2_n32768/jvp_tiles_131072.json | 322.418 | 10.215 | full size | PASS |
| tile_v2_n32768/jvp_tiles_262144.json | 318.987 | 12.715 | full size | PASS |
| tile_v2_n32768/jvp_tiles_32768.json | 343.379 | 8.739 | full size | PASS |
| tile_v2_n32768/jvp_tiles_65536.json | 329.453 | 8.739 | full size | PASS |

## Same-card source/destination radial controls

| Atoms | Source CSR ms | Destination ms | Throughput ratio | Time reduction | Reserved GB |
|---:|---:|---:|---:|---:|---:|
| 100000 | 1018.621 | 872.166 | 1.168 | 14.38% | 26.523 |
| 131072 | 1342.112 | 1150.643 | 1.166 | 14.27% | 34.664 |

## Regression and qualification

- tests_complete_v7.xml: 218 tests, 0 failures, 0 skipped.
- tests_destination_v2.xml: 70 tests, 0 failures, 0 skipped.
- tests_forcecsr_v6.xml: 88 tests, 0 failures, 0 skipped.
- tests_inline_v4.xml: 84 tests, 0 failures, 0 skipped.
- tests_v1.xml: 63 tests, 0 failures, 0 skipped.
- Public destination path, held-out checkpoint E/F: PASS.
- Final large-size near-zero-force failures remain unqualified; repeated default execution also crossed the component threshold.
- Independent local original-formula closures are diagnostic only and do not replace the registered oracle or qualify failed points.

DPA rows with a smaller original audit retain full-size full/lower API parity and finite outputs; they do not claim a full-size unaccelerated oracle.
The actual compile/Triton dispatch, source snapshot, same-allocation ratios, failure stage and regression counts are retained in capacity_summary.json.
Failed/OOM records are retained. A smaller-memory path is not automatically a higher-throughput path, and no experimental route is promoted by this report.
