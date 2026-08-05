# CHORUS–TECE Transition1x training throughput

Single RTX 4090 D, serial execution, fp32, batch size 16, conservative force
training, and the same reaction-disjoint Transition1x 50k/10k split. Each run
contains 3,125 optimizer steps. These are the deployed model configurations,
not an iso-parameter comparison.

| Model | Acceleration | Parameters | Steady train (steps/s) | Full wall (s) | End-to-end (steps/s) |
|---|---|---:|---:|---:|---:|
| CHORUS | MakeFX, four buckets | 168,634 | **50.30** | 209.04 | **14.95** |
| TECE | eager | 263,361 | 10.10 | 426.35 | 7.33 |
| TECE | CUE on first CGTP layer | 263,361 | 9.97 | 396.21 | 7.89 |
| TECE | AOTI | 263,361 | unsupported | failed at 98.74 | — |

CHORUS is 4.98× faster than TECE eager and 5.05× faster than TECE-CUE in
steady training. The CHORUS steady value is the timestamp slope from steps
201–3001, after all four MakeFX buckets were compiled. The TECE values are the
Lightning progress rates at step 3109, before validation.

The end-to-end numbers include raw-data loading, graph construction, initial
compilation, validation, and checkpoint writing. CHORUS spent roughly 98 s
compiling its four MakeFX buckets in this deliberately short run, but still
finished 2.04× sooner than TECE eager.

The apparently lower full wall time of TECE-CUE than TECE eager is not evidence
of an acceleration: its steady training rate was slightly lower, while its
startup benefited from filesystem cache warmed by the preceding eager run.
CUE only replaces TECE's first-layer O(3) scatter tensor product; its second
SO(2)/attention layer remains unchanged.

TECE AOTI did not produce a throughput result. On the tested master revision it
failed during Lightning's validation sanity check because Dynamo could not
capture the dynamic-shape `aten.bincount.default` operation. This row should be
reported as unsupported, not assigned a zero or extrapolated speed.

Machine-readable values and the exact measurement definitions are in
`transition1x_training_throughput.json`.
