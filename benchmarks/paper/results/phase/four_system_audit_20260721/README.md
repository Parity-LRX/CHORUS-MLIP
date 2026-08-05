# Four-system CHORUS result audit (2026-07-21)

This snapshot contains the remote validation logs for revised benzene,
revised ethanol, revised aspirin, and Cheng liquid water.  Metrics use one
checkpoint-selection rule throughout: select the epoch with minimum validation
loss, then report force and energy MAE from that same epoch.

## Data-quality summary

- Parsed physical training runs: 91
- Complete runs: 87
- Incomplete runs: 4
- Raw logs synchronized: 124 (including status/driver logs)
- The 300-epoch eager campaign and 500-epoch make_fx campaign are separate
  protocols and must not be averaged together.
- The current density-preserving attention operator is distinct from the older
  attention control in the 300-epoch campaign.

Incomplete runs:

1. Ethanol positive gate, seed 20260618: 85/300 epochs.
2. Water diagonal density, seed 20260617: 76/300 epochs.
3. Aspirin full-U1, seed 20260617: 227/500 epochs.
4. Water legacy attention, seed 20260616: 474/500 epochs.

## Primary 300-epoch eager mechanism results

Values are mean plus/minus sample standard deviation.  `n` is the number of
complete seeds, so water and ethanol are not yet balanced three-seed results.

| System | Mode | n | Validation loss | Force MAE | Energy MAE |
|---|---|---:|---:|---:|---:|
| Benzene | full-U1 | 3 | 0.000925 +/- 0.000178 | 0.001874 +/- 0.000064 | 0.000204 +/- 0.000158 |
| Benzene | diagonal j=k | 3 | 0.001563 +/- 0.000409 | 0.002192 +/- 0.000112 | 0.000211 +/- 0.000197 |
| Benzene | legacy attention | 3 | 0.001364 +/- 0.000247 | 0.002363 +/- 0.000154 | 0.000491 +/- 0.000371 |
| Ethanol | full-U1 | 2 | 0.017369 +/- 0.000462 | 0.008460 +/- 0.000240 | 0.000249 +/- 0.000038 |
| Ethanol | diagonal j=k | 2 | 0.019097 +/- 0.000775 | 0.008699 +/- 0.000143 | 0.000749 +/- 0.000233 |
| Ethanol | legacy attention | 2 | 0.023088 +/- 0.000504 | 0.009910 +/- 0.000113 | 0.000994 +/- 0.000112 |
| Aspirin | full-U1 | 3 | 0.058540 +/- 0.001629 | 0.016296 +/- 0.000268 | 0.000859 +/- 0.000642 |
| Aspirin | diagonal j=k | 3 | 0.062127 +/- 0.000978 | 0.016808 +/- 0.000118 | 0.001300 +/- 0.000159 |
| Aspirin | legacy attention | 3 | 0.083969 +/- 0.001373 | 0.019844 +/- 0.000385 | 0.001356 +/- 0.001093 |
| Water | full-U1 | 2 | 0.240903 +/- 0.013987 | 0.027996 +/- 0.001409 | 0.004296 +/- 0.001667 |
| Water | diagonal j=k | 1 | 0.220694 | 0.027816 | 0.001359 |
| Water | legacy attention | 1 | 0.319684 | 0.034374 | 0.008514 |

## Current density-preserving attention, 500 epochs make_fx

| System | n | Validation loss | Force MAE | Energy MAE |
|---|---:|---:|---:|---:|
| Benzene | 3 | 0.001079 +/- 0.000276 | 0.002002 +/- 0.000111 | 0.000040 +/- 0.000017 |
| Ethanol | 3 | 0.019228 +/- 0.001588 | 0.008966 +/- 0.000308 | 0.000242 +/- 0.000110 |
| Aspirin | 3 | 0.068976 +/- 0.005497 | 0.016983 +/- 0.000159 | 0.000415 +/- 0.000048 |
| Water | 3 | 0.207980 +/- 0.003737 | 0.025982 +/- 0.000189 | 0.001418 +/- 0.000057 |

## Evidence boundary

The complete 300-epoch results support full-U1 over diagonal density and the
older attention control on benzene and aspirin, and over both controls on the
two completed ethanol seeds.  Water does not yet establish full-U1 over the
diagonal control: on the only complete paired seed, diagonal force MAE is
slightly lower.  A matched four-system 500-epoch table cannot yet be formed,
because benzene and ethanol lack completed baseline/full-U1/diagonal runs and
water lacks a completed 500-epoch diagonal run.

See `analysis/runs.csv` for every parsed run, `analysis/aggregates_300_eager.csv`
for all seven 300-epoch controls, and `analysis/aggregates_500_makefx.csv` for
the available 500-epoch records.
