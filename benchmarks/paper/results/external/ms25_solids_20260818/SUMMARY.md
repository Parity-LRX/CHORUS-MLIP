# MS25 solid-state benchmark: audited summary

## Scope and protocol

This benchmark compares 11 configurations on the official MS25 Set1 splits for HEA, Zr-O, and MgO-2x2, with an additional MgO-2x2 to MgO-4x4 size-transfer test. Every training run uses 45,000 optimizer steps, strict float32, no EMA or TF32, and an Energy:Force loss weight of 1:100. A single checkpoint is selected by the minimum full-validation Force MAE and then evaluated on test; test data never select a checkpoint. Zr-O energy offsets are fitted on the training split only.

The collected matrix is complete: 33 unique dataset/model records, 11 Zr-O train-only calibrations, and 11 MgO size-transfer evaluations. The automated audit reports `PASS` with no errors or warnings.

## Main test results

Units are meV/atom for energy and meV/Angstrom for force.

| Dataset | Best Energy MAE | Best Force MAE | Main reading |
|:--|:--|:--|:--|
| HEA | TECE C48: 9.23 | MACE-ICTC C128: 6.74 | The ICTC host is already strongest on force; adding CHORUS is neutral to slightly adverse here. |
| Zr-O | DPA-4 C48: 21.40 | MACE-CHORUS-Persistent: 64.39 | CHORUS gives the strongest force model, while DPA-4 retains a clear energy advantage. |
| MgO-2x2 | DPA-4 C48: 0.04 | DPA-4 C48: 2.56 | DPA-4 leads; CHORUS still improves both host families in-domain. |
| MgO 2x2 to 4x4 | DPA-4 C32: 0.12 | NequIP-SH C84: 16.89 | Size transfer changes the ranking and exposes a NequIP-CHORUS force-generalisation failure. |

## What CHORUS contributes

### MACE-ICTC host

| Dataset | Off F MAE | Final F MAE | Persistent F MAE | Best CHORUS change from Off |
|:--|--:|--:|--:|--:|
| HEA | 6.739 | 6.750 | 6.899 | -0.16% (no gain) |
| Zr-O | 68.118 | 64.801 | 64.389 | +5.47% |
| MgO-2x2 | 3.948 | 3.680 | 3.690 | +6.79% |
| MgO 2x2 to 4x4 | 33.922 | 34.254 | 32.008 | +5.64% |

Final adds 16.3-23.5% parameters over MACE-ICTC across these datasets; Persistent adds 49.7-61.2%. Final is therefore the better compact accuracy/cost choice in-domain, while Persistent is the only MACE variant that improves the MgO size-transfer force error.

### NequIP-SH host

| Dataset | SH F MAE | Final F MAE | Persistent F MAE | Best CHORUS change from SH |
|:--|--:|--:|--:|--:|
| HEA | 12.204 | 9.614 | 9.184 | +24.75% |
| Zr-O | 84.033 | 76.401 | 75.813 | +9.78% |
| MgO-2x2 | 5.177 | 4.543 | 4.635 | +12.25% |
| MgO 2x2 to 4x4 | 16.894 | 37.839 | 37.013 | -119.09% |

The NequIP result is the cleanest in-domain transfer evidence: both CHORUS modes improve Force MAE on all three training domains. That conclusion does not extend to cell-size transfer. On MgO-4x4, both variants more than double the NequIP-SH force error, even though their energy error falls sharply.

## Comparison with external architectures

- MACE-CHORUS Final and Persistent each beat both TECE C36 and C48 on Force MAE for all three in-domain datasets.
- They beat both DPA-4 C32 and C48 on HEA and Zr-O force, but lose on MgO-2x2: two wins out of three datasets.
- Energy has a different ranking. TECE C48 leads HEA, while DPA-4 C48 leads Zr-O and MgO-2x2. The benchmark therefore supports a force-accuracy claim for CHORUS, not universal metric dominance.
- On the force-versus-parameter Pareto frontier, MACE-CHORUS Final and Persistent appear for Zr-O. HEA is instead fronted by DPA-4 C32 and the smaller MACE-ICTC host, and MgO by NequIP-SH, MACE-ICTC, and DPA-4.
- The large gap between MACE-ICTC and MACE-SH is evidence about the host representation, not an incremental CHORUS gain, and must not be attributed to CHORUS.

## Bottom line

Under the unified protocol, each CHORUS mode reduces in-domain Force MAE in five of six host-dataset comparisons. The signal is strongest and most consistent on NequIP, and MACE-CHORUS supplies the best Zr-O force result while outperforming TECE on every in-domain force endpoint. The result is competitive rather than uniformly dominant: the MACE host does not improve on HEA, DPA-4 remains strongest on MgO and on two energy endpoints, and NequIP-CHORUS fails the MgO cell-size force transfer.

The defensible conclusion is that CHORUS is a broadly useful in-domain force-accuracy module whose benefit depends on host and propagation mode. Persistent propagation adds value in the hardest MACE cases, including MgO size transfer, but is not a universal replacement for Final. The size-transfer reversal is the principal open issue before claiming general out-of-domain robustness.

These are point estimates from the declared common run protocol. They establish the observed ranking but do not quantify seed uncertainty.

## Artifacts

- Full tables: `README.md`
- Machine-readable records: `results.json`
- Protocol: `protocol.json`
- Automated data-quality audit: `quality_audit.json`
- Raw source metrics: `raw/`
