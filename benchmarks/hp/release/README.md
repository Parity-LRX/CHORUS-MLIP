# Paired HP release qualification

Paper versions are unchanged: `chorus-mlip==0.1.0`, `ictc-o3-hp==0.2.12`.
The optional `hp` dependency pins Git revision
`9fdea47eef620013e9ea591f7d90534bb27671c8`, HP API contract 1. A package-version
match alone is not sufficient to select the new operator implementation.

The ordinary CHORUS dependency list, existing extras, CLI names and default
reference policy remain unchanged. The HP package uses separate `ictc-hp-*`
console scripts, retains its original license and `mace_ictc` import namespace,
and is not vendored into CHORUS.

All builds, installation checks and numerical tests run through
`sbatch / GPU-A100 / XHPC-A100`. Task root:
`/public/home/sps-xia/rxlin/chorus_hp_release_20261019`.
The final source snapshot is `source_v3/{chorus,ictc_o3_hp}`; the committed CHORUS
reference before this HP round is saved separately in `reference_before`.
No shared environment or running MD source is modified.

## Final qualification

A100 batch **92645** completed with exit code `0:0` on
`NVIDIA A100-SXM4-40GB`, Python 3.11 / PyTorch `2.7.1+cu128`.

- [Wheel metadata](results/wheel_metadata.json): original package versions,
  unchanged base dependencies/existing extras, exact HP Git dependency and
  non-overlapping console scripts.
- [Reference regression](results/reference_compatibility.json): Final and
  Persistent E/F and force-loss parameter gradients match the preceding
  committed implementation with HP imports blocked. Both force maximum
  absolute errors are `0.0`; checkpoint contents remain unchanged.
- [Installation](results/install_validation.json): `[hp]` fetched the pinned
  HP revision from GitHub; a base installation did not select HP. No new
  dependency issues were introduced. The inherited MACE/e3nn mismatch and
  missing optional MACE dependencies are listed separately in the report.
- [Native regression](results/native_tests.json), [JUnit](results/native_tests.xml):
  **59 passed**, no failures or skips. This packaging check retains all
  compatibility/density cases, both model scopes and representative tests
  for each operator family; 259 historical tuning variants were deselected.
  It runs the installed wheels with CUDA Graph capture/replay and compiler
  cudagraphs prohibited, and TF32 disabled. It is not a new throughput benchmark.

The paired wheels and source distributions built in this batch are retained
locally in `dist/paired-wheels/` and `dist/sdists/` (not committed as binaries),
and remotely in `wheels_source_v3/` and `sdists_source_v3/` under the task root.
Provisional version-bump artifacts are isolated under `dist/superseded-version-bump-*`.

## Run history and checks

Attempt 92642 built both packages and passed metadata/reference checks. Its
offline HP resolution exposed a missing Ninja wheel in the inherited test
environment. The retry stages that ordinary build dependency in the task
wheel directory before offline project resolution; the HP requirement is
retained. The original snapshot and failed-attempt log are preserved.
Attempt 92644 passed installation/reference checks and reached numerical
regressions before being cancelled when the user required preserving the
paper version numbers. Its provisional 0.2.0/0.3.0 artifacts are superseded,
not published releases. The final attempt validates the original version
numbers with the immutable Git dependency instead.

The batch builds both wheels and sdists, then checks:

- Wheel metadata contains the exact HP source pin only under the `hp` extra;
  existing reference dependencies/extras are unchanged and console scripts
  do not overlap.
- A normal wheel installation does not select HP. With `mace_ictc` imports
  blocked, Final/Persistent reference E/F and force-loss parameter gradients
  match the preceding committed implementation using its original state.
- One `pip install 'chorus-mlip[hp]==0.1.0'` selects the local CHORUS wheel
  and fetches the pinned HP commit from GitHub. Pip's installation report
  must record that exact source commit. Conflicting HP requirements are rejected.
- Installed-wheel native operator and whole-network tests execute on A100,
  with CUDA Graph capture/replay prohibited and TF32/AMP disabled. Source
  trees must not shadow either installed package.
- Missing, wrong-version and overwritten legacy namespaces are diagnosed;
  `auto` falls back and `generated` rejects unsupported installations.

`pip check` is recorded for the inherited base environment and the isolated
HP environment separately. Existing base-environment problems are reported,
not repaired in shared storage; any new dependency problem fails qualification.

From a CHORUS checkout, installation is:

```bash
python -m pip install -e '.[hp]'
# or install the CHORUS wheel, which carries the same source dependency:
python -m pip install '/path/to/chorus_mlip-0.1.0-py3-none-any.whl[hp]'
```

This requires Git/HTTPS access to the pinned source commit; an HP wheel with
the same version number does not override a direct Git dependency offline.
Ordinary dependencies come from the configured index/current environment.
No PyPI upload or replacement of the paper's existing package release is needed.
