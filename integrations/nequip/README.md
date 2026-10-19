# NequIP integration

This directory contains the CHORUS implementation for two NequIP v0.6.2
interaction backends:

- `interaction_backend: e3nn` — the standard NequIP equivariant convolution;
- `interaction_backend: ictc` — the channelwise ICTC interaction block.

Both backends call the same `HermitianDensityResidual` implementation and
support the unmodified backbone, Final CHORUS, and Persistent CHORUS. The four
off/on × e3nn/ICTC construction paths are covered by
`overlay/tests/unit/nn/test_chorus.py`.

## Why this is an overlay

NequIP is an independently versioned MIT-licensed project.  Copying an entire
NequIP checkout into CHORUS would obscure which lines implement the new
operator and would make upstream updates difficult to audit.  This integration
therefore uses the current overlay as its installation source:

- `overlay/` contains every file added or changed by the integration, preserving
  its path relative to the NequIP repository root;
- `patches/nequip-v0.6.2-chorus.patch` preserves the historical integration
  snapshot. It does not include later phase-initialization corrections or the
  native OpenEquivariance optimization below; use the overlay for current work.

The overlay also includes the dataset adapters, evaluation changes, benchmark
configuration writers, Slurm launchers, and unit tests used for the transfer
experiments.  It is source code, not a vendored NequIP installation.

## Apply to a clean NequIP checkout

```bash
git clone --branch v0.6.2 https://github.com/mir-group/nequip.git
cd nequip
rsync -a /path/to/CHORUS-MLIP/integrations/nequip/overlay/ ./
```

This copies only the integration files onto the NequIP v0.6.2 source tree.

Install this CHORUS repository in the same environment before selecting the
ICTC backend:

```bash
pip install -e /path/to/CHORUS-MLIP
```

The captured integration revision is
`b20792fc72265bcd6c847117bafe88a630093e70`; its historical MACE-ICTC imports
were migrated mechanically to the canonical `chorus` namespace when the code
entered this repository. NequIP's upstream license is preserved in
`LICENSE.nequip`.

## Model switches

The minimal CHORUS-specific configuration is:

```yaml
interaction_backend: e3nn  # or ictc
chorus_enabled: true
chorus_scope: final        # or persistent
chorus_rank: 16
chorus_hidden_channels: 32
chorus_scale_init: 0.05
```

The two example fragments in `examples/` make the backend difference explicit.
All ordinary NequIP architecture and training options remain in the NequIP
configuration.

## Native OpenEquivariance execution

For the spherical-harmonic backend, add:

```yaml
interaction_backend: e3nn
openequivariance_enabled: true
```

This enables ordinary OEQ tensor products and the following CHORUS execution
changes, with the original parameters, CG paths, rank and checkpoint keys:

- Equal-rank Hermitian contractions use rank as a batch dimension, avoiding
  the old dense rank-by-rank identity-weight problem.
- The first phase-network affine map projects node features before neighbor
  gather, without caching activations between calls.
- Element-conditioned output maps use explicit 2-D bank GEMMs, then mix by
  the original node attributes. Soft attributes and their gradients remain valid.
- Neutral layers with non-scalar inputs preceding Final CHORUS use OEQ fused
  TP convolution. The scalar first layer keeps the smaller-workspace TP path;
  charged layers keep the edge messages required by the phase-dependent operator.
- Phase diagnostics/intervention hooks retain the original phase-network call.
- Irrep Linear maps use one feature/weight split and native channel GEMMs with
  the original path normalization on the weight side. This reduces independent
  slice-backward zero fills and full-message gradient accumulation.
- Linear neighbor normalization is applied after reduction to node rows;
  quadratic Hermitian density normalization retains its original placement.
  Hooked, unsupported, non-CUDA and scripted Linear calls use the original module.

The measured runtime is OpenEquivariance 0.6.8 with PyTorch 2.11, native
execution, FP32 network/forces and TF32/AMP off. It uses neither CUDA Graph
nor AOTI. NequIP's existing stress/virial definition and dtype are preserved.
The non-OEQ path retains its original operators and model equations.

`NEQUIP_CHORUS_NATIVE_DATAFLOW=0` disables the new Linear execution and linear
normalization reordering. Irrep sections still use the equivalent single-split
layout. Cached execution plans contain only Python layout metadata, not weights
or activations; the checkpoint schema is unchanged.

Full-output and force-loss mixed-gradient qualification, same-card inference
and AdamW measurements, and the existing optimized OEQ baseline are recorded
in [the optimization report](../../benchmarks/nequip_hp/README.md).
The latest bounded round includes a no-CHORUS same-backbone baseline with the
same generic dataflow optimizations; see the
[dataflow protocol](../../benchmarks/nequip_hp/dataflow_round/README.md).
