# NequIP integration

This directory contains the CHORUS implementation for two NequIP v0.6.2
interaction backends:

- `interaction_backend: e3nn` — the standard NequIP equivariant convolution;
- `interaction_backend: ictc` — the channelwise ICTC interaction block.

Both backends call the same `HermitianDensityResidual` implementation and
support CHORUS-off, final-layer CHORUS, and all-layer CHORUS.  The four
off/on × e3nn/ICTC construction paths are covered by
`overlay/tests/unit/nn/test_chorus.py`.

## Why this is an overlay

NequIP is an independently versioned MIT-licensed project.  Copying an entire
NequIP checkout into CHORUS would obscure which lines implement the new
operator and would make upstream updates difficult to audit.  This integration
therefore ships in two equivalent forms:

- `overlay/` contains every file added or changed by the integration, preserving
  its path relative to the NequIP repository root;
- `patches/nequip-v0.6.2-chorus.patch` is the complete patch against the tagged
  NequIP v0.6.2 source tree.

The overlay also includes the dataset adapters, evaluation changes, benchmark
configuration writers, Slurm launchers, and unit tests used for the transfer
experiments.  It is source code, not a vendored NequIP installation.

## Apply to a clean NequIP checkout

```bash
git clone --branch v0.6.2 https://github.com/mir-group/nequip.git
cd nequip
git apply /path/to/CHORUS-MLIP/integrations/nequip/patches/nequip-v0.6.2-chorus.patch
```

Alternatively, copy the contents of `overlay/` onto the root of a NequIP
v0.6.2 checkout.

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
chorus_scope: final        # or all
chorus_rank: 16
chorus_hidden_channels: 32
chorus_scale_init: 0.05
```

The two example fragments in `examples/` make the backend difference explicit.
All ordinary NequIP architecture and training options remain in the NequIP
configuration.
