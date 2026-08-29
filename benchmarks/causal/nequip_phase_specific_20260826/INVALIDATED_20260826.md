# Invalidated NequIP causal batch

Slurm array `57448` and its dependent ZrO retry `57522` were cancelled after
an implementation audit found that the NequIP phase head was initialised at
exactly zero.  Because the Hermitian phase kernel is stationary at aligned
phase, the learned-phase row then received zero gradient.  Full and Aligned
therefore followed the same training trajectory and did not implement the
intended causal contrast.

The completed outputs under
`/public/home/sps-xia/rxlin/nequip_chorus_runs/causal_phase_specific_20260826`
are retained only as an audit record.  They must not enter any scientific
summary, figure, table, or checkpoint pool.

The corrected campaign uses a small nonzero phase-row initialisation, checks
that Full differs from Aligned and that the phase gradient is nonzero, and
writes to the separate `causal_phase_specific_v2_20260826` root.
