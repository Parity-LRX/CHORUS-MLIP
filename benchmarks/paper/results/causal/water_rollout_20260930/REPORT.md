# Paired causal analysis of CHORUS liquid-water rollouts

## Design and estimand

The architectural intervention is the independently retrained model:
MACE-ICTC, Final, or Persistent.  The five common velocity seeds
20260616--20260620 are matched rollout blocks.  Every block is observed under
all three interventions, so the primary estimands are within-seed total
checkpoint effects, averaged over the declared velocity-seed distribution.
The 300 ps production estimands use the common 100--300 ps window.  The 1 ns
run at seed 20260616 is a horizon extension and is not counted as a sixth
replicate.

## Accuracy-to-deployment chain

The validation-selected water checkpoints have held-out force MAEs of
21.891, 20.906, and
20.454 meV Angstrom^-1 for MACE-ICTC, Final, and Persistent.
Final and Persistent therefore reduce force MAE by 4.50% and 6.56% relative
to MACE-ICTC before any rollout outcome is inspected.  All fifteen 300 ps matched
rollouts and all three 1 ns horizon extensions reach their target lengths.

## Primary structural effect: tetrahedral order

The five-seed mean tetrahedral order is
0.643215 for MACE-ICTC,
0.641672 for Final, and
0.640412 for Persistent.
Final - MACE-ICTC is -0.001543
(95% paired-t interval -0.002554 to -0.000532);
Persistent - MACE-ICTC is -0.002803
(-0.003494 to -0.002113).
The directions agree in 5/5 and
5/5 paired seeds, respectively.
Relative to MACE-ICTC seed-to-seed variation, the effects are
-2.45 and
-4.46 SD.  A strict
MACE-ICTC--Final--Persistent ordering occurs in
4/5 seeds, while the fitted
scope slope has the same sign in
5/5 seeds.
The direct Persistent - Final effect is
-0.001260
(-0.002673 to
+0.000152), with
4/5 paired seeds in the
modal direction; the pairwise evidence therefore separates both CHORUS scopes
from MACE-ICTC more clearly than it separates Persistent from Final.
With five pairs, the two-sided exact sign-flip p-value for a unanimous direction
is 0.0625; the evidence is summarized by
the paired effect, interval, and direction agreement rather than a binary
p-value threshold.

The mean paired-q effect is negative in
4/4 blocks for Final - MACE-ICTC and
4/4 blocks for the
Persistent - MACE-ICTC contrast.  The individual 50 ps intervals are wide: they exclude zero in
0/4 and
1/4 blocks, respectively.  The
robust estimand is therefore the pre-specified 100--300 ps trajectory summary,
supported by 5/5 paired-seed direction agreement, rather than any individual
short block.

## Secondary structure and dynamics

O--O peak-height effects are -0.02377 (Final - MACE-ICTC) and
-0.02990 (Persistent - MACE-ICTC).  The corresponding
H-bond coordination effects are -0.00147
(-0.00749 to +0.00456) and
-0.00475
(-0.00725 to -0.00224); the
Persistent - MACE-ICTC direction agrees in
5/5 seeds.  Across 1.5--6.0 A,
the seed-mean RMS
O--O RDF contrasts are 0.01677 for
Final - MACE-ICTC and 0.01761 for
Persistent - MACE-ICTC.  These RMS values describe total curve displacement; the
paired peak-height intervals and pointwise RDF bands do not support treating
RDF amplitude as the primary causal endpoint.

Oxygen diffusion changes from
2.648 for MACE-ICTC to
2.680 for Final and
2.788 x10^-9
m^2 s^-1 for Persistent.  The paired effects are
+0.032 and +0.140, but their
95% intervals (-0.329, +0.392) and
(-0.095, +0.376) include zero.
Only 1/5 seeds show a strict
MACE-ICTC--Final--Persistent diffusion ordering.  Diffusion is therefore a noisy
secondary rollout response, not the primary causal endpoint.

## Deployment stability and one-million-step horizon

The causal contrasts in absolute potential-energy drift are
+0.576 and -1.644
meV H2O^-1 ns^-1 for Final - MACE-ICTC and Persistent - MACE-ICTC.  Completion, bounded
force excursions, and the 1 ns block summaries establish that all three
checkpoints remain usable under closed-loop integration.

At 200--400 ps, the 1 ns block-mean q values are
0.643028, 0.642471, and
0.640969; at 800--1000 ps they are
0.640420, 0.642814, and
0.640797.  The short-horizon ordering is not preserved
as a simple monotone 1 ns separation.  The combined result is therefore:
the selected CHORUS checkpoints produce a repeatable finite-horizon change in
higher-order liquid structure while all scopes remain stable over one million
steps; the 1 ns evidence supports deployment persistence rather than a claim
of monotone long-time transport ranking.

## Manuscript-level conclusion

The water study extends the paper's designed interventions downstream.  The
replicated training panel establishes the algorithmic effect on force error;
the matched velocity-seed analysis shows that the selected lower-error
checkpoints cause reproducible changes in trajectory-level many-body
statistics; and the one-million-step extension establishes closed-loop
deployability.

## Files

- `causal_effect_summary.csv`: paired scalar estimands and intervals.
- `paired_seed_differences.csv`: every observed potential-outcome contrast.
- `time_resolved_effects.csv`: 50 ps paired effects for q and H-bond order.
- `long_1ns_blocks.csv`: 200 ps horizon-extension blocks.
- `functional_effect_summary.csv`: RMS RDF intervention contrasts.
- `causal_rollout_main.pdf`: main-text candidate figure.
- `causal_effect_forest.pdf`: standardized endpoint forest plot.
- `functional_rollout_contrasts.pdf`: paired RDF and MSD curves.
