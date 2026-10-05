# Claim–evidence matrix

| Claim | Direct evidence | Publication-safe wording | Boundary |
|---|---|---|---|
| Hermitian self-density improves aggregation | Diagonal vs Off, 20 paired models per backbone | The Hermitian second-order density is the only component that improves all 20 pairs in both tested backbones. | Total module effect; active capacity also changes. |
| Cross-neighbour closure contributes | Aligned vs Diagonal | Cross-neighbour Hermitian closure adds a backbone-dependent gain. | It is strong in NequIP-SH and heterogeneous in MACE-ICTC. |
| Learned relative phase is used by fitted models | Frozen alignment, dose, and permutation | Learned relative-phase organisation is functionally embedded in the predictor. | Frozen reliance can be much larger than the net retraining gain because optimisation compensates. |
| U(1) structure is implemented correctly | Common-shift negative control | Predictions are invariant to a common phase shift within float32 tolerance. | This validates global U(1) invariance; it does not assign θ a physical observable. |
| Persistence increases phase reliance | Persistent-minus-Final frozen interaction | Propagating the charged state across depth increases functional dependence on relative phase. | Retraining interactions may remain smaller and heterogeneous. |
| The mechanism transports across backbones | Matched MACE-ICTC and NequIP-SH panels | The same intervention ladder is reproduced in two O(3)-equivariant backbones. | Two backbones do not establish universality over all equivariant models. |
| Phase histograms alone explain accuracy | Observational association rung | No stable scalar phase-statistic association is claimed. | High-dimensional phase organisation is not identified by one marginal statistic. |
