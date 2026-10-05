# Frozen-phase effect heterogeneity

## Result

Aligning the learned relative phase increases Force MAE in every reported
environmental stratum of every completed MACE-Final predictor.  The absolute
Q4-minus-Q1 intervention consequence is positive for 4/4
datasets when environments are ordered by reference-force magnitude.  The
scale-normalised Q4-versus-Q1 contrast is positive for only
2/4 datasets.  Harder configurations therefore incur a
larger absolute penalty, while the current panel does not support a universal
claim that their *relative* dependence on phase is stronger.

Neighbour count is a more heterogeneous moderator: the proportional Q4-versus-Q1
contrast is positive in 1/3 datasets
with identifiable Q1 and Q4 strata.  Crowding alone is consequently not a
sufficient explanation of phase use.  Element-resolved results are retained in
`stratum_effects.csv`; they are descriptive within each chemical system rather
than population-level element effects.

The equal-dataset frozen alignment effect on energy is
156.96%
(95% hierarchical-bootstrap interval
18.81% to
479.94%).

## Interpretation

This analysis localises the consequence of an intervention on a fitted
predictor.  It does not turn atoms into independent causal units.  Each stratum
first reduces all nested atom or frame outcomes to one value per trained
dataset-by-seed predictor; bootstrap resampling is then performed over those
predictors.  Absolute and proportional effects are shown together because an
absolute error difference can rise mechanically with target scale.

## Audit

- Trained-model pairs: 20.
- Atomic repeated outcomes: 4,964,675.
- Frame repeated outcomes: 112,455.
- PyArrow: 24.0.0.
- All required files present: **True**.
- All numeric outcomes finite: **True**.
