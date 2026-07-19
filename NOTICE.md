# Software lineage and attribution

CHORUS-MLIP is developed by the MACE-ICTC authors as a continuation of the
MACE-ICTC research and software stack.

## CHORUS

The CHORUS contribution introduces phase-coherent real-doublet aggregation,
low-rank full-\(L\) Hermitian density contractions, coherent/diagonal controls,
and an experimental persistent charged stream.

The public distribution is named `chorus-mlip`. The internal Python namespace
remains `mace_ictc` to preserve compatibility with MACE-ICTC checkpoints,
compiled extensions, converters, and deployment interfaces.

## MACE-ICTC

MACE-ICTC provides the Irreducible Cartesian Tensor Decomposition
representation, MACE-compatible interaction and product path, training and
checkpoint conversion, compilation, ASE integration, and LAMMPS deployment.

Software record:

```text
MACE-ICTC. Zenodo. https://doi.org/10.5281/zenodo.20690950
```

## MACE

MACE-ICTC and CHORUS build on the MACE architecture and contain code derived
from the MIT-licensed MACE implementation:

```text
Copyright (c) 2022 ACEsuit/mace
https://github.com/ACEsuit/mace
```

Relevant scientific reference:

```text
I. Batatia et al.
MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and
Accurate Force Fields.
NeurIPS 2022.
```

The repository-level MIT license retains both the upstream MACE notice and the
MACE-ICTC/CHORUS notice. Pretrained models, benchmark datasets, and external
dependencies may be governed by their own licenses.
