# Software lineage and attribution

CHORUS-MLIP is developed by the MACE-ICTC authors as a continuation of the
MACE-ICTC research and software stack.

## CHORUS

The CHORUS contribution introduces phase-coherent real-doublet aggregation,
low-rank full-\(L\) Hermitian density contractions, coherent/diagonal controls,
and an experimental persistent charged stream.

The public distribution is named `chorus-mlip` and the canonical Python
namespace is `chorus`. A minimal deprecated `mace_ictc` shim is distributed
only to resolve historical checkpoint module paths.

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

## NequIP

The optional integration under `integrations/nequip/` is a patch and source
overlay against the MIT-licensed NequIP v0.6.2 codebase:

```text
Copyright (c) 2021 Harvard University
https://github.com/mir-group/nequip
```

Its upstream license is reproduced in
`integrations/nequip/LICENSE.nequip`. The overlay is not a bundled NequIP
installation.
