# Upstream provenance

| Item | Value |
|---|---|
| Upstream project | NequIP |
| Upstream repository | `https://github.com/mir-group/nequip` |
| Base revision | tag `v0.6.2` |
| CHORUS integration revision | `b20792fc72265bcd6c847117bafe88a630093e70` |
| License | MIT; see `LICENSE.nequip` |

Files under `overlay/` retain their repository-relative paths. The historical
patch under `patches/` captures the original integration and namespace migration
from `mace_ictc` to `chorus`; it is not the current installation artifact.
The overlay additionally contains phase-initialization fixes and the native
OpenEquivariance execution update. It is the canonical current implementation.
The OEQ update preserves NequIP's model parameters and equations, and uses
the upstream OpenEquivariance tensor-product/convolution APIs without a vendor fork.
