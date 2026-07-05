# MACE-ICTC: ICTC algorithm contributions and e3nn/MACE numerical-parity mechanism

Scope: this note documents the *algorithmic contribution* of MACE-ICTC — the irreducible
Cartesian tensor decomposition (ICTC) angular basis, the `Q` basis change to e3nn/MACE, the
ICTC Clebsch–Gordan (CG) and symmetric-contraction operators, and the **bridge-U** path that
makes the model numerically reproduce native `mace-torch`. It is written to be read alongside the
code but does not require it. File/line references are to commit `414aa25` of the local repo
(`/Users/sara/Desktop/code/MACE-ICTC`); the 4090 deployment `/home/ylzhang/lrx/MACE-ICTC` is a
content copy of the same tree.

Conventions used below:
- `l` is an angular degree (irrep order); a degree-`l` feature has `2l+1` components indexed by `m`.
- A node/edge feature tensor is stored as `(..., C, 2l+1)` per degree (channel-major), or flattened
  over degrees into `(..., C, (L+1)^2)` with `L = lmax`.
- `D_l := dim Sym^l(R^3) = (l+1)(l+2)/2` is the dimension of the degree-`l` *symmetric* Cartesian
  tensor space (the monomial space), distinct from the harmonic dimension `2l+1`.
- All fixed operators are built in float64 and cached; runtime dtype is a final cast.

A one-line map of "who owns the math":
- `mace_ictc/models/ictd_irreps.py` — ICTC basis, ICTC CG, ICTC symmetric-contraction `U`.
- `mace_ictc/mace_basis.py` — the per-degree orthogonal `Q_l` (ICTC ↔ e3nn).
- `mace_ictc/models/_mace_cg.py`, `_mace_symmetric_contraction.py` — verbatim-equivalent MACE
  (e3nn) CG / symmetric contraction (the reference and bridge-U core).
- `mace_ictc/models/pure_cartesian_ictd_fix.py` — the model, the four product backends, and the
  `SO3ToE3NNBasisBridge`.
- `mace_ictc/interfaces/mace_converter.py` — load a real `ScaleShiftMACE` into a MACE-ICTC model.

---

## A. ICTC basis construction

### A.1 What a degree-`l` ICTC feature *is*

ICTC stores a degree-`l` equivariant feature as the **harmonic part of a symmetric Cartesian
tensor of rank `l`**, expressed in a fixed orthonormal basis of that harmonic space. Concretely:

1. The rank-`l` symmetric Cartesian tensors are identified with homogeneous degree-`l` polynomials
   in `(x,y,z)`, i.e. the monomial space `Sym^l` with basis `{ x^a y^b z^c : a+b+c=l }`,
   `dim Sym^l = D_l = (l+1)(l+2)/2` (`sym_dim`, [ictd_irreps.py:597](mace_ictc/models/ictd_irreps.py)).

2. The **harmonic (traceless) subspace** `Harm^l ⊂ Sym^l` is the null space of the Laplacian
   `Δ : Sym^l → Sym^{l-2}`; `dim Harm^l = 2l+1`. A basis `B_l ∈ R^{D_l × (2l+1)}` is obtained as the
   trailing right-singular vectors of `Δ` (`_harmonic_basis_cpu_f64`,
   [ictd_irreps.py:655](mace_ictc/models/ictd_irreps.py)).

3. That basis is **orthonormalized under an O(3)-invariant inner product** `G_l` on `Sym^l`, the
   Gaussian-moment Gram matrix
   ```
   G_l[α, β] = E_{u ~ N(0, I_3)} [ m_α(u) · m_β(u) ]
             = E[x^{a+a'}] · E[y^{b+b'}] · E[z^{c+c'}]          (m_α = x^a y^b z^c)
   ```
   (`_gram_gaussian`, [ictd_irreps.py:637](mace_ictc/models/ictd_irreps.py)). Each 1-D factor is a
   Gaussian moment (`E[x^{2p}] = (2p-1)!!`). Because the isotropic Gaussian measure is rotation
   invariant, `G_l` is an O(3)-invariant metric, so the orthonormalization is rotation-covariant.
   The orthonormalization itself is `B_l ← B_l · M^{-1/2}` with `M = B_l^T G_l B_l`, giving
   `B_l^T G_l B_l = I_{2l+1}` (eigendecomposition of `M`, [ictd_irreps.py:672](mace_ictc/models/ictd_irreps.py)).

So the **ICTC basis is intrinsic and Cartesian**: it is defined purely from polynomial algebra and a
Gaussian metric, with **no reference to spherical harmonics or e3nn**. This is the key contribution
— every angular operator below (harmonics, CG, symmetric contraction) is then derived *inside* this
basis.

### A.2 ICTC angular features (`direction_harmonics`)

For a unit direction `n`, the degree-`l` ICTC harmonic coordinates are the harmonic projection of the
symmetric power `n^{⊗ l}`:
```
t_α(n) = multinomial(l; a,b,c) · n_x^a n_y^b n_z^c          # monomial coeffs of (n·r)^l, shape (D_l,)
c_l(n) = s_l · B_l^T G_l t(n)                                # shape (2l+1,)
```
(`direction_harmonics`, [ictd_irreps.py:765](mace_ictc/models/ictd_irreps.py); the code evaluates it
as `c = einsum("...d,md,mc->...c", t, G, B)`, i.e. `c = t^T (G_l B_l)`).

The scalar `s_l = _direction_component_scale` ([ictd_irreps.py:830](mace_ictc/models/ictd_irreps.py))
rescales the Gaussian-orthonormal basis to **e3nn "component" normalization** — each component has
unit RMS over directions uniform on `S^2`:
```
s_l = ( trace(P_l^T  G^sphere_l  P_l) / (2l+1) )^{-1/2},     P_l = G_l B_l,
```
where `G^sphere_l` is the analogous Gram under the *uniform-sphere* measure
(`_sphere_monomial_moment_3d`, [ictd_irreps.py:618](mace_ictc/models/ictd_irreps.py)). This matters
because MACE/e3nn edge spherical harmonics use `normalization="component"`; matching the RMS makes the
later `Q_l` an *orthogonal* (not merely invertible) matrix.

### A.3 Relation of the ICTC basis to e3nn/spherical basis — the `Q` transform

For every `l` there is a **fixed orthogonal** matrix `Q_l ∈ R^{(2l+1)×(2l+1)}` with
```
direction_harmonics(n, l) @ Q_l  ==  Y_l^{e3nn}(n)            (machine precision, all n)   [*]
```
where `Y_l^{e3nn}` are e3nn real spherical harmonics in `component` normalization
(`mace_basis.py` docstring + `_q_per_l_f64`, [mace_basis.py:31](mace_ictc/mace_basis.py); the same
fit is reproduced inside the model as `SO3ToE3NNBasisBridge`,
[pure_cartesian_ictd_fix.py:669](mace_ictc/models/pure_cartesian_ictd_fix.py)).

`Q_l` is computed by **orthogonal Procrustes** on `M = 8192` sampled directions
(`seed = 20260426`):
```
A = direction_harmonics(dirs, l)         # (M, 2l+1)   ICTC
B = Y_l^{e3nn}(dirs)                      # (M, 2l+1)   e3nn
U, Σ, V^T = svd(A^T B)                    # (2l+1 × 2l+1)
Q_l = U V^T                               #  -> argmin_Q ||A Q - B||_F  over O(2l+1)
```
Because both `A` and `B` are component-orthonormal frames of the *same* `(2l+1)`-dim irrep, the
Procrustes minimizer is exactly orthogonal: `Q_l^T Q_l = I`, and the residual `||A Q_l − B||` is at
the float64 floor (verified below: `8.3e-17`). `Q_0 = [[1]]` (the `l=0` invariant is unchanged).

**Is `Q_l` orthogonal?** Yes — exactly (up to float64), by construction (`U`, `V` orthogonal ⇒ `UV^T`
orthogonal). It is *not* in general a permutation or a signed permutation; it genuinely rotates the
`2l+1` components (the test measures `||x − xQ|| ≈ 5e-2` for the model's `l≥1` features).

**Conversion formulas (shapes and side verified against code):**

Let `x` carry the `(L+1)^2` angular components on its **last** axis (channel/batch axes to the left),
and `Q = blockdiag(Q_0, …, Q_L)` (`orthogonal_Q`, [mace_basis.py:59](mace_ictc/mace_basis.py)).
```
ICTC -> e3nn :   x_e3nn  = x_ictd @ Q          # to_mace_basis;   einsum("ncm,mp->ncp", x, Q_l) per block
e3nn -> ICTC :   x_ictd  = x_e3nn @ Q^T        # to_ictd_basis ;   einsum("ncm,pm->ncp", x, Q_l) per block
```
Both are **right-multiplications on the last (m) axis**; the channel axis is untouched. `e3nn→ICTC`
right-multiplies by `Q_l^T` (the inverse, since `Q_l` is orthogonal). This is the *only* place a
transpose appears, and the two bridge directions in
`SO3ToE3NNBasisBridge.ictd_flat_to_e3nn_blocks` / `e3nn_flat_to_ictd_blocks`
([pure_cartesian_ictd_fix.py:710](mace_ictc/models/pure_cartesian_ictd_fix.py)) use exactly
`einsum("ncm,mp->ncp", ·, q)` and `einsum("ncm,pm->ncp", ·, q)` respectively, confirming the
`Q` / `Q^T` directions above.

**Physical invariants are unchanged by `Q`.** Energy, forces and virial are SO(3) invariants /
physical Cartesian tensors; `Q` only re-expresses the `l≥1` angular *internal* features. Hence any
global `ICTC↔e3nn` change is energy/force-preserving (this is what makes the bridge exact, §C/§E).

---

## B. Angular features: `angular_basis="ictd"` vs `"e3nn"`

`angular_basis` is a **global** switch for which orthogonal frame the model's `l≥1` features live in:

- `angular_basis="ictd"` (default): edge harmonics are `direction_harmonics`, the interaction CG is the
  ICTC CG (§C.1), and all internal `l≥1` features are in the ICTC frame.
- `angular_basis="e3nn"`: `Q` is **folded once** into the fixed angular operators — the edge harmonics
  become `Y^{e3nn}` and the interaction CG is rotated by `Q` — so every internal `l≥1` feature is
  emitted *natively* in the e3nn frame. The numerically delicate **order-≥2 symmetric contraction is
  still evaluated in the stable ICTC frame**, with an exact `e3nn↔ICTC` rotation applied only at its
  input/output (`SO3ToE3NNBasisBridge`).

Because this is a single global orthogonal change of an angular basis, it is provably output-preserving
(`mace_ictc/test/test_angular_basis.py` docstring):
- energy / forces / virial are **bit-identical** to `angular_basis="ictd"`;
- every intermediate `l≥1` node feature equals the `ictd` feature re-expressed in e3nn (`x @ Q`) to
  machine precision.

Why both exist: `angular_basis="ictd"` is the native ICTC computation; `angular_basis="e3nn"` is what
lets the **cuEquivariance product backend** (which expects the e3nn convention) run on the model's
features without per-edge re-rotation, i.e. it is a *performance* convenience, not a different model.

| path | role |
|---|---|
| `angular_basis="ictd"` | native ICTC numerics; default |
| `angular_basis="e3nn"` | e3nn-convention features for the cuEq product path; output-identical to `ictd` |

---

## C. Tensor product / symmetric contraction in the ICTC basis

### C.1 ICTC CG intertwiner (the convolution tensor product core)

The e3nn/MACE convolution couples two features by Clebsch–Gordan:
```
z_{l3,m3} = Σ_{m1,m2}  C^{l3 m3}_{l1 m1, l2 m2}  x_{l1,m1}  y_{l2,m2}.
```
MACE-ICTC builds the analogous coupling tensor **inside the ICTC basis**, two ways:

1. **Natural-parity paths** (`l1+l2+l3` even) — `build_cg_tensor(l1,l2,l3)`
   ([ictd_irreps.py:1143](mace_ictc/models/ictd_irreps.py)). It is *polynomial multiplication then
   harmonic projection*:
   ```
   outer  = B_{l1} ⊗ B_{l2}                    # harmonic basis vectors as polynomials
   t_L    = M_poly · outer                     # multiply polynomials -> Sym^{L},  L=l1+l2
   C[·,·,m3] = P_{L→l3} · t_L                  # project to the degree-l3 trace block
   ```
   where `M_poly` is the exact monomial-product map ([ictd_irreps.py:1124](mace_ictc/models/ictd_irreps.py))
   and `P_{L→l3}` is the harmonic projector of the symmetric-trace chain
   `Sym^L ≅ ⊕_k r^{2k} Harm^{L-2k}` (`build_harmonic_projectors`,
   [ictd_irreps.py:722](mace_ictc/models/ictd_irreps.py)). Output `C` has shape
   `(2l1+1, 2l2+1, 2l3+1)` and satisfies `c[m3] = Σ a[m1] b[m2] C[m1,m2,m3]`.

2. **All SO(3) paths incl. antisymmetric** (`|l1−l2| ≤ l3 ≤ l1+l2`, any parity chain, e.g. `1×1→1`) —
   `build_full_cg_tensor_so3(l1,l2,l3)` ([ictd_irreps.py:1224](mace_ictc/models/ictd_irreps.py)). The
   higher-order contraction uses intermediate irreps with independent parity, which the polynomial
   construction (1) does not reach. Instead of calling e3nn `wigner_3j`, the unique intertwiner is
   found by **solving the infinitesimal equivariance equations** in the ICTC basis:
   ```
   J3 · C − C · (J1 ⊗ I + I ⊗ J2) = 0      for the SO(3) generators J along x,y,z
   ```
   `C` is the 1-D null space of the stacked constraint matrix (robust SVD), reshaped to
   `(2l1+1, 2l2+1, 2l3+1)` and **sign-canonicalized** (largest-magnitude entry made positive). The
   generators themselves are built in the ICTC basis from the monomial rotation generators
   (`_harmonic_rotation_generator_cpu_f64`, [ictd_irreps.py:1216](mace_ictc/models/ictd_irreps.py)).

   *Contribution note:* (2) means the ICTC CG/`U` machinery is **self-contained** — it reproduces the
   full real-CG algebra (including the antisymmetric `1×1→1` path MACE needs at body order ≥3) from
   Cartesian polynomial algebra + the Lie-algebra equivariance constraint, with no e3nn/Wigner call.

### C.2 ICTC symmetric-contraction `U` (MACE Eq. 10–11, "pure-U")

MACE's product block is a learnable symmetric contraction of body order up to `correlation = ν`. For
a target output degree `l_out` and order `ν`, the fixed coupling is a tensor
`U^{(ν)}_{l_out}` with shape `(2l_out+1, D, …, D, n_paths)` — `ν` copies of the **flattened one-body
input dimension** `D = Σ_{l=0}^{lmax}(2l+1)` and `n_paths` independent coupling paths.

MACE-ICTC builds this **purely from the ICTC CG of §C.1** (`ictd_u_matrix_so3` →
`_ictd_so3_coupled_basis`, [ictd_irreps.py:1306](mace_ictc/models/ictd_irreps.py)):
```
order 1 : basis = { identity block for each l }                       # (2l+1, D)
order ν : for each accumulated (l_left, tensor) and each l_right,
          for l_out in |l_left-l_right| .. l_left+l_right:
             cg = build_full_cg_tensor_so3(l_left, l_right, l_out)     # §C.1(2)
             coupled = einsum("ar,abq->qrb", flat_left, cg)            # couple in one more body
          accumulate (l_out, parity, tensor)
U^{(ν)}_{l_out} = stack over paths of the tensors with l_out & canonical parity   # last axis = paths
```
This is the **ictd-pure-u** operator. It is mathematically an SO(3)-symmetric-contraction basis; it
agrees with MACE up to a basis change and per-path normalization, but is **not guaranteed
bit-identical** to `mace-torch` (the converter refuses it for exact parity — see §F).

### C.3 The MACE (e3nn) symmetric contraction kept verbatim

`_mace_cg.py::U_matrix_real` and `_mace_symmetric_contraction.py::MaceSymmetricContraction` are
**local minimal copies of MACE's own code** (Batatia et al., MACE Eq. 10–11). `U_matrix_real` builds
`U^{(ν)}` from e3nn `wigner_3j` (or, for `correlation==4` / reduced CG, from cuEquivariance's
`reduced_symmetric_tensor_product_basis`). This is the reference operator and the **core of bridge-U**.
The contraction itself is the standard opt-einsum'd MACE recursion (with a scalar-`l_out=0`,
`correlation=3` fast path, [_mace_symmetric_contraction.py:260](mace_ictc/models/_mace_symmetric_contraction.py)).

### C.4 bridge-U: folding the MACE `U` into the ICTC basis (the parity path)

The **bridge-U** product block keeps MACE's `U^{(ν)}` buffers **verbatim** (built by `U_matrix_real`,
weights copied 1:1 from the source MACE model) and wraps the native contraction with the orthogonal
bridge at its I/O (`BridgeUMaceSymmetricContractionSO3.forward`,
[pure_cartesian_ictd_fix.py:761](mace_ictc/models/pure_cartesian_ictd_fix.py)):
```
x_e3nn   = bridge.ictd_flat_to_e3nn_features(x_ictd)        # = x_ictd @ Q_in
out_e3nn = MaceSymmetricContraction(x_e3nn, node_attrs)     # native MACE U, e3nn frame
out_ictd = bridge.e3nn_flat_to_ictd_flat(out_e3nn)          # = out_e3nn @ Q_out^T   (if model is ICTC-frame)
```

**Why this is exact e3nn/MACE parity.** The contraction is multilinear in `x` and `Q` is orthogonal,
so the feature bridge is *algebraically identical* to pre-folding `Q` into `U`. Writing the order-`ν`
term schematically (suppressing channel & path-weight indices),
```
out_e3nn[m_out] = Σ_{m_1..m_ν}  U^{(ν)}_e3nn[m_out, m_1, …, m_ν]  Π_k x_e3nn[m_k],
```
substituting `x_e3nn[m_k] = Σ_{m'_k} x_ictd[m'_k] Q_in[m'_k, m_k]` and
`out_ictd[m'_out] = Σ_{m_out} out_e3nn[m_out] Q_out[m'_out, m_out]` gives an **effective folded
operator**
```
U^{(ν)}_ictd[m'_out, m'_1, …, m'_ν]
   = Σ_{m_out, m_1..m_ν}  Q_out[m'_out, m_out] · U^{(ν)}_e3nn[m_out, m_1..m_ν] · Π_k Q_in[m'_k, m_k].
```
In the compact "fold" shorthand of the task brief, this is
`U_ictd = Q_out · U_e3nn · (Q_in ⊗ … ⊗ Q_in)` with `Q_out` acting on the output leg and one `Q_in`
on each of the `ν` input legs (each input leg contracts the e3nn index of `U_e3nn`, i.e. it is the
`Q_in^T`-side in input-first index order — matching the `einsum("ncm,pm->ncp")`=`@Q^T` output map).

Two important implementation facts (verified in code):
- **The fold is realized at runtime by the feature bridge, not by mutating a `U` buffer.** The MACE
  `U_matrix_{ν}` buffers are stored unchanged; `Q` is applied to features. The "pre-folded" phrasing in
  the converter docstring is the *algebraic* description; both forms are identical because `Q` is
  orthogonal. (This avoids ever double-folding a fixed buffer on checkpoint reload / AOTI export.)
- For an `angular_basis="e3nn"` model the inputs are already in the e3nn frame, so `bridge.ictd_flat_to_e3nn`
  becomes the identity for the already-e3nn legs (the bridge only rotates legs that are still ICTC-frame).

**Correlation dependence.** bridge-U inherits MACE's behavior for all `ν`: `correlation = 1,2,3` use
the standard real-CG basis; `correlation = 4` uses the parity-filtered / reduced-CG basis
(`filter_ir_mid` or cuEq reduced product, [_mace_cg.py:128](mace_ictc/models/_mace_cg.py)). The
*pure-U* path (§C.2) builds every `ν` recursively from ICTC CG; the *cueq* path (§F) consumes the same
MACE weights through a fused kernel.

---

## D. Native MACE conversion (`convert_mace_to_ictd`)

`convert_mace_to_ictd(mace_model, ictd_model)`
([mace_converter.py:263](mace_ictc/interfaces/mace_converter.py)) copies every learnable weight from a
built `ScaleShiftMACE` (mace-torch 0.3.16) into a built `PureCartesianICTDFix`, **in float64, in
place**. Supported target backends for exact parity: `native-mace`, `ictd-bridge-u` (default), `cueq`
([mace_converter.py:279](mace_ictc/interfaces/mace_converter.py)). The mapping block-by-block:

**Directly copied (after an exact reparameterization):**
- **`o3.Linear` blocks** (`node_embedding`, `linear_up`, post-conv `linear`, product `linear`). e3nn's
  path-normalization + flat weight layout are bypassed by probing the block with an identity input:
  `W_eff = block(I)`. An SO(3)-equivariant linear is block-diagonal in `l` and `m`-diagonal, so the
  `l`-block of `W_eff` is exactly `M_l ⊗ I_{2l+1}`; `M_l` drops into the ICTC adapter as
  `weight = M_l^T`. Exact to machine precision and version-independent
  ([mace_converter.py:15-23](mace_ictc/interfaces/mace_converter.py)).
- **Symmetric-contraction weights** (`weights_max`, `weights.{k}`) copy **1:1** — both sides use the
  identical MACE parameterization. For bridge-U the inner `.symmetric_contractions` owns them
  (`_copy_symmetric_contraction`, [mace_converter.py:736](mace_ictc/interfaces/mace_converter.py)).

**Re-constructed / calibrated (not a raw copy):**
- **Convolution `conv_tp`.** MACE `conv_tp` and ICTC `tp` evaluate the same `(l1,l2)→l3` paths but in a
  different **path order** and with a different **per-path scalar** (e3nn carries an element
  path-normalization that the ICTC CG basis does not). Both are recovered *empirically at convert time*
  (`_calibrate_conv_tp`, [mace_converter.py:~160-257](mace_ictc/interfaces/mace_converter.py)): random
  node features (bridged ICTC→e3nn by `Q`) and a random direction are pushed through the **isolated**
  MACE `conv_tp` and ICTC `tp`; the path permutation `perm` and per-path scalar `c_p` (with
  `mace_path_out == c_p · (ictd_path_out @ Q_{l3})`) are solved by least squares. The residual is
  required `< 1e-8` (observed `~1e-15`), which is itself a parity check that the two path bases
  correspond.
- **Radial MLP.** MACE `conv_tp_weights` (e3nn `FullyConnectedNet`, `normalize2mom(silu)`, no bias) →
  ICTC `fc` (`[Linear,SiLU]×3,Linear`, with bias). Each weight is transposed to `nn.Linear` layout and
  divided by `sqrt(h_in)`; the `normalize2mom` constant `K` is folded into the next layer; the last
  (un-activated) layer additionally absorbs `K`, is **reordered into ICTC path order**, and each path
  row is scaled by `c_p`. ICTC's own per-path base weight `tp.weight` is set to all-ones (no MACE
  equivalent). All biases zeroed ([mace_converter.py:331-375](mace_ictc/interfaces/mace_converter.py)).
- **First-layer self-connection `sc0`.** MACE's `h_1 = linear(symcontract(message)) + sc0(Z)` has a
  pure per-element `l=0` constant that the ICTC baseline's first `product` (called with `sc=None`)
  structurally omits; it is installed as a model **buffer** added explicitly in `forward` (avoids
  forward hooks, which are unreliable under `torch.export`/AOTI;
  [mace_converter.py:700](mace_ictc/interfaces/mace_converter.py)).
- **Energy scale/shift.** `ScaleShiftMACE` `scale`/`shift` installed into the ICTC model; `E0` atomic
  energies returned in the report for the caller's total-energy assembly
  ([mace_converter.py:718](mace_ictc/interfaces/mace_converter.py)).

**`use_reduced_cg`.** Selects whether `U_matrix_real` builds the symmetric-contraction basis from the
plain real-CG product or from cuEquivariance's *reduced* symmetric tensor-product basis. It must match
between the source MACE contraction and the ICTC target (relevant mainly at `correlation=4`). The cueq
backend additionally needs a weight projection helper to map MACE/e3nn contraction weights into cueq's
reduced weight space ([pure_cartesian_ictd_fix.py:882](mace_ictc/models/pure_cartesian_ictd_fix.py)).

**Stated limits (do not over-generalize).** The converter targets the **MACE-style baseline configs**
used in this repo: one readout per interaction (the last interaction has the two-layer readout, with
its hidden `l=0` folded into `linear_2`), the standard first-interaction structure, and
`hidden_lmax / max_ell / correlation` within the baseline grid. It is **not** a claim that every
mace-torch architectural variant converts exactly.

---

## E. Numerical parity — what is tested, and the measured residuals (RTX 4090, this run)

All numbers below are from the parity suite re-run on the 4090 in this task
(`parity_tests.log`). torch 2.7.1+cu128, e3nn 0.5.9, mace 0.3.16, cuequivariance 0.10.0.

| test | what it checks | result |
|---|---|---|
| `test_angular_basis.py` (run as **script**) | `angular_basis="e3nn"` vs `"ictd"`: output bit-identity + intermediate feature `e3nn == ictd @ Q` | **PASS**. f64: `dE=0`, `dF=4.2e-21`, feature `\|e3nn − ictd@Q\|=8.3e-17` (Q rotates l≥1 by `5.4e-2`). f32: `dE=0`, `dF=7.3e-12`, feature `4.7e-9`. |
| `test_mace_converter.py` | whole-model `ScaleShiftMACE → PureCartesianICTDFix` parity across backends / atom counts / boxes / seeds, **and** rotation invariance | **PASS** (1 passed, 58.8 s). Asserts rel`\|dE\| < 1e-9`, max`\|dF\| < 1e-6`; rotation `d_rot_i < 1e-6`, MACE-vs-ICTC(rotated) `< 1e-5`. Default backend `ictd-bridge-u`. |
| `test_cueq_product_backend.py` | cuEq fused contraction vs reference MACE contraction; training-grad routing; eval refresh | **PASS** (4 passed, 1 skipped, 15.8 s). Asserts `max_abs < 2e-5` (fp32) for corr∈{1,2,4} × (lmax,target) combos. |

Notes on coverage and the skip:
- `test_angular_basis.py` defines `run()`/`main()` and **no `test_*` functions**, so `pytest` collects
  "no tests" — it must be run as `python -m mace_ictc.test.test_angular_basis` (done here; PASS). This
  is a harness quirk, **not** a failure. (Recommendation for the paper repo: rename to `test_run()` or
  add a thin `def test_angular_basis(): main()` so CI exercises it.)
- The single **skip** in `test_cueq_product_backend.py` is the *CPU* case: when CUDA is available the
  cuEq backend intentionally builds CUDA-only fast kernels, so the CPU reference-vs-fast comparison is
  skipped by design (the CUDA cases run and pass). The reduced-CG cases also skip gracefully if the
  local reduced-CG projection helper is absent.
- **Symmetric-contraction / radial / conv_tp parity** are exercised *inside* `test_mace_converter`
  (the conv_tp calibration residual `< 1e-8` gate and the 1:1 contraction-weight copy are part of the
  whole-model parity), plus the fast-path unit tests `test_scalar_corr3_contraction_fastpath.py` and
  `test_scalar_path_tp_fastpath.py` for the scalar fast paths.
- **Reload / AOTI double-fold avoidance.** Fixed operators (`U`, `Q`, scalar-corr3 fast buffers) are
  either persistent deterministic buffers or regenerated by load hooks (`_load_from_state_dict`,
  `register_load_state_dict_post_hook` for cueq weight mirroring,
  `refresh_scalar_corr3_fast_buffers`). Because the `Q` fold lives in the runtime *feature* bridge (not
  in a mutated `U` buffer), a checkpoint reload or `torch.export`/AOTI trace cannot double-apply it.

**Strict vs test-coverage statements (for the paper):**
- *Strict numerical correspondence (by construction + machine-precision verification):* the
  `angular_basis` switch (global orthogonal change ⇒ invariants unchanged) and the bridge-U contraction
  (orthogonal feature bridge around verbatim MACE `U`). These are exact up to float rounding for **any**
  input, not just the sampled tests.
- *Test-coverage statements (empirically verified on the tested configs):* whole-model converter parity
  and cuEq-fast-vs-reference parity hold to the tolerances above **for the MACE-style baseline configs
  and (lmax, correlation) combinations covered by the suite**; they are not a universal guarantee for
  all mace-torch variants.

---

## F. Backend distinction (which paths can be promised e3nn/MACE-exact)

| backend (`ictd_fix_product_backend`) | what it computes | parity status |
|---|---|---|
| **`ictd-bridge-u`** (default, conversion target) | native MACE `U` (e3nn) wrapped by the orthogonal `Q` feature bridge | **exact e3nn/MACE parity** (orthogonal bridge + verbatim MACE U; multilinear ⇒ exact). Primary path for native-MACE conversion. |
| **`native-mace`** | literal `MaceSymmetricContraction` on e3nn-frame features | **reference**; exact MACE by definition. Debug/reference backend. |
| **`cueq`** | cuEquivariance fused symmetric contraction; MACE weights mirrored/projected in; can pair with `angular_basis="e3nn"` | **parity to `<2e-5` (fp32)** vs the reference contraction (tested). Performance path. |
| **`ictd-pure-u`** | `U` built intrinsically from ICTC CG (`ictd_u_matrix_so3`) | **diagnostic** — close but **not bit-exact** to mace-torch; the converter explicitly **refuses** it for exact parity ([mace_converter.py:279-284](mace_ictc/interfaces/mace_converter.py)). |

Summary for claims:
- Paths that can be stated as strictly aligned with e3nn/MACE under the tested configurations:
  **`ictd-bridge-u`**, **`native-mace`**, and **`cueq`** (the last within its `<2e-5` fp32 tolerance).
- **`ictd-pure-u`** is the from-scratch ICTC reconstruction; it validates that the ICTC CG/`U` algebra
  is correct and self-contained, but it is a **diagnostic** path and should not be advertised as
  bit-exact to mace-torch.
- All parity statements are scoped to the **MACE-style baseline configurations** exercised by the
  converter and tests; this is not a claim of exact support for every possible MACE variant.

---

### Appendix: notation cross-reference to code

| symbol here | code object |
|---|---|
| `B_l` (harmonic basis in monomials) | `_harmonic_basis_t(l)` |
| `G_l` (Gaussian Gram) | `_gram_gaussian(l)` |
| `c_l(n)` (ICTC harmonics) | `direction_harmonics(n, l)` / `direction_harmonics_all` |
| `s_l` (component rescale) | `_direction_component_scale_cpu_f64(l)` |
| `Q_l` (ICTC→e3nn orthogonal) | `mace_basis._q_per_l_f64(lmax)[l]`, `SO3ToE3NNBasisBridge.q_{l}` |
| `C[m1,m2,m3]` (ICTC CG, natural parity) | `build_cg_tensor(l1,l2,l3)` |
| `C[m1,m2,m3]` (ICTC CG, all SO(3) paths) | `build_full_cg_tensor_so3(l1,l2,l3)` |
| `U^{(ν)}_{l_out}` (ICTC pure-U) | `ictd_u_matrix_so3(...)` |
| `U^{(ν)}` (MACE/e3nn) | `_mace_cg.U_matrix_real(...)` |
| MACE contraction (verbatim) | `MaceSymmetricContraction` |
| feature bridge | `SO3ToE3NNBasisBridge`, `BridgeUMaceSymmetricContractionSO3` |
| converter | `convert_mace_to_ictd` |
