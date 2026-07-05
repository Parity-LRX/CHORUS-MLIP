from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch


ROOT = Path(__file__).resolve().parent
FSCETP = Path("/Users/sara/Desktop/code/rebuild/FSCETP")
DOUBLE_COVER = Path("/Users/sara/Desktop/code/rebuild/ictd_doublecover_hamiltonian")
sys.path.insert(0, str(FSCETP))
sys.path.insert(0, str(DOUBLE_COVER))

from molecular_force_field.models.ictd_irreps_2d import (  # noqa: E402
    HarmonicFullyConnectedTensorProductO2,
    HarmonicFullyConnectedTensorProductSO2,
    build_cg_tensor_so2,
    parse_o2_active_irreps,
    so2_irrep_dim,
)
from ictd_doublecover_hamiltonian.backends import (  # noqa: E402
    DoubleCoverO3Irrep,
    build_double_cover_o3_matrix_basis,
    check_double_cover_o3_equivariance,
    double_cover_o3_rotation,
)
from ictd_doublecover_hamiltonian.backends.ictd_orbital import harmonic_values  # noqa: E402


def _rel(diff: torch.Tensor, ref: torch.Tensor) -> float:
    return float(diff.norm().item()) / max(float(ref.norm().item()), 1.0e-30)


def _so2_row_rotation(m: int, angle: float, dtype: torch.dtype) -> torch.Tensor:
    if m == 0:
        return torch.ones(1, 1, dtype=dtype)
    c = math.cos(m * angle)
    s = math.sin(m * angle)
    return torch.tensor([[c, s], [-s, c]], dtype=dtype)


def _so2_reflection(m: int, dtype: torch.dtype) -> torch.Tensor:
    if m == 0:
        return torch.ones(1, 1, dtype=dtype)
    return torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=dtype)


def _tp_so2(x: torch.Tensor, y: torch.Tensor, cg: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bi,bj,ijk->bk", x, y, cg)


def validate_so2_cg(mmax: int = 4, batch: int = 7) -> Dict[str, float | str]:
    dtype = torch.float64
    gen = torch.Generator().manual_seed(20260615)
    angle = 0.731
    max_rot_abs = 0.0
    max_rot_rel = 0.0
    max_ref_abs = 0.0
    max_ref_rel = 0.0
    npaths = 0
    for m1 in range(mmax + 1):
        for m2 in range(mmax + 1):
            candidates = {abs(m1 - m2), m1 + m2}
            for m3 in sorted(c for c in candidates if c <= mmax):
                cg = build_cg_tensor_so2(m1, m2, m3).to(dtype=dtype)
                x = torch.randn(batch, so2_irrep_dim(m1), generator=gen, dtype=dtype)
                y = torch.randn(batch, so2_irrep_dim(m2), generator=gen, dtype=dtype)
                base = _tp_so2(x, y, cg)

                d1 = _so2_row_rotation(m1, angle, dtype)
                d2 = _so2_row_rotation(m2, angle, dtype)
                d3 = _so2_row_rotation(m3, angle, dtype)
                lhs = _tp_so2(x @ d1, y @ d2, cg)
                rhs = base @ d3
                diff = lhs - rhs
                max_rot_abs = max(max_rot_abs, float(diff.abs().max().item()))
                max_rot_rel = max(max_rot_rel, _rel(diff, rhs))

                f1 = _so2_reflection(m1, dtype)
                f2 = _so2_reflection(m2, dtype)
                f3 = _so2_reflection(m3, dtype)
                lhs = _tp_so2(x @ f1, y @ f2, cg)
                rhs = base @ f3
                diff = lhs - rhs
                max_ref_abs = max(max_ref_abs, float(diff.abs().max().item()))
                max_ref_rel = max(max_ref_rel, _rel(diff, rhs))
                npaths += 1
    return {
        "test": f"SO2 CG paths m<= {mmax} ({npaths} paths)",
        "max_abs": max(max_rot_abs, max_ref_abs),
        "max_rel": max(max_rot_rel, max_ref_rel),
        "detail": f"rotation abs={max_rot_abs:.3e}, reflection abs={max_ref_abs:.3e}",
    }


def _random_so2_blocks(
    mmax: int,
    *,
    batch: int,
    mul: int,
    dtype: torch.dtype,
    gen: torch.Generator,
) -> Dict[int, torch.Tensor]:
    return {
        m: torch.randn(batch, mul, so2_irrep_dim(m), generator=gen, dtype=dtype)
        for m in range(mmax + 1)
    }


def _transform_so2_blocks(
    blocks: Dict[int, torch.Tensor],
    *,
    angle: float,
) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    for m, value in blocks.items():
        D = _so2_row_rotation(m, angle, value.dtype)
        out[m] = torch.einsum("bcd,dk->bck", value, D)
    return out


def validate_so2_module(mmax: int = 4) -> Dict[str, float | str]:
    dtype = torch.float64
    gen = torch.Generator().manual_seed(20260618)
    tp = HarmonicFullyConnectedTensorProductSO2(
        mul_in1=2,
        mul_in2=3,
        mul_out=2,
        mmax=mmax,
        internal_weights=True,
        normalization="none",
        internal_compute_dtype=torch.float64,
    ).to(dtype=dtype)
    with torch.no_grad():
        tp.weight.copy_(torch.randn(tp.weight.shape, generator=gen, dtype=dtype))

    x = _random_so2_blocks(mmax, batch=5, mul=2, dtype=dtype, gen=gen)
    y = _random_so2_blocks(mmax, batch=5, mul=3, dtype=dtype, gen=gen)
    base = tp(x, y)

    angle = -0.419
    lhs = tp(_transform_so2_blocks(x, angle=angle), _transform_so2_blocks(y, angle=angle))
    rhs = _transform_so2_blocks(base, angle=angle)
    abs_err, rel_err = _block_residual(lhs, rhs)
    return {
        "test": f"SO2 fully connected TP module m<= {mmax}",
        "max_abs": abs_err,
        "max_rel": rel_err,
        "detail": f"{tp.num_paths} weighted paths, random channel weights",
    }


def _random_o2_blocks(
    active: Iterable[str],
    *,
    batch: int,
    mul: int,
    dtype: torch.dtype,
    gen: torch.Generator,
) -> Dict[Tuple[str, int], torch.Tensor]:
    blocks: Dict[Tuple[str, int], torch.Tensor] = {}
    for key in parse_o2_active_irreps(list(active)):
        dim = 1 if key[0] == "scalar" else 2
        blocks[key] = torch.randn(batch, mul, dim, generator=gen, dtype=dtype)
    return blocks


def _transform_o2_blocks(
    blocks: Dict[Tuple[str, int], torch.Tensor],
    *,
    angle: float | None,
    reflection: bool,
) -> Dict[Tuple[str, int], torch.Tensor]:
    out: Dict[Tuple[str, int], torch.Tensor] = {}
    for key, value in blocks.items():
        dtype = value.dtype
        if key[0] == "scalar":
            sign = -1.0 if reflection and int(key[1]) < 0 else 1.0
            out[key] = value * sign
            continue
        m = int(key[1])
        if reflection:
            D = _so2_reflection(m, dtype)
        else:
            assert angle is not None
            D = _so2_row_rotation(m, angle, dtype)
        out[key] = torch.einsum("bcd,dk->bck", value, D)
    return out


def _block_residual(
    lhs: Dict[Tuple[str, int], torch.Tensor],
    rhs: Dict[Tuple[str, int], torch.Tensor],
) -> Tuple[float, float]:
    max_abs = 0.0
    sq_diff = 0.0
    sq_ref = 0.0
    for key in lhs:
        diff = lhs[key] - rhs[key]
        max_abs = max(max_abs, float(diff.abs().max().item()))
        sq_diff += float(diff.square().sum().item())
        sq_ref += float(rhs[key].square().sum().item())
    return max_abs, math.sqrt(sq_diff) / max(math.sqrt(sq_ref), 1.0e-30)


def validate_o2_module() -> Dict[str, float | str]:
    dtype = torch.float64
    active = ["0e", "0o", "1", "2", "3"]
    gen = torch.Generator().manual_seed(20260616)
    probe = HarmonicFullyConnectedTensorProductO2(
        mul_in1=2,
        mul_in2=3,
        mul_out=2,
        active_irreps=active,
        internal_weights=True,
        normalization="none",
    )
    allowed_paths = [
        path
        for path in probe.paths
        if not (
            (path[0][0] == "scalar" and int(path[0][1]) < 0 and path[1][0] == "freq")
            or (path[1][0] == "scalar" and int(path[1][1]) < 0 and path[0][0] == "freq")
        )
    ]
    tp = HarmonicFullyConnectedTensorProductO2(
        mul_in1=2,
        mul_in2=3,
        mul_out=2,
        active_irreps=active,
        internal_weights=True,
        normalization="none",
        allowed_paths=allowed_paths,
        internal_compute_dtype=torch.float64,
    ).to(dtype=dtype)
    with torch.no_grad():
        tp.weight.copy_(torch.randn(tp.weight.shape, generator=gen, dtype=dtype))

    x = _random_o2_blocks(active, batch=5, mul=2, dtype=dtype, gen=gen)
    y = _random_o2_blocks(active, batch=5, mul=3, dtype=dtype, gen=gen)
    base = tp(x, y)

    angle = -0.419
    lhs = tp(
        _transform_o2_blocks(x, angle=angle, reflection=False),
        _transform_o2_blocks(y, angle=angle, reflection=False),
    )
    rhs = _transform_o2_blocks(base, angle=angle, reflection=False)
    rot_abs, rot_rel = _block_residual(lhs, rhs)

    lhs = tp(
        _transform_o2_blocks(x, angle=None, reflection=True),
        _transform_o2_blocks(y, angle=None, reflection=True),
    )
    rhs = _transform_o2_blocks(base, angle=None, reflection=True)
    ref_abs, ref_rel = _block_residual(lhs, rhs)

    return {
        "test": "O2 TP reflection-closed subset, active 0e/0o/1/2/3",
        "max_abs": max(rot_abs, ref_abs),
        "max_rel": max(rot_rel, ref_rel),
        "detail": (
            f"rotation abs={rot_abs:.3e}, reflection abs={ref_abs:.3e}; "
            "excludes 0o x frequency -> frequency paths"
        ),
    }


def validate_double_cover_cg() -> Dict[str, float | str]:
    paths = [
        (
            DoubleCoverO3Irrep(0, 1, 1),
            DoubleCoverO3Irrep(1, 2, -1, two_s=0),
            DoubleCoverO3Irrep(1, 3, -1),
        ),
        (
            DoubleCoverO3Irrep(0, 1, 1),
            DoubleCoverO3Irrep(0, 1, 1),
            DoubleCoverO3Irrep(0, 0, 1, two_s=0),
        ),
        (
            DoubleCoverO3Irrep(2, 2, 1, two_s=2),
            DoubleCoverO3Irrep(1, 2, -1, two_s=0),
            DoubleCoverO3Irrep(2, 4, -1, two_s=2),
        ),
    ]
    max_abs = 0.0
    max_rel = 0.0
    details = []
    for idx, path in enumerate(paths):
        result = check_double_cover_o3_equivariance(
            *path,
            device="cpu",
            dtype=torch.complex128,
            ntrials=6,
            seed=881 + idx,
        )
        max_abs = max(max_abs, float(result["max_abs"]))
        max_rel = max(max_rel, float(result["max_rel"]))
        details.append(f"{result['path']}: abs={float(result['max_abs']):.3e}")
    return {
        "test": "double-cover O3 CG tensor products",
        "max_abs": max_abs,
        "max_rel": max_rel,
        "detail": "; ".join(details),
    }


def validate_double_cover_matrix_basis() -> Dict[str, float | str]:
    paths = [
        (
            DoubleCoverO3Irrep(0, 1, 1),
            DoubleCoverO3Irrep(0, 1, 1),
            DoubleCoverO3Irrep(0, 0, 1, two_s=0),
        ),
        (
            DoubleCoverO3Irrep(0, 1, 1),
            DoubleCoverO3Irrep(1, 1, -1),
            DoubleCoverO3Irrep(1, 2, -1, two_s=0),
        ),
        (
            DoubleCoverO3Irrep(1, 1, -1),
            DoubleCoverO3Irrep(1, 3, -1),
            DoubleCoverO3Irrep(1, 2, 1, two_s=0),
        ),
    ]
    axis = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float64)
    angle = 0.37
    max_abs = 0.0
    max_rel = 0.0
    for row_ir, col_ir, out_ir in paths:
        basis = build_double_cover_o3_matrix_basis(row_ir, col_ir, out_ir)
        dr = double_cover_o3_rotation(row_ir, axis, angle)
        dc = double_cover_o3_rotation(col_ir, axis, angle)
        dout = double_cover_o3_rotation(out_ir, axis, angle)
        for q in range(out_ir.dim):
            lhs = dr @ basis[q] @ dc.conj().T
            rhs = sum(dout[p, q] * basis[p] for p in range(out_ir.dim))
            diff = lhs - rhs
            max_abs = max(max_abs, float(diff.abs().max().item()))
            max_rel = max(max_rel, _rel(diff, rhs))
    return {
        "test": "double-cover O3 matrix bases row x col* -> out",
        "max_abs": max_abs,
        "max_rel": max_rel,
        "detail": f"{len(paths)} covariance paths",
    }


def validate_orbital_harmonics() -> Dict[str, float | str]:
    vectors = torch.randn(8, 3, generator=torch.Generator().manual_seed(20260617), dtype=torch.float64)
    axis = torch.tensor([0.4, -0.1, 0.2], dtype=torch.float64)
    angle = -0.23
    d1 = double_cover_o3_rotation(DoubleCoverO3Irrep(1, 2, -1, two_s=0), axis, angle).real
    perm = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    r_xyz = perm.T @ d1 @ perm
    max_abs = 0.0
    max_rel = 0.0
    for ell in range(4):
        d_l = double_cover_o3_rotation(
            DoubleCoverO3Irrep(ell, 2 * ell, 1 if ell % 2 == 0 else -1, two_s=0),
            axis,
            angle,
        ).real
        lhs = harmonic_values(ell, vectors @ r_xyz.T)
        rhs = harmonic_values(ell, vectors) @ d_l.T
        diff = lhs - rhs
        max_abs = max(max_abs, float(diff.abs().max().item()))
        max_rel = max(max_rel, _rel(diff, rhs))
    return {
        "test": "ICTC orbital harmonic values l<=3",
        "max_abs": max_abs,
        "max_rel": max_rel,
        "detail": "ordinary O3 parent carrier used by double-cover backend",
    }


def main() -> None:
    rows = [
        validate_so2_cg(),
        validate_so2_module(),
        validate_o2_module(),
        validate_orbital_harmonics(),
        validate_double_cover_cg(),
        validate_double_cover_matrix_basis(),
    ]
    lines = [
        "# SO2/O2 and double-cover O3 equivariance validation",
        "",
        "All tests were run in float64/complex128 on CPU. Residuals compare",
        "transform-then-apply against apply-then-transform.",
        "",
        "| test | max abs residual | max relative residual | detail |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['test']} | {float(row['max_abs']):.3e} | "
            f"{float(row['max_rel']):.3e} | {row['detail']} |"
        )
    text = "\n".join(lines) + "\n"
    out = ROOT / "representation_equivariance_validation.md"
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
