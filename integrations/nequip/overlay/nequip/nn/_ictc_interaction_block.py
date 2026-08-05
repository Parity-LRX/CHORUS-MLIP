"""NequIP interaction block using the ICTC Cartesian coupling basis."""

from typing import Callable, Dict, Optional

import torch
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct, Linear
from torch_runstats.scatter import scatter

from nequip.data import AtomicDataDict
from nequip.nn.nonlinearities import ShiftedSoftPlus
from ._chorus import (
    CHORUS_CHARGED_IMAG_KEY,
    CHORUS_CHARGED_REAL_KEY,
    HermitianDensityResidual,
    PersistentChargedUpdate,
)
from ._graph_mixin import GraphModuleMixin


def _natural_parity(l_value: int) -> int:
    return 1 if int(l_value) % 2 == 0 else -1


def _validate_natural_irreps(irreps, *, name: str, common_mul: bool) -> int:
    irreps = o3.Irreps(irreps).simplify()
    seen = set()
    multiplicities = set()
    for mul, ir in irreps:
        if ir.l in seen:
            raise ValueError(f"{name} must contain one block per angular degree")
        seen.add(ir.l)
        multiplicities.add(mul)
        if ir.p != _natural_parity(ir.l):
            raise ValueError(
                f"{name} must use natural parity (-1)^l; got {mul}x{ir}"
            )
    if common_mul and len(multiplicities) != 1:
        raise ValueError(f"{name} must use one common channel multiplicity")
    return next(iter(multiplicities))


def _split_by_l(tensor: torch.Tensor, irreps) -> Dict[int, torch.Tensor]:
    irreps = o3.Irreps(irreps).simplify()
    blocks: Dict[int, torch.Tensor] = {}
    for (mul, ir), component_slice in zip(irreps, irreps.slices()):
        blocks[int(ir.l)] = tensor[..., component_slice].reshape(
            *tensor.shape[:-1], int(mul), int(ir.dim)
        )
    return blocks


def _merge_by_l(blocks: Dict[int, torch.Tensor], irreps) -> torch.Tensor:
    irreps = o3.Irreps(irreps).simplify()
    pieces = []
    for mul, ir in irreps:
        block = blocks[int(ir.l)]
        pieces.append(block.reshape(*block.shape[:-2], int(mul) * int(ir.dim)))
    return torch.cat(pieces, dim=-1)


class ICTCInteractionBlock(GraphModuleMixin, torch.nn.Module):
    """Path-preserving ICTC replacement for NequIP's e3nn interaction TP.

    The surrounding NequIP architecture, radial network, self connection,
    nonlinearities, and readout are unchanged.  Only the angular tensor-product
    projector is replaced.
    """

    avg_num_neighbors: Optional[float]
    use_sc: bool

    def __init__(
        self,
        irreps_in,
        irreps_out,
        invariant_layers=1,
        invariant_neurons=8,
        avg_num_neighbors=None,
        use_sc=True,
        nonlinearity_scalars: Dict[int, Callable] = {"e": "silu"},
        chorus_enabled: bool = False,
        chorus_rank: int = 8,
        chorus_hidden_channels: int = 32,
        chorus_scale_init: float = 0.0,
        chorus_persistent: bool = False,
        chorus_update_enabled: bool = True,
        chorus_clear_state: bool = False,
    ) -> None:
        super().__init__()
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                AtomicDataDict.EDGE_EMBEDDING_KEY,
                AtomicDataDict.EDGE_ATTRS_KEY,
                AtomicDataDict.NODE_FEATURES_KEY,
                AtomicDataDict.NODE_ATTRS_KEY,
            ],
            my_irreps_in={
                AtomicDataDict.EDGE_EMBEDDING_KEY: o3.Irreps(
                    [
                        (
                            irreps_in[
                                AtomicDataDict.EDGE_EMBEDDING_KEY
                            ].num_irreps,
                            (0, 1),
                        )
                    ]
                )
            },
            irreps_out={AtomicDataDict.NODE_FEATURES_KEY: irreps_out},
        )

        self.avg_num_neighbors = avg_num_neighbors
        self.use_sc = use_sc
        feature_irreps_in = self.irreps_in[AtomicDataDict.NODE_FEATURES_KEY]
        feature_irreps_out = self.irreps_out[AtomicDataDict.NODE_FEATURES_KEY]
        edge_irreps = self.irreps_in[AtomicDataDict.EDGE_ATTRS_KEY]
        channels = _validate_natural_irreps(
            feature_irreps_in, name="ICTC node irreps", common_mul=True
        )
        _validate_natural_irreps(
            feature_irreps_out, name="ICTC output irreps", common_mul=False
        )
        edge_mul = _validate_natural_irreps(
            edge_irreps, name="ICTC edge irreps", common_mul=False
        )
        if edge_mul != 1 or any(mul != 1 for mul, _ in edge_irreps.simplify()):
            raise ValueError("ICTC edge attributes require one harmonic per l")

        input_by_l = {ir.l: ir for _, ir in feature_irreps_in.simplify()}
        edge_by_l = {ir.l: ir for _, ir in edge_irreps.simplify()}
        output_by_l = {ir.l: ir for _, ir in feature_irreps_out.simplify()}
        allowed_paths = []
        for l1, ir1 in input_by_l.items():
            for l2, ir2 in edge_by_l.items():
                for l3, ir3 in output_by_l.items():
                    if ir3 in ir1 * ir2:
                        allowed_paths.append((int(l1), int(l2), int(l3)))
        if not allowed_paths:
            raise ValueError("ICTC interaction has no allowed angular paths")
        lmax = max(max(path) for path in allowed_paths)

        try:
            from chorus.mace_basis import orthogonal_Q_blocks
            from chorus.models.ictd_irreps import (
                EdgeWeightedPathPreservingTensorProduct,
            )
        except ImportError as exc:
            raise ImportError(
                "ICTCInteractionBlock requires the MACE-ICTC/CHORUS operator "
                "package on PYTHONPATH"
            ) from exc

        self.linear_1 = Linear(
            feature_irreps_in,
            feature_irreps_in,
            internal_weights=True,
            shared_weights=True,
        )
        self.tp = EdgeWeightedPathPreservingTensorProduct(
            channels=int(channels),
            lmax=int(lmax),
            allowed_paths=allowed_paths,
            internal_compute_dtype=torch.get_default_dtype(),
        )
        with torch.no_grad():
            self.tp.weight.fill_(1.0)
        self.tp.weight.requires_grad_(False)
        self.tp.fold_cg_to_e3nn(
            orthogonal_Q_blocks(lmax, dtype=torch.float64, device="cpu")
        )

        irreps_mid = o3.Irreps(
            [
                (
                    int(channels) * int(self.tp.path_counts_by_l.get(l_value, 0)),
                    output_by_l[l_value],
                )
                for l_value in sorted(output_by_l)
                if int(self.tp.path_counts_by_l.get(l_value, 0)) > 0
            ]
        ).simplify()
        self.fc = FullyConnectedNet(
            [
                self.irreps_in[
                    AtomicDataDict.EDGE_EMBEDDING_KEY
                ].num_irreps,
            ]
            + invariant_layers * [invariant_neurons]
            + [self.tp.num_paths * int(channels)],
            {
                "ssp": ShiftedSoftPlus,
                "silu": torch.nn.functional.silu,
            }[nonlinearity_scalars["e"]],
        )
        self.linear_2 = Linear(
            irreps_mid,
            feature_irreps_out,
            internal_weights=True,
            shared_weights=True,
        )
        self.chorus = None
        self.chorus_update = None
        self.chorus_persistent = bool(chorus_persistent)
        self.chorus_clear_state = bool(chorus_clear_state)
        if chorus_enabled:
            self.chorus = HermitianDensityResidual(
                irreps_message=irreps_mid,
                irreps_node=feature_irreps_in,
                num_elements=self.irreps_in[
                    AtomicDataDict.NODE_ATTRS_KEY
                ].num_irreps,
                edge_invariant_dim=self.irreps_in[
                    AtomicDataDict.EDGE_EMBEDDING_KEY
                ].num_irreps,
                rank=chorus_rank,
                hidden_channels=chorus_hidden_channels,
                avg_num_neighbors=avg_num_neighbors,
                scale_init=chorus_scale_init,
            )
            if self.chorus_persistent and bool(chorus_update_enabled):
                self.chorus_update = PersistentChargedUpdate(
                    self.chorus.irreps_charged
                )
        elif self.chorus_persistent:
            raise ValueError("persistent CHORUS requires chorus_enabled=True")

        self.sc = None
        if self.use_sc:
            self.sc = FullyConnectedTensorProduct(
                feature_irreps_in,
                self.irreps_in[AtomicDataDict.NODE_ATTRS_KEY],
                feature_irreps_out,
            )
        self._feature_irreps_in = feature_irreps_in.simplify()
        self._edge_irreps = edge_irreps.simplify()
        self._irreps_mid = irreps_mid

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_invariants = data[AtomicDataDict.EDGE_EMBEDDING_KEY]
        gates = self.fc(edge_invariants)
        x = data[AtomicDataDict.NODE_FEATURES_KEY]
        edge_src = data[AtomicDataDict.EDGE_INDEX_KEY][1]
        edge_dst = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        sc = None
        if self.sc is not None:
            sc = self.sc(x, data[AtomicDataDict.NODE_ATTRS_KEY])

        x = self.linear_1(x)
        node_blocks = _split_by_l(x[edge_src], self._feature_irreps_in)
        edge_blocks = _split_by_l(
            data[AtomicDataDict.EDGE_ATTRS_KEY], self._edge_irreps
        )
        message_blocks = self.tp(node_blocks, edge_blocks, gates)
        edge_features = _merge_by_l(message_blocks, self._irreps_mid)

        chorus_residual = None
        if self.chorus is not None:
            if self.chorus_persistent:
                charged_real, charged_imag = self.chorus.aggregate_charged(
                    edge_messages=edge_features,
                    node_features=x,
                    edge_invariants=edge_invariants,
                    edge_src=edge_src,
                    edge_dst=edge_dst,
                    num_nodes=len(x),
                )
                if CHORUS_CHARGED_REAL_KEY in data:
                    if self.chorus_update is None:
                        raise RuntimeError(
                            "persistent charged state reached a layer without "
                            "a charged update"
                        )
                    charged_real, charged_imag = self.chorus_update(
                        data[CHORUS_CHARGED_REAL_KEY],
                        data[CHORUS_CHARGED_IMAG_KEY],
                        charged_real,
                        charged_imag,
                    )
                data[CHORUS_CHARGED_REAL_KEY] = charged_real
                data[CHORUS_CHARGED_IMAG_KEY] = charged_imag
                chorus_residual = self.chorus.neutral_from_charged(
                    charged_real,
                    charged_imag,
                    node_attrs=data[AtomicDataDict.NODE_ATTRS_KEY],
                )
                if self.chorus_clear_state:
                    data.pop(CHORUS_CHARGED_REAL_KEY)
                    data.pop(CHORUS_CHARGED_IMAG_KEY)
            else:
                chorus_residual = self.chorus(
                    edge_messages=edge_features,
                    node_features=x,
                    node_attrs=data[AtomicDataDict.NODE_ATTRS_KEY],
                    edge_invariants=edge_invariants,
                    edge_src=edge_src,
                    edge_dst=edge_dst,
                    num_nodes=len(x),
                )
        if self.avg_num_neighbors is not None:
            edge_features = edge_features.div(
                float(self.avg_num_neighbors) ** 0.5
            )
        x = scatter(edge_features, edge_dst, dim=0, dim_size=len(x))
        if chorus_residual is not None:
            x = x + chorus_residual
        x = self.linear_2(x)
        if sc is not None:
            x = x + sc
        data[AtomicDataDict.NODE_FEATURES_KEY] = x
        return data
