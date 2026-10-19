"""Parameter-preserving adapter to the separately installed ICTC-O3-HP library.

The neutral library's model configurator is deliberately not applied to CHORUS:
it owns neither the charged streams nor the recurrent Hermitian state. Only
documented operators are borrowed. Execution handles are unregistered objects,
so parameter identities, optimizers and state_dict keys remain unchanged.
"""

from __future__ import annotations

import importlib.util
import torch


def _core(model):
    from chorus.models.pure_cartesian_ictd_fix import PureCartesianICTDFix
    seen = set()
    while not isinstance(model, PureCartesianICTDFix) and id(model) not in seen:
        seen.add(id(model))
        child = getattr(model, "module", None)
        if child is None:
            child = getattr(model, "model", None)
        if not isinstance(child, torch.nn.Module):
            break
        model = child
    return model


def check_model(model, *, training=False, capture=False, symmetric_contraction=True,
                checkpoint_contraction=False):
    """Read-only structural coverage, not a numerical/performance qualification."""
    from chorus.models.pure_cartesian_ictd_fix import (
        PureCartesianICTDFix, ICTDBridgeUSymmetricContractionSO3,
    )
    core = _core(model)
    reasons = []
    if not isinstance(core, PureCartesianICTDFix):
        return {"supported": False, "reasons": ["not a CHORUS Cartesian backbone"]}
    from ._compat import backend_reasons
    reasons.extend(backend_reasons())
    if importlib.util.find_spec("triton") is None:
        reasons.append("Triton is unavailable")
    if (isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
            or (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1)):
        reasons.append("multi-device training is not qualified by this single-device adapter")
    parameters = tuple(core.parameters())
    if not parameters or any(p.dtype != torch.float32 or not p.is_cuda for p in parameters):
        reasons.append("all model parameters must already be CUDA FP32")
    elif len({p.device for p in parameters}) != 1:
        reasons.append("all parameters must be on the same CUDA device")
    if core.edge_compute_dtype != torch.float32:
        reasons.append("internal compute dtype must already be FP32")
    if getattr(core, "angular_basis", "ictd") != "ictd":
        reasons.append("requires the original Cartesian ICTC angular basis")
    if max(core.lmax, core.ictd_fix_edge_lmax) > 3:
        reasons.append("this CHORUS adapter currently qualifies hidden/edge L<=3")
    if getattr(core, "ictd_fix_route", "baseline") != "baseline":
        reasons.append("requires the baseline interaction route")
    if capture:
        reasons.append("native fusion runs without CUDA Graph; configure with capture=False")
    for index, layer in enumerate(core.interactions):
        if layer.tp.weight.requires_grad and not training:
            reasons.append(f"interaction {index}: TP base coefficients are trainable; not frozen automatically")
        if layer.tp_cueq is not None:
            reasons.append(f"interaction {index}: select native ICTC convolution before HP configuration")
        if layer.interaction_attn_heads:
            reasons.append(f"interaction {index}: interaction attention is not covered")
    for module in core.modules():
        if symmetric_contraction and isinstance(module, ICTDBridgeUSymmetricContractionSO3):
            if module.use_reduced_cg and not checkpoint_contraction:
                reasons.append("HP symmetric contraction requires native full CG")
            if any(c.correlation not in (1, 2, 3)
                   for c in module.symmetric_contractions.contractions):
                reasons.append("HP symmetric contraction supports correlation 1/2/3")
    return {"supported": not reasons, "reasons": list(dict.fromkeys(reasons))}


def _borrow_tp(source, *, streams=1, unit_weights=False):
    from mace_ictc.models.ictd_irreps import EdgeWeightedPathPreservingTensorProduct
    # Constructor initialization must not advance the caller's training RNG.
    with torch.random.fork_rng(devices=[]):
        tp = EdgeWeightedPathPreservingTensorProduct(
            channels=source.channels * streams, lmax=source.lmax,
            allowed_paths=list(source.paths), path_policy="full",
            irrep_normalization=source._normalization,
            path_normalization=source._path_normalization,
            internal_compute_dtype=source.internal_compute_dtype,
        ).to(device=source.weight.device, dtype=source.weight.dtype)
    if (tp.paths != source.paths or tp.path_offset != source.path_offset
            or tp.path_counts_by_l != source.path_counts_by_l):
        raise ValueError("ICTC-O3-HP TP path/layout schema does not match this CHORUS module")
    # Use the checkpoint's actual Cartesian constants, not regenerated approximations.
    source._get_cg_list(device=source.weight.device, dtype=source.weight.dtype)
    tp._cg_cpu_f64 = source._cg_cpu_f64
    tp._cg_cache_by_dev_dtype.clear()
    tp._proj_group_cache_by_dev_dtype.clear()
    tp._proj_group_view_cache_by_dev_dtype.clear()
    tp._refresh_scalar_identity_projector_fast_path()
    if unit_weights:
        # A trainable path coefficient is multiplied into the live radial gate.
        # Only this unregistered execution mirror is fixed, never the model's
        # coefficient. Autograd retains dW and the force-loss mixed derivative.
        tp.weight = torch.nn.Parameter(torch.ones_like(source.weight).repeat(1, streams),
                                       requires_grad=False)
    elif streams == 1:
        tp.weight = source.weight
    else:
        tp.weight = torch.nn.Parameter(source.weight.repeat(1, streams), requires_grad=False)
    return tp


class _PreweightedGates:
    """Per-call tensor marker; never cached on a module or across steps."""
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


class _LiveRadial:
    def __init__(self, hidden, bank):
        self.hidden, self.bank = hidden, bank


class _Message:
    def __init__(self, layer, *, layout, reduction, stream_fusion, coherent_fusion,
                 radial_fusion, radial_tile_edges, radial_gate_pipeline, radial_jvp, radial_destination,
                 radial_cache_gates, radial_options,
                 topology_backend, large_graph_fusion, gate_coefficient_fusion,
                 radial_coefficient_fusion, training_mixed_threads, training_split_first,
                 training_mixed_grouped, training_csr_splits, training_mixed_destination,
                 training_destination_first, training_mixed_compact, training_radial_tp_fusion,
                 training_mixed_cache, training_radial_cached, training_first_cache,
                 training_mixed_fuse_jvp, training_first_schedule, training_radial_first_bounded):
        from mace_ictc.hp import EdgeTensorProduct, GraphMessage
        self.source = layer.tp
        self.channels = layer.channels
        self.input_lmax = layer.input_lmax
        self.target_lmax = layer.target_lmax
        self.heads = layer.phase_heads
        self.angular = layer.phase_angular_channels
        self.phase = layer.phase_enabled
        self.stream_fusion = stream_fusion
        self.radial_fusion = radial_fusion
        self.radial_tile_edges = radial_tile_edges
        self.radial_gate_pipeline = radial_gate_pipeline
        # Keep the original first-layer radial path. Retain only the wide,
        # live-input gate bank from the current forward, never across calls.
        self.radial_jvp = radial_jvp and (not radial_cache_gates or layer.input_lmax > 0)
        self.radial_plan = None
        self.fused_radial_trunk = False
        self.radial_trunk_tile = 16
        self.fold_trainable_weight = self.source.weight.requires_grad
        self.gate_coefficient_fusion = gate_coefficient_fusion
        self.radial_coefficient_fusion = radial_coefficient_fusion
        self.training_radial_tp_fusion = training_radial_tp_fusion
        if radial_jvp:
            from .radial_jvp import RadialJVP
            self.radial_plan = RadialJVP(layer.fc, **radial_options)
        tp = _borrow_tp(self.source, unit_weights=self.fold_trainable_weight)
        self.edge = EdgeTensorProduct(tp, input_lmax=layer.input_lmax,
                                      edge_lmax=layer.lmax, layout="mul_ir")
        self.neutral = GraphMessage(tp, input_lmax=layer.input_lmax,
                                    edge_lmax=layer.lmax, layout=layout,
                                    reduction=reduction, topology_backend=topology_backend)
        self.neutral_large = None
        if (large_graph_fusion and not self.phase and self.channels == 128
                and self.input_lmax == 0 and self.target_lmax == 2
                and tuple(tp.paths) == ((0, 0, 0), (0, 1, 1), (0, 2, 2))):
            self.neutral_large = GraphMessage(tp, input_lmax=layer.input_lmax,
                edge_lmax=layer.lmax, layout=layout, edge_chunk_size=1 << 30,
                reduction=reduction, topology_backend=topology_backend)
            self.neutral_large._options['_csr_neighbor_warps'] = 4
        self.streams = None
        self.coherent = None
        if self.phase and stream_fusion and coherent_fusion and self.heads <= 4:
            from mace_ictc.hp import CoherentGraphMessage
            self.coherent = CoherentGraphMessage(
                tp, input_lmax=layer.input_lmax, edge_lmax=layer.lmax,
                heads=self.heads, angular=self.angular, large_graph_fusion=large_graph_fusion,
                mixed_threads=training_mixed_threads, training_split_first=training_split_first,
                mixed_grouped=training_mixed_grouped, training_csr_splits=training_csr_splits,
                mixed_destination=training_mixed_destination,
                training_destination_first=training_destination_first,
                mixed_compact=training_mixed_compact, mixed_cache=training_mixed_cache,
                first_cache=training_first_cache, mixed_fuse_jvp=training_mixed_fuse_jvp,
                first_schedule=training_first_schedule)
            self.coherent.distance_destination = radial_destination
            self.coherent.radial_cached = training_radial_cached
            self.coherent.radial_first_bounded = training_radial_first_bounded
            self.coherent.distance_cached_gates = radial_cache_gates and layer.input_lmax > 0
            self.coherent.frozen_input_fusion = bool(radial_cache_gates and large_graph_fusion
                                                     and layer.input_lmax == 0)
        elif self.phase and stream_fusion:
            triple = _borrow_tp(self.source, streams=3, unit_weights=self.fold_trainable_weight)
            self.streams = GraphMessage(triple, input_lmax=layer.input_lmax,
                                        edge_lmax=layer.lmax, layout=layout,
                                        reduction=reduction, topology_backend=topology_backend)
        self.phase_indices = torch.tensor(
            [((out_l*self.heads if self.angular else 0)+head)
             for _, _, out_l in self.source.paths for head in range(self.heads)],
            dtype=torch.long, device=self.source.weight.device)
        self._version = self.source.weight._version

    def _validate(self):
        weight = self.source.weight
        if ((not self.fold_trainable_weight and weight is not self.edge.tp.weight)
                or weight.requires_grad != self.fold_trainable_weight
                or weight.device != self.edge._mask.device or weight.dtype != torch.float32):
            raise RuntimeError("HP model device/dtype/TP trainability changed: reconfigure or select reference")
        if self.streams is not None and not self.fold_trainable_weight and weight._version != self._version:
            with torch.no_grad():
                self.streams.tp.weight.copy_(weight.repeat(1, 3))
            self._version = weight._version

    def edges(self, blocks, harmonics, gates, src):
        self._validate()
        if self.fold_trainable_weight:
            gates = self._weighted_gates(gates)
        packed = self.edge.pack_input({l: value.index_select(0, src) for l, value in blocks.items()})
        sh = torch.cat([harmonics[l].squeeze(-2) for l in range(self.edge.edge_lmax + 1)], -1)
        return self.edge.unpack_output(self.edge(packed, sh, gates))

    def radial_eligible(self, fc):
        return (self.radial_fusion and self.coherent is not None and self.input_lmax > 0
                and not fc.training
                and isinstance(fc[-1], torch.nn.Linear) and fc[-1].bias is None
                and not fc[-1].weight.requires_grad and fc[-1].in_features <= 128)

    def _weighted_gates(self, gates):
        if self.gate_coefficient_fusion:
            from mace_ictc.hp import path_weighted_gates
            return path_weighted_gates(gates, self.source.weight)
        return gates*self.source.weight.reshape(1, -1)

    def radial_coefficient_eligible(self, fc):
        return (self.radial_coefficient_fusion and self.fold_trainable_weight
                and isinstance(fc[-1], torch.nn.Linear) and fc[-1].bias is None)

    def radial_inputs(self, edge_feats, fc, edge_length=None):
        if self.radial_coefficient_eligible(fc):
            from mace_ictc.hp import path_weighted_linear
            hidden = edge_feats
            for layer in tuple(fc.children())[:-1]:
                hidden = layer(hidden)
            if self.training_radial_tp_fusion and self.coherent is not None and self.input_lmax > 0:
                return _LiveRadial(hidden, fc[-1].weight*self.source.weight.reshape(-1, 1))
            return _PreweightedGates(path_weighted_linear(hidden, fc[-1].weight, self.source.weight))
        if (self.radial_jvp and self.coherent is not None and edge_length is not None
                and self.radial_plan.eligible()):
            return edge_length, self.radial_plan, "distance"
        if self.radial_eligible(fc):
            hidden = edge_feats
            for layer in tuple(fc.children())[:-1]:
                hidden = layer(hidden)
            return hidden, fc[-1].weight
        if self.fused_radial_trunk:
            from .radial_trunk import FrozenRadialTrunk
            if FrozenRadialTrunk.eligible(fc, edge_feats):
                return fc[-1](FrozenRadialTrunk(self.radial_trunk_tile)(fc, edge_feats))
        return fc(edge_feats)

    def _coefficients(self, cos, sin):
        c, s = [], []
        for _, _, out_l in self.source.paths:
            start = out_l * self.heads if self.angular else 0
            shape = (-1, self.heads, 1)
            c.append(cos[:, start:start+self.heads].reshape(shape).expand(
                -1, -1, self.channels // self.heads).reshape(-1, self.channels))
            s.append(sin[:, start:start+self.heads].reshape(shape).expand(
                -1, -1, self.channels // self.heads).reshape(-1, self.channels))
        return torch.stack(c, 1), torch.stack(s, 1)

    def __call__(self, blocks, harmonics, gates, src, dst, mask, cos, sin):
        self._validate()
        if isinstance(gates, _LiveRadial):
            return self.coherent.radial_training(blocks, harmonics, gates.hidden, gates.bank,
                src, dst, mask, cos, sin, tile_edges=self.radial_tile_edges)
        preweighted = isinstance(gates, _PreweightedGates)
        if preweighted:
            gates = gates.value
        if self.fold_trainable_weight and not preweighted:
            if isinstance(gates, tuple):
                raise ValueError("Frozen radial inference fusion is not a trainable TP path")
            gates = self._weighted_gates(gates)
        if not self.phase:
            operation = self.neutral_large if self.neutral_large is not None and src.numel() >= 327680 else self.neutral
            return operation(blocks, harmonics, gates, src, dst, mask)
        if self.coherent is not None:
            if isinstance(gates, tuple):
                if len(gates) == 3:
                    return self.coherent.distance(blocks, harmonics, gates[0], gates[1], src, dst,
                                                  mask, cos, sin, tile_edges=self.radial_tile_edges,
                                                  gate_pipeline=self.radial_gate_pipeline)
                return self.coherent.radial(blocks, harmonics, *gates, src, dst, mask, cos, sin,
                                            tile_edges=self.radial_tile_edges,
                                            gate_pipeline=self.radial_gate_pipeline)
            return self.coherent(blocks, harmonics, gates, src, dst, mask, cos, sin)
        if self.streams is None:
            from .phase_gates import modulate_gates
            pieces = [self.neutral(blocks, harmonics, gates, src, dst, mask)]
            for coefficient in (cos, sin):
                modulated = modulate_gates(gates, coefficient, self.phase_indices,
                                           self.channels, self.heads)
                pieces.append(self.neutral(blocks, harmonics, modulated, src, dst, mask))
            return {l: torch.stack([piece[l] for piece in pieces], 1)
                    for l in range(self.target_lmax + 1)}
        # Expand only node channels and scalar gates, never [E,3,P*C,2L+1].
        from .phase_gates import pack_phase_gates
        triple_blocks = {l: value.repeat(1, 3, 1) for l, value in blocks.items()}
        triple_gates = pack_phase_gates(gates, cos, sin, self.phase_indices,
                                        self.channels, self.heads)
        output = self.streams(triple_blocks, harmonics, triple_gates, src, dst, mask)
        result = {}
        for l in range(self.target_lmax + 1):
            paths = self.source.path_counts_by_l[l]
            result[l] = output[l].reshape(-1, paths, 3, self.channels, 2*l+1).permute(
                0, 2, 1, 3, 4).reshape(blocks[0].shape[0], 3, paths*self.channels, 2*l+1)
        return result


def _borrow_contraction(source, *, training, checkpoint_contraction):
    from mace_ictc.models.pure_cartesian_ictd_fix import ICTDBridgeUSymmetricContractionSO3
    from mace_ictc.hp import configure_symmetric_contraction

    class BorrowedContraction(ICTDBridgeUSymmetricContractionSO3):
        def __init__(self):
            torch.nn.Module.__init__(self)
            for name in ("channels", "lmax", "target_lmax", "use_reduced_cg",
                         "hidden_irreps", "target_irreps"):
                setattr(self, name, getattr(source, name))
            self.basis_bridge = source.basis_bridge
            self.symmetric_contractions = source.symmetric_contractions

    borrowed = BorrowedContraction()
    borrowed.train(source.training)
    configure_symmetric_contraction(borrowed, policy="generated", training=training,
                                    reference_buffer_policy="device", species_schedule="indexed",
                                    **({"allow_reduced_cg": True} if checkpoint_contraction else {}))
    return borrowed


_HANDLES = ("_hp_message", "_hp_contraction", "_hp_channel_linear",
            "_hp_activation", "_hp_density", "_hp_harmonics",
            "_hp_species_linear", "_hp_phase_coefficients", "_hp_phase_trunk",
            "_hp_terminal_phase", "_hp_terminal_rank_message", "_hp_large_graph_fusion")


def _clear(core):
    for module in core.modules():
        for name in _HANDLES:
            if name in module.__dict__:
                object.__delattr__(module, name)


def configure_model(model, *, training=False, policy="auto", layout="fused_pack",
                    reduction="auto", stream_fusion=True, density_fusion=True,
                    symmetric_contraction=True, checkpoint_contraction=False,
                    species_fusion=True, phase_fusion=True, coherent_fusion=True,
                    phase_context_mixed_fusion=False,
                    phase_coefficient_mixed_fusion=False,
                    gate_coefficient_fusion=False,
                    radial_coefficient_fusion=False,
                    training_species_fusion=False,
                    training_mixed_threads=0,
                    training_split_first=False,
                    training_mixed_grouped=False,
                    training_mixed_compact=False,
                    training_mixed_cache=0,
                    training_mixed_fuse_jvp=False,
                    training_first_cache=False,
                    training_first_schedule='legacy',
                    training_radial_tp_fusion=False,
                    training_radial_cached=False,
                    training_radial_first_bounded=False,
                    training_terminal_fusion=False,
                    training_csr_splits=0,
                    training_mixed_destination=False,
                    training_destination_first=0,
                    terminal_fusion=True, radial_fusion=False, radial_tile_edges=65536,
                    radial_gate_pipeline=False,
                    radial_jvp=False, radial_destination=False, radial_cache_gates=False,
                    topology_backend="native", capture=False, large_graph_fusion=True):
    """Enable optional eager HP operators; keep all model parameters/checkpoint keys.

    generated requires structural coverage. auto reports a reference fallback.
    Neither choice freezes coefficients, casts the model, retrains, or changes
    the charged-state equations. Configure after .to()/load_state_dict(); handles
    are intentionally not serialized. Reconfigure after weights/device/dtype change.
    """
    if policy not in ("reference", "auto", "generated"):
        raise ValueError("policy must be reference, auto or generated")
    if layout not in ("packed", "fused_pack") or reduction not in ("auto", "atomic", "csr"):
        raise ValueError("invalid HP layout or reduction")
    if topology_backend not in ("native", "device"):
        raise ValueError("topology_backend must be native or device")
    for value in (training, stream_fusion, density_fusion, symmetric_contraction,
                  checkpoint_contraction, species_fusion, phase_fusion, coherent_fusion,
                  phase_context_mixed_fusion,
                  phase_coefficient_mixed_fusion,
                  gate_coefficient_fusion,
                  radial_coefficient_fusion,
                  training_species_fusion,
                  training_split_first,
                  training_mixed_grouped,
                  training_mixed_compact,
                  training_mixed_fuse_jvp,
                  training_first_cache,
                  training_radial_tp_fusion,
                  training_radial_cached,
                  training_radial_first_bounded,
                  training_terminal_fusion,
                  training_mixed_destination,
                  terminal_fusion, radial_fusion, radial_gate_pipeline, radial_jvp, radial_destination,
                  radial_cache_gates,
                  capture, large_graph_fusion):
        if not isinstance(value, bool):
            raise TypeError("HP execution switches must be booleans")
    core = _core(model)
    if not isinstance(radial_tile_edges, int) or isinstance(radial_tile_edges, bool) or radial_tile_edges < 1:
        raise ValueError("radial_tile_edges must be a positive integer")
    if radial_destination and not (radial_jvp and radial_gate_pipeline):
        raise ValueError('radial_destination requires radial_jvp and radial_gate_pipeline')
    if radial_cache_gates and not (radial_jvp and radial_gate_pipeline and radial_destination):
        raise ValueError('radial_cache_gates requires radial_jvp, radial_gate_pipeline and radial_destination')
    if phase_context_mixed_fusion and not (training and phase_fusion):
        raise ValueError('phase_context_mixed_fusion requires training=True and phase_fusion=True')
    if phase_coefficient_mixed_fusion and not (training and phase_fusion):
        raise ValueError('phase_coefficient_mixed_fusion requires training=True and phase_fusion=True')
    if gate_coefficient_fusion and not training:
        raise ValueError('gate_coefficient_fusion requires training=True')
    if radial_coefficient_fusion and not training:
        raise ValueError('radial_coefficient_fusion requires training=True')
    if training_species_fusion and not (training and species_fusion):
        raise ValueError('training_species_fusion requires training=True and species_fusion=True')
    if isinstance(training_mixed_threads, bool) or training_mixed_threads not in (0, 32, 64, 128):
        raise ValueError('training_mixed_threads must be 0, 32, 64 or 128')
    if training_mixed_threads and not training:
        raise ValueError('training_mixed_threads requires training=True')
    if training_split_first and not training:
        raise ValueError('training_split_first requires training=True')
    if training_mixed_grouped and not training:
        raise ValueError('training_mixed_grouped requires training=True')
    if training_mixed_compact and not training:
        raise ValueError('training_mixed_compact requires training=True')
    if isinstance(training_mixed_cache, bool) or training_mixed_cache not in (0, 8, 16, 32, 64):
        raise ValueError('training_mixed_cache must be 0, 8, 16, 32 or 64')
    if training_mixed_cache and (not training or (training_radial_tp_fusion and not training_radial_cached)):
        raise ValueError('training_mixed_cache requires training and excludes inline, but supports cached radial fusion')
    if training_mixed_fuse_jvp and not training_mixed_cache:
        raise ValueError('training_mixed_fuse_jvp requires training_mixed_cache')
    if training_first_cache and not training:
        raise ValueError('training_first_cache requires training=True')
    if training_first_schedule not in ('legacy', 'pruned', 'grouped', 'compact', 'compact_grouped'):
        raise ValueError('Invalid training_first_schedule')
    if training_first_schedule != 'legacy' and (not training or training_first_cache):
        raise ValueError('training_first_schedule requires training and excludes training_first_cache')
    if training_radial_tp_fusion and not (training and radial_coefficient_fusion and coherent_fusion):
        raise ValueError('training_radial_tp_fusion requires training, radial_coefficient_fusion and coherent_fusion')
    if training_radial_cached and not training_radial_tp_fusion:
        raise ValueError('training_radial_cached requires training_radial_tp_fusion')
    if training_radial_first_bounded and not training_radial_cached:
        raise ValueError('training_radial_first_bounded requires training_radial_cached')
    if training_terminal_fusion and not (training and terminal_fusion and density_fusion):
        raise ValueError('training_terminal_fusion requires training, terminal_fusion and density_fusion')
    if training_mixed_destination and not training:
        raise ValueError('training_mixed_destination requires training=True')
    if isinstance(training_csr_splits, bool) or training_csr_splits not in (0, 1, 2, 4, 8):
        raise ValueError('training_csr_splits must be 0, 1, 2, 4 or 8')
    if training_csr_splits and (not training or training_split_first):
        raise ValueError('training_csr_splits requires training and excludes training_split_first')
    if isinstance(training_destination_first, bool) or training_destination_first not in (0, 1, 2, 4):
        raise ValueError('training_destination_first must be 0, 1, 2 or 4')
    if training_destination_first and (not training or training_split_first or training_csr_splits):
        raise ValueError('training_destination_first requires training and excludes other explicit first-adjoint schedules')
    if policy == "reference":
        _clear(core)
        return {"requested": policy, "selected": "reference", "reasons": [], "counts": {}}
    coverage = check_model(model, training=training, capture=capture,
                           symmetric_contraction=symmetric_contraction,
                           checkpoint_contraction=checkpoint_contraction)
    if not coverage["supported"]:
        if policy == "generated":
            raise ValueError("CHORUS HP unavailable: " + "; ".join(coverage["reasons"]))
        _clear(core)
        return {"requested": policy, "selected": "reference", "reasons": coverage["reasons"], "counts": {}}
    try:
        from mace_ictc.hp import channel_linear, normalized_silu, direction_harmonics
    except ImportError as error:
        raise ImportError("mace_ictc is not the required ICTC-O3-HP installation") from error
    from chorus.models.pure_cartesian_ictd_fix import (
        ICTDBridgeUSymmetricContractionSO3, PhaseHermitianFullLResidual,
        PathPreservingLinearSO3, _Normalize2MomSiLU, ElementConditionedLinearSO3,
        ElementConditionedScalarLinear,
        PersistentChargedUpdate,
    )
    from chorus.models.ictd_irreps import EquivariantChannelLinearSO3, EquivariantChannelLinearSO3Rect
    from .density import DensityPlan
    from .phase_coefficients import phase_coefficients
    if phase_coefficient_mixed_fusion:
        from functools import partial
        phase_coefficients = partial(phase_coefficients, mixed_adjoint=True)
    from .phase_trunk import PhaseTrunk
    species_channel_linear = None
    if species_fusion and core.num_elements <= 4:
        if training_species_fusion:
            from mace_ictc.hp import training_species_channel_linear as species_channel_linear
        else:
            from mace_ictc.hp import species_channel_linear
    from mace_ictc.models.fp32_policy import strict_fp32
    strict_fp32()
    before = {name: (id(value), value.requires_grad) for name, value in core.named_parameters()}
    keys = tuple(core.state_dict())
    prepared = []
    large_enabled = bool(large_graph_fusion and not training and not core.training
        and not any(p.requires_grad for p in core.parameters())
        and torch.cuda.get_device_capability(next(core.parameters()).device) == (8, 0))
    prepared.append((core, '_hp_large_graph_fusion', large_enabled))
    counts = {"graph_messages": 0, "three_stream_messages": 0, "hermitian_density": 0,
              "symmetric_contraction": 0, "channel_maps": 0, "activations": 0,
              "species_maps": 0, "phase_coefficients": 0, "coherent_messages": 0,
              "terminal_projections": 0, "radial_messages": 0, "radial_jvp_messages": 0,
              "radial_destination_messages": 0, "radial_cached_gate_messages": 0,
              "trainable_tp_gate_folds": 0}
    counts["phase_context_mixed_adjoints"] = 0
    counts["fused_trainable_gate_adjoints"] = 0
    counts["radial_coefficient_fused_heads"] = 0
    for layer in core.interactions:
        prepared.append((layer, "_hp_message", _Message(
            layer, layout=layout, reduction=reduction, stream_fusion=stream_fusion,
            coherent_fusion=coherent_fusion, radial_fusion=radial_fusion,
            radial_tile_edges=radial_tile_edges,
            radial_gate_pipeline=radial_gate_pipeline,
            radial_jvp=radial_jvp, radial_destination=radial_destination,
            radial_cache_gates=radial_cache_gates,
            radial_options=dict(r_max=core.max_radius,
                number_of_basis=core.number_of_basis, function_type=core.function_type,
                polynomial_cutoff_p=core.polynomial_cutoff_p,
                sqrt_num_basis_norm=core.radial_sqrt_num_basis),
            topology_backend=topology_backend, large_graph_fusion=large_enabled,
            gate_coefficient_fusion=gate_coefficient_fusion,
            radial_coefficient_fusion=radial_coefficient_fusion,
            training_mixed_threads=training_mixed_threads, training_split_first=training_split_first,
            training_mixed_grouped=training_mixed_grouped, training_csr_splits=training_csr_splits,
            training_mixed_compact=training_mixed_compact,
            training_radial_tp_fusion=training_radial_tp_fusion,
            training_radial_cached=training_radial_cached,
            training_radial_first_bounded=training_radial_first_bounded,
            training_mixed_cache=training_mixed_cache,
            training_mixed_fuse_jvp=training_mixed_fuse_jvp,
            training_first_cache=training_first_cache,
            training_first_schedule=training_first_schedule,
            training_mixed_destination=training_mixed_destination,
            training_destination_first=training_destination_first)))
        counts["graph_messages"] += 1
        counts["trainable_tp_gate_folds"] += int(layer.tp.weight.requires_grad)
        counts["fused_trainable_gate_adjoints"] += int(gate_coefficient_fusion and layer.tp.weight.requires_grad)
        counts["radial_coefficient_fused_heads"] += int(prepared[-1][2].radial_coefficient_eligible(layer.fc))
        counts["three_stream_messages"] += int(layer.phase_enabled and stream_fusion)
        counts["coherent_messages"] += int(prepared[-1][2].coherent is not None)
        counts["radial_messages"] += int(not radial_jvp and prepared[-1][2].radial_eligible(layer.fc))
        counts["radial_jvp_messages"] += int(prepared[-1][2].radial_jvp and prepared[-1][2].coherent is not None
                                              and prepared[-1][2].radial_plan.eligible())
        counts["radial_destination_messages"] += int(radial_destination and prepared[-1][2].radial_jvp
            and prepared[-1][2].coherent is not None and prepared[-1][2].radial_plan.eligible())
        counts["radial_cached_gate_messages"] += int(radial_cache_gates and prepared[-1][2].radial_jvp
            and prepared[-1][2].coherent is not None and prepared[-1][2].radial_plan.eligible())
        if layer.phase_enabled and phase_fusion:
            prepared.append((layer, "_hp_phase_coefficients", phase_coefficients))
            prepared.append((layer, "_hp_phase_trunk", PhaseTrunk(layer, auto_network=large_enabled,
                mixed_adjoint=phase_context_mixed_fusion)))
            counts["phase_coefficients"] += 1
            counts["phase_context_mixed_adjoints"] += int(phase_context_mixed_fusion)
    for module in core.modules():
        if species_channel_linear is not None and isinstance(
                module, (ElementConditionedLinearSO3, ElementConditionedScalarLinear,
                         PhaseHermitianFullLResidual)):
            prepared.append((module, "_hp_species_linear", species_channel_linear))
            counts["species_maps"] += 1
        if isinstance(module, (EquivariantChannelLinearSO3, EquivariantChannelLinearSO3Rect)):
            for adapter in module.adapters.values():
                prepared.append((adapter, "_hp_channel_linear", channel_linear))
                counts["channel_maps"] += 1
        if isinstance(module, (PathPreservingLinearSO3, PhaseHermitianFullLResidual,
                               PersistentChargedUpdate)):
            prepared.append((module, "_hp_channel_linear", channel_linear))
            counts["channel_maps"] += 1
        if isinstance(module, PhaseHermitianFullLResidual) and density_fusion:
            prepared.append((module, "_hp_density", DensityPlan(module)))
            counts["hermitian_density"] += 1
        if isinstance(module, ICTDBridgeUSymmetricContractionSO3) and symmetric_contraction:
            prepared.append((module, "_hp_contraction", _borrow_contraction(
                module, training=training, checkpoint_contraction=checkpoint_contraction)))
            counts["symmetric_contraction"] += 1
        if isinstance(module, _Normalize2MomSiLU):
            prepared.append((module, "_hp_activation", normalized_silu))
            counts["activations"] += 1
    prepared.append((core, "_hp_harmonics", direction_harmonics))
    terminal_key = str(len(core.interactions)-1)
    if (terminal_fusion and density_fusion and (training_terminal_fusion or
            (not training and not core.training and not any(p.requires_grad for p in core.parameters())))
            and core.ictd_fix_phase_scope == "persistent"
            and core.ictd_fix_phase_density_pairs in {"full", "full-nonlinear", "full-nonlinear-readout"}
            and terminal_key in core.charged_updates
            and terminal_key in core.phase_adapters
            and isinstance(core.phase_adapters[terminal_key], PhaseHermitianFullLResidual)):
        from .terminal import TerminalChargedProjection
        update = core.charged_updates[terminal_key]
        prepared.append((update, "_hp_terminal_phase", TerminalChargedProjection(
            update, core.phase_adapters[terminal_key], live=training_terminal_fusion)))
        counts["terminal_projections"] += 1
    if (training_terminal_fusion and core.ictd_fix_phase_scope == "final"
            and core.ictd_fix_phase_density_pairs in {"full", "full-nonlinear", "full-nonlinear-readout"}
            and terminal_key in core.phase_adapters
            and isinstance(core.phase_adapters[terminal_key], PhaseHermitianFullLResidual)):
        from .terminal import TerminalRankMessage
        rank = TerminalRankMessage(core.interactions[-1], core.phase_adapters[terminal_key], live=True)
        if not rank.eligible():
            raise ValueError("Live terminal message/rank fold does not cover this normalization")
        prepared.append((core.interactions[-1], "_hp_terminal_rank_message", rank))
        counts["terminal_projections"] += 1
    _clear(core)
    for module, name, value in prepared:
        object.__setattr__(module, name, value)
    after = {name: (id(value), value.requires_grad) for name, value in core.named_parameters()}
    if before != after or keys != tuple(core.state_dict()):
        _clear(core)
        raise RuntimeError("HP configuration changed parameter identity/trainability or checkpoint schema")
    return {"requested": policy, "selected": "generated", "counts": counts, "reasons": [],
            "layout": layout, "stream_fusion": stream_fusion, "density_fusion": density_fusion,
            "symmetric_contraction": symmetric_contraction,
            "checkpoint_contraction": checkpoint_contraction,
            "species_fusion": species_fusion, "phase_fusion": phase_fusion,
            "coherent_fusion": coherent_fusion,
            "phase_context_mixed_fusion": phase_context_mixed_fusion,
            "phase_coefficient_mixed_fusion": phase_coefficient_mixed_fusion,
            "gate_coefficient_fusion": gate_coefficient_fusion,
            "radial_coefficient_fusion": radial_coefficient_fusion,
            "training_species_fusion": training_species_fusion,
            "training_mixed_threads": training_mixed_threads,
            "training_split_first": training_split_first,
            "training_mixed_grouped": training_mixed_grouped,
            "training_mixed_compact": training_mixed_compact,
            "training_mixed_cache": training_mixed_cache,
            "training_mixed_fuse_jvp": training_mixed_fuse_jvp,
            "training_first_cache": training_first_cache,
            "training_first_schedule": training_first_schedule,
            "training_radial_tp_fusion": training_radial_tp_fusion,
            "training_radial_cached": training_radial_cached,
            "training_radial_first_bounded": training_radial_first_bounded,
            "training_terminal_fusion": training_terminal_fusion,
            "training_csr_splits": training_csr_splits,
            "training_mixed_destination": training_mixed_destination,
            "training_destination_first": training_destination_first,
            "terminal_fusion": terminal_fusion,
            "radial_fusion": radial_fusion,
            "radial_tile_edges": radial_tile_edges,
            "radial_gate_pipeline": radial_gate_pipeline,
            "radial_jvp": radial_jvp,
            "radial_destination": radial_destination,
            "radial_cache_gates": radial_cache_gates,
            "large_graph_fusion": large_enabled,
            "topology_backend": topology_backend,
            "execution": "eager", "arithmetic": "IEEE FP32; TF32 disabled"}
