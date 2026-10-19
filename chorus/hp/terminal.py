"""Exact frozen-inference fold of terminal charged update and rank projection."""
from __future__ import annotations

import torch


class RankProjectedDoublet:
    """Internal carrier: the existing rank projection has already been applied."""
    _hp_rank_doublet = True

    def __init__(self, owner, blocks, reference):
        self.owner, self.blocks, self.reference = owner, blocks, reference


class TerminalRankMessage:
    """Fold the last message map into its existing charged rank projection.

    The neutral stream is unchanged. Charged streams do not materialize the
    full-width message merely to project it to the checkpoint's original rank.
    This is exact linear reassociation, not a new rank or model parameter.
    """
    def __init__(self, layer, adapter, *, live=False):
        self.layer, self.adapter = layer, adapter
        self.live = live
        self._signature, self._weights = None, None

    def eligible(self):
        from chorus.models.pure_cartesian_ictd_fix import PerLScaleSO3
        return ((self.live or (not self.layer.training and not self.adapter.training))
                and self.layer.phase_norm is None
                and self.layer.phase_normalization == 'avg-neighbors'
                and isinstance(self.layer.message_output_scale, (torch.nn.Identity, PerLScaleSO3))
                and (self.live or not any(p.requires_grad for p in (*self.layer.message_linear.parameters(),
                    *self.layer.message_output_scale.parameters(), *self.adapter.parameters()))))

    def __call__(self, blocks, denominator):
        if not self.eligible():
            raise RuntimeError('Terminal rank fold became ineligible: reconfigure HP execution')
        from mace_ictc.hp import channel_linear
        parameters = (*self.layer.message_linear.parameters(),
                      *self.layer.message_output_scale.parameters(), *self.adapter.parameters())
        signature = tuple((id(p), p._version, p.dtype, p.device) for p in parameters)
        def weights():
            scales = (self.layer.message_output_scale.log_scale.exp()
                if hasattr(self.layer.message_output_scale, 'log_scale') else None)
            return {l: (self.adapter.rank_projections[str(l)].weight
                        @ self.layer.message_linear.weights[str(l)])
                        * (1. if scales is None else scales[l])
                    for l in range(self.adapter.lmax+1)}
        if self.live:
            bank = weights()
        else:
            if signature != self._signature:
                with torch.no_grad():
                    self._weights = weights()
                self._signature = signature
            bank = self._weights
        neutral = self.layer.message_linear({l: value[:, 0] for l, value in blocks.items()})/denominator
        rank = {l: channel_linear(value[:, 1:], bank[l])/denominator
                for l, value in blocks.items()}
        return neutral, RankProjectedDoublet(self, rank, neutral)

    def finish(self, projected, *, node_attrs, node_type_idx, gate_features=None,
               species_embedding=None):
        density = self.adapter._hp_density(projected.blocks)
        return self.adapter._mix_density_blocks(density, reference=projected.reference,
            node_attrs=node_attrs, node_type_idx=node_type_idx, gate_features=gate_features,
            species_embedding=species_embedding)


class TerminalChargedProjection:
    """Consume the last full-width charged state directly at Hermitian rank.

    The original parameters remain owned by their modules. Only their frozen
    product is cached, with identity/version/dtype/device checks on every eager
    call. Trainable weights and training mode use the ordinary model path.
    CUDA graph replay is guarded by CapturedInference's complete model signature.
    """
    def __init__(self, update, adapter, *, live=False):
        self.update, self.adapter = update, adapter
        self.live = live
        self._signature = None
        self._weights = None

    def __call__(self, previous, incoming, *, node_attrs, node_type_idx,
                 gate_features=None, species_embedding=None):
        parameters = tuple(self.update.parameters()) + tuple(self.adapter.parameters())
        if (not self.live and (self.update.training or self.adapter.training
                or any(p.requires_grad for p in parameters))):
            return None
        from chorus.models.pure_cartesian_ictd_fix import _split_irreps
        from mace_ictc.hp import channel_linear
        signature = tuple((id(p), p._version, p.dtype, p.device) for p in parameters)
        def weights():
            gates = self.update.memory_logits.sigmoid()
            return {
                    l: torch.cat((
                        gates[l] * (self.adapter.rank_projections[str(l)].weight
                                    @ self.update.previous_linear.adapters[str(l)].weight),
                        (1.0-gates[l]) * (self.adapter.rank_projections[str(l)].weight
                                          @ self.update.incoming_linear.adapters[str(l)].weight),
                    ), dim=-1)
                    for l in range(self.adapter.lmax+1)
                }
        if self.live:
            bank = weights()
        else:
            if signature != self._signature:
                with torch.no_grad():
                    self._weights = weights()
                self._signature = signature
            bank = self._weights
        before = _split_irreps(previous, self.adapter.channels, self.adapter.lmax)
        added = _split_irreps(incoming, self.adapter.channels, self.adapter.lmax)
        rank_doublet = {
            l: channel_linear(torch.cat((before[l], added[l]), dim=-2), bank[l])
            for l in range(self.adapter.lmax+1)
        }
        density = self.adapter._hp_density(rank_doublet)
        return self.adapter._mix_density_blocks(
            density, reference=incoming, node_attrs=node_attrs,
            node_type_idx=node_type_idx, gate_features=gate_features,
            species_embedding=species_embedding,
        )
