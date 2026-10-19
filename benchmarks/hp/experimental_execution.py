"""Explicit remote-screen configurations; not library defaults."""
import re


def apply_experiment(model, name):
    options = {'force_reduction': 'csr'} if name.endswith('_forcecsr') else {}
    name = name.removesuffix('_forcecsr')
    if name == 'none':return options
    if name.startswith('persistent_'):
        for layer in model.interactions:
            message = getattr(layer, '_hp_message', None)
            coherent = getattr(message, 'coherent', None)
            if coherent is not None and name in ('persistent_distance_destination',
                    'persistent_distance_inline_forward', 'persistent_distance_inline',
                    'persistent_distance_cached', 'persistent_distance_cached_deep'):
                coherent.distance_destination = True
                coherent.distance_projection = ('forward' if name.endswith('_inline_forward')
                    else 'both' if name.endswith('_inline') else 'gemm')
                coherent.distance_cached_gates = name.startswith('persistent_distance_cached')
            if (message is not None and name == 'persistent_distance_cached_deep'
                    and layer.input_lmax == 0):message.radial_jvp = False
            if message is not None:message.fused_radial_trunk = 'trunk' in name
            if message is not None and re.search(r'trunk([0-9]+)', name):
                message.radial_trunk_tile = int(re.search(r'trunk([0-9]+)', name).group(1))
            if coherent is not None and layer.input_lmax == 0:
                coherent.frozen_input_fusion = True
            if coherent is not None and 'split' in name:
                coherent.destination_splits = int(re.search(r'split([0-9]+)', name).group(1))
            if coherent is not None and 'groups' in name:
                coherent.destination_group_paths = int(re.search(r'groups([0-9]+)', name).group(1))
        forward_match = re.search(r'forward([0-9]+)', name)
        if forward_match:
            import mace_ictc.hp.coherent as coherent
            from mace_ictc.hp._coherent_forward_tuned import forward as tuned_forward
            warps = int(forward_match.group(1))
            def forward(ctx, x, sh, w, g, real, imag, src, dst, mask, p, projection):
                ctx.set_materialize_grads(False)
                ctx.save_for_backward(x, sh, w, g, real, imag, src, dst, mask)
                ctx.p, ctx.projection = p, projection
                out = tuple((x.new_empty if src.numel() else x.new_zeros)(
                    x.shape[0], 3, 2*l+1, width//(2*l+1)).transpose(-1, -2)
                    for l, _, width in coherent._outputs(p))
                tuned_forward(x, sh, w, g, real, imag, src, dst, mask, p, projection, out,
                    accumulator_limit=16, neighbor_warps=warps, readonly=True)
                return out
            coherent._Coherent.forward = staticmethod(forward)
        return options
    import mace_ictc.models.message_topology as topology_module
    from chorus.hp.density import DensityPlan
    topology_module.topology = topology_module.stable_topology
    DensityPlan.vjp = DensityPlan.gather_vjp
    if 'dest' in name:
        import mace_ictc.hp._coherent_csr_split as split_module
        from mace_ictc.hp._coherent_destination import adjoint as destination_adjoint
        splits = int(re.search(r'dest([0-9]+)', name).group(1))
        def adjoint(*args, **kw):
            kw.update(splits=splits, atomic_x='atomic' in name, factored='factor' in name,
                      coalesced='coalesce' in name)
            return destination_adjoint(*args, **kw)
        split_module.adjoint = adjoint
    for layer in model.interactions:
        if getattr(layer, '_hp_phase_trunk', None) is not None:
            layer._hp_phase_trunk.fused_tail = 'phase' in name
            layer._hp_phase_trunk.fused_network = 'network' in name
            layer._hp_phase_trunk.network_preprojected = 'preproject' in name
    if 'nativeharm' in name:object.__setattr__(model, '_hp_harmonics', None)
    if 'first4' in name or 'fullcsr' in name:
        for layer in model.interactions:
            if not layer.phase_enabled:
                options = layer._hp_message.neutral._options
                if 'first4' in name:options['_csr_neighbor_warps'] = 4
                if 'fullcsr' in name:options['_message_chunk_edges'] = 1 << 30
    if 'forward4' in name:
        import mace_ictc.hp.coherent as coherent
        from mace_ictc.hp._coherent_forward_tuned import forward as tuned_forward
        def forward(ctx, x, sh, w, g, real, imag, src, dst, mask, p, projection):
            ctx.set_materialize_grads(False)
            ctx.save_for_backward(x, sh, w, g, real, imag, src, dst, mask)
            ctx.p, ctx.projection = p, projection
            out = tuple((x.new_empty if src.numel() else x.new_zeros)(
                x.shape[0], 3, 2*l+1, width//(2*l+1)).transpose(-1, -2)
                for l, _, width in coherent._outputs(p))
            tuned_forward(x, sh, w, g, real, imag, src, dst, mask, p, projection, out,
                accumulator_limit=16, neighbor_warps=4, readonly=True)
            return out
        coherent._Coherent.forward = staticmethod(forward)
    if 'rank' in name:
        from chorus.hp.terminal import TerminalRankMessage
        if model.ictd_fix_phase_scope != 'final':
            raise ValueError('Final terminal rank fold cannot be applied to Persistent')
        layer = model.interactions[-1]
        adapter = model.phase_adapters[str(len(model.interactions)-1)]
        handle = TerminalRankMessage(layer, adapter)
        if not handle.eligible():raise ValueError('Checkpoint ineligible for exact terminal rank fold')
        object.__setattr__(layer, '_hp_terminal_rank_message', handle)
    return {**options, 'fused_geometry': 'geom' in name, 'fused_forces': 'nativeforces' not in name}
