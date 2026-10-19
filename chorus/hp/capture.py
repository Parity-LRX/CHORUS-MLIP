"""Fixed-capacity CUDA graph execution of the native HP energy/force model."""
from __future__ import annotations

import torch
from .runtime import _core, configure_model


class CapturedInference:
    """Replay live positions/species/topology/cells at fixed N/E/B.

    Model parameters must already be CUDA FP32, frozen and in eval mode.
    This class neither casts nor freezes them. Outputs are borrowed device
    buffers overwritten by the next call; use clone_outputs=True to retain them.
    Neighbor construction and atomic reference energies (E0) are not included.
    New capacities or model updates require constructing a new capture.
    Calls must be serialized on the CUDA stream used for construction.
    """
    def __init__(self, model, inputs, *, clone_outputs=False, validate_values=True,
                 virial=False, **hp_options):
        from mace_ictc.hp import energy_and_forces
        if virial:
            from .inference import energy_forces_virial
            evaluate = energy_forces_virial
        else:
            evaluate = energy_and_forces
        if torch.is_inference_mode_enabled():
            raise ValueError("Forces require autograd; do not use inference_mode")
        self.model, self.core = model, _core(model)
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("Capture requires an already frozen eval model")
        if "capture" in hp_options or hp_options.get("policy", "generated") == "reference":
            raise ValueError("CapturedInference requires the native generated HP path")
        hp_options.setdefault("policy", "generated")
        hp_options.setdefault("topology_backend", "device")
        if hp_options["topology_backend"] != "device":
            raise ValueError("Capture requires live device topology")
        self.plan = configure_model(model, **hp_options)
        self.clone_outputs = bool(clone_outputs)
        self.validate_values = bool(validate_values)
        self.closed = False
        self._inputs = tuple(v.detach().clone() for v in inputs)
        self._validate(inputs, initial=True)
        device = self._inputs[0].device
        self._stream = torch.cuda.current_stream(device)
        self._state = self._signature()
        self._handles = self._handle_signature()
        self._python_state = self._python_signature()
        previous = {name: self.core.__dict__.get(name, None)
                    for name in ("skip_input_validation", "preserve_edge_order")}
        # These are capture-time Python guards only. Replay never executes
        # Python forward, so restore the ordinary model's guards afterwards.
        self.core.skip_input_validation = True
        self.core.preserve_edge_order = True
        try:
            warm = torch.cuda.Stream(device=device)
            warm.wait_stream(self._stream)
            with torch.cuda.stream(warm):
                for _ in range(3):
                    evaluate(model, *self._inputs)
            self._stream.wait_stream(warm)
            self._stream.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._outputs = evaluate(model, *self._inputs)
            self.graph.replay()
        finally:
            for name, value in previous.items():
                if value is None:
                    self.core.__dict__.pop(name, None)
                else:
                    setattr(self.core, name, value)
        self._state = self._signature()
        self._handles = self._handle_signature()
        self._python_state = self._python_signature()

    def _signature(self):
        return tuple((id(v), v._version, v.requires_grad, v.dtype, v.device)
                     for v in (*self.model.parameters(), *self.model.buffers()))

    def _handle_signature(self):
        return tuple((id(module), key, id(value)) for module in self.core.modules()
                     for key, value in module.__dict__.items() if key.startswith("_hp_"))

    def _python_signature(self):
        # Phase controls and other Python execution choices are baked into a
        # captured graph just as surely as tensor parameters are. Version-only
        # checks would silently replay an old intervention after such an edit.
        simple = (str, int, float, bool, type(None), torch.dtype, torch.device)
        return tuple((id(module), key, value) for module in self.core.modules()
                     for key, value in module.__dict__.items() if isinstance(value, simple))

    def _validate(self, inputs, *, initial=False):
        if len(inputs) != 7:
            raise ValueError("Expected positions, atomic_numbers, batch, src, dst, shifts, cell")
        p, z, batch, src, dst, shifts, cell = inputs
        n, e = p.shape[0], src.numel()
        if (p.shape != (n, 3) or z.shape != (n,) or batch.shape != (n,)
                or src.shape != (e,) or dst.shape != (e,) or shifts.shape != (e, 3)
                or cell.ndim != 3 or cell.shape[1:] != (3, 3) or n == 0):
            raise ValueError("Invalid graph shape or empty atom set")
        sample = next(self.model.parameters())
        for index, value in enumerate(inputs):
            dtype = torch.int64 if index in (1, 2, 3, 4) else torch.float32
            if value.device != sample.device or value.dtype != dtype or not value.is_cuda:
                raise ValueError("Graph inputs must be CUDA FP32 (indices int64) on the model device")
            if not initial and value.shape != self._inputs[index].shape:
                raise ValueError("N/E/B changed; construct a new capture")
        if self.validate_values:
            if not bool(torch.isfinite(p).all() & torch.isfinite(shifts).all() & torch.isfinite(cell).all()):
                raise ValueError("Nonfinite geometry")
            if (bool(((src < 0) | (src >= n) | (dst < 0) | (dst >= n)).any())
                    or bool(((batch < 0) | (batch >= cell.shape[0])).any())):
                raise ValueError("Node or graph index outside bounds")
            if bool(((z < 0) | (z >= self.core.atomic_number_to_index.numel())).any()):
                raise ValueError("Atomic number outside mapping")
            if bool((self.core.atomic_number_to_index[z] < 0).any()):
                raise ValueError("Unsupported atomic species")

    def __call__(self, *inputs):
        if self.closed:
            raise RuntimeError("Capture has been closed")
        if (self.model.training or self._signature() != self._state
                or self._handle_signature() != self._handles
                or self._python_signature() != self._python_state):
            raise RuntimeError("Model state/execution changed; construct a new capture")
        if torch.cuda.current_stream(self._inputs[0].device) != self._stream:
            raise RuntimeError("Use the capture's construction stream; calls must not overlap")
        self._validate(inputs)
        for target, value in zip(self._inputs, inputs):
            target.copy_(value)
        self.graph.replay()
        return ({k: v.clone() for k, v in self._outputs.items()}
                if self.clone_outputs else dict(self._outputs))

    def close(self):
        """Release the private graph and its owned buffers; the model is retained."""
        if not self.closed:
            self._stream.synchronize()
            self.graph.reset()
            self._outputs = None
            self._inputs = ()
            self.closed = True


def capture_inference(model, inputs, **options):
    """Construct CapturedInference after loading/finalizing the original model."""
    return CapturedInference(model, inputs, **options)
