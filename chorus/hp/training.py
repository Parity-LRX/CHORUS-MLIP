"""Ordinary autograd with explicit optimizer-facing gradient targets."""
import torch


def backward_parameters(loss, model):
    """Compute all live parameter gradients, not unused input-leaf gradients.

    Conservative forces and their complete parameter mixed derivatives stay in
    the graph. This is not valid for a caller that also consumes position/input
    leaf gradients after backward. Single-device execution is qualified here.
    """
    if (isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
            or (torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1)):
        raise ValueError("Parameter-target backward is qualified for single-device training only")
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not parameters:
        raise ValueError("Parameter-target backward needs live trainable parameters")
    # autograd.grad supplies the engine's explicit target metadata to native
    # mixed adjoints. Accumulate exactly as backward would, so trainer gradient
    # accumulation and optimizer state continue to have their original meaning.
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    for parameter, gradient in zip(parameters, gradients):
        if gradient is not None:
            if parameter.grad is None:
                parameter.grad = gradient
            else:
                parameter.grad.add_(gradient)
