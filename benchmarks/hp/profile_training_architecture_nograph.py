"""Train fixed architecture initializations with the established no-graph step.

The saved inference-scaling states are reused verbatim. Constructor trainability
is restored for training, including live convolution TP coefficients. These are
architecture-throughput experiments, not trained accuracy-matched checkpoints.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

from architecture_cases import CASES, chorus_spec
from chorus.synthetic import build_model
from no_cuda_graph import forbid_cuda_graph
import profile_training_nograph as benchmark


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--architecture-case", choices=CASES, required=True)
    parser.add_argument("--scope", choices=("final", "persistent"), required=True)
    args, remaining = parser.parse_known_args()
    spec = chorus_spec(args.architecture_case, args.scope)
    original_loader = benchmark.load_trainable

    def load_architecture(initialization):
        saved = torch.load(initialization, map_location="cpu", weights_only=False)
        if saved["spec"] != spec:
            raise ValueError("Fixed architecture initialization specification mismatch")
        # Reuse the state's actual coupling basis. Regenerating U from a newer
        # constructor/cache can have a different number of paths, and must not
        # silently redefine the architecture or discard its saved coefficients.
        u_state = []
        for index in range(spec["interactions"]):
            prefix = f"products.{index}.symmetric_contractions.symmetric_contractions.contractions."
            tensors = {key.removeprefix(prefix): value for key, value in saved["state"].items()
                       if key.startswith(prefix) and key.rsplit(".", 1)[-1] in
                       {"U_matrix_1", "U_matrix_2", "U_matrix_3"}}
            if not tensors:
                raise ValueError(f"Missing original product-U buffers at interaction {index}")
            u_state.append(tensors)
        model = build_model(
            channels=spec["channels"], lmax=spec["lmax"],
            num_interaction=spec["interactions"], route="baseline",
            product_backend="ictd-bridge-u", dtype=torch.float32,
            device=torch.device("cuda"), correlation=3, avg_num_neighbors=50.,
            ictd_fix_use_reduced_cg=True,
            ictd_fix_phase_mode="final-full-l-residual",
            ictd_fix_phase_amplitude="softplus",
            ictd_fix_phase_placement="pre-product-full-l",
            ictd_fix_phase_density_pairs="full-nonlinear",
            ictd_fix_phase_density_rank=16, ictd_fix_phase_scope=args.scope,
            ictd_fix_u_tensor_state=u_state,
            ictd_fix_freeze_conv_tp_weight=False)
        model.load_state_dict(saved["state"], strict=True)
        trainability = {name: p.requires_grad for name, p in model.named_parameters()}
        if not all(layer.tp.weight.requires_grad for layer in model.interactions):
            raise RuntimeError("Architecture benchmark must train every convolution TP coefficient")
        return model.train(), trainability

    sys.argv = [sys.argv[0], *remaining]
    benchmark.load_trainable = load_architecture
    try:
        with forbid_cuda_graph() as graph_policy:
            benchmark.main(graph_policy)
    finally:
        benchmark.load_trainable = original_loader
    output = Path(remaining[remaining.index("--out") + 1])
    result = json.loads(output.read_text())
    result.update(architecture=spec, initialization="fixed_constructor_state_not_trained",
                  coupling_basis="original_initialization_U_buffers",
                  convolution_tp_coefficients="trainable_not_frozen")
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
