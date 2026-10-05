# Validation report

## Verdict

The compiled causal dataset is internally consistent and suitable for the stated mechanism-panel analysis. All expected dataset-by-seed cells are present, all reported outcomes are finite, nominal Full/Aligned parameter counts match, and test data were not used for checkpoint selection.

## Automated checks

```json
{
  "aligned_rows_expected_20": true,
  "all_aligned_metrics_finite": true,
  "all_primary_metrics_finite": true,
  "all_probe_force_metrics_finite": true,
  "all_test_selection_flags_false": true,
  "common_shift_within_0.05_mev_per_angstrom": true,
  "dataset_seed_coverage": {
    "aligned": {
      "3bpa": [
        20260616,
        20260617,
        20260618,
        20260619,
        20260620
      ],
      "bucky": [
        20260616,
        20260617,
        20260618,
        20260619,
        20260620
      ],
      "mal": [
        20260616,
        20260617,
        20260618,
        20260619,
        20260620
      ],
      "zro": [
        20260616,
        20260617,
        20260618,
        20260619,
        20260620
      ]
    },
    "observational": {
      "3bpa": [
        20260616,
        20260617,
        20260618,
        20260619,
        20260620
      ],
      "bucky": [
        20260616,
        20260617,
        20260618,
        20260619,
        20260620
      ],
      "mal": [
        20260616,
        20260617,
        20260618,
        20260619,
        20260620
      ],
      "zro": [
        20260616,
        20260617,
        20260618,
        20260619,
        20260620
      ]
    },
    "primary": {
      "3bpa": [
        20260616,
        20260617,
        20260618,
        20260619,
        20260620
      ],
      "bucky": [
        20260616,
        20260617,
        20260618,
        20260619,
        20260620
      ],
      "mal": [
        20260616,
        20260617,
        20260618,
        20260619,
        20260620
      ],
      "zro": [
        20260616,
        20260617,
        20260618,
        20260619,
        20260620
      ]
    },
    "probes": {
      "3bpa": [
        20260616,
        20260617,
        20260618,
        20260619,
        20260620
      ],
      "bucky": [
        20260616,
        20260617,
        20260618,
        20260619,
        20260620
      ],
      "mal": [
        20260616,
        20260617,
        20260618,
        20260619,
        20260620
      ],
      "zro": [
        20260616,
        20260617,
        20260618,
        20260619,
        20260620
      ]
    }
  },
  "full_aligned_nominal_parameter_counts_equal": true,
  "maximum_common_shift_force_change_mev_per_angstrom": 0.004291534423828125,
  "maximum_native_replay_force_mae_difference_mev_per_angstrom": 0.0004408348512576765,
  "native_replay_within_0.1_mev_per_angstrom": true,
  "observational_rows_expected_20": true,
  "passed": true,
  "primary_rows_expected_80": true,
  "probe_rows_expected_20": true
}
```

## Counterfactual audit

```json
{
  "aligned_checkpoint_sha256": "1e6f083d586d8412f4e32c3e9157b081159c2935fad9dde7f6bf8d7f2db982ea",
  "aligned_state_elements": 181131,
  "aligned_state_tensor_count": 73,
  "deterministic_algorithms_forced": false,
  "full_checkpoint_sha256": "a13e506bd6166184dc891fc863cf64f0ee9adb786cbf1ff1da29b354c70dcfd4",
  "full_state_elements": 181131,
  "full_state_tensor_count": 73,
  "inactive_aligned_phase_head_elements": 32,
  "interpretation": "Nominal architecture and state schema are matched. Aligned-Full makes the 32-element phase head functionally inactive; CUDA deterministic algorithms were not forced, so the paired retraining is a learning-system potential-outcome estimate rather than bitwise replay.",
  "phase_head_key": "interactions.1.phase_head.weight",
  "same_state_tensor_schema": true
}
```

## Parquet audit

```json
{
  "atomic_finite": true,
  "atomic_rows": 4964675,
  "frame_finite": true,
  "frame_rows": 112455,
  "pyarrow_version": "24.0.0"
}
```

## Residual limitations

1. The four datasets form a deliberately selected mechanism panel, so the equal-dataset macro is not an estimate for all materials or all equivariant networks.
2. Five seeds per dataset resolve paired consistency but leave dataset-level heterogeneity only coarsely estimated.
3. The observational rung is post hoc and association-only.
4. GPU deterministic algorithms were not forced. Pairing controls the declared data and optimiser randomness, while bitwise trajectory identity is not guaranteed.
