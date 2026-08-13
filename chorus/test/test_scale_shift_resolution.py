from unittest.mock import patch

from chorus.cli.train import _resolve_atomic_inter_scale_shift


def test_explicit_scale_and_shift_skip_h5_scan():
    with patch(
        "chorus.cli.train._atomic_inter_scale_shift_from_h5",
        side_effect=AssertionError("H5 scan must not run"),
    ):
        result = _resolve_atomic_inter_scale_shift(
            "/does/not/exist.h5",
            atomic_energy_keys=[1],
            atomic_energy_values=[-1.0],
            scaling="std_scaling",
            atomic_inter_scale=0.75,
            atomic_inter_shift=0.125,
        )
    assert result == (0.75, 0.125)


def test_explicit_scale_and_zero_shift_skip_h5_scan():
    with patch(
        "chorus.cli.train._atomic_inter_scale_shift_from_h5",
        side_effect=AssertionError("H5 scan must not run"),
    ):
        result = _resolve_atomic_inter_scale_shift(
            "/does/not/exist.h5",
            atomic_energy_keys=[1],
            atomic_energy_values=[-1.0],
            scaling="std_scaling",
            atomic_inter_scale=0.75,
            atomic_inter_shift=None,
            no_atomic_inter_shift=True,
        )
    assert result == (0.75, 0.0)


def test_partial_override_still_computes_missing_value():
    with patch(
        "chorus.cli.train._atomic_inter_scale_shift_from_h5",
        return_value=(2.0, 3.0),
    ) as scan:
        result = _resolve_atomic_inter_scale_shift(
            "train.h5",
            atomic_energy_keys=[1],
            atomic_energy_values=[-1.0],
            scaling="std_scaling",
            atomic_inter_scale=0.75,
            atomic_inter_shift=None,
            max_samples=100,
        )
    assert result == (0.75, 3.0)
    scan.assert_called_once()
