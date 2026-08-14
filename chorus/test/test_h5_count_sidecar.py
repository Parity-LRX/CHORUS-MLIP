import h5py
import numpy as np

from chorus.data.datasets import H5Dataset


def test_makefx_buckets_use_count_sidecar_before_resolving_external_links(tmp_path):
    """A valid sidecar must make dataset startup independent of sample H5 links."""

    path = tmp_path / "processed_train.h5"
    with h5py.File(path, "w") as handle:
        for index in range(4):
            handle[f"sample_{index}"] = h5py.ExternalLink(
                str(tmp_path / "deliberately-missing-shard.h5"),
                f"sample_{index}",
            )

    np.savez(
        str(path) + ".counts.npz",
        node_counts=np.asarray([3, 7, 5, 11], dtype=np.int64),
        edge_counts=np.asarray([12, 42, 24, 88], dtype=np.int64),
    )

    dataset = H5Dataset(
        prefix="train",
        data_dir=tmp_path,
        makefx_buckets=2,
        pad_nodes_to_max=True,
        pad_edges_to_max=True,
    )

    assert dataset.max_atoms == 11
    assert dataset.max_edges == 88
    assert dataset.sample_bucket == [0, 1, 0, 1]
    assert dataset._bucket_n_max == [5, 11]
    assert dataset._bucket_e_max == [24, 88]
