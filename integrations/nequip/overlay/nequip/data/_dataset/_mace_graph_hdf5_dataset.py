"""Read fixed-topology MACE graph HDF5 files without rebuilding neighbors."""

from pathlib import Path
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

from .. import AtomicData, AtomicDataDict
from ..transforms import TypeMapper
from ._base_datasets import AtomicDataset


class MACEGraphHDF5Dataset(AtomicDataset):
    """Lazy reader for ``sample_N`` graph groups written by MACE-ICTC tools.

    These files already contain the 4.5 Å neighbor graph.  Reusing it is both
    exact and avoids NequIP 0.6's expensive ASE/forkserver preprocessing path.
    """

    def __init__(
        self,
        root: str,
        file_name: str,
        AtomicData_options: Dict[str, Any] = {},
        type_mapper: Optional[TypeMapper] = None,
    ) -> None:
        super().__init__(root=root, type_mapper=type_mapper)
        self.file_name = str(Path(file_name).expanduser().resolve())
        self.r_max = float(AtomicData_options["r_max"])
        self._h5 = None
        self._group_names = self._read_group_names()

    def _read_group_names(self):
        import h5py

        with h5py.File(self.file_name, "r") as handle:
            names = list(handle.keys())
        return sorted(names, key=lambda name: int(name.rsplit("_", 1)[-1]))

    def _handle(self):
        if self._h5 is None:
            import h5py

            self._h5 = h5py.File(self.file_name, "r")
        return self._h5

    def len(self) -> int:
        return len(self._group_names)

    def get(self, idx: int) -> AtomicData:
        group = self._handle()[self._group_names[int(idx)]]
        # MACE stores neighbor -> center as (edge_src, edge_dst), while NequIP
        # stores center in edge_index[0] and neighbor in edge_index[1].
        edge_index = np.stack(
            (group["edge_dst"][...], group["edge_src"][...]), axis=0
        )
        return AtomicData(
            pos=group["pos"][...],
            edge_index=edge_index,
            atomic_numbers=group["A"][...],
            total_energy=np.asarray(group["y"][...]).reshape(1),
            forces=group["force"][...],
        )

    def statistics(
        self,
        fields: List[Union[str, Callable]],
        modes: List[str],
        stride: int = 1,
        unbiased: bool = True,
        kwargs: Dict[str, dict] = {},
    ) -> List[tuple]:
        """Compute the statistics required by NequIP model builders."""
        del kwargs
        if len(fields) != len(modes):
            raise ValueError("fields and modes must have equal length")
        results = []
        for field, mode in zip(fields, modes):
            chunks = []
            counts = Counter()
            for index in range(0, len(self), int(stride)):
                sample = self[index]
                if callable(field):
                    values, _ = field(sample)
                else:
                    values = sample[field]
                values = torch.as_tensor(values).detach().cpu().numpy()
                if mode == "per_atom_mean_std":
                    values = values / int(sample.num_nodes)
                flat = values.reshape(-1)
                if mode == "count":
                    counts.update(flat.tolist())
                else:
                    chunks.append(flat.astype(np.float64, copy=False))
            if mode == "count":
                keys = sorted(counts)
                results.append(
                    (
                        torch.tensor(keys),
                        torch.tensor([counts[key] for key in keys]),
                    )
                )
                continue
            values = np.concatenate(chunks)
            if mode == "rms":
                results.append(torch.tensor((np.sqrt(np.mean(values**2)),)))
            elif mode in ("mean_std", "per_atom_mean_std"):
                results.append(
                    (
                        torch.tensor(np.mean(values)),
                        torch.tensor(np.std(values, ddof=int(unbiased))),
                    )
                )
            else:
                raise NotImplementedError(f"unsupported statistics mode {mode!r}")
        return results

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()
