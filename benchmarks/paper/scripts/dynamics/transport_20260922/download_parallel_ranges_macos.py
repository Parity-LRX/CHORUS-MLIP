#!/usr/bin/env python3
"""Download a large HTTP object with parallel byte-range requests on macOS."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--md5", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-mib", type=int, default=2)
    args = parser.parse_args()

    chunk_bytes = args.chunk_mib * 2**20
    parts = args.output.with_name(args.output.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    ranges = [
        (index, start, min(args.size - 1, start + chunk_bytes - 1))
        for index, start in enumerate(range(0, args.size, chunk_bytes))
    ]

    def download(specification: tuple[int, int, int]) -> tuple[int, int]:
        index, start, end = specification
        destination = parts / f"{index:06d}.part"
        expected = end - start + 1
        if destination.is_file() and destination.stat().st_size == expected:
            return index, expected
        temporary = destination.with_suffix(".tmp")
        temporary.unlink(missing_ok=True)
        subprocess.run(
            [
                "/usr/bin/curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--retry",
                "8",
                "--retry-delay",
                "2",
                "--retry-all-errors",
                "--range",
                f"{start}-{end}",
                "--output",
                str(temporary),
                args.url,
            ],
            check=True,
        )
        observed = temporary.stat().st_size
        if observed != expected:
            raise RuntimeError(
                f"chunk {index}: received {observed} bytes, expected {expected}"
            )
        temporary.replace(destination)
        return index, observed

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download, item): item for item in ranges}
        for future in concurrent.futures.as_completed(futures):
            index, count = future.result()
            completed += count
            print(
                f"chunk={index + 1}/{len(ranges)} "
                f"downloaded={completed}/{args.size}",
                flush=True,
            )

    assembling = args.output.with_suffix(args.output.suffix + ".assembling")
    digest = hashlib.md5()
    with assembling.open("wb") as output_handle:
        for index, start, end in ranges:
            part = parts / f"{index:06d}.part"
            expected = end - start + 1
            if part.stat().st_size != expected:
                raise RuntimeError(f"chunk {index} changed size before assembly")
            with part.open("rb") as input_handle:
                while chunk := input_handle.read(2**20):
                    output_handle.write(chunk)
                    digest.update(chunk)
    if assembling.stat().st_size != args.size:
        raise RuntimeError(
            f"assembled size {assembling.stat().st_size} != {args.size}"
        )
    if digest.hexdigest() != args.md5:
        raise RuntimeError(f"MD5 {digest.hexdigest()} != {args.md5}")
    os.replace(assembling, args.output)
    print(f"DOWNLOAD_COMPLETE path={args.output} md5={digest.hexdigest()}")


if __name__ == "__main__":
    main()
