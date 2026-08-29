#!/usr/bin/env python3
"""Fetch selected members from the MS25 ZIP without downloading all 3.9 GB.

The official MS25 archive is a regular (non-ZIP64) ZIP served by Zenodo.  This
utility reads its central directory with HTTP range requests, downloads only
the requested compressed members, and verifies both size and CRC32.  It is
intended for reproducible benchmark setup, not as a general ZIP replacement.
"""

from __future__ import annotations

import argparse
import binascii
import json
import os
import struct
import tempfile
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path


ARCHIVE_URL = "https://zenodo.org/records/10901820/files/Datasets.zip?download=1"
ARCHIVE_MD5 = "3074335fe9083d5fc84701fc011c3ac2"
ARCHIVE_DOI = "10.5281/zenodo.10901820"

DATASET_MEMBERS = {
    "hea": "Datasets/HEA/set{set_id}/{name}",
    "zr-o": "Datasets/Zr-O/set{set_id}/{name}",
    "mgo-2x2": "Datasets/MgO-2x2/set{set_id}/{name}",
    "mgo-4x4": "Datasets/MgO-4x4/set{set_id}/{name}",
}


@dataclass(frozen=True)
class ZipMember:
    name: str
    compression: int
    flags: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


def request_bytes(request: urllib.request.Request) -> tuple[int, dict[str, str], bytes]:
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.status, dict(response.headers.items()), response.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            last_error = error
            if attempt == 5:
                break
            time.sleep(2**attempt)
    raise RuntimeError("MS25 range request failed after six attempts") from last_error


def range_get(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "CHORUS-MS25-benchmark/1.0",
        },
    )
    status, headers, payload = request_bytes(request)
    content_range = headers.get("Content-Range", "")
    if status != 206 and not content_range:
        raise RuntimeError(
            f"server ignored byte range {start}-{end} (status={status})"
        )
    expected = end - start + 1
    if len(payload) != expected:
        raise RuntimeError(
            f"short range response for {start}-{end}: got {len(payload)}, expected {expected}"
        )
    return payload


def archive_size(url: str) -> int:
    request = urllib.request.Request(
        url,
        headers={
            "Range": "bytes=0-0",
            "User-Agent": "CHORUS-MS25-benchmark/1.0",
        },
    )
    _, headers, _ = request_bytes(request)
    content_range = headers.get("Content-Range", "")
    if "/" not in content_range:
        raise RuntimeError(f"missing Content-Range in response: {content_range!r}")
    return int(content_range.rsplit("/", 1)[1])


def read_directory(url: str) -> dict[str, ZipMember]:
    size = archive_size(url)
    tail_size = min(size, 65_557)
    tail = range_get(url, size - tail_size, size - 1)
    eocd_at = tail.rfind(b"PK\x05\x06")
    if eocd_at < 0:
        raise RuntimeError("ZIP end-of-central-directory record not found")
    eocd = struct.unpack_from("<4s4H2LH", tail, eocd_at)
    _, disk_no, cd_disk, disk_entries, total_entries, cd_size, cd_offset, _ = eocd
    if disk_no or cd_disk or disk_entries != total_entries:
        raise RuntimeError("multi-disk ZIP archives are not supported")
    central = range_get(url, cd_offset, cd_offset + cd_size - 1)

    members: dict[str, ZipMember] = {}
    cursor = 0
    for _ in range(total_entries):
        if central[cursor : cursor + 4] != b"PK\x01\x02":
            raise RuntimeError(f"invalid central-directory signature at byte {cursor}")
        fields = struct.unpack_from("<4s6H3L5H2L", central, cursor)
        (
            _,
            _,
            _,
            flags,
            compression,
            _,
            _,
            crc32,
            compressed_size,
            uncompressed_size,
            name_len,
            extra_len,
            comment_len,
            _,
            _,
            _,
            local_header_offset,
        ) = fields
        name_bytes = central[cursor + 46 : cursor + 46 + name_len]
        encoding = "utf-8" if flags & (1 << 11) else "cp437"
        name = name_bytes.decode(encoding)
        members[name] = ZipMember(
            name=name,
            compression=compression,
            flags=flags,
            crc32=crc32,
            compressed_size=compressed_size,
            uncompressed_size=uncompressed_size,
            local_header_offset=local_header_offset,
        )
        cursor += 46 + name_len + extra_len + comment_len
    return members


def fetch_member(url: str, member: ZipMember, destination: Path) -> None:
    if member.flags & 0x1:
        raise RuntimeError(f"encrypted member is unsupported: {member.name}")
    header = range_get(
        url,
        member.local_header_offset,
        member.local_header_offset + 29,
    )
    fields = struct.unpack("<4s5H3L2H", header)
    if fields[0] != b"PK\x03\x04":
        raise RuntimeError(f"invalid local header for {member.name}")
    name_len, extra_len = fields[-2:]
    data_offset = member.local_header_offset + 30 + name_len + extra_len
    compressed = range_get(
        url,
        data_offset,
        data_offset + member.compressed_size - 1,
    )
    if member.compression == 0:
        payload = compressed
    elif member.compression == 8:
        payload = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise RuntimeError(
            f"unsupported compression method {member.compression} for {member.name}"
        )
    if len(payload) != member.uncompressed_size:
        raise RuntimeError(
            f"size mismatch for {member.name}: {len(payload)} != {member.uncompressed_size}"
        )
    crc32 = binascii.crc32(payload) & 0xFFFFFFFF
    if crc32 != member.crc32:
        raise RuntimeError(
            f"CRC mismatch for {member.name}: {crc32:08x} != {member.crc32:08x}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def requested_members(datasets: list[str], set_id: int) -> list[str]:
    selected = ["Datasets/README.md"]
    for dataset in datasets:
        pattern = DATASET_MEMBERS[dataset]
        names = ["train-1000.traj", "valid.traj", "test.traj"]
        if dataset == "mgo-4x4":
            names = ["test.traj"]
        selected.extend(
            pattern.format(set_id=set_id, name=name) for name in names
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url", default=ARCHIVE_URL)
    parser.add_argument("--set", dest="set_id", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_MEMBERS),
        default=["hea", "zr-o", "mgo-2x2", "mgo-4x4"],
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    directory = read_directory(args.url)
    selected = requested_members(args.datasets, args.set_id)
    missing = [name for name in selected if name not in directory]
    if missing:
        raise RuntimeError(f"members absent from archive: {missing}")

    manifest = {
        "source": "MS25",
        "record_doi": ARCHIVE_DOI,
        "archive_url": args.url,
        "archive_md5_from_record": ARCHIVE_MD5,
        "set": args.set_id,
        "members": [],
    }
    for name in selected:
        member = directory[name]
        destination = args.output_dir / name
        if destination.exists() and not args.force:
            payload = destination.read_bytes()
            crc32 = binascii.crc32(payload) & 0xFFFFFFFF
            if len(payload) == member.uncompressed_size and crc32 == member.crc32:
                print(f"verified existing {name}")
            else:
                raise RuntimeError(
                    f"existing file failed verification: {destination}; use --force to replace"
                )
        else:
            print(f"fetching {name} ({member.uncompressed_size / 2**20:.2f} MiB)")
            fetch_member(args.url, member, destination)
        manifest["members"].append(asdict(member))

    manifest_path = args.output_dir / f"ms25_set{args.set_id}_source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
