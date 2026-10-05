#!/usr/bin/env python3
"""Resume an HTTP download while refreshing short-lived redirect signatures."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import requests


CONTENT_RANGE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=100)
    parser.add_argument("--chunk-bytes", type=int, default=2**20)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    expected_total: int | None = None
    for attempt in range(1, args.attempts + 1):
        offset = args.output.stat().st_size if args.output.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(
                args.url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=(30, 120),
            ) as response:
                if response.status_code == 416 and expected_total == offset:
                    break
                response.raise_for_status()
                if offset and response.status_code != 206:
                    # A server that ignores Range cannot safely append.  Restart
                    # once from byte zero rather than corrupting the file.
                    args.output.unlink(missing_ok=True)
                    print(
                        f"attempt={attempt} server ignored Range; restarting",
                        flush=True,
                    )
                    continue
                content_range = response.headers.get("Content-Range", "")
                match = CONTENT_RANGE.fullmatch(content_range)
                if match:
                    start, _, total = map(int, match.groups())
                    if start != offset:
                        raise RuntimeError(
                            f"server resumed at {start}, expected {offset}"
                        )
                    expected_total = total
                elif response.headers.get("Content-Length"):
                    length = int(response.headers["Content-Length"])
                    expected_total = offset + length

                mode = "ab" if offset else "wb"
                with args.output.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=args.chunk_bytes):
                        if chunk:
                            handle.write(chunk)
                observed = args.output.stat().st_size
                print(
                    f"attempt={attempt} downloaded={observed} total={expected_total}",
                    flush=True,
                )
                if expected_total is None or observed == expected_total:
                    break
                if observed > expected_total:
                    raise RuntimeError(
                        f"download larger than expected: {observed} > {expected_total}"
                    )
        except (requests.RequestException, OSError) as error:
            observed = args.output.stat().st_size if args.output.exists() else 0
            print(
                f"attempt={attempt} interrupted_at={observed}: {error}",
                flush=True,
            )
            if attempt == args.attempts:
                raise
            time.sleep(min(60, 5 * attempt))
    else:
        raise RuntimeError(f"download did not complete after {args.attempts} attempts")

    observed = args.output.stat().st_size
    if expected_total is not None and observed != expected_total:
        raise RuntimeError(f"incomplete download: {observed} != {expected_total}")
    print(f"DOWNLOAD_COMPLETE bytes={observed}")


if __name__ == "__main__":
    main()
