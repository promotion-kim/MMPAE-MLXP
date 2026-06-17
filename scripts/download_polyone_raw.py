#!/usr/bin/env python
"""Download the polyOne parquet shards used by the MMPAE scale-up run."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


FILES = [
    "polyOne_ar.parquet",
    "polyOne_az.parquet",
    "polyOne_bg.parquet",
    "polyOne_bo.parquet",
    "polyOne_bq.parquet",
    "polyOne_em.parquet",
    "polyOne_fx.parquet",
    "polyOne_gk.parquet",
    "polyOne_hk.parquet",
    "polyOne_ho.parquet",
    "polyOne_hu.parquet",
    "polyOne_hv.parquet",
    "polyOne_hw.parquet",
    "polyOne_hx.parquet",
]


def download(url: str, path: Path, chunk_size: int = 1024 * 1024) -> None:
    req = Request(url, headers={"User-Agent": "mmpae-download"})
    with urlopen(req, timeout=120) as response, path.open("wb") as out:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header else None
        done = 0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"  {done / 1024**2:.1f}/{total / 1024**2:.1f} MiB", flush=True)
            else:
                print(f"  {done / 1024**2:.1f} MiB", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="/data/polyone")
    parser.add_argument("--record", default="7766806")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        path = dest / filename
        if path.exists() and path.stat().st_size > 0 and not args.overwrite:
            print(f"skip existing {path} ({path.stat().st_size / 1024**2:.1f} MiB)", flush=True)
            continue

        url = f"https://zenodo.org/records/{args.record}/files/{filename}?download=1"
        print(f"download {filename}", flush=True)
        try:
            download(url, path)
        except (HTTPError, URLError) as exc:
            if path.exists() and path.stat().st_size == 0:
                path.unlink()
            raise SystemExit(f"failed to download {filename}: {exc}") from exc


if __name__ == "__main__":
    main()
