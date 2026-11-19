"""Checksum manifest of every input on the data drive (provenance for "regenerable" data).

Hashes every file under $AKP_DATA: probe samples and stage caches such as a1/. AppleDouble sidecars
are skipped, and so is the SPECFEM2D smoke run dir, which pr_f_solvers.py rewrites on every run. The source URL of each file is
in the probe script that fetched it. Writes akp/setup/data_manifest.csv.
Run: python -m akp.setup.manifest
"""
from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from akp.paths import data_glob, data_root

OUT = Path(__file__).with_name("data_manifest.csv")
SKIP = ("specfem2d_smoke",)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    root = data_root()
    rows = []
    for p in data_glob(root, "**/*"):
        rel = p.relative_to(root)
        if not p.is_file() or any(q in SKIP or q.startswith("._") for q in rel.parts):
            continue
        mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
        rows.append(dict(path=str(rel), bytes=p.stat().st_size, sha256=sha256(p), mtime_utc=mtime))
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256", "mtime_utc"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} files -> {OUT.name}")


if __name__ == "__main__":
    main()
