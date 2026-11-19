"""P3 G0 input: survey boundaries for the Track A surveys whose footprint is only a metadata bounding box.

The DGGS "vector data" package of each survey (URLs from the cached publication records,
$AKP_DATA/p3/inventory/meta/pubs) holds the line path, the survey boundary and data contours.
Only boundary members are extracted: names matching BOUNDARY, as .shp sidecars or .kml/.kmz. The
contour and line-path members are never opened, so no measured value of a Track A survey is seen
before the P3 freeze. Each package is recorded with its URL, bytes and sha256, plus the list
of member names. Writes p3_vector_fetch.csv. Packages are kept in $AKP_DATA/p3/inventory/vector_sealed/
(sealed like the line data: they hold Track A line paths and contours); only the extracted boundary
members live in $AKP_DATA/p3/inventory/vector/.
Run: python -m akp.stage_p3.p3_vector_fetch
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from akp.paths import data_dir

HERE = Path(__file__).parent
BASE = "https://dggs.alaska.gov"
BOUNDARY = re.compile(r"bound|outline|extent|perimeter", re.I)
UA = {"User-Agent": "research-data-client/1.0"}


def package_url(pub: str) -> str | None:
    m = re.search(r"GPR\s?(\d{4})-(\d+)", pub)
    if not m:
        return None
    pat = re.compile(rf'"href":\s*"(/webpubs/data/gpr{m.group(1)}[_-]0*{m.group(2)}[^"]*vector[^"]*\.zip)"', re.I)
    for f in sorted(data_dir("p3/inventory/meta/pubs").glob("*.json")):
        hit = pat.search(f.read_text(errors="ignore"))
        if hit:
            return BASE + hit.group(1)
    return None


def main() -> None:
    inv = pd.read_csv(HERE / "p3_survey_inventory.csv")
    sel = inv[(inv.intersects_track_a.astype(str).isin(["yes", "unknown"]))
              & ~inv.footprint_source.astype(str).str.startswith("inventory v5.0")]
    out_d = data_dir("p3/inventory/vector")
    sealed_d = data_dir("p3/inventory/vector_sealed")
    rows, seen = [], set()
    for pub in sel.publication.dropna().unique():
        url = package_url(pub)
        if url is None or url in seen:
            continue
        seen.add(url)
        dst = sealed_d / url.rsplit("/", 1)[1]
        if not dst.exists():
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
                dst.write_bytes(r.read())
            time.sleep(1.0)
        data = dst.read_bytes()
        with zipfile.ZipFile(dst) as z:
            names = z.namelist()
            keep = [n for n in names if BOUNDARY.search(Path(n).name) and not n.endswith("/")]
            for n in keep:
                z.extract(n, out_d / dst.stem)
        rows.append(dict(publication=pub, url=url, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                         n_members=len(names), boundary_members=json.dumps(keep)))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "p3_vector_fetch.csv", index=False)
    print(out[["publication", "bytes", "n_members", "boundary_members"]].to_string(index=False))


if __name__ == "__main__":
    main()
