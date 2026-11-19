"""P2a: size the thematic maps that control areas are matched on.

HTTP HEAD by default: status, content type and served Content-Length per file. With --fetch
the files go to $AKP_DATA/p2a/maps/<file>, verified against the
served Content-Length (akp.stage_p2.p2b_fetch_ifsar.fetch); a file already complete is skipped.
The maps are statewide thematic maps, not imagery. The DEM source and vintage criterion
comes from TNM IfSAR item metadata (akp.setup.pr_d_dem.tnm_items), not from a file here.
With --extract, each archive is unpacked into $AKP_DATA/p2a/maps/<archive stem>/ with the system
tar (libarchive: zip and 7z; an inner .tar is unpacked too); a directory already present is kept.
Writes p2a_map_sources.csv. Run: python -m akp.stage_p2.p2a_map_sources [--fetch] [--extract]
"""
from __future__ import annotations

import csv
import subprocess
import sys
import urllib.request
from pathlib import Path

from akp.paths import data_dir, data_glob
from akp.stage_p2.p2b_fetch_ifsar import fetch

OUT = Path(__file__).with_suffix(".csv")
UA = {"User-Agent": "akp-research/0.1 (map size probe)"}
SB = "https://www.sciencebase.gov/catalog/file/get/5602ab5ae4b03bc34f5448b4?f=__disk__"
SOURCES = [
    # (criterion, source, citation, url, file)
    ("permafrost", "Pastick et al. 2015, near-surface permafrost probability, 30 m",
     "doi:10.5066/F7C53HX6", SB + "28%2F5d%2F17%2F285d1775c7f7ae08967bd5668a1b67cf47aa6663",
     "AK_PFProbability_1m.7z"),
    ("permafrost", "Pastick et al. 2015, FGDC metadata", "doi:10.5066/F7C53HX6",
     SB + "b5%2F84%2Fb1%2Fb584b11a5c8bd25d8c4046e397038aac7aa0ad78", "AK_ProbofPF_1m_clipped_FGDC.xml"),
    ("surficial_geology", "Wilson et al. 2015, Geologic map of Alaska (SIM 3340), geodatabase",
     "doi:10.3133/sim3340", "https://pubs.usgs.gov/sim/3340/sim3340_gdb.zip", "sim3340_gdb.zip"),
    ("land_cover", "NLCD 2016 Land Cover, Alaska (MRLC, 2020-07-24 bundle), 30 m",
     "mrlc.gov NLCD 2016 Alaska",
     "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/NLCD_2016_Land_Cover_AK_20200724.zip",
     "NLCD_2016_Land_Cover_AK_20200724.zip"),
]


def head(url: str) -> tuple[int, str, int]:
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, r.headers.get("Content-Type", ""), int(r.headers.get("Content-Length", -1))


def extract(archive: Path) -> Path:
    out = archive.with_name(archive.name.split(".")[0])
    if not out.exists():
        part = out.with_name(out.name + ".part")
        part.mkdir(exist_ok=True)
        subprocess.run(["tar", "-xf", str(archive), "-C", str(part)], check=True)
        for inner in data_glob(part, "*.tar"):  # the permafrost 7z holds one tar
            subprocess.run(["tar", "-xf", str(inner), "-C", str(part)], check=True)
            inner.unlink()
        part.rename(out)
    return out


def main(do_fetch: bool = False, do_extract: bool = False) -> None:
    rows = []
    for criterion, source, cite, url, name in SOURCES:
        status, ctype, n = head(url)
        ok = status == 200 and n > 0 and not ctype.startswith("text/html")  # not an error page
        fetched = ""
        if ok and do_fetch:
            got, fetched = fetch(url, data_dir("p2a/maps") / name, n)
            ok = got == n
        rows.append(dict(criterion=criterion, source=source, cite=cite, url=url, file=name,
                         status=status, content_type=ctype, bytes=n, ok=int(ok), fetched=fetched))
        print(f"{status} {n / 1e6:9.1f} MB {ctype:30s} {fetched:8s} {source}", flush=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"total {sum(r['bytes'] for r in rows) / 1e9:.2f} GB; wrote {OUT.name}")
    if not all(r["ok"] for r in rows):
        raise SystemExit(1)
    if do_extract:
        for name in (r["file"] for r in rows if r["file"].endswith((".zip", ".7z"))):
            print("extracted", extract(data_dir("p2a/maps") / name), flush=True)


if __name__ == "__main__":
    main(do_fetch="--fetch" in sys.argv[1:], do_extract="--extract" in sys.argv[1:])
