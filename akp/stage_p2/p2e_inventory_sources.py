"""P2e / B-gate inventory test: size the landform inventories that could explain a flag.

The B-gate's single-method route needs "no inventoried pingo or cataloged landform within an
inventory match radius". No open pingo or frost-mound inventory was found for the
searched area (research leads, 2026-11-11: every one lies north of 65.3 N); these are the
sources that do cover it, or stand in for it. HEAD requests only, so the sizes are the
served Content-Length. With --fetch, files go to $AKP_DATA/p2e/inventories/<file>,
verified against the served Content-Length (akp.stage_p2.p2b_fetch_ifsar.fetch).
Writes p2e_inventory_sources.csv. Run: python -m akp.stage_p2.p2e_inventory_sources [--fetch]
"""
from __future__ import annotations

import csv
import sys
import urllib.request
from pathlib import Path

OUT = Path(__file__).with_suffix(".csv")
UA = {"User-Agent": "akp-research/0.1 (inventory size probe)"}
SOURCES = [
    # (kind, source, citation, url, file)
    ("volcanic_vents", "Cameron & Nye 2014, Preliminary database of Quaternary vents in Alaska (DGGS MP 153)",
     "doi:10.14509/27357", "https://dggs.alaska.gov/webpubs/data/mp153_AK-Quaternary-vents.zip",
     "mp153_AK-Quaternary-vents.zip"),
    ("volcanic_vents_report", "Cameron & Nye 2014, MP 153 report (defines the vents' Loc_conf codes)",
     "doi:10.14509/27357", "https://dggs.alaska.gov/webpubs/dggs/mp/text/mp153.pdf", "mp153.pdf"),
    ("glacial_limits", "Kaufman et al. 2011, Alaska Palaeo-Glacier Atlas v2 (NOAA WDS-Paleo 2011-104)",
     "doi:10.1016/B978-0-444-53447-7.00033-7",
     "https://www.ncei.noaa.gov/pub/data/paleo/reconstructions/alaska-glacier-atlas/arcgis/APG_v2_ArcGIS.zip",
     "APG_v2_ArcGIS.zip"),
    ("pingo_suitability", "Karjalainen et al. 2020, spatial predictions of pingo occurrence (modeled, ~1 km)",
     "doi:10.1594/PANGAEA.922771", "https://download.pangaea.de/dataset/922771/files/Datasets.zip",
     "Karjalainen2020_Datasets.zip"),
    ("pingo_locations", "Holmes 1967, Location of pingos and pingolike mounds in interior Alaska (USGS OFR 67-115)",
     "doi:10.3133/ofr67115", "https://pubs.usgs.gov/of/1967/0115/report.pdf", "Holmes1967_ofr67115.pdf"),
    ("mines_prospects", "USGS Alaska Resource Data File (ARDF): mines, prospects and occurrences",
     "USGS Mineral Resources Data System, mrdata.usgs.gov/ardf", "https://mrdata.usgs.gov/ardf/ardf.zip", "ardf.zip"),
]


def head(url: str) -> tuple[int, str, int]:
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, r.headers.get("Content-Type", ""), int(r.headers.get("Content-Length", -1))


def main(do_fetch: bool = False) -> None:
    rows = []
    for kind, source, cite, url, name in SOURCES:
        try:
            status, ctype, n = head(url)
        except OSError as e:
            status, ctype, n = 0, f"error: {e}", -1
        ok = status == 200 and n > 0 and not ctype.startswith("text/html")
        fetched = ""
        if ok and do_fetch:
            from akp.paths import data_dir
            from akp.stage_p2.p2b_fetch_ifsar import fetch

            got, fetched = fetch(url, data_dir("p2e/inventories") / name, n)
            ok = got == n
        rows.append(dict(kind=kind, source=source, cite=cite, url=url, file=name, status=status,
                         content_type=ctype, bytes=n, ok=int(ok), fetched=fetched))
        print(f"{status} {n / 1e6:8.2f} MB {ctype[:28]:28s} {fetched:8s} {kind}", flush=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main(do_fetch="--fetch" in sys.argv[1:])
