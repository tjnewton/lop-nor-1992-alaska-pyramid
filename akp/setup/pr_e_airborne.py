"""S5-e: airborne EM / magnetic survey coverage of the probe boxes, and reader access to one survey.

Footprints: USGS Airborne Geophysical Survey Inventory v5.0 (Johnson et al. 2019, doi:10.5066/P9K8YTW1),
which includes Alaska DGGS surveys. Coverage per box is metadata only. Reader access is tested on
the smallest downloadable Alaska EM survey that intersects NO probe box (no claim-zone data
viewed in setup). min_line_spacing_km is the inventory spacing1 field (km per its data dictionary, whose floor is 0.1).
Writes pr_e_airborne.csv. Run: python -m akp.setup.pr_e_airborne
"""
from __future__ import annotations

import csv
import re
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box as shp_box

from akp.paths import data_dir
from akp.zones import BOXES

ITEM = "https://www.sciencebase.gov/catalog/file/get/5d38aac0e4b01d82ce8b940a?f=__disk__"
FILES = {"footprint.zip": "58%2F80%2F9b%2F58809bc767cd54e6e72dd07d48ea3db56099da19",
         "DataLinks.csv": "4e%2F26%2Fd2%2F4e26d228e558ac9273af6a19ddcb2c8610f93bef"}
OUT = Path(__file__).with_suffix(".csv")


def fetch(d: Path, name: str, url: str) -> Path:
    p = d / name
    if not p.exists():
        urllib.request.urlretrieve(url, p)
    return p


def main() -> None:
    d = data_dir("probes/airborne_inventory")
    for name, key in FILES.items():
        fetch(d, name, ITEM + key)
    g = gpd.read_file(f"zip://{d / 'footprint.zip'}!AirborneGeophysicalSurvey_footprint.shp")
    ak = g[g["state"] == "AK"].copy()
    rows = []
    zone_geoms = {z: shp_box(b["minlongitude"], b["minlatitude"], b["maxlongitude"], b["maxlatitude"])
                  for z, b in BOXES.items()}
    hit_any = pd.Series(False, index=ak.index)
    for z, geom in zone_geoms.items():
        hit = ak.intersects(geom)
        hit_any |= hit
        for kind in ("EM", "Mag"):
            sel = ak[hit & (ak[kind] == 1)]
            zone_aea = gpd.GeoSeries([geom], crs=4326).to_crs(3338).iloc[0]
            union = sel.to_crs(3338).union_all() if len(sel) else None  # surveys overlap; count area once
            area = union.intersection(zone_aea).area / 1e6 if union is not None else 0.0
            rows.append(dict(scope=z, kind=kind, n_surveys=len(sel), overlap_km2=round(area),
                             zone_km2=round(zone_aea.area / 1e6),
                             min_line_spacing_km=sel["spacing1"][sel["spacing1"] > 0].min() if len(sel) else "",
                             median_line_spacing_km=sel["spacing1"][sel["spacing1"] > 0].median() if len(sel) else "",
                             median_year=sel["Year"][sel["Year"] > 0].median() if len(sel) else "",
                             surveys=";".join(sorted(sel["name"].astype(str)))))
    # reader-access test on an out-of-zone Alaska DGGS EM survey: its ER Mapper grid package via GDAL
    links = pd.read_csv(d / "DataLinks.csv", encoding_errors="replace")
    cand = ak[(ak["EM"] == 1) & ~hit_any].merge(links[["Survey", "download"]], on="Survey")
    cand = cand[cand["download"].str.contains("dggs.alaska.gov|10.14509", na=False)].sort_values("Survey")
    access = dict(scope="access_test", kind="EM", n_surveys=0, overlap_km2="", zone_km2="",
                  min_line_spacing_km="", median_line_spacing_km="", median_year="", surveys="")
    for _, c in cand.iterrows():
        try:
            with urllib.request.urlopen(c["download"].replace("http://", "https://"), timeout=120) as r:
                html = r.read().decode("utf-8", "replace")
            href = next(iter(re.findall(r'href="([^"]+_grids_ermapper\.zip)"', html)), None)
            if href is None:
                continue
            url = "https://dggs.alaska.gov" + href if href.startswith("/") else href
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=120) as r:
                size = int(r.headers["Content-Length"])
            if size > 1_000_000_000:
                continue
            out = data_dir(f"probes/airborne_access/{c['Survey']}")
            pkg = fetch(out, Path(url).name, url)
            ext = out / "extracted"  # archive contents: own directory, read as data only
            with zipfile.ZipFile(pkg) as zf:
                zf.extractall(ext)
            ers = sorted(q for q in ext.rglob("*.ers") if not q.name.startswith("._"))
            with rasterio.open(ers[0]) as ds:
                desc = f"{ers[0].name} {ds.width}x{ds.height} crs={ds.crs} dtype={ds.dtypes[0]}"
            access.update(n_surveys=1, surveys=f"{c['name']} ({c['Survey']}); {Path(url).name} {size} bytes; "
                                                f"{len(ers)} .ers grids; GDAL opened {desc}")
            break
        except Exception as exc:  # try the next candidate; record what failed
            access["surveys"] += f"FAILED {c['Survey']}: {type(exc).__name__}: {exc}; "
    rows.append(access)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print({k: (str(v)[:200]) for k, v in r.items()})


if __name__ == "__main__":
    main()
