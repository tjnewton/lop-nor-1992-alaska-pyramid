"""P2e / B-gate inventory test: Quaternary volcanic vents in and near the searched area.

Reads the DGGS MP 153 vent table (Cameron & Nye 2014; fetched by p2e_inventory_sources.py)
straight from its zip and keeps the vents inside the searched area (p2a_searched_area.geojson),
inside either probe box, or within MARGIN_KM of the searched area (an inventory radius is set
at calibration; the margin only bounds this table). Writes p2e_vents.csv (one row per kept vent:
vent_id, volcano, feature, lat, lon, zone, km_outside) and prints the counts.
Run: python -m akp.stage_p2.p2e_vents
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

from pyproj import Geod
from shapely.geometry import Point, shape

from akp.paths import data_dir
from akp.zones import BOXES, in_box

HERE = Path(__file__).parent
MARGIN_KM = 100.0
GEOD = Geod(ellps="WGS84")


def vents() -> list[dict]:
    z = data_dir("p2e/inventories") / "mp153_AK-Quaternary-vents.zip"
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if n.endswith("AK_Q_vents.csv") and "/._" not in n)
        with zf.open(name) as f:  # Windows-1252 text (byte 0x85 is an ellipsis)
            return list(csv.DictReader(io.TextIOWrapper(f, encoding="cp1252")))


def km_outside(lat: float, lon: float, zones) -> float:
    """Geodesic km from the point to the nearest searched-area boundary vertex (0 inside)."""
    p = Point(lon, lat)
    if any(z.contains(p) for z in zones):
        return 0.0
    best = min((GEOD.inv(lon, lat, x, y)[2] for z in zones for x, y in z.exterior.coords), default=float("inf"))
    return best / 1e3


def main() -> None:
    gj = json.loads((HERE / "p2a_searched_area.geojson").read_text())
    zones = [shape(f["geometry"]) for f in gj["features"]]
    names = [f["properties"]["zone"] for f in gj["features"]]
    rows = []
    for v in vents():
        lat, lon = float(v["LatDegree"]), float(v["LongDegree"])
        inside = [n for n, z in zip(names, zones) if z.contains(Point(lon, lat))]
        boxes = [b for b, box in BOXES.items() if in_box(lat, lon, box)]
        d = km_outside(lat, lon, zones)
        if inside or boxes or d <= MARGIN_KM:
            rows.append(dict(vent_id=v["vent_id"], volcano=v["Volcano"], feature=v["Feature"],
                             lat=lat, lon=lon, zone=(inside or boxes or ["near"])[0], km_outside=round(d, 1)))
    out = HERE / "p2e_vents.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["vent_id", "volcano", "feature", "lat", "lon", "zone", "km_outside"])
        w.writeheader()
        w.writerows(rows)
    by_zone: dict[str, int] = {}
    for r in rows:
        by_zone[r["zone"]] = by_zone.get(r["zone"], 0) + 1
    print(f"{len(rows)} vents kept; by zone {by_zone}; wrote {out.name}")


if __name__ == "__main__":
    main()
