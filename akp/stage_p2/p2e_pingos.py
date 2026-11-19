"""P2e / B-gate inventory test: Holmes (1967) pingos in and near the searched area.

Holmes (1967, USGS OFR 67-115, doi:10.3133/ofr67115) lists observed pingos and pingolike
mounds in interior Alaska by 1:250,000 quadrangle, sub-quadrangle (row A-D from the south,
column 1-6 from the east), reference number and deg-min coordinates (report pp. 1-13).
p2e_holmes1967_table.csv is a hand transcription of every row in the five quadrangles that
reach west of 147 W; `note` marks where the print is ambiguous. The six eastern quadrangles
(EAST, not transcribed) are checked whole: each must lie more than MARGIN_KM from the searched
area, which bounds every row in them. Each transcribed row must fall inside its
sub-quadrangle (a transcription check). Writes p2e_pingos.csv: the rows inside a searched
zone or a probe box, or within MARGIN_KM. Run: python -m akp.stage_p2.p2e_pingos
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import shapely
from shapely.geometry import box, shape

from akp.stage_p2.p2e_vents import MARGIN_KM, km_outside
from akp.zones import BOXES, in_box

HERE = Path(__file__).parent
# 1:250,000 quadrangle -> (south latitude, east longitude); each spans 1 deg x 3 deg here
QUADS = {"Fairbanks": (64, -147), "Kantishna River": (64, -150), "Kateel River": (65, -156),
         "Livengood": (65, -147), "Tanana": (65, -150)}
EAST = {"Big Delta": (64, -144), "Charley River": (65, -141), "Circle": (65, -144),
        "Eagle": (64, -141), "Mt. Hayes": (63, -144), "Tanacross": (63, -141)}


def in_subquad(lat: float, lon: float, quad: str, sub: str) -> bool:
    lat0, lon_e = QUADS[quad]
    s = lat0 + 0.25 * "ABCD".index(sub[0])
    e = lon_e - 0.5 * (int(sub[2:]) - 1)
    return s <= lat <= s + 0.25 and e - 0.5 <= lon <= e


def quad_km(lat0: float, lon_e: float, zones) -> float:
    """Approximate km from a quadrangle to the searched area: its edge sampled every 0.1 deg."""
    edge = shapely.segmentize(box(lon_e - 3, lat0, lon_e, lat0 + 1), 0.1).exterior.coords
    return min(km_outside(y, x, zones) for x, y in edge)


def main() -> None:
    gj = json.loads((HERE / "p2a_searched_area.geojson").read_text())
    zones = [shape(f["geometry"]) for f in gj["features"]]
    names = [f["properties"]["zone"] for f in gj["features"]]
    for q, (lat0, lon_e) in EAST.items():
        d = quad_km(lat0, lon_e, zones)
        assert d > MARGIN_KM, (q, d)
        print(f"{q}: whole quadrangle {d:.0f} km from the searched area (not transcribed)")
    rows = []
    with (HERE / "p2e_holmes1967_table.csv").open() as f:
        for r in csv.DictReader(f):
            lat = int(r["lat_deg"]) + float(r["lat_min"]) / 60
            lon = -(int(r["lon_deg"]) + float(r["lon_min"]) / 60)
            assert in_subquad(lat, lon, r["quadrangle"], r["subquad"]), r
            inside = [n for n, z in zip(names, zones) if z.contains(shapely.Point(lon, lat))]
            boxes = [b for b, bx in BOXES.items() if in_box(lat, lon, bx)]
            d = km_outside(lat, lon, zones)
            if inside or boxes or d <= MARGIN_KM:
                rows.append(dict(quadrangle=r["quadrangle"], subquad=r["subquad"], ref=r["ref"],
                                 lat=round(lat, 4), lon=round(lon, 4), veg=r["veg"],
                                 zone=(inside or boxes or ["near"])[0], km_outside=round(d, 1)))
    out = HERE / "p2e_pingos.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by_zone: dict[str, int] = {}
    for r in rows:
        by_zone[r["zone"]] = by_zone.get(r["zone"], 0) + 1
    print(f"{len(rows)} pingos kept; by zone {by_zone}; wrote {out.name}")


if __name__ == "__main__":
    main()
