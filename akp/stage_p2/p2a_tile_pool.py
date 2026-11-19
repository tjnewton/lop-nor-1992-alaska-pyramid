"""P2a: the statewide IfSAR tile list, flagged for the control-area pool (procedure).

TNM item metadata only (no raster is fetched or read). A tile is excluded from the control pool
(controls lie outside both probe boxes, wider than the searched band) when its bounding box
meets either probe box in akp.zones.BOXES, or when it was fetched as a claim-box tile
(p2b_fetch_ifsar.csv): TNM's box query also returns tiles ending a few thousandths of a degree
short of a box edge. Columns: tile_id (row number; rows sorted by project, subproject and file,
so a rebuild from an unchanged catalog keeps the ids), project, subproject (one tile name can recur
in two collection years of a project), file, url, publication_date, bytes,
the tile's lon/lat bounds, bbox_area_km2 (geodesic area of the bounding box) and in_probe_box.
Writes p2a_tile_pool.csv. Run: python -m akp.stage_p2.p2a_tile_pool
"""
from __future__ import annotations

import csv
from pathlib import Path

from pyproj import Geod

from akp.setup.pr_d_dem import tnm_items
from akp.zones import BOXES

HERE = Path(__file__).parent
OUT = Path(__file__).with_suffix(".csv")
ALASKA = (-180.0, 51.0, -129.0, 72.0)  # (minlon, minlat, maxlon, maxlat); west of 180 skipped
GEOD = Geod(ellps="WGS84")


def meets(b: dict, box: dict) -> bool:
    return not (b["maxX"] < box["minlongitude"] or b["minX"] > box["maxlongitude"]
                or b["maxY"] < box["minlatitude"] or b["minY"] > box["maxlatitude"])


def bbox_area_km2(b: dict) -> float:
    lons = [b["minX"], b["maxX"], b["maxX"], b["minX"]]
    lats = [b["minY"], b["minY"], b["maxY"], b["maxY"]]
    return abs(GEOD.polygon_area_perimeter(lons, lats)[0]) / 1e6


def main() -> None:
    with (HERE / "p2b_fetch_ifsar.csv").open() as f:
        claim_box = {r["url"] for r in csv.DictReader(f)}
    rows, seen = [], set()
    for it in tnm_items(ALASKA):
        url = it["downloadURL"]
        if url in seen:
            continue
        seen.add(url)
        b = it["boundingBox"]
        rows.append(dict(
            project=url.split("/Projects/")[1].split("/")[0],
            subproject=url.split("/Projects/")[1].split("/")[1], file=url.rsplit("/", 1)[1], url=url,
            publication_date=it["publicationDate"], bytes=int(it["sizeInBytes"]),
            min_lon=round(b["minX"], 5), max_lon=round(b["maxX"], 5),
            min_lat=round(b["minY"], 5), max_lat=round(b["maxY"], 5),
            bbox_area_km2=round(bbox_area_km2(b), 1),
            in_probe_box=int(url in claim_box or any(meets(b, box) for box in BOXES.values())),
        ))
    assert len({(r["project"], r["subproject"], r["file"]) for r in rows}) == len(rows), "duplicate names"
    rows.sort(key=lambda r: (r["project"], r["subproject"], r["file"]))
    rows = [{"tile_id": i, **r} for i, r in enumerate(rows)]
    assert claim_box <= seen, "a claim-box tile is missing from the statewide list"
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    pool = [r for r in rows if not r["in_probe_box"]]
    print(f"{len(rows)} tiles; pool {len(pool)} tiles, {sum(r['bytes'] for r in pool) / 1e9:.1f} GB, "
          f"bbox area {sum(r['bbox_area_km2'] for r in pool):.0f} km2 (vintages overlap); wrote {OUT.name}")


if __name__ == "__main__":
    main()
