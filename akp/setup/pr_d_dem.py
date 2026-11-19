"""S5-d: DEM access and size -- USGS IfSAR 5 m DTM (TNM API) and ArcticDEM v4.1 2 m mosaic (PGC STAC).

Access is confirmed by a windowed COG read of the size-probe tile (outside every claim box).
Claim-box sizes come from metadata only (TNM sizeInBytes; HTTP HEAD on a sample of ArcticDEM tiles):
no claim-zone imagery is read. Writes pr_d_dem.csv. Run: python -m akp.setup.pr_d_dem
"""
from __future__ import annotations

import csv
import json
import statistics
import urllib.parse
import urllib.request
from pathlib import Path

import rasterio
from rasterio.windows import Window

from akp.zones import BOXES, DEM_PROBE_BBOX

TNM = "https://tnmaccess.nationalmap.gov/api/v1/products"
IFSAR = "Alaska IFSAR 5 meter DEM"
STAC = "https://stac.pgc.umn.edu/api/v1/collections/arcticdem-mosaics-v4.1-2m/items"
HEAD_SAMPLE = 10
OUT = Path(__file__).with_suffix(".csv")


def get_json(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def head_bytes(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=120) as r:
        return int(r.headers["Content-Length"])


def bbox_str(b):
    return ",".join(str(x) for x in b)


def box_bbox(box):
    return (box["minlongitude"], box["minlatitude"], box["maxlongitude"], box["maxlatitude"])


def tnm_items(bbox):
    items, offset = [], 0
    while True:
        q = urllib.parse.urlencode(dict(datasets=IFSAR, bbox=bbox_str(bbox), max=100, offset=offset))
        d = get_json(f"{TNM}?{q}")
        items += d["items"]
        offset += len(d["items"])
        if not d["items"] or offset >= d["total"]:
            return items


def stac_items(bbox):
    feats, url = [], f"{STAC}?{urllib.parse.urlencode(dict(bbox=bbox_str(bbox), limit=100))}"
    while url:
        d = get_json(url)
        feats += d["features"]
        url = next((link["href"] for link in d.get("links", []) if link.get("rel") == "next"), None)
    return feats


def projects(items):
    return {i["downloadURL"].split("/Projects/")[1].split("/")[0] for i in items}


def window_read(url):
    with rasterio.open(f"/vsicurl/{url}") as ds:
        w = Window(ds.width // 2, ds.height // 2, 256, 256)
        a = ds.read(1, window=w)
        return ds.res[0], ds.crs.to_epsg(), ds.width, ds.height, ds.profile.get("tiled"), a.shape


def main() -> None:
    rows = []
    # probe tile: access + bytes
    it = tnm_items(DEM_PROBE_BBOX)[0]
    res, epsg, wd, ht, tiled, shp = window_read(it["downloadURL"])
    rows.append(dict(product="ifsar_5m", scope="probe_tile", n_tiles=1, bytes=it["sizeInBytes"],
                     note=f"{it['title']}; res={res} epsg={epsg} {wd}x{ht} tiled={tiled} window={shp}"))
    ft = stac_items(DEM_PROBE_BBOX)[0]
    url = ft["assets"]["dem"]["href"]
    res, epsg, wd, ht, tiled, shp = window_read(url)
    rows.append(dict(product="arcticdem_2m_v4.1", scope="probe_tile", n_tiles=1, bytes=head_bytes(url),
                     note=f"{ft['id']}; res={res} epsg={epsg} {wd}x{ht} tiled={tiled} window={shp}"))
    # claim boxes: metadata only
    for zone, box in BOXES.items():
        its = tnm_items(box_bbox(box))
        rows.append(dict(product="ifsar_5m", scope=zone, n_tiles=len(its),
                         bytes=sum(int(i["sizeInBytes"]) for i in its),
                         note=f"exact TNM sizeInBytes sum over all {len(projects(its))} project vintages; "
                              "vintages overlap, so this is an upper bound on one-vintage coverage"))
        fts = stac_items(box_bbox(box))
        sample = [head_bytes(f["assets"]["dem"]["href"]) for f in fts[:: max(1, len(fts) // HEAD_SAMPLE)][:HEAD_SAMPLE]]
        rows.append(dict(product="arcticdem_2m_v4.1", scope=zone, n_tiles=len(fts),
                         bytes=round(statistics.mean(sample) * len(fts)) if sample else 0,
                         note=f"projected: mean of {len(sample)} HEAD sizes x n_tiles; "
                              f"sample min/max={min(sample, default=0)}/{max(sample, default=0)}"))
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
