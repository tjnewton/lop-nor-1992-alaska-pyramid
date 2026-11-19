"""P2a: control-area strata.

Builds one 1 km grid in NAD83 / Alaska Albers (EPSG:3338, equal area, so a cell is 1 km2) and
classifies every land cell into a joint stratum ``pf|geo|lc``:
- pf: near-surface permafrost probability (Pastick et al. 2015), "pf" at or above the source's
  50% presence cut, "nopf" below it, "unmapped" where the source masks the cell (values > 100);
- geo: SIM 3340 unit, from QCLASS: "bedrock" (0, 99), "glacial" (surficial units, QCLASS >= 1000,
  whose name in the nsaqkey table names a glacial deposit), "surficial" (other QCLASS >= 1000),
  "other" (ice fields, water, unmapped, unlabeled: 90, 98, 101, 102);
- lc: NLCD 2016 level-1 group, "forest" (41-43), "shrub" (51, 52), "herb_wet" (71-74, 81-82,
  90, 95, and the developed classes 21-24), "barren_ice" (12, 31).
Land is NLCD not 0 (outside the map) and not 11 (open water). The rasters are read at their
x32 overview (about 960 m) and reprojected to the grid by nearest neighbor; the geology is
rasterized at the grid (cell center). Statewide thematic maps only (no imagery).

Units: the searched area (p2a_searched_area.geojson: band and Valdez box), and every tile of
p2a_tile_pool.csv (its lon/lat bounding box, densified). Writes p2a_strata.csv in long form:
unit ("searched" or the tile_id of p2a_tile_pool.csv), stratum, cells (land cells, km2);
a "_total" stratum row counts every grid cell of the unit, land or not.
Run: python -m akp.stage_p2.p2a_strata
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import rasterio
import shapely
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from pyproj import Transformer
from shapely.geometry import box, mapping, shape

from akp.paths import data_dir

HERE = Path(__file__).parent
OUT = HERE / "p2a_strata.csv"
CELL = 1000.0  # m
X0, X1, Y0, Y1 = -2_233_000.0, 1_495_000.0, 344_000.0, 2_381_000.0  # NLCD extent, snapped to 1 km
OVERVIEW = 4  # index of the x32 overview
PF_CUT = 50
GLACIAL = r"glaci|drift|till|moraine|outwash|kame|esker"
LC = {"forest": (41, 42, 43), "shrub": (51, 52),
      "herb_wet": (21, 22, 23, 24, 71, 72, 73, 74, 81, 82, 90, 95), "barren_ice": (12, 31)}
GEO = ("bedrock", "glacial", "surficial", "other")
PF = ("pf", "nopf", "unmapped")


def grid():
    w, h = int((X1 - X0) / CELL), int((Y1 - Y0) / CELL)
    return from_origin(X0, Y1, CELL, CELL), (h, w)


def warp_overview(path: Path, transform, shape_hw, nodata: int) -> np.ndarray:
    with rasterio.open(path, OVERVIEW_LEVEL=OVERVIEW) as ds:
        src = ds.read(1)
        dst = np.full(shape_hw, nodata, dtype=src.dtype)
        # NLCD declares no nodata but fills outside the map with 0: without src_nodata, GDAL
        # would nudge those source 0s to 1 to keep them apart from the destination nodata
        src_nodata = nodata if ds.nodata is None else ds.nodata
        reproject(src, dst, src_transform=ds.transform, src_crs=ds.crs, src_nodata=src_nodata,
                  dst_transform=transform, dst_crs="EPSG:3338", dst_nodata=nodata,
                  resampling=Resampling.nearest)
    return dst


def geology(maps: Path, transform, shape_hw) -> np.ndarray:
    gdb = maps / "sim3340_gdb/sim3340_gdb/AKgeol_web_gdb/geologic_data/AKStategeol.gdb"
    key = pyogrio.read_dataframe(gdb, layer="nsaqkey", read_geometry=False)
    glacial = set(key.QCLASS[key.GEOLOGIC_UNIT.str.contains(GLACIAL, case=False, na=False)
                             & (key.QCLASS >= 1000)])
    poly = pyogrio.read_dataframe(gdb, layer="AKStategeol_poly", columns=["QCLASS"])
    assert poly.crs.to_epsg() == 3338
    q = poly.QCLASS.to_numpy()
    code = np.where(np.isin(q, (0, 99)), 0, np.where(q >= 1000, np.where(np.isin(q, list(glacial)), 1, 2), 3))
    return rasterize(zip(poly.geometry, code), out_shape=shape_hw, transform=transform,
                     fill=3, dtype="uint8")


def strata_grid():
    transform, hw = grid()
    maps = data_dir("p2a/maps")
    nlcd = warp_overview(maps / "NLCD_2016_Land_Cover_AK_20200724/NLCD_2016_Land_Cover_AK_20200724.img",
                         transform, hw, 0)
    pf = warp_overview(maps / "AK_PFProbability_1m/AK_ProbofPF_1m_clipped.img", transform, hw, 255)
    geo = geology(maps, transform, hw)
    land = (nlcd != 0) & (nlcd != 11)
    lc = np.full(hw, -1, dtype=np.int8)
    for i, codes in enumerate(LC.values()):
        lc[np.isin(nlcd, codes)] = i
    pfc = np.where(pf <= 100, np.where(pf >= PF_CUT, 0, 1), 2).astype(np.int8)
    # one integer per stratum: pf * 16 + geo * 4 + lc; -1 off land
    s = np.where(land & (lc >= 0), pfc * 16 + geo.astype(np.int8) * 4 + lc, -1).astype(np.int16)
    assert not (land & (lc < 0)).any(), "an NLCD land class is missing from LC"
    return transform, s


TO3338 = Transformer.from_crs("EPSG:4326", "EPSG:3338", always_xy=True)


def to_3338(g):
    return shapely.transform(g, lambda xy: np.column_stack(TO3338.transform(xy[:, 0], xy[:, 1])))


def label(k: int) -> str:
    return f"{PF[k // 16]}|{GEO[(k // 4) % 4]}|{list(LC)[k % 4]}"


def counts(mask: np.ndarray, s: np.ndarray) -> dict[str, int]:
    v, n = np.unique(s[mask], return_counts=True)
    out = {label(int(k)): int(c) for k, c in zip(v, n) if k >= 0}
    out["_total"] = int(mask.sum())
    return out


def main() -> None:
    transform, s = strata_grid()
    hw = s.shape
    rows = []
    zones = [shape(f["geometry"]) for f in json.loads((HERE / "p2a_searched_area.geojson").read_text())["features"]]
    searched = ~geometry_mask([mapping(to_3338(z)) for z in zones], hw, transform)
    rows += [("searched", k, n) for k, n in counts(searched, s).items()]
    tiles = pd.read_csv(HERE / "p2a_tile_pool.csv")
    for t in tiles.itertuples():
        g = box(t.min_lon, t.min_lat, t.max_lon, t.max_lat).segmentize(0.01)
        m = ~geometry_mask([mapping(to_3338(g))], hw, transform)
        rows += [(str(t.tile_id), k, n) for k, n in counts(m, s).items()]
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("unit", "stratum", "cells"))
        w.writerows(rows)
    sa = {k: n for u, k, n in rows if u == "searched"}
    land = sum(n for k, n in sa.items() if k != "_total")
    print(f"searched area: {sa['_total']} cells, {land} land; strata: {len(sa) - 1}; "
          f"{len(tiles)} tiles; wrote {OUT.name}")


if __name__ == "__main__":
    main()
