"""P2c feasibility: how much searched land and control-eligible land airborne surveys cover.

Footprints only (USGS Airborne Geophysical Survey Inventory v5.0, Johnson et al. 2019,
doi:10.5066/P9K8YTW1, fetched by akp.setup.pr_e_airborne), so no survey data is read.
Kinds: "EM" (every Alaska survey with EM == 1) and "Mag_fine" (Mag == 1 with line spacing at
most FINE_KM; coarser magnetics cannot resolve a body of the claimed base).
Land and strata come from the P2a 1 km grid (p2a_strata.strata_grid). Units: "searched"
(p2a_searched_area.geojson) and "pool" (land outside both probe boxes, where controls may lie).
Writes p2c_coverage.csv in long form: level ("survey", "stratum" or "total"), unit, kind, key
(survey id, stratum label, or "_all"), name, year, spacing_km, covered_km2, land_km2, frac.
covered_km2 counts land cells whose center lies in the footprint (a union for stratum and total).
Run: python -m akp.stage_p2.p2c_coverage
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import geometry_mask
from shapely.geometry import box, mapping, shape

from akp.paths import data_dir
from akp.stage_p2.p2a_strata import counts, strata_grid, to_3338
from akp.zones import BOXES

HERE = Path(__file__).parent
OUT = HERE / "p2c_coverage.csv"
FINE_KM = 0.4  # densest common DGGS spacing; about two claimed bases


def masks():
    transform, s = strata_grid()
    hw = s.shape

    def inside(geoms):
        return ~geometry_mask([mapping(g) for g in geoms], hw, transform)

    zones = [to_3338(shape(f["geometry"]))
             for f in json.loads((HERE / "p2a_searched_area.geojson").read_text())["features"]]
    boxes = [to_3338(box(b["minlongitude"], b["minlatitude"], b["maxlongitude"], b["maxlatitude"]).segmentize(0.01))
             for b in BOXES.values()]
    land = s >= 0
    return transform, s, inside, {"searched": inside(zones) & land, "pool": ~inside(boxes) & land}


def surveys() -> gpd.GeoDataFrame:
    g = gpd.read_file(f"zip://{data_dir('probes/airborne_inventory') / 'footprint.zip'}"
                      "!AirborneGeophysicalSurvey_footprint.shp")
    ak = g[g.state == "AK"].to_crs(3338)
    em = ak[ak.EM == 1].assign(kind="EM")
    mag = ak[(ak.Mag == 1) & (ak.spacing1 > 0) & (ak.spacing1 <= FINE_KM)].assign(kind="Mag_fine")
    return pd.concat([em, mag])


def main() -> None:
    _, s, inside, units = masks()
    sv = surveys()
    rows = []
    union = {k: np.zeros(s.shape, bool) for k in sv.kind.unique()}
    for r in sv.itertuples():
        m = inside([r.geometry])
        union[r.kind] |= m
        for u, um in units.items():
            n = int((m & um).sum())
            if n:
                rows.append(dict(level="survey", unit=u, kind=r.kind, key=r.Survey, name=r.name, year=r.Year,
                                 spacing_km=r.spacing1, covered_km2=n, land_km2=int(um.sum())))
    for kind, cov in union.items():
        for u, um in units.items():
            land, covered = counts(um, s), counts(um & cov, s)
            rows += [dict(level="stratum", unit=u, kind=kind, key=k, covered_km2=covered.get(k, 0), land_km2=n)
                     for k, n in land.items() if k != "_total"]
            rows.append(dict(level="total", unit=u, kind=kind, key="_all", covered_km2=covered["_total"],
                             land_km2=land["_total"]))
    df = pd.DataFrame(rows)
    df["frac"] = (df.covered_km2 / df.land_km2).round(4)
    tot = df[df.level == "total"]
    # self-check: a union covers no more than its unit, and at least its largest survey
    assert (tot.covered_km2 <= tot.land_km2).all()
    for (u, k), g in df[df.level == "survey"].groupby(["unit", "kind"]):
        assert g.covered_km2.max() <= tot[(tot.unit == u) & (tot.kind == k)].covered_km2.iloc[0]
    df.to_csv(OUT, index=False)
    print(tot.to_string(index=False))


if __name__ == "__main__":
    main()
