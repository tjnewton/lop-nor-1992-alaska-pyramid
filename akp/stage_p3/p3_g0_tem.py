"""P3 G0 robustness: Track A prior mass inside the time-domain EM survey outlines (not modeled in G0).

If every cell inside these outlines counted as fully searched, G0 could rise by at most this mass.
Outlines come from the DGGS vector packages (p3_vector_fetch, boundary members only). Writes
p3_g0_tem.csv (survey, land_km2, mass_cover, mass_none). Run: python -m akp.stage_p3.p3_g0_tem
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from rasterio.features import geometry_mask
from shapely.geometry import mapping

from akp.paths import data_dir
from akp.stage_p3.p3_prior import out_dir

HERE = Path(__file__).parent
TEM = {"DGGS_GPR2025-2": "kuskokwim_em_kaiyuh_survey_boundary", "DGGS_GPR2026-3": "candle_hills_boundary",
       "DGGS_GPR2024-2": "seward_pen_em_kigluaik_survey_boundary"}


def main() -> None:
    pr, cv = np.load(out_dir() / "p3_prior.npz"), np.load(out_dir() / "p3_cover.npz")
    land, transform = pr["land"], Affine(*pr["transform"])
    vec = data_dir("p3/inventory/vector")
    rows = []
    for sid, stem in TEM.items():
        g = gpd.read_file(next(vec.glob(f"*/**/{stem}.shp"))).to_crs(3338).union_all()
        m = geometry_mask([mapping(g)], land.shape, transform, invert=True)[land]
        rows.append(dict(survey=sid, land_km2=int(m.sum()), mass_cover=round(float(cv["mass_cover"][m].sum()), 4),
                         mass_none=round(float(pr["mass"][m].sum()), 4)))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "p3_g0_tem.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
