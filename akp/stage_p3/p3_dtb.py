"""P3 cover map input: depth to bedrock for Alaska from Shangguan et al. (2017), regridded to the P2a grid.

Source: SoilGrids "BDTICM" (absolute depth to bedrock, cm, 250 m; Shangguan et al. 2017,
doi:10.1002/2016MS000686), read over HTTP from ISRIC. The clip of Alaska is kept under
300 MB. The file is stored in full-width strips, so a full-resolution window would fetch whole global
rows. Only its internal 4x overview (about 1 km) is read, for the window WINDOW, and then averaged
onto the P2a 1 km grid (EPSG:3338). The model explains about 59% of the variance of absolute depth
to bedrock (Shangguan et al. 2017), so p3_cover treats it as a weak likelihood, not as truth.
Writes $AKP_DATA/p3/prior/p3_dtb.npz (dtb_m per land cell, NaN where missing) and p3_dtb.csv
(depth quantiles by P2a geology class within Track A). Run: python -m akp.stage_p3.p3_dtb
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds

from akp.stage_p2.p2a_strata import GEO, strata_grid
from akp.stage_p3.p3_prior import out_dir

HERE = Path(__file__).parent
URL = "/vsicurl/https://files.isric.org/soilgrids/former/2017-03-10/data/BDTICM_M_250m_ll.tif"
WINDOW = (-180.0, 51.0, -129.0, 72.0)  # lon/lat bounds of Alaska (the Aleutians west of 180 are not needed)
FACTOR = 4


def main() -> None:
    os.environ.setdefault("GDAL_HTTP_USERAGENT", "research-data-client/1.0")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    with rasterio.open(URL) as r:
        win = from_bounds(*WINDOW, transform=r.transform).round_offsets().round_lengths()
        shape = (int(win.height) // FACTOR, int(win.width) // FACTOR)
        a = r.read(1, window=win, out_shape=shape, resampling=Resampling.average).astype(np.float32)
        src_t = r.window_transform(win) * Affine.scale(win.width / shape[1], win.height / shape[0])
        nodata = r.nodata
    a[a == nodata] = np.nan
    a /= 100.0  # cm to m
    pr = np.load(out_dir() / "p3_prior.npz")
    land, dst_t = pr["land"], Affine(*pr["transform"])
    dst = np.full(land.shape, np.nan, np.float32)
    reproject(a, dst, src_transform=src_t, src_crs="EPSG:4326", dst_transform=dst_t, dst_crs="EPSG:3338",
              resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
    dtb = dst[land]
    np.savez_compressed(out_dir() / "p3_dtb.npz", dtb_m=dtb)
    _, s = strata_grid()
    geo = ((s[land] // 4) % 4).astype(int)
    ta = pr["track_a"]
    rows = []
    for i, g in enumerate(GEO):
        v = dtb[(geo == i) & ta & np.isfinite(dtb)]
        rows.append(dict(geo=g, cells=int(v.size), dtb_m_p10=round(float(np.percentile(v, 10)), 1) if v.size else np.nan,
                         dtb_m_median=round(float(np.median(v)), 1) if v.size else np.nan,
                         dtb_m_p90=round(float(np.percentile(v, 90)), 1) if v.size else np.nan,
                         share_ge_50m=round(float((v >= 50).mean()), 4) if v.size else np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "p3_dtb.csv", index=False)
    print(out.to_string(index=False))
    print(f"land cells with a value: {np.isfinite(dtb).mean():.3f}")


if __name__ == "__main__":
    main()
