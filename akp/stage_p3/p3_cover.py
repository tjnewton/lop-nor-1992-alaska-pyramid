"""P3 cover map: P(loose cover >= 50 m) per land cell, and the cover-weighted Track A prior.

Two versions:
- "class": the coarse W1 placeholders. Each SIM 3340 class of the P2a strata (bedrock, glacial,
  surficial, other) gets a factor COVER (inputs: both sediment classes share one value; bedrock and
  other keep the plan's low floor).
- "dtb" (the default from W2): a weak likelihood from the Shangguan et al. (2017) depth to
  bedrock (p3_dtb). P(cover >= MIN_COVER_M) = 1 - Phi(ln(MIN_COVER_M / dtb) / SIGMA_LN), a
  log-normal error of SIGMA_LN about the model's prediction (an input: the model explains about 59%
  of the variance). Cells without a value fall back to the class factor.
The cover-weighted prior is mass x cover, renormalized over land. Writes $AKP_DATA/p3/prior/p3_cover.npz:
cover (dtb version), cover_class, and the cover-weighted masses for W and W_EQUAL (mass_cover and
mass_equal_cover use the dtb version; *_class keep the placeholders). Also writes p3_cover.csv (check,
scope, value). Run: python -m akp.stage_p3.p3_cover
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from akp.stage_p2.p2a_strata import GEO, strata_grid
from akp.stage_p3.p3_prior import out_dir

HERE = Path(__file__).parent
COVER = {"bedrock": 0.02, "glacial": 0.25, "surficial": 0.25, "other": 0.02}
MIN_COVER_M = 50.0
SIGMA_LN = 1.0  # log-normal error of the depth-to-bedrock model (input)


def main() -> None:
    pr = np.load(out_dir() / "p3_prior.npz")
    _, s = strata_grid()
    land = s >= 0
    assert (land == pr["land"]).all()
    geo = ((s[land] // 4) % 4).astype(int)
    cover_class = np.array([COVER[g] for g in GEO])[geo]
    dtb = np.load(out_dir() / "p3_dtb.npz")["dtb_m"].astype(np.float64)
    ok = np.isfinite(dtb) & (dtb > 0)
    cover = cover_class.copy()
    cover[ok] = norm.sf(np.log(MIN_COVER_M / dtb[ok]) / SIGMA_LN)
    rows = [dict(check="dtb_cells_used", scope="_all", value=int(ok.sum()))]
    out = {"cover": cover.astype(np.float32), "cover_class": cover_class.astype(np.float32)}
    for key in ("mass", "mass_equal"):
        m = pr[key].astype(np.float64)
        for tag, cf in (("", cover), ("_class", cover_class)):
            mc = m * cf / np.sum(m * cf)
            out[f"{key}_cover{tag}"] = mc.astype(np.float32)
            for i, g in enumerate(GEO):
                rows.append(dict(check=f"{key}_cover{tag}_by_class", scope=g, value=round(float(mc[geo == i].sum()), 4)))
        for i, g in enumerate(GEO):
            rows.append(dict(check=f"{key}_by_class", scope=g, value=round(float(m[geo == i].sum()), 4)))
    ta = pr["track_a"]
    for i, g in enumerate(GEO):
        sel = (geo == i) & ta
        rows.append(dict(check="track_a_mean_cover_dtb", scope=g, value=round(float(cover[sel].mean()), 4)))
    ta = pr["track_a"]
    for i, g in enumerate(GEO):
        rows.append(dict(check="track_a_land_km2_by_class", scope=g, value=int(((geo == i) & ta).sum())))
    np.savez_compressed(out_dir() / "p3_cover.npz", **out)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "p3_cover.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
