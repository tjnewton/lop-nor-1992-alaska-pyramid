"""P2b calibration, part 2: detection probability by injection into control tiles.

Each trial plants a square faceted relief (swe.topo.faceted_template) of base B and height H
at a random rotation in [0, 90) deg at a random location of a control tile (tiles drawn in
proportion to their valid land, from the control-run checkpoints), reads a WINDOW m window of
the native 5 m DTM centered there, and runs the full detector (all bases, scores down to 0)
with and without the plant. Its score at each detector base is the highest flag within half
of max(B, detector base) of the plant center; the no-plant score at the same place is the
null. A window must be fully valid (no nodata).
Grid (proposed): B in TARGET_BASES (213 m = 700 ft, the base claimed in C3/C4), H in
HEIGHTS; N_TRIALS per cell; seeded per trial.
Writes p2b_injection.csv (one row per trial) and p2b_pd.csv (B, H, threshold, n, detected,
p_detect, ci_low, ci_high: detected when any detector base scores at or above the threshold).
Run: python -m akp.stage_p2.p2b_injection
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from swe.detection import clopper_pearson
from swe.topo import faceted_template

from akp.stage_p2 import p2b_detector as det
from akp.stage_p2.p2b_controls_run import tile_path

HERE = Path(__file__).parent
SEED = 20261111
TARGET_BASES = (50.0, 100.0, 213.0, 400.0)  # m
HEIGHTS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)  # m
N_TRIALS = 100
WINDOW = 2400.0  # m
THRESHOLDS = (0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
WORKERS = 4  # small windows (about 0.5 GB per worker); see p2b_controls_run for the budget
MAX_TRIES = 200


def near_scores(fl: list[dict], x0: float, y0: float, target_base: float) -> dict[str, float]:
    out = {}
    for b in det.BASES:
        r = target_base / 2 + det.RES[b]  # within the target's half-base, plus one detector cell
        s = [f["score"] for f in fl if f["base"] == b and (f["x"] - x0) ** 2 + (f["y"] - y0) ** 2 <= r * r]
        out[f"s{int(b)}"] = max(s) if s else np.nan
    return out


@lru_cache(maxsize=1)
def tiles_and_weights() -> tuple[list[dict], np.ndarray]:
    """Control tiles and draw weights (on-land valid area, p2b_control_land.csv)."""
    tiles = pd.read_csv(HERE / "p2a_controls.csv", dtype={"tile_id": str}).to_dict("records")
    land = pd.read_csv(HERE / "p2b_control_land.csv", dtype={"tile_id": str}).set_index("tile_id").land_km2
    assert set(land.index) == {t["tile_id"] for t in tiles}, "run p2b_controls_run first"
    w = np.array([land[t["tile_id"]] for t in tiles])
    return tiles, w / w.sum()


def trial(args) -> dict:
    i, base, height = args
    tiles, weights = tiles_and_weights()
    rng = np.random.default_rng([SEED, i])
    rotation = float(rng.uniform(0, 90))
    for _ in range(MAX_TRIES):
        t = tiles[rng.choice(len(tiles), p=weights)]
        with rasterio.Env(GDAL_CACHEMAX=det.GDAL_CACHE_MB), det.open_tile(tile_path(t)) as ds:
            n = int(WINDOW / abs(ds.res[0])) | 1
            if ds.width < n or ds.height < n:
                continue
            col, row = int(rng.integers(0, ds.width - n + 1)), int(rng.integers(0, ds.height - n + 1))
            g = det.read_native(ds, rasterio.windows.Window(col, row, n, n))
        if np.isfinite(g.z).all():
            break
    else:
        raise RuntimeError(f"trial {i}: no fully valid window in {MAX_TRIES} tries")
    tpl = faceted_template(n_sides=4, base=base, height=height, dx=g.res, rotation=rotation)
    k, m = tpl.shape
    r0, c0 = (n - k) // 2, (n - m) // 2
    planted = g.z.copy()
    planted[r0:r0 + k, c0:c0 + m] += tpl
    cx, cy = rasterio.transform.xy(g.transform, r0 + k // 2, c0 + m // 2)
    x0, y0 = Transformer.from_crs(g.crs, "EPSG:3338", always_xy=True).transform(cx, cy)
    with_plant = near_scores(det.tile_flags(det.Grid(planted, g.transform, g.crs, g.res), s_min=0.0), x0, y0, base)
    null = near_scores(det.tile_flags(g, s_min=0.0), x0, y0, base)
    return dict(trial=i, base=base, height=height, rotation=round(rotation, 2), tile_id=t["tile_id"],
                x=round(x0), y=round(y0), **with_plant, **{f"null_{k}": v for k, v in null.items()})


def main() -> None:
    tiles_and_weights()  # fail early if the control run is incomplete
    jobs, i = [], 0
    for base in TARGET_BASES:
        for height in HEIGHTS:
            for _ in range(N_TRIALS):
                jobs.append((i, base, height))
                i += 1
    with ProcessPoolExecutor(WORKERS) as ex:
        rows = list(ex.map(trial, jobs, chunksize=4))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "p2b_injection.csv", index=False)
    cols = [f"s{int(b)}" for b in det.BASES]
    best = df[cols].max(axis=1)
    out = []
    for (base, height), idx in df.groupby(["base", "height"]).groups.items():
        for t in THRESHOLDS:
            k, n = int((best[idx] >= t).sum()), len(idx)
            lo, hi = clopper_pearson(k, n)
            out.append(dict(base=base, height=height, threshold=t, n=n, detected=k,
                            p_detect=round(k / n, 3), ci_low=round(lo, 3), ci_high=round(hi, 3)))
    pd.DataFrame(out).to_csv(HERE / "p2b_pd.csv", index=False)
    print(pd.DataFrame(out).pivot_table(index=["base", "height"], columns="threshold", values="p_detect").to_string())


if __name__ == "__main__":
    main()
