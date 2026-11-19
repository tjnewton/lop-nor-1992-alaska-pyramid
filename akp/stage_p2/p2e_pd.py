"""Final detection probability, P2b alone and through the B-gate's one-method route.

Each trial plants a square faceted relief (as p2b_injection) at a random valid pixel of a
control tile drawn by land, in a WINDOW m window read boundless, so tile edges and nodata stay
in the window as in the real run (replaces the fixed fully valid window for this run). The
chosen variant's detector (`--variant`, p2b_explore.VARIANTS) runs on the planted window. For
each tabulated threshold t, the flags at or above t are merged as in p2e_anomalies; the plant's
anomaly is the one holding the highest-scoring flag within the detection radius (target
half-base plus one detector cell, as p2b_injection), and its representative flag's x
(delta_bic per window pixel, NaN for an invalid window) is recorded.
Writes p2e_pd_trials.csv (one row per trial: plant, stratum, vetoed, and per t the
representative's score and x), from which `summary(c)` gives P_d(t, c), weighted by the
searched area's stratum mix, plants inside a veto dropped, times one minus the blind fraction.
Seed: FINAL_SEED (fresh; the fixed third seed). `--summary` writes p2e_pd.csv: P_d at the operating t,
P2b alone (vetoed plants dropped, stratum-weighted, no blind fraction) and through the one-method
route (cut c from p2e_cut.csv, times one minus the blind fraction).
Run: python -m akp.stage_p2.p2e_pd --variant V | --summary  [--fourth: FOURTH_SEED, files tagged _t2]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.io import MemoryFile
from swe.detection import clopper_pearson
from swe.topo import faceted_template

from akp.stage_p2 import p2b_detector as det
from akp.stage_p2.p2b_controls_run import tile_path
from akp.stage_p2.p2b_explore import VARIANTS
from functools import lru_cache

from akp.stage_p2.p2b_injection import HEIGHTS, MAX_TRIES, N_TRIALS, TARGET_BASES, WINDOW, tiles_and_weights
from akp.stage_p2.p2e_anomalies import window
from akp.stage_p2.p2e_model_select import delta_bic

HERE = Path(__file__).parent
FINAL_SEED = 20261115
FOURTH_SEED = 20261116  # P_d at the second operating point
T_GRID = tuple(float(t) for t in np.round(np.arange(0.6, 1.0, 0.01), 2))
WORKERS = 4


@lru_cache(maxsize=1)
def tiles_by_pixels() -> tuple[list[dict], np.ndarray]:
    """Control tiles, drawn in proportion to raster pixels: a uniform pixel then rejected if
    nodata leaves the accepted plants uniform over valid pixels (land-weighted draws would
    weight tiles by land times their valid fraction)."""
    tiles, _ = tiles_and_weights()
    n = []
    for t in tiles:
        with det.open_tile(tile_path(t)) as ds:
            n.append(float(ds.width) * ds.height * abs(ds.res[0] * ds.res[1]))
    w = np.array(n)
    return tiles, w / w.sum()


def members(flags: list[dict]) -> list[int]:
    """Index of each flag's representative under the greedy merge of p2e_anomalies.merge."""
    order = sorted(range(len(flags)), key=lambda i: -flags[i]["score"])
    kept, rep = [], [0] * len(flags)
    for i in order:
        f = flags[i]
        k = next((j for j in kept if np.hypot(f["x"] - flags[j]["x"], f["y"] - flags[j]["y"])
                  <= max(f["base"], flags[j]["base"]) / 2), None)
        if k is None:
            kept.append(i)
            k = i
        rep[i] = k
    return rep


def read_window(ds, col: int, row: int):
    """The WINDOW m window centered on pixel (col, row), read boundless: (Grid, n)."""
    n = int(WINDOW / abs(ds.res[0])) | 1
    win = rasterio.windows.Window(col - n // 2, row - n // 2, n, n)
    a = ds.read(1, window=win, boundless=True, masked=True)
    z = np.ma.filled(a.astype(np.float64), np.nan)
    return det.Grid(z, ds.window_transform(win), ds.crs, abs(ds.res[0])), n


def score_plant(g, n: int, base: float, height: float, rotation: float) -> dict:
    """Plant at the window center and run the detector: the plant's EPSG:3338 position, and per
    tabulated t the representative flag's score, x, base and position (rx, ry)."""
    tpl = faceted_template(n_sides=4, base=base, height=height, dx=g.res, rotation=rotation)
    k, m = tpl.shape
    r0, c0 = (n - k) // 2, (n - m) // 2
    planted = g.z.copy()
    planted[r0:r0 + k, c0:c0 + m] += tpl  # NaN stays NaN: nodata is not land
    cx, cy = rasterio.transform.xy(g.transform, r0 + k // 2, c0 + m // 2)
    x0, y0 = Transformer.from_crs(g.crs, "EPSG:3338", always_xy=True).transform(cx, cy)
    fl = det.tile_flags(det.Grid(planted, g.transform, g.crs, g.res), s_min=min(T_GRID))
    out = dict(x=round(x0, 1), y=round(y0, 1))
    xs: dict[int, float] = {}
    prof = dict(driver="GTiff", width=n, height=n, count=1, dtype="float64", crs=g.crs, transform=g.transform,
                nodata=np.nan)
    with MemoryFile() as mem, mem.open(**prof) as ds:
        ds.write(planted, 1)
        for tt in T_GRID:
            sub = [f for f in fl if f["score"] >= tt]
            rep = members(sub)
            near = [j for j, f in enumerate(sub)
                    if (f["x"] - x0) ** 2 + (f["y"] - y0) ** 2 <= (base / 2 + det.RES[f["base"]]) ** 2]
            if not near:
                for col in ("score", "x", "base", "rx", "ry"):
                    out[f"{col}_{tt:.2f}"] = np.nan
                continue
            r = sub[rep[max(near, key=lambda j: sub[j]["score"])]]
            key = id(r)
            if key not in xs:
                w = window(ds, r)
                xs[key] = delta_bic(w, r["base"], det.RES[r["base"]]) / w.size if w is not None else np.nan
            out[f"score_{tt:.2f}"], out[f"x_{tt:.2f}"], out[f"base_{tt:.2f}"] = r["score"], xs[key], r["base"]
            out[f"rx_{tt:.2f}"], out[f"ry_{tt:.2f}"] = r["x"], r["y"]
    return out


def trial(args) -> dict:
    variant, i, base, height, seed = args
    det.configure(**VARIANTS[variant])
    tiles, weights = tiles_by_pixels()
    rng = np.random.default_rng([seed, i])
    rotation = float(rng.uniform(0, 90))
    for _ in range(MAX_TRIES):
        t = tiles[rng.choice(len(tiles), p=weights)]
        with rasterio.Env(GDAL_CACHEMAX=det.GDAL_CACHE_MB), det.open_tile(tile_path(t)) as ds:
            col, row = int(rng.integers(0, ds.width)), int(rng.integers(0, ds.height))
            g, n = read_window(ds, col, row)
        if np.isfinite(g.z[n // 2, n // 2]):  # the plant center sits on valid land
            break
    else:
        raise RuntimeError(f"trial {i}: no valid center in {MAX_TRIES} tries")
    return dict(trial=i, variant=variant, base=base, height=height, rotation=round(rotation, 2),
                tile_id=t["tile_id"]) | score_plant(g, n, base, height, rotation)


def summary(df: pd.DataFrame, t: float, c: float, blind: float) -> pd.DataFrame:
    """P_d(t, c) per (base, height): weighted by w_s, vetoed plants dropped, times 1 - blind."""
    d = df[~df.vetoed & (df.w > 0)]  # offland plants carry no weight
    hit = (d[f"score_{t:.2f}"] >= t) & (d[f"x_{t:.2f}"] > c)
    rows = []
    for (b, h), idx in d.groupby(["base", "height"]).groups.items():
        w, k = d.loc[idx, "w"], hit.loc[idx]
        p = float((w * k).sum() / w.sum())
        n_eff = float(w.sum() ** 2 / (w ** 2).sum())  # Kish effective size, for the interval
        lo, hi = clopper_pearson(int(round(p * n_eff)), int(round(n_eff)))
        rows.append(dict(base=b, height=h, t=t, c=c, blind=round(blind, 3), n=len(idx), p_detect=round(p * (1 - blind), 3),
                         ci_low=round(lo * (1 - blind), 3), ci_high=round(hi * (1 - blind), 3)))
    return pd.DataFrame(rows)


def compare() -> None:
    """Both seeds at both operating points, so a change of point is read on common trials.

    Routes: P2b alone; its ceiling through the gate (times one minus the blind fraction); the
    one-method route at the point's cut c; and a diagnostic cut at the largest control statistic
    at that point (the least strict cut with no control candidate). Also the representative
    flag's base for detected plants at the claimed base with the largest relief.
    Writes p2e_pd_compare.csv and p2e_pd_repbase.csv.
    """
    v = pd.read_csv(HERE / "p2e_vetoes.csv").set_index("stratum").loc["_total"]
    blind = float(1 - v.searched_unvetoed_km2 / v.searched_km2)
    rows, reps = [], []
    for seed_tag, seed in (("", "third"), ("_t2", "fourth")):
        df = pd.read_csv(HERE / f"p2e_pd_trials{seed_tag}.csv", dtype={"tile_id": str})
        for cut_tag in ("", "_t2"):
            cut = pd.read_csv(HERE / f"p2e_cut{cut_tag}.csv").iloc[0]
            an = pd.read_csv(HERE / f"p2e_anomalies{cut_tag}.csv", dtype={"tile_id": str})
            x_max = float(an[~an.vetoed & an.valid & (an.score >= cut.t)].x.max())
            t = float(cut.t)
            for route, c, bl in (("P2b alone", -np.inf, 0.0), ("P2b ceiling", -np.inf, blind),
                                 ("one-method", float(cut.c), blind), ("cut at control max", x_max, blind)):
                rows.append(summary(df, t, c, bl).assign(seed=seed, route=route))
            d = df[~df.vetoed & (df.w > 0) & (df.base == 213.0) & (df.height == 16.0) & (df[f"score_{t:.2f}"] >= t)]
            for b, n in d[f"base_{t:.2f}"].value_counts().items():
                reps.append(dict(seed=seed, t=t, rep_base=b, n=int(n),
                                 median_x=round(float(d[d[f"base_{t:.2f}"] == b][f"x_{t:.2f}"].median()), 3)))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(HERE / "p2e_pd_compare.csv", index=False)
    pd.DataFrame(reps).to_csv(HERE / "p2e_pd_repbase.csv", index=False)
    print(out[(out.base == 213.0) & (out.height == 16.0)][["seed", "t", "route", "c", "p_detect", "ci_low", "ci_high"]]
          .to_string(index=False))
    print(pd.DataFrame(reps).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS))
    ap.add_argument("--summary", action="store_true", help="P_d from p2e_pd_trials.csv and p2e_cut.csv")
    ap.add_argument("--compare", action="store_true", help="both seeds at both operating points (p2e_pd_compare.csv)")
    ap.add_argument("--fourth", action="store_true", help="second operating point: FOURTH_SEED, files tagged _t2")
    a = ap.parse_args()
    tag, seed = ("_t2", FOURTH_SEED) if a.fourth else ("", FINAL_SEED)
    if a.summary:
        df = pd.read_csv(HERE / f"p2e_pd_trials{tag}.csv", dtype={"tile_id": str})
        cut = pd.read_csv(HERE / f"p2e_cut{tag}.csv").iloc[0]
        v = pd.read_csv(HERE / "p2e_vetoes.csv").set_index("stratum").loc["_total"]
        blind = 1 - v.searched_unvetoed_km2 / v.searched_km2
        t = float(cut.t)
        out = pd.concat([summary(df, t, -np.inf, 0.0).assign(route="P2b alone"),
                         summary(df, t, float(cut.c), float(blind)).assign(route="one-method")], ignore_index=True)
        out.to_csv(HERE / f"p2e_pd{tag}.csv", index=False)
        print(out.pivot_table(index=["base", "height"], columns="route", values="p_detect").to_string())
        return
    if a.compare:
        compare()
        return
    assert a.variant, "--variant is required for the trials"
    tiles_and_weights()
    jobs = [(a.variant, i, b, h, seed) for i, (b, h) in enumerate(
        (b, h) for b in TARGET_BASES for h in HEIGHTS for _ in range(N_TRIALS))]
    with ProcessPoolExecutor(WORKERS) as ex:
        df = pd.DataFrame(list(ex.map(trial, jobs, chunksize=4)))
    import shapely

    from akp.stage_p2 import p2e_cut as pc
    from akp.stage_p2.p2a_strata import strata_grid
    from akp.stage_p2.p2b_controls_run import stratum_namer
    from akp.stage_p2.p2e_vetoes import veto_shapes

    transform, s = strata_grid()
    _, _, name = stratum_namer(s)
    inv = ~transform
    df["stratum"] = [name(int(np.floor(gy)), int(np.floor(gx))) for gx, gy in (inv * (x, y) for x, y in zip(df.x, df.y))]
    tree = shapely.STRtree(np.concatenate(list(veto_shapes().values())))
    df["vetoed"] = [tree.query(shapely.Point(x, y), predicate="intersects").size > 0 for x, y in zip(df.x, df.y)]
    w = pc.weights(pd.read_csv(HERE / "p2e_tile_land.csv", dtype={"tile_id": str}), pc._searched())
    df["w"] = df.stratum.map(w).fillna(0.0)
    df.to_csv(HERE / f"p2e_pd_trials{tag}.csv", index=False)
    print(f"{len(df)} trials; {int(df.vetoed.sum())} inside a veto; wrote p2e_pd_trials{tag}.csv")


if __name__ == "__main__":
    main()
