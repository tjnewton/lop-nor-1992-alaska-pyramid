"""T1: the frozen terrain detector on one real mound.

Question. Does the terrain detector, exactly as frozen, flag and keep a real man-made mound of
about the body's base? Until now it was tested only on planted bodies of its own faceted shape.

Rules fixed before the check: the
target is Mound A at Poverty Point, with a center rule, a void rule, a hit rule, the stage-two
route, a flag-density test and four readings. Every constant the rule names is a constant below.

Phase 1 (default run) reads no elevation data of the target. It writes, before the data:
- t1_scale_response.csv. The detector's own shape (the square faceted pyramid that
  p2b_injection plants) on flat clean ground, for each base in SCAN_BASES and each rotation in
  SCAN_ROTS. Two rows per base: the rotation with the highest and with the lowest best score.
  Columns: s50..s400 are the best score at each detector base within half the planted base
  plus one detector cell (the frozen hit rule); best_score and best_base are their maximum and
  the base that gave it; reaches_threshold compares best_score with the frozen threshold;
  x, rep_base and kept are stage two (the anomaly's statistic, its base, and x above the frozen
  cut); frac_reach and frac_kept are shares of the scanned rotations.
- t1_synthetic_prediction.csv. Idealized mounds of the published size (platform, cone, mix) on
  clean ground and at two noise levels, over MOUND_ROTS. Per row: the median best score per
  base, the stage-two statistic over the hits, and the count of each reading.
- t1_checks.csv. Named scalar checks (see "Self-check").

Phase 2 (`--real`, not run in phase 1) refuses unless this module and the phase-1 tables are
committed unchanged. It then downloads the two files the rule names into $AKP_DATA/p4/real_mound,
joins and block-averages them, applies the rules, and plants the mound's relief in the Alaskan
control tiles. It will write t1_download.csv (url, bytes, sha256), t1_real_mound.csv (center,
scores, statistic, density, reading), t1_real_flags.csv (flags at or above the threshold, and
all flags inside the hit radius), t1_alaska_trials.csv and t1_alaska_pd.csv.

Two further runs, on trials already defined. Neither changes a table above.
- `--paired` writes t1_alaska_paired.csv and t1_alaska_paired_checks.csv. On the control
  windows of t1_alaska_trials.csv (rebuilt from the seed and asserted equal row by row) the
  frozen route plants its own pyramid twice: at the frozen planted base and relief nearest the
  mound, and at the mound's length and measured relief (t1_alaska_pd.csv). One row per body and
  stage: n is the trials the frozen summary counts (not vetoed, with weight); hits are
  unweighted; p_detect and its interval are the frozen summary's (the "both" row carries its
  blind fraction). The template rows also carry the pair against the mound: trials where only
  the template is detected, where only the mound is, and the exact two-sided binomial test on
  them. The mound's trials are read from the committed table and not rerun.
- `--searched` writes t1_searched_paired.csv and t1_searched_checks.csv. The falsifier:
  the mound's relief (rebuilt from the same lidar files, checked against the committed center
  and height) is planted on the searched-area windows of the frozen template cell SELF_CELL
  (p2e_claim_trials.csv), by the frozen `p2e_claim_run.trial_at` and the counting lines of its
  `inject`: both stages, the representative flag owned by its tile and outside every veto, all
  trials counted. One row per trial: the template's committed outcome beside the mound's. The
  reading is `searched_reading`. Only outcomes at the operating threshold are written. The
  frozen route holds the flags of each planted window in memory to score the plant; they are
  written nowhere, and no window is run without a plant.
  The guard (`claim_guard`): the frozen planting run called `check_freeze` and then opened the
  tiles with `claim_zone=True`. `check_freeze` lists the whole akp tree, so it refuses now that
  later stages have added files. What it protects is checked instead: no code or environment
  file that existed at the P2e freeze differs from it, the P3 freeze (which covers akp/stage_p2,
  the paths, the environment and the installed SWE commit) passes, and the tile checksums
  were verified for the freeze commit. The frozen template's trials are then rerun on those
  windows and must equal the committed rows.

Inputs. The frozen code and tables of akp/stage_p2, by import; nothing there is edited. The
threshold is p2e_cut.csv column t. The cut is p2e_vintage.frozen_cut. The control flag rate is
p2b_control_fa.csv (base "all", n_flags at the threshold) over the on-land control area of
p2b_control_area.csv. The Alaskan self-check and phase 2 read the cached control tiles.

Method.
- Both stages on a grid (`flags_of`, `route`): `p2b_detector.tile_flags` in the frozen blocks;
  flags at or above the threshold merged by `p2e_pd.members`; the anomaly holding the best hit
  flag inside the radius; its window by `p2e_anomalies.window`; its statistic by
  `p2e_model_select.delta_bic` per window pixel. These are the lines of `p2e_pd.score_plant`
  without the plant, which that function cannot omit.
- The frozen code measures distances after a reprojection to EPSG:3338. Every grid is handed
  to it declared as EPSG:3338, so that reprojection is the identity and distances are in the
  file's own coordinates, as the rule requires.
- Real target (`join`, `center`, `void`, `target`, `density`, `reading`): rule by rule.
- The mound on Alaskan ground (`relief`, `planting`, `alaska_trials`, `alaska_pd`): the frozen
  `p2e_pd.trial` runs unchanged, with the pyramid it would plant replaced at run time by the
  mound's relief. Window choice, weighting, detector, merge, window, statistic and the summary
  are the frozen lines themselves.

Run, from the repository root:
  phase 1: PYTHONPATH=. conda run -n swe-akp python -m akp.stage_p4.t1_real_mound
  phase 2: PYTHONPATH=. conda run -n swe-akp python -m akp.stage_p4.t1_real_mound --real
the same with --paired, then with --searched

Self-check (every default run, before the tables). It proves that:
- `route` gives the same score, statistic and base as the frozen `score_plant` for a planted
  pyramid on rolling ground, and `planting` leaves `score_plant` unchanged for its own shape;
- the score does not depend on the planted height on clean ground, and rotations mirror
  about 45 degrees, so SCAN_ROTS covers every case;
- on a synthetic site (a mound on rolling ground, a second smaller mound, a road-like ridge,
  two overlapping 1 m tiles with a nodata hole) the join equals the block mean of the whole,
  the center rule finds the mound, the void rule fires on a hole and not otherwise, blocked
  flags equal whole-grid flags, and the relief recovers the mound's height;
- on the Alaskan control tiles, the frozen trials of one cell are reproduced exactly, and the
  same pyramid planted through the relief route on the same windows gives outcomes in line
  with them. If the tiles are not reachable the check says so and the rest still runs.
"""
from __future__ import annotations

import argparse
import resource
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import shapely
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter, rotate
from scipy.stats import binomtest, poisson
from swe.topo import faceted_template

from akp.paths import data_dir, data_glob, data_root
from akp.setup.manifest import sha256
from akp.stage_p2 import p2b_detector as det
from akp.stage_p2 import p2e_claim_run as cr
from akp.stage_p2 import p2e_cut as pc
from akp.stage_p2 import p2e_pd as pd2
from akp.stage_p2.p2b_controls_run import stratum_namer
from akp.stage_p2.p2b_explore import VARIANTS
from akp.stage_p2.p2b_fetch_ifsar import fetch
from akp.stage_p2.p2b_injection import HEIGHTS, N_TRIALS, TARGET_BASES, WINDOW, near_scores
from akp.stage_p2.p2e_anomalies import window
from akp.stage_p2.p2e_model_select import delta_bic
from akp.stage_p2.p2e_vetoes import veto_shapes
from akp.stage_p2.p2e_vintage import frozen_cut
from akp.stage_p3 import p3_freeze
from akp.stage_p3.p3_g1b_run import strata

HERE = Path(__file__).parent
P2 = Path(det.__file__).parent
PHASE1 = ("t1_scale_response.csv", "t1_synthetic_prediction.csv", "t1_checks.csv")

# Frozen values, read from the frozen tables.
T = float(pd.read_csv(P2 / "p2e_cut.csv").iloc[0].t)  # operating threshold on the score
C = float(frozen_cut().c)  # cut on delta_bic per window pixel
CELL = min(det.RES.values())  # m: the Alaskan product's native cell
CRS = rasterio.crs.CRS.from_epsg(3338)
SCOLS = [f"s{int(b)}" for b in det.BASES]

# As written there.
FILES = ("USGS_1M_15_x64y362_LA_NortheastDOTD_2017_C20.tif", "USGS_1M_15_x65y362_LA_NortheastDOTD_2017_C20.tif")
SITE_LON, SITE_LAT = -91.41139, 32.63667  # deg (cite: published site coordinate)
MOUND_LONG, MOUND_SHORT, MOUND_HEIGHT = 215.0, 200.0, 22.0  # m (cite: published dimensions)
SEARCH_R = 1000.0  # m (input: search radius about the site coordinate)
CENTROID_R = 150.0  # m (input: radius about the summit for the centroid)
VOID_R = 1200.0  # m (input: data needed about the center)
EXCLUDE_R = 1000.0  # m (input: exclusion radius for the density)
ALPHA = 0.01  # (input: level for "denser than the controls")
HIT_BASES = det.BASES[-2:]  # A hit "of this base" is at one of the two largest bases
# The public USGS store's path for 1 m products, the store that holds the Alaskan tiles. The
# project folder is the tail of the file name. Not requested in phase 1, so not yet tested.
STORE = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/{project}/TIFF/{file}"

# Fixed here, before the lidar files are read.
# One native cell: a finer step would not move the planted raster's edge by a whole cell.
SCAN_BASES = np.arange(min(det.BASES), max(det.BASES) + 0.1, CELL)
# The detector turns its template every 15 deg and a square grid mirrors about 45 deg (checked
# below), so 0 to 45 holds every case; a sixth of the detector's step resolves the worst one.
SCAN_ROTS = np.arange(0.0, 45.1, 2.5)
SCAN_HEIGHT = max(HEIGHTS)  # m: any height gives the same score on clean ground (checked below)
# A rectangle of unequal sides repeats every 180 deg and mirrors about 90.
MOUND_ROTS = np.arange(0.0, 90.1, 5.0)
# (sd in m, correlation length in m; input: chosen here, leaves the levels to this module).
# "sensor": white, of the order of lidar height error in one cell. "ground": a meter of relief
# correlated over 20 m, far rougher than a flat field, to show the sign of the effect.
NOISE = {"clean": (0.0, 0.0), "sensor": (0.1, 0.0), "ground": (1.0, 20.0)}
# The flat top's share of the shorter side is not published: three values show what it does.
PLATFORM_TOPS = (0.1, 0.3, 0.5)
MIX = 0.5  # the "mix" is this share of the cone plus the rest of the platform, cell by cell
SHAPES = [("platform", t) for t in PLATFORM_TOPS] + [("cone", np.nan)] + [("mix", t) for t in PLATFORM_TOPS]
# The ring for the mound's relief. A corner of a rectangle of the published sides lies
# hypot(long, short) / 2 from its center (under 150 m), and the fixed center is a centroid of the
# upper half, which may sit tens of meters off the plan's center: the ring starts at 200 m.
# 100 m of width holds thousands of cells for a three-parameter plane, and the outer radius
# stays far inside the VOID_R that must hold data.
RING_IN, RING_OUT = 200.0, 300.0  # m
MOUND_SEED = 20261119  # new: the seeds up to 20261118 are used by the frozen stages
# As many as each frozen cell, so the mound's interval is as wide as the one it sits beside.
N_MOUND_TRIALS = N_TRIALS
SELF_CELL = (TARGET_BASES[2], max(HEIGHTS))  # the frozen cell the Alaskan self-check repeats
N_SELF = 24  # trials of that cell in the self-check: small, under half a minute
BLOCK = det.BLOCK // 4  # cells: the frozen block peaked at 2.7 GB per process; flags do not depend on it (checked)
STRIP = 500  # native rows per read of the join: a few hundred MB for two 10 km tiles
PAIRED_ALPHA = 0.05  # (input: level for the paired test)


def control_rate() -> float:
    """Raw control flags at the threshold, all bases, per km2 of on-land control area."""
    fa = pd.read_csv(P2 / "p2b_control_fa.csv", dtype={"base": str})
    area = pd.read_csv(P2 / "p2b_control_area.csv")
    n = fa[(fa.base == "all") & np.isclose(fa.threshold, T)].n_flags.iloc[0]
    return float(n / area[area.stratum != "offland"].control_km2.sum())  # the table counts on-land flags only


# ---------------------------------------------------------------- both stages on a grid

@contextmanager
def as_file(g: det.Grid):
    """The grid as an in-memory raster declared EPSG:3338 (see the docstring: distances in the
    file's own coordinates). The frozen window code reads from a raster, as score_plant does."""
    prof = dict(driver="GTiff", width=g.z.shape[1], height=g.z.shape[0], count=1, dtype="float64", crs=CRS,
                transform=g.transform, nodata=np.nan)
    with MemoryFile() as mem, mem.open(**prof) as ds:
        ds.write(g.z, 1)
        yield ds


def flags_of(ds, block: int = BLOCK) -> list[dict]:
    """Every flag of the raster, scores down to 0: the loop of p2b_controls_run.run_tile."""
    fl = []
    for win, core in det.blocks(ds, block):
        b = det.read_native(ds, win)
        if np.isfinite(b.z).any():
            fl += [f for f in det.tile_flags(b, s_min=0.0) if det.in_core(f, ds, core)]
    return fl


def route(ds, fl: list[dict], x0: float, y0: float, half: float, bases=det.BASES) -> dict:
    """Stage one and stage two at (x0, y0): score_plant's lines at the frozen threshold, no plant.

    A hit is a flag at or above T, at one of `bases`, within `half` plus that base's cell.
    `same_any_base` says whether the best flag of any base inside the radius leads to the same
    anomaly (score_plant takes that one; takes the hit flag)."""
    near = lambda f: (f["x"] - x0) ** 2 + (f["y"] - y0) ** 2 <= (half + det.RES[f["base"]]) ** 2  # noqa: E731
    out = near_scores(fl, x0, y0, 2 * half)  # the best score at each base, whatever the threshold
    sub = [f for f in fl if f["score"] >= T]
    # simplification: the frozen merge is quadratic in the flags at or above T. Fine up to some
    # thousands; if a file holds far more, merge only those within a few bases of the point.
    rep = pd2.members(sub)
    inside = [j for j, f in enumerate(sub) if near(f)]
    hits = [j for j in inside if sub[j]["base"] in bases]
    out |= dict(hit=bool(hits), score=np.nan, hit_base=np.nan, x=np.nan, rep_base=np.nan, same_any_base=np.nan,
                other_base_flags=len(inside) - len(hits))
    if hits:
        best = lambda js: max(js, key=lambda j: sub[j]["score"])  # noqa: E731
        j = best(hits)
        r = sub[rep[j]]
        w = window(ds, r)
        out |= dict(score=sub[j]["score"], hit_base=sub[j]["base"], rep_base=r["base"],
                    same_any_base=rep[best(inside)] == rep[j],
                    x=delta_bic(w, r["base"], det.RES[r["base"]]) / w.size if w is not None else np.nan)
    out["kept"] = bool(out["x"] > C)  # NaN (no hit, or no window) is not kept:
    return out


def both_stages(g: det.Grid, x0: float, y0: float, half: float, bases=det.BASES) -> dict:
    with as_file(g) as ds:
        return route(ds, flags_of(ds), x0, y0, half, bases)


def reading(void: bool, hit: bool, kept: bool, dense: bool = False, rank: int | None = None) -> str:
    """The fixed readings, by the numbers alone."""
    if void:
        return "void"
    if not hit:
        return "no hit"
    many = f", in a file where the detector flags many other features (rank {rank} among flags of its base)"
    return ("hit and kept" if kept else "hit and removed") + (many if dense else "")


# ---------------------------------------------------------------- phase 1: synthetic shapes

N = int(WINDOW / CELL) | 1  # the frozen planting window
YY, XX = (np.mgrid[0:N, 0:N] - N // 2) * CELL  # m from the center cell
X0 = Y0 = (N // 2 + 0.5) * CELL  # the center cell's coordinates in the grid below


def grid(z: np.ndarray) -> det.Grid:
    return det.Grid(z, from_origin(0.0, N * CELL, CELL, CELL), CRS, CELL)


def plant(z: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """`shape` added at the center of z, placed as the frozen routine places its template."""
    k, m = shape.shape
    r0, c0 = (z.shape[0] - k) // 2, (z.shape[1] - m) // 2
    out = z.copy()
    out[r0:r0 + k, c0:c0 + m] += shape
    return out


def pyramid(base: float, height: float, rotation: float) -> np.ndarray:
    return faceted_template(n_sides=4, base=base, height=height, dx=CELL, rotation=rotation)


def mound(xx, yy, kind: str, top: float, rot: float = 0.0) -> np.ndarray:
    """An idealized mound of the published size. Platform: four flat sides up to a flat top
    that spans `top` of the shorter side. Cone: an elliptical cone on the published axes."""
    th = np.radians(rot)
    u, v = xx * np.cos(th) + yy * np.sin(th), -xx * np.sin(th) + yy * np.cos(th)
    plat = np.clip(np.minimum(MOUND_LONG / 2 - np.abs(u), MOUND_SHORT / 2 - np.abs(v)) / ((1 - top) * MOUND_SHORT / 2), 0, 1)
    cone = np.clip(1 - np.hypot(u / (MOUND_LONG / 2), v / (MOUND_SHORT / 2)), 0, None)
    return MOUND_HEIGHT * {"platform": plat, "cone": cone, "mix": MIX * cone + (1 - MIX) * plat}[kind]


def noise(rng, shape, sd: float, corr: float):
    if not sd:
        return 0.0
    e = rng.normal(size=shape)
    if corr:
        e = gaussian_filter(e, corr / CELL)
    return sd * e / e.std()


def scale_response() -> pd.DataFrame:
    rows = [dict(base=b, rotation=r) | both_stages(grid(plant(np.zeros((N, N)), pyramid(b, SCAN_HEIGHT, r))), X0, Y0, b / 2)
            for b in SCAN_BASES for r in SCAN_ROTS]
    df = pd.DataFrame(rows)
    df["best_score"] = df[SCOLS].max(axis=1)
    df["best_base"] = df[SCOLS].idxmax(axis=1).str[1:].astype(float)
    df["reaches_threshold"] = df.best_score >= T
    out = []
    for _, d in df.groupby("base"):
        for case, i in (("best", d.best_score.idxmax()), ("worst", d.best_score.idxmin())):
            out.append(dict(case=case, **d.loc[i], n_rotations=len(d), frac_reach=d.reaches_threshold.mean(),
                            frac_kept=d.kept.mean()))
    cols = ["base", "case", "rotation", *SCOLS, "best_score", "best_base", "reaches_threshold", "x", "rep_base", "kept",
            "n_rotations", "frac_reach", "frac_kept"]
    return pd.DataFrame(out)[cols].round(4)


def synthetic_prediction() -> pd.DataFrame:
    rows = []
    for kind, top in SHAPES:
        for li, (level, (sd, corr)) in enumerate(NOISE.items()):
            r = pd.DataFrame([both_stages(grid(mound(XX, YY, kind, top, rot)
                                               + noise(np.random.default_rng([MOUND_SEED, li, k]), XX.shape, sd, corr)),
                                          X0, Y0, MOUND_LONG / 2, HIT_BASES) for k, rot in enumerate(MOUND_ROTS)])
            read = pd.Series([reading(False, h, k) for h, k in zip(r.hit, r.kept)]).value_counts()
            rows.append(dict(shape=kind, top=top, noise=level, noise_sd_m=sd, noise_corr_m=corr, n=len(r))
                        | r[SCOLS].median().round(4).to_dict()
                        | dict(x_min=r.x.min(), x_median=r.x.median(), x_max=r.x.max())
                        | {k.replace(" ", "_"): int(read.get(k, 0)) for k in ("hit and kept", "hit and removed", "no hit")})
    return pd.DataFrame(rows).round(4)


# ---------------------------------------------------------------- the real target (phase 2)

def join(paths) -> det.Grid:
    """The files on their common grid, block-averaged to CELL by the detector's own block mean.

    Read in strips of STRIP rows. Where files overlap, the first with data is taken. The block
    origin is the upper-left corner of the union. The grid keeps the files' own CRS."""
    dss = [rasterio.open(p) for p in paths]
    try:
        d0 = dss[0]
        res = abs(d0.res[0])
        f = int(round(CELL / res))
        left, top = min(d.bounds.left for d in dss), max(d.bounds.top for d in dss)
        off = [((d.bounds.left - left) / res, (top - d.bounds.top) / res) for d in dss]
        assert all(d.crs == d0.crs and np.allclose(d.res, d0.res) for d in dss), "the files differ in CRS or cell"
        assert np.allclose(off, np.round(off), atol=1e-6) and f * res == CELL, "the files do not share one grid"
        w = int(round((max(d.bounds.right for d in dss) - left) / res)) // f * f
        h = int(round((top - min(d.bounds.bottom for d in dss)) / res)) // f * f
        tr = from_origin(left, top, res, res)
        out = np.full((h // f, w // f), np.nan)
        for r0 in range(0, h, STRIP - STRIP % f):
            n = min(STRIP - STRIP % f, h - r0)
            z = np.full((n, w), np.nan)
            for d, (dc, dr) in zip(dss, off):
                a = d.read(1, window=Window(-int(round(dc)), r0 - int(round(dr)), w, n), boundless=True, masked=True)
                z = np.where(np.isfinite(z), z, np.ma.filled(a.astype(np.float64), np.nan))
            out[r0 // f:(r0 + n) // f] = det.coarsen(det.Grid(z, tr, d0.crs, res), CELL).z
        return det.Grid(out, tr @ rasterio.Affine.scale(f, f), d0.crs, CELL)
    finally:
        for d in dss:
            d.close()


def xy(g: det.Grid):
    """Cell-center coordinates: x as a row, y as a column."""
    h, w = g.z.shape
    return g.transform.c + (np.arange(w) + 0.5) * g.transform.a, (g.transform.f + (np.arange(h) + 0.5) * g.transform.e)[:, None]


def center(g: det.Grid) -> dict | None:
    """The fixed center rule; None if no cell with data lies within SEARCH_R of the site."""
    px, py = Transformer.from_crs("EPSG:4326", g.crs, always_xy=True).transform(SITE_LON, SITE_LAT)
    x, y = xy(g)
    z = np.where(np.hypot(x - px, y - py) <= SEARCH_R, g.z, np.nan)
    if not np.isfinite(z).any():
        return None
    r, c = np.unravel_index(np.nanargmax(z), z.shape)
    disc = np.hypot(x - x[c], y - y[r, 0]) <= CENTROID_R
    rr, cc = np.nonzero(disc & (g.z > (g.z[r, c] + np.nanmedian(g.z[disc])) / 2))
    return dict(site_x=px, site_y=py, summit_x=x[c], summit_y=y[r, 0], summit_z=g.z[r, c],
                cx=x[cc].mean(), cy=y[rr, 0].mean())


def void(g: det.Grid, cx: float, cy: float) -> str:
    """The fixed void rule: the reason, or "" if the check stands."""
    x, y = xy(g)
    h, w = g.z.shape
    reach = min(cx - g.transform.c, g.transform.c + w * g.res - cx, g.transform.f - cy, cy - (g.transform.f - h * g.res))
    if reach < VOID_R or not np.isfinite(g.z[np.hypot(x - cx, y - cy) <= VOID_R]).all():
        return "a cell without data within the void radius"
    # The score at the center, at each base: a crop of one frozen halo, cut on the coarsest
    # cell so that its coarse cells are those of the whole grid.
    k = int(max(det.RES.values()) / g.res)
    col, row = (int(v) for v in ~g.transform * (cx, cy))
    r0, c0 = (row - det.HALO) // k * k, (col - det.HALO) // k * k
    crop = det.Grid(g.z[r0:row + det.HALO, c0:col + det.HALO], g.transform @ rasterio.Affine.translation(c0, r0), CRS, g.res)
    for b in det.BASES:
        cg = det.coarsen(crop, det.RES[b])
        cc, rr = (int(v) for v in ~cg.transform * (cx, cy))
        if not np.isfinite(det.score_map(cg, b)[0][rr, cc]):
            return f"the score at the center is undefined at base {b:g}"
    return ""


def density(g: det.Grid, fl: list[dict], cx: float, cy: float) -> dict:
    """The fixed flag-density test. The count leaves out EXCLUDE_R about the center, so the area
    does too; the whole valid area is given beside it."""
    x, y = xy(g)
    valid = np.isfinite(g.z)
    km2 = float((valid & (np.hypot(x - cx, y - cy) > EXCLUDE_R)).sum()) * g.res ** 2 / 1e6
    n = sum(f["score"] >= T and np.hypot(f["x"] - cx, f["y"] - cy) > EXCLUDE_R for f in fl)
    rate = control_rate()
    p = float(poisson.sf(n - 1, rate * km2))
    return dict(far_flags=n, far_valid_km2=km2, grid_valid_km2=float(valid.sum()) * g.res ** 2 / 1e6,
                control_flags_per_km2=rate, far_expected=rate * km2, density_p=p, dense=p < ALPHA)


def target(g: det.Grid) -> tuple[dict, list[dict]]:
    """Every rule on the joined grid: (named results with the reading, the flags)."""
    c = center(g)
    why = "no data within the search radius of the site" if c is None else void(g, c["cx"], c["cy"])
    out = dict(crs=g.crs.to_string(), rows=g.z.shape[0], cols=g.z.shape[1], threshold=T, cut=C) | (c or {}) | dict(void=why)
    if why:
        return out | dict(reading=reading(True, False, False)), []
    with as_file(g) as ds:
        fl = flags_of(ds)
        out |= route(ds, fl, c["cx"], c["cy"], MOUND_LONG / 2, HIT_BASES)
    out |= density(g, fl, c["cx"], c["cy"])
    rank = 1 + sum(f["base"] == out["hit_base"] and f["score"] > out["score"] for f in fl) if out["hit"] else None
    return out | dict(rank_at_base=rank, reading=reading(False, out["hit"], out["kept"], out["dense"], rank)), fl


def download() -> list[Path]:
    """The two files the rule names, into their own directory; address, size and checksum tabled."""
    d = data_dir("p4/real_mound")
    extra = [p.name for p in data_glob(d, "*") if p.name not in FILES]
    if extra:
        raise SystemExit(f"{d} must hold nothing but the two files; found {extra[:5]}")
    rows = []
    for name in FILES:
        url = STORE.format(project=name.split("_", 4)[4].removesuffix(".tif"), file=name)
        n, status = fetch(url, d / name, -1)  # no listed size: the served length decides
        if not status.startswith(("ok", "cached")):
            raise SystemExit(f"{name}: {status}")
        rows.append(dict(file=name, url=url, bytes=n, sha256=sha256(d / name)))
    pd.DataFrame(rows).to_csv(HERE / "t1_download.csv", index=False)
    return [d / name for name in FILES]


# ---------------------------------------------------------------- the mound on Alaskan ground

def relief(g: det.Grid, cx: float, cy: float) -> np.ndarray:
    """Elevation above the plane fitted to the ring, in an odd square with the center's cell in
    the middle. A raised cosine takes it to zero across the ring, so no step is planted."""
    col, row = (int(v) for v in ~g.transform * (cx, cy))
    m = int(RING_OUT / g.res)
    z = g.z[row - m:row + m + 1, col - m:col + m + 1]
    yy, xx = np.mgrid[-m:m + 1, -m:m + 1] * g.res
    r = np.hypot(xx, yy)
    ring = (r >= RING_IN) & (r <= RING_OUT)
    a, b, c = np.linalg.lstsq(np.column_stack([np.ones(ring.sum()), xx[ring], yy[ring]]), z[ring], rcond=None)[0]
    taper = 0.5 - 0.5 * np.cos(np.pi * np.clip((RING_OUT - r) / (RING_OUT - RING_IN), 0, 1))
    return (z - (a + b * xx + c * yy)) * taper


@contextmanager
def planting(shape: np.ndarray):
    """While open, the frozen routine plants `shape`, turned by its own rotation draw, where it
    would plant a pyramid. No frozen file is edited: the name score_plant looks up is swapped
    in this process only. The turn is clockwise as displayed, as faceted_template's (checked)."""
    keep = pd2.faceted_template
    pd2.faceted_template = lambda *, rotation, **_: rotate(shape, -rotation, reshape=False, order=1)
    try:
        yield
    finally:
        pd2.faceted_template = keep


def alaska_trials(shape: np.ndarray | None, base: float, seed: int, idx, height: float | None = None) -> pd.DataFrame:
    """Frozen trials (p2e_pd.trial) on the control tiles, one process. `shape` None plants the
    frozen pyramid of `height`. A real mound has no quarter-turn symmetry, and the frozen draw
    covers a quarter turn, so each trial also draws a number of quarter turns."""
    rows = []
    for i in idx:
        if shape is None:
            rows.append(pd2.trial(("V0", i, base, height, seed)))
            continue
        q = int(np.random.default_rng([seed, i, 1]).integers(4))
        with planting(np.rot90(shape, q)):
            rows.append(pd2.trial(("V0", i, base, float(shape.max()), seed)) | dict(quarter=q))
    df = pd.DataFrame(rows)
    # Stratum, veto and weight of each plant: the lines of p2e_pd.main, which is not a function.
    # The strata grid comes from the frozen P3 cache of p2a_strata.strata_grid (its hash is in
    # the P3 freeze): building it afresh peaks near 6 GB.
    transform, s = strata()
    _, _, name = stratum_namer(s)
    inv = ~transform
    df["stratum"] = [name(int(np.floor(gy)), int(np.floor(gx))) for gx, gy in (inv * (x, y) for x, y in zip(df.x, df.y))]
    tree = shapely.STRtree(np.concatenate(list(veto_shapes().values())))
    df["vetoed"] = [tree.query(shapely.Point(x, y), predicate="intersects").size > 0 for x, y in zip(df.x, df.y)]
    w = pc.weights(pd.read_csv(P2 / "p2e_tile_land.csv", dtype={"tile_id": str}), pc._searched())
    df["w"] = df.stratum.map(w).fillna(0.0)
    return df


def alaska_pd(df: pd.DataFrame) -> pd.DataFrame:
    """The frozen summary at the frozen threshold: stage one alone, and through the frozen cut
    with the blind fraction (the control-side value of p2e_vintage_cut.csv, for the pyramid)."""
    v = pd.read_csv(P2 / "p2e_vetoes.csv").set_index("stratum").loc["_total"]
    blind = float(1 - v.searched_unvetoed_km2 / v.searched_km2)
    return pd.concat([pd2.summary(df, T, -np.inf, 0.0).assign(route="P2b alone"),
                      pd2.summary(df, T, C, blind).assign(route="one-method, frozen cut")], ignore_index=True)


# ---------------------------------------------------------------- self-check

CHECKS: list[dict] = []


def check(name: str, value, ok: bool = True, must: bool = True) -> None:
    CHECKS.append(dict(check=name, value=value, ok=bool(ok)))
    assert ok or not must, (name, value)


def _site():
    """A synthetic site as two overlapping in-memory 1 m tiles: a "mix" mound on gently rolling
    ground, a second smaller mound, a road-like ridge, and a nodata hole in a far corner.
    Returns (tiles, the block mean of the whole at CELL, the mound's true center)."""
    px, py = (int(v) for v in Transformer.from_crs("EPSG:4326", "EPSG:26915", always_xy=True).transform(SITE_LON, SITE_LAT))
    left, top, w, h, seam, nodata = px - 1700, py + 1400, 3600, 2800, 1850, -999999.0
    cx, cy = px + 120.5, py - 90.5  # off the site coordinate, and cut by the seam
    # float32 throughout: at 1 m the site has ten million cells, and memory must stay low
    xx = np.broadcast_to((left + np.arange(w) + 0.5 - cx).astype("float32"), (h, w))
    yy = np.broadcast_to((top - np.arange(h) - 0.5 - cy).astype("float32")[:, None], (h, w))
    z = 20 + 0.004 * xx + 1.5 * np.sin(xx / 400) * np.cos(yy / 550)
    z += 0.05 * np.random.default_rng(MOUND_SEED).standard_normal((h, w), dtype="float32")
    near = (slice(int(top - cy) - 300, int(top - cy) + 300), slice(int(cx - left) - 300, int(cx - left) + 300))
    z[near] += mound(xx[near], yy[near], "mix", 0.3, 20.0)  # zero farther out; the whole site would cost memory
    z += 7 * np.clip(1 - np.hypot(xx + 420, yy - 300) / 45, 0, None)
    z += 1.5 * np.exp(-((xx * np.float32(np.cos(0.5)) + yy * np.float32(np.sin(0.5)) - 500) ** 2) / (2 * 6.0 ** 2))
    z = z.astype("float32")
    z[:40, :40] = np.nan
    tiles = []
    for c0, c1 in ((0, seam + 6), (seam - 6, w)):
        mem = MemoryFile()
        with mem.open(driver="GTiff", width=c1 - c0, height=h, count=1, dtype="float32", crs="EPSG:26915",
                      transform=from_origin(left + c0, top, 1.0, 1.0), nodata=nodata) as ds:
            ds.write(np.where(np.isfinite(z[:, c0:c1]), z[:, c0:c1], nodata), 1)
        tiles.append(mem)
    f = int(CELL)
    return tiles, z.astype(np.float64).reshape(h // f, f, w // f, f).mean(axis=(1, 3)), (cx, cy)


def selfcheck() -> None:
    check("frozen threshold (p2e_cut.csv, t)", T)
    check("frozen cut (p2e_vintage.frozen_cut, c)", C)
    check("control flags per km2 at the threshold", control_rate())
    check("ring starts beyond the published plan's corner, m to spare", RING_IN - np.hypot(MOUND_LONG, MOUND_SHORT) / 2,
          RING_IN - np.hypot(MOUND_LONG, MOUND_SHORT) / 2 > 0)
    for void_, hit, kept, dense, want in ((True, True, True, True, "void"), (False, False, False, True, "no hit"),
                                          (False, True, True, False, "hit and kept"), (False, True, False, False, "hit and removed")):
        assert reading(void_, hit, kept, dense, 1) == want
    assert reading(False, True, True, True, 3).startswith("hit and kept, in a file where") and "rank 3" in reading(False, True, True, True, 3)

    # The repeated lines against the frozen route, on rolling ground with noise.
    base, height = SELF_CELL
    ground = 0.02 * XX + 15.0 * np.sin(XX / 700.0) * np.cos(YY / 900.0) + np.random.default_rng(MOUND_SEED).normal(0, 0.3, XX.shape)
    for rot in (0.0, 20.0):
        frozen = pd2.score_plant(grid(ground), N, base, height, rot)
        mine = both_stages(grid(plant(ground, pyramid(base, height, rot))), frozen["x"], frozen["y"], base / 2)
        same = np.allclose([mine["score"], mine["x"], mine["rep_base"]], [frozen[f"{k}_{T:.2f}"] for k in ("score", "x", "base")],
                           equal_nan=True)
        check(f"route equals frozen score_plant, rotation {rot:g}: score, statistic", f"{mine['score']:.4f}, {mine['x']:.4f}", same)
    with planting(pyramid(base, height, 0.0)):
        swapped = pd2.score_plant(grid(ground), N, base, height, 0.0)
    plain = pd2.score_plant(grid(ground), N, base, height, 0.0)
    check("planting leaves score_plant unchanged for its own shape", True, all(np.isclose(swapped[k], plain[k], equal_nan=True) for k in plain))
    square = plant(np.zeros((N, N)), pyramid(base, height, 0.0))
    turned = plant(np.zeros((N, N)), pyramid(base, height, 20.0))
    with planting(square):
        mine = pd2.faceted_template(rotation=20.0)
    check("planting turns as faceted_template does: correlation at 20 deg", round(float(np.corrcoef(mine.ravel(), turned.ravel())[0, 1]), 5),
          np.corrcoef(mine.ravel(), turned.ravel())[0, 1] > 0.999)

    # What the scan assumes.
    s = lambda b, h, r: both_stages(grid(plant(np.zeros((N, N)), pyramid(b, h, r))), X0, Y0, b / 2)  # noqa: E731
    d = max(abs(s(280.0, SCAN_HEIGHT, 10.0)[k] - s(280.0, 2 * SCAN_HEIGHT, 10.0)[k]) for k in SCOLS)
    check("score change when the planted height doubles, clean ground", d, d < 1e-6)
    d = max(abs(s(280.0, SCAN_HEIGHT, 10.0)[k] - s(280.0, SCAN_HEIGHT, 80.0)[k]) for k in SCOLS)
    check("score change between rotations 10 and 80 deg (mirror about 45)", d, d < 1e-2)

    # The real-target functions on the synthetic site.
    tiles, truth, (tx, ty) = _site()
    g = join([m.name for m in tiles])
    for m in tiles:
        m.close()
    d = float(np.nanmax(np.abs(g.z - truth)))
    check("join: largest difference from the block mean of the whole, m", d, d < 1e-3 and (np.isnan(g.z) == np.isnan(truth)).all())
    check("join: cells without data (the hole)", int(np.isnan(g.z).sum()), np.isnan(g.z).sum() == (40 // int(CELL)) ** 2)
    res, fl = target(g)
    d = float(np.hypot(res["cx"] - tx, res["cy"] - ty))
    check("center rule: distance from the true center, m", round(d, 2), d <= CELL * np.sqrt(2))
    check("void rule on the whole site", res["void"], res["void"] == "")
    holed = det.Grid(g.z.copy(), g.transform, g.crs, g.res)
    col, row = (int(v) for v in ~g.transform * (tx + 900.0, ty))
    holed.z[row, col] = np.nan
    check("void rule with one empty cell inside the radius", void(holed, tx, ty), void(holed, tx, ty) != "")
    shifted = det.Grid(g.z[:, :col], g.transform, g.crs, g.res)
    check("void rule with the grid ending inside the radius", void(shifted, tx, ty), void(shifted, tx, ty) != "")
    for k in SCOLS:
        check(f"site: best score at base {k[1:]} inside the hit radius", round(res[k], 4))
    check("site: stage-two statistic of the hit's anomaly", round(float(res["x"]), 4), must=False)
    check("site: best flag of any base leads to the same anomaly", res["same_any_base"], must=False)
    check("site: flags at or above the threshold beyond the exclusion radius", res["far_flags"])
    check("site: expected there at the control rate", res["far_expected"], 0 < res["far_expected"] < res["grid_valid_km2"] * control_rate())
    check("site: reading", res["reading"], res["reading"].split(",")[0] in ("hit and kept", "hit and removed", "no hit"))
    with as_file(g) as ds:
        small = flags_of(ds, 256)
    key = lambda fs: sorted((f["base"], round(f["x"], 1), round(f["y"], 1), round(f["score"], 6)) for f in fs)  # noqa: E731
    check("blocked flags equal whole-grid flags, scores down to 0 (flags)", len(fl), key(small) == key(fl))
    rel = relief(g, res["cx"], res["cy"])
    check("relief: height recovered, m (true height in the constants)", round(float(rel.max()), 2), abs(rel.max() - MOUND_HEIGHT) < 1.0)
    check("relief: largest value on the border, m", float(np.abs(rel[[0, -1]]).max()), np.abs(rel[[0, -1]]).max() < 1e-9)


def alaska_selfcheck() -> None:
    """The frozen trials of SELF_CELL, repeated, then the same pyramid through the relief route."""
    if not (data_root() / "p2b/ifsar_controls").is_dir():
        check("Alaskan control tiles reachable", False, must=False)
        return
    base, height = SELF_CELL
    old = pd.read_csv(P2 / "p2e_pd_trials.csv", dtype={"tile_id": str})
    cell = old[(old.base == base) & (old.height == height)]
    old = cell.head(N_SELF).reset_index(drop=True)
    sc, xc = f"score_{T:.2f}", f"x_{T:.2f}"
    kept = lambda d: (d[sc] >= T) & (d[xc] > C)  # noqa: E731
    t0 = time.time()
    new = alaska_trials(None, base, pd2.FINAL_SEED, old.trial, height)
    same = (all(np.allclose(new[k], old[k], equal_nan=True) for k in ("rotation", "x", "y", sc, xc, "w"))
            and (new.tile_id == old.tile_id).all() and (new.stratum == old.stratum).all() and (new.vetoed == old.vetoed).all())
    check(f"Alaska: frozen trials of base {base:g}, height {height:g} reproduced exactly (trials)", len(new), same)
    shape = relief(grid(plant(np.zeros((N, N)), pyramid(base, height, 0.0))), X0, Y0)
    rel = alaska_trials(shape, base, pd2.FINAL_SEED, old.trial)
    check("Alaska: seconds per trial", round((time.time() - t0) / (2 * len(old)), 2))
    check("Alaska: same windows as the frozen trials", True, np.allclose(rel[["x", "y"]], old[["x", "y"]]))
    check("Alaska: frozen pyramid at or above the threshold, on these windows", int((old[sc] >= T).sum()))
    check("Alaska: relief-route pyramid at or above the threshold", int((rel[sc] >= T).sum()))
    check("Alaska: frozen pyramid kept through both stages", int(kept(old).sum()))
    check("Alaska: relief-route pyramid kept through both stages", int(kept(rel).sum()))
    check("Alaska: trials with the same outcome through both stages", int((kept(rel) == kept(old)).sum()))
    both = (old[sc] >= T) & (rel[sc] >= T)
    check("Alaska: largest score difference where both are flagged", float((rel[sc] - old[sc])[both].abs().max()))
    for tag, d in (("frozen pyramid", new), ("relief-route pyramid", rel)):  # the summary phase 2 will use, exercised
        p = alaska_pd(d)
        check(f"Alaska: {tag}, frozen summary through the frozen cut, these windows only", float(p[p.route != "P2b alone"].p_detect.iloc[0]))
    rate = float(kept(cell).mean())
    p = float(binomtest(int(kept(rel).sum()), len(rel), rate).pvalue)
    check("Alaska: frozen cell's rate through both stages (all its trials, unweighted)", rate)
    check("Alaska: relief-route count against that rate, binomial p", p, p >= ALPHA, must=False)


# ---------------------------------------------------------------- paired comparisons

SC, XC = f"score_{T:.2f}", f"x_{T:.2f}"


def found(d: pd.DataFrame) -> pd.Series:
    """Stage one: the plant's anomaly is at or above the threshold."""
    return d[SC] >= T


def kept(d: pd.DataFrame) -> pd.Series:
    """Both stages: found, and its statistic above the frozen cut."""
    return found(d) & (d[XC] > C)


def pair(template, other) -> dict:
    """Discordant pairs of two outcome columns on the same trials, and the exact two-sided
    binomial test on them (each discordant pair equally likely to fall either way)."""
    a, b = np.asarray(template, bool), np.asarray(other, bool)
    t_only, m_only = int((a & ~b).sum()), int((~a & b).sum())
    p = float(binomtest(t_only, t_only + m_only, 0.5).pvalue) if t_only + m_only else 1.0
    return dict(template_only=t_only, mound_only=m_only, exact_p=p)


def searched_reading(template_hits: int, mound_hits: int, p: float) -> str:
    """The fixed readings of "the mound is detected less often than the template"."""
    if mound_hits >= template_hits:
        return "false"
    return "less often, significant" if p < PAIRED_ALPHA else "less often, does not decide"


def paired_selfcheck() -> None:
    got = pair([1, 1, 1, 0, 0, 1], [0, 0, 1, 0, 1, 0])
    check("pair on a made-up case: template only, mound only, p", f"{got['template_only']}, {got['mound_only']}, {got['exact_p']:.4f}",
          got == dict(template_only=3, mound_only=1, exact_p=float(binomtest(3, 4, 0.5).pvalue)) and pair([0, 1], [0, 1])["exact_p"] == 1.0)
    want = ((11, 11, 1.0, "false"), (11, 12, 0.01, "false"), (11, 0, 0.001, "less often, significant"),
            (11, 8, 0.5, "less often, does not decide"), (11, 10, PAIRED_ALPHA, "less often, does not decide"))
    check("reading rule on made-up counts", True, all(searched_reading(t, m, p) == r for t, m, p, r in want))


def same_windows(a: pd.DataFrame, b: pd.DataFrame, cols) -> bool:
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    num = [c for c in cols if a[c].dtype.kind in "fiu"]
    return (len(a) == len(b) and np.allclose(a[num].astype(float), b[num].astype(float), equal_nan=True)
            and all((a[c].fillna("") == b[c].fillna("")).all() for c in cols if c not in num))


def paired() -> None:
    """a: the template's own shape on the control windows already used for the mound."""
    paired_selfcheck()
    mound_ = pd.read_csv(HERE / "t1_alaska_trials.csv", dtype={"tile_id": str})
    was = pd.read_csv(HERE / "t1_alaska_pd.csv")
    now = alaska_pd(mound_)
    cols = ["n", "p_detect", "ci_low", "ci_high"]
    check("mound: the committed summary follows from the committed trials", True, np.allclose(now[cols], was[cols]))
    base, height = float(was.base[0]), float(was.height[0])
    bodies = {"mound": mound_,
              "template, nearest planted base and relief": alaska_trials(None, SELF_CELL[0], MOUND_SEED, mound_.trial, SELF_CELL[1]),
              "template, the mound's length and relief": alaska_trials(None, base, MOUND_SEED, mound_.trial, height)}
    rows = []
    for name, df in bodies.items():
        check(f"{name}: tile, window, rotation, stratum, veto and weight as the mound's trials (rows)", len(df),
              same_windows(df, mound_, ["tile_id", "x", "y", "rotation", "stratum", "vetoed", "w"]))
        count = ~df.vetoed & (df.w > 0)  # the trials the frozen summary counts; the same for every body
        summ = alaska_pd(df)
        for stage, hit, route_ in (("first", found, "P2b alone"), ("both", kept, "one-method, frozen cut")):
            p = summ[summ.route == route_].iloc[0]
            assert p.n == count.sum()
            rows.append(dict(body=name, base=p.base, height=p.height, stage=stage, n=int(count.sum()), hits=int(hit(df)[count].sum()),
                             p_detect=p.p_detect, ci_low=p.ci_low, ci_high=p.ci_high)
                        | (pair(hit(df)[count], hit(mound_)[count]) if name != "mound" else {}))
    rows = pd.DataFrame(rows).astype({"template_only": "Int64", "mound_only": "Int64"})  # counts; empty for the mound
    rows.to_csv(HERE / "t1_alaska_paired.csv", index=False)
    # A synthetic body: the template's shape through the relief route must pair with no discordance.
    few = mound_.trial[:N_SELF]
    shape = relief(grid(plant(np.zeros((N, N)), pyramid(*SELF_CELL, 0.0))), X0, Y0)
    got = pair(kept(bodies["template, nearest planted base and relief"].head(N_SELF)), kept(alaska_trials(shape, SELF_CELL[0], MOUND_SEED, few)))
    check(f"template through the relief route against itself, {N_SELF} windows: discordant pairs", got["template_only"] + got["mound_only"],
          must=False)
    finish("t1_alaska_paired_checks.csv")
    print(rows.to_string(index=False))


def claim_guard() -> None:
    """What the frozen check_freeze protects, for the planting route (see the docstring)."""
    sha, swe = (P2 / "p2e_freeze.txt").read_text().split()[:2]
    try:
        cr.check_freeze()
        refusal = ""
    except SystemExit as e:
        refusal = str(e)
    check("guard: the frozen check_freeze, as called by the frozen planting run", (refusal[:60] + "...") if refusal else "passes", must=False)
    git = lambda *a: subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, text=True)  # noqa: E731
    check("guard: the P2e freeze commit is an ancestor of HEAD", sha[:7], git("merge-base", "--is-ancestor", sha, "HEAD").returncode == 0)
    d = git("diff", "--name-only", "--diff-filter=a", sha, "--", ":(top,glob)akp/**/*.py", ":(top)environment-akp.yml",
            ":(top)env-akp.lock.txt", ":(top)scripts/repro_check.py")
    check("guard: code and environment files of the P2e freeze that differ now", d.stdout.split(), d.returncode == 0 and not d.stdout.strip())
    check("guard: the P3 freeze passes (akp/stage_p2, paths, environment, installed SWE)", p3_freeze.check()[:7],
          p3_freeze.FREEZE.read_text().split()[1] == swe)
    check("guard: tile checksums were verified for the P2e freeze commit", True, (cr.out_dir() / f"checksums_ok_{sha}").exists())


def searched_trials(shape: np.ndarray | None, idx) -> pd.DataFrame:
    """The frozen searched-area plants (p2e_claim_run.trial_at, its seed, claim_zone=True as its
    claim_trial) and the counting lines of its inject(), which is not a function. `shape` None
    plants the frozen pyramid of SELF_CELL; else the shape, with a drawn number of quarter turns."""
    det.configure(**VARIANTS["V0"])
    state = cr._inject_state()
    rows = []
    for i in idx:
        if shape is None:
            rows.append(cr.trial_at(i, *SELF_CELL, *state, claim_zone=True))
            continue
        q = int(np.random.default_rng([cr.CLAIM_SEED, i, 1]).integers(4))
        with planting(np.rot90(shape, q)):
            rows.append(cr.trial_at(i, MOUND_LONG, float(shape.max()), *state, claim_zone=True) | dict(quarter=q))
    df = pd.DataFrame(rows)
    tree = shapely.STRtree(np.concatenate(list(veto_shapes().values())))
    vet = lambda x, y: np.isfinite(x) and tree.query(shapely.Point(x, y), predicate="intersects").size > 0  # noqa: E731
    own, gt = cr._owners(), cr._grid()
    rx, ry = df[f"rx_{T:.2f}"], df[f"ry_{T:.2f}"]
    df["rep_vetoed"] = [vet(x, y) for x, y in zip(rx, ry)]
    df["rep_owned"] = [np.isfinite(x) and own.get(tuple(int(np.floor(v)) for v in (~gt * (x, y))[::-1])) == tid
                       for x, y, tid in zip(rx, ry, df.tile_id)]
    df["hit"] = kept(df) & df.rep_owned & ~df.rep_vetoed
    return df


def searched() -> None:
    """b: the mound's relief on the searched-area windows of the frozen template cell."""
    paired_selfcheck()
    claim_guard()
    # The patch of phase 2, rebuilt from the same files and checked against the committed tables.
    files = pd.read_csv(HERE / "t1_download.csv").set_index("file").sha256
    paths = [data_root() / "p4/real_mound" / f for f in FILES]
    check("lidar files as downloaded in phase 2 (sha256)", len(paths), all(sha256(p) == files[p.name] for p in paths))
    g = join(paths)
    was = pd.read_csv(HERE / "t1_real_mound.csv").set_index("name").value
    c = center(g)
    check("center as committed, m apart", float(np.hypot(c["cx"] - float(was.cx), c["cy"] - float(was.cy))),
          np.hypot(c["cx"] - float(was.cx), c["cy"] - float(was.cy)) < 1e-6)
    shape = relief(g, c["cx"], c["cy"])
    height = float(pd.read_csv(HERE / "t1_alaska_pd.csv").height[0])
    check("relief patch: height as committed, m apart", abs(float(shape.max()) - height), abs(shape.max() - height) < 1e-9)
    del g
    # The frozen template's windows, and its committed outcomes.
    old = pd.read_csv(P2 / "p2e_claim_trials.csv", dtype={"tile_id": str})
    old = old[(old.base == SELF_CELL[0]) & (old.height == SELF_CELL[1])].reset_index(drop=True)
    tab = pd.read_csv(P2 / "p2e_claim_pd.csv")
    tab = tab[(tab.base == SELF_CELL[0]) & (tab.height == SELF_CELL[1]) & (tab.scope == "all")].iloc[0]
    check("template: committed hits of committed trials, as p2e_claim_pd.csv (scope all)", f"{int(old.hit.sum())} of {len(old)}",
          old.hit.sum() == tab.hits and len(old) == tab.n)
    where = ["cell_row", "cell_col", "px", "py", "tile_id", "miss", "rotation", "x", "y"]
    again = searched_trials(None, old.trial)
    check("template: frozen trials rerun on these windows equal the committed rows (rows)", len(again),
          same_windows(again, old, [*where, SC, XC, "rep_vetoed", "rep_owned", "hit"]))
    new = searched_trials(shape, old.trial)
    check("mound: cell, point, tile, miss, rotation and plant position as the template's trials (rows)", len(new),
          same_windows(new, old, where))
    out = old[["trial", "tile_id", "cell_row", "cell_col", "px", "py", "miss", "rotation"]].assign(
        quarter=new.quarter.to_numpy(), template_found=found(old), template_both=kept(old), template_hit=old.hit,
        mound_found=found(new).to_numpy(), mound_both=kept(new).to_numpy(), mound_hit=new.hit.to_numpy())
    out.to_csv(HERE / "t1_searched_paired.csv", index=False)
    pr = pair(out.template_hit, out.mound_hit)
    check("trials counted (the route counts every trial)", len(out))
    check("template: found at stage one", int(out.template_found.sum()))
    check("mound: found at stage one", int(out.mound_found.sum()))
    check("template: hits by the route's rule", int(out.template_hit.sum()))
    check("mound: hits by the route's rule", int(out.mound_hit.sum()))
    check("discordant pairs: template only", pr["template_only"])
    check("discordant pairs: mound only", pr["mound_only"])
    check("exact two-sided binomial test on the discordant pairs, p", pr["exact_p"])
    check("reading", searched_reading(int(out.template_hit.sum()), int(out.mound_hit.sum()), pr["exact_p"]))
    finish("t1_searched_checks.csv")


def finish(name: str) -> None:
    check("peak memory of this run, GB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2))  # bytes on macOS
    pd.DataFrame(CHECKS).to_csv(HERE / name, index=False)
    print(pd.DataFrame(CHECKS).to_string(index=False))


# ---------------------------------------------------------------- runs

def real() -> None:
    names = [Path(__file__).name, *PHASE1]
    st = subprocess.run(["git", "-C", str(HERE), "status", "--porcelain", "--", *names], capture_output=True, text=True)
    if st.returncode or st.stdout.strip() or not all((HERE / n).exists() for n in names):
        raise SystemExit("The module and its phase-1 tables must be committed, unchanged, before the files are read")
    g = join(download())
    res, fl = target(g)
    pd.Series(res, name="value").rename_axis("name").to_csv(HERE / "t1_real_mound.csv")
    r2 = (MOUND_LONG / 2 + max(det.RES.values())) ** 2
    keep = [f for f in fl if f["score"] >= T or (f["x"] - res["cx"]) ** 2 + (f["y"] - res["cy"]) ** 2 <= r2]
    pd.DataFrame(keep, columns=["base", "x", "y", "score", "rotation"]).to_csv(HERE / "t1_real_flags.csv", index=False)
    print(pd.Series(res).to_string())
    if res["void"]:
        return
    df = alaska_trials(relief(g, res["cx"], res["cy"]), MOUND_LONG, MOUND_SEED, range(N_MOUND_TRIALS))
    df.to_csv(HERE / "t1_alaska_trials.csv", index=False)
    alaska_pd(df).to_csv(HERE / "t1_alaska_pd.csv", index=False)
    print(alaska_pd(df).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="phase 2: download the two files and run the check")
    ap.add_argument("--paired", action="store_true", help="a: the template on the mound's control windows")
    ap.add_argument("--searched", action="store_true", help="b: the mound on the template's searched-area windows")
    a = ap.parse_args()
    with rasterio.Env(GDAL_CACHEMAX=det.GDAL_CACHE_MB):  # as the frozen readers: GDAL's default cache is 5% of RAM
        if a.real or a.paired or a.searched:
            return real() if a.real else paired() if a.paired else searched()
        selfcheck()
        scale_response().to_csv(HERE / PHASE1[0], index=False)
        synthetic_prediction().to_csv(HERE / PHASE1[1], index=False)
        alaska_selfcheck()
    check("peak memory of this run, GB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2))  # bytes on macOS
    pd.DataFrame(CHECKS).to_csv(HERE / PHASE1[2], index=False)
    print(pd.DataFrame(CHECKS).to_string(index=False))


if __name__ == "__main__":
    main()
