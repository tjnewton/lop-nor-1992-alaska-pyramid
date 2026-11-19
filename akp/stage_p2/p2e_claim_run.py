"""P2e on the claim zone: owners, candidate search, injection, checks.

Every stage first calls `check_freeze`, which refuses unless:
- p2e_freeze.txt holds the freeze commit (an ancestor of HEAD) and the SWE commit, and the
  installed SWE is that commit;
- no FROZEN path differs from the freeze commit, in the index, the working tree or as an
  untracked file (the claim run's own outputs excepted);
- the claim tiles, veto sources and thematic maps match data_manifest.csv checksums (checked
  once per freeze commit; a stamp in the output directory records it).

Stages, in order (one heavy job at a time):
- `--owners`: each claim tile's valid area per 1 km P2a cell, read at OVERVIEW decimation. Each
  searched land cell goes to the most recently published tile with at least OWN_KM2 valid there,
  failing that to the tile with the most valid area, ties to the file name (`owner_table`).
  Also the claim unvetoed land per stratum and per project, and the cut recomputed with the
  owner table's vintage mix (p2e_claim_owner_mix.csv).
- `--search`: the detector on every owning tile, as on the controls (p2b_controls_run.run_all,
  claim_zone=True). Each tile's flags outside its owned cells are dropped, then one greedy merge
  runs across tiles at t_low (`owned_flags`, p2e_anomalies.merge). Vetoed anomalies and those
  without a valid window are not usable; a candidate is usable, at or above t, with x above the
  frozen cut (p2e_vintage.frozen_cut). Writes p2e_claim_anomalies.csv.
- `--inject`: N_TRIALS plants per target grid cell, each at a uniform point of a uniform searched
  land cell, set into its owner tile's window and scored as p2e_pd.score_plant. A plant with no
  owner or on nodata is a miss; a detection needs its representative flag in a cell its tile owns
  and outside every veto. Writes p2e_claim_trials.csv and p2e_claim_pd.csv.
- `--checks`: the pre-stated transfer checks and the candidate report (p2e_claim_checks.csv,
  p2e_claim_candidates.csv: positions at 0.1 degree only, with distances to the nearest Holmes
  pingo, veto and mapped mine or prospect).
Run: python -m akp.stage_p2.p2e_claim_run --owners | --search | --inject | --checks | --selfcheck
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask
from shapely.geometry import box, mapping, shape

from akp.paths import data_dir, data_root
from akp.stage_p2 import p2b_detector as det
from akp.stage_p2.p2b_controls_run import compact_cells

HERE = Path(__file__).parent
REPO = HERE.parents[1]
OWN_KM2 = 0.5  # At least half of a 1 km2 cell valid
OVERVIEW = 8  # decimation of the validity read: 40 m pixels for 5 m tiles, as the control run's VBLOCK
CLAIM_SEED = 20261117  # fresh: the claim-zone injection
ALPHA = 0.01  # Revisions: transfer-check level
WORKERS = 4  # injection windows, as p2e_pd
FROZEN = ["akp", "environment-akp.yml", "env-akp.lock.txt",
          ":(exclude)akp/stage_p2/p2e_claim_*.csv", ":(exclude)akp/stage_p2/p2e_freeze.txt",
          ":(exclude)**/__pycache__/**"]
CHECKED_DATA = ("p2a/maps/", "p2e/inventories/")  # plus the claim tiles


def out_dir() -> Path:
    return data_dir("p2e/claim_run")


def _git(*a) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True)


def check_freeze() -> str:
    """The freeze commit's SHA, or SystemExit if the run is not frozen."""
    f = HERE / "p2e_freeze.txt"
    if not f.exists():
        raise SystemExit("not frozen: the claim-zone run needs the freeze record")
    sha, swe_sha = f.read_text().split()[:2]
    if _git("merge-base", "--is-ancestor", sha, "HEAD").returncode:
        raise SystemExit(f"freeze commit {sha} is not an ancestor of HEAD")
    changed = (_git("diff", "--name-only", sha, "--", *FROZEN).stdout.split()
               + _git("diff", "--name-only", "--cached", sha, "--", *FROZEN).stdout.split()
               + _git("ls-files", "--others", "--exclude-standard", "--", *FROZEN).stdout.split())
    if changed:
        raise SystemExit(f"frozen paths changed since {sha}: {sorted(set(changed))}; log a dated deviation first")
    url = json.loads(importlib.metadata.distribution("seismic-workflow-engine").read_text("direct_url.json"))
    installed = url.get("vcs_info", {}).get("commit_id")
    if installed != swe_sha:
        raise SystemExit(f"installed SWE {installed} is not the frozen SWE commit {swe_sha}")
    stamp = out_dir() / f"checksums_ok_{sha}"
    if not stamp.exists():
        verify_checksums()
        stamp.write_text("ok\n")
    return sha


def verify_checksums() -> None:
    from akp.setup.manifest import sha256

    man = pd.read_csv(REPO / "akp/setup/data_manifest.csv")
    root = data_root()
    tiles = {str(Path(p).relative_to(root)) for p in claim_tiles().path}
    want = man[man.path.isin(tiles) | man.path.str.startswith(CHECKED_DATA)]
    assert tiles <= set(man.path), f"claim tiles missing from the manifest: {sorted(tiles - set(man.path))[:3]}"
    bad = [r.path for r in want.itertuples() if sha256(root / r.path) != r.sha256]
    if bad:
        raise SystemExit(f"{len(bad)} files differ from data_manifest.csv: {bad[:5]}")
    print(f"checksums OK: {len(want)} files", flush=True)


def claim_tiles() -> pd.DataFrame:
    f = pd.read_csv(HERE / "p2b_fetch_ifsar.csv")
    p = pd.read_csv(HERE / "p2a_tile_pool.csv")[["url", "tile_id", "min_lon", "max_lon", "min_lat", "max_lat"]]
    m = f.merge(p, on="url", validate="one_to_one")
    zones = [shape(g["geometry"]) for g in json.loads((HERE / "p2a_searched_area.geojson").read_text())["features"]]
    hit = [any(box(r.min_lon, r.min_lat, r.max_lon, r.max_lat).intersects(z) for z in zones) for r in m.itertuples()]
    out = m[hit].copy()
    out["tile_id"] = out.tile_id.astype(str)
    out["path"] = [str(data_dir("p2b/ifsar") / r.project / r.file) for r in out.itertuples()]
    return out


def valid_cells(path: Path, grid_transform: tuple, claim_zone: bool = True) -> list[list[float]]:
    """(row, col, valid km2) per 1 km P2a cell, from a read decimated by OVERVIEW."""
    inv = ~rasterio.Affine(*grid_transform)
    with rasterio.Env(GDAL_CACHEMAX=det.GDAL_CACHE_MB), det.open_tile(path, claim_zone=claim_zone) as ds:
        assert ds.crs.is_projected, path
        h, w = max(1, ds.height // OVERVIEW), max(1, ds.width // OVERVIEW)
        a = ds.read(1, out_shape=(h, w), masked=True)  # GDAL uses the tile's overviews where present
        ok = ~np.ma.getmaskarray(a) & np.isfinite(np.ma.filled(a.astype(np.float64), np.nan))
        t = ds.transform * rasterio.Affine.scale(ds.width / w, ds.height / h)
        r, c = np.nonzero(ok)
        xs, ys = rasterio.transform.xy(t, r, c)
        gx, gy = inv * Transformer.from_crs(ds.crs, "EPSG:3338", always_xy=True).transform(np.asarray(xs), np.asarray(ys))
        km2 = abs(t.a * t.e) / 1e6
    return compact_cells([np.column_stack([np.floor(gy), np.floor(gx), np.full(len(r), km2)])])


def owner_table(cells: pd.DataFrame, searched: set[tuple[int, int]]) -> pd.DataFrame:
    """One owner per searched land cell from (row, col, tile_id, km2, publication_date, file) rows."""
    d = cells[[(r, c) in searched for r, c in zip(cells.row, cells.col)]].copy()
    d["full"] = d.km2 >= OWN_KM2
    d["date_key"] = d.publication_date.where(d.full, "")  # the newest full tile wins
    d["km2_key"] = d.km2.where(~d.full, 0.0)  # else the most valid
    d = d.sort_values(["row", "col", "full", "date_key", "km2_key", "file"],
                      ascending=[True, True, False, False, False, True])
    return d.drop_duplicates(["row", "col"])[["row", "col", "tile_id", "km2", "full"]]


def searched_cells() -> set[tuple[int, int]]:
    from akp.stage_p2.p2a_strata import strata_grid, to_3338

    transform, s = strata_grid()
    zones = [to_3338(shape(f["geometry"])) for f in json.loads((HERE / "p2a_searched_area.geojson").read_text())["features"]]
    m = ~geometry_mask([mapping(z) for z in zones], s.shape, transform) & (s >= 0)
    return set(map(tuple, np.argwhere(m)))


def owned_flags(flags_by_tile: dict[str, list[dict]], own: dict[tuple[int, int], str], inv, s_min: float) -> list[dict]:
    """Flags at or above s_min that lie in a cell their own tile owns, tagged with the tile; then
    one greedy merge across tiles."""
    from akp.stage_p2.p2e_anomalies import merge

    keep = []
    for tid, fl in flags_by_tile.items():
        for f in fl:
            gx, gy = inv * (f["x"], f["y"])
            if f["score"] >= s_min and own.get((int(np.floor(gy)), int(np.floor(gx)))) == tid:
                keep.append(f | {"tile_id": tid})
    return merge(keep)


def _grid():
    from akp.stage_p2.p2a_strata import grid

    return grid()[0]


def _owners() -> dict[tuple[int, int], str]:
    own = pd.read_csv(out_dir() / "owners.csv", dtype={"tile_id": str})
    return {(int(r), int(c)): t for r, c, t in zip(own.row, own.col, own.tile_id)}


def owners() -> None:
    from akp.stage_p2.p2a_strata import strata_grid
    from akp.stage_p2.p2b_controls_run import stratum_namer
    from akp.stage_p2.p2e_vintage import vintage

    check_freeze()
    gt = tuple(_grid())[:6]
    tiles = claim_tiles()
    rows = []
    for t in tiles.itertuples():
        rows += [dict(row=int(r), col=int(c), tile_id=t.tile_id, km2=a, publication_date=t.publication_date,
                      file=t.file) for r, c, a in valid_cells(Path(t.path), gt)]
        print(f"{t.file}: {len(rows)} cell rows", flush=True)
    cells = searched_cells()
    own = owner_table(pd.DataFrame(rows), cells)
    own.to_csv(out_dir() / "owners.csv", index=False)
    np.save(out_dir() / "searched_cells.npy", np.array(sorted(cells), dtype=np.int32))
    # claim unvetoed land: per stratum (transfer check 2) and per project (the owner-table vintage mix)
    transform, s = strata_grid()
    _, _, name = stratum_namer(s)
    keep = 1.0 - np.load(data_dir("p2e") / "veto_fraction.npy")
    own["stratum"] = [name(r, c) for r, c in zip(own.row, own.col)]
    own["unvetoed_km2"] = [min(a, 1.0) * float(keep[r, c]) for r, c, a in zip(own.row, own.col, own.km2)]  # valid land, as the controls
    own = own.merge(tiles[["tile_id", "project"]], on="tile_id")
    own.groupby("stratum").unvetoed_km2.sum().to_csv(out_dir() / "claim_land.csv")
    tab, cut = vintage(own.groupby("project").unvetoed_km2.sum())
    from akp.stage_p2.p2e_vintage import frozen_cut

    c0 = float(frozen_cut().c)
    mix = tab.reset_index().assign(kind="group")
    cut = cut.assign(kind="cut", frozen_c=c0, deviation=cut.c > c0 + 1e-12)
    pd.concat([mix, cut], ignore_index=True).to_csv(HERE / "p2e_claim_owner_mix.csv", index=False)
    print(f"{len(own)} owned cells over {own.tile_id.nunique()} of {len(tiles)} tiles; "
          f"{(~own.full).sum()} owned without a full tile; owner-mix cut {cut.c.max():.3f} vs frozen {c0:.3f}")


def search() -> None:
    import shapely

    from akp.stage_p2.p2a_strata import strata_grid
    from akp.stage_p2.p2b_controls_run import run_all, stratum_namer
    from akp.stage_p2.p2e_vetoes import veto_shapes
    from akp.stage_p2.p2e_vintage import frozen_cut

    check_freeze()
    run = pd.read_csv(HERE / "p2e_cut.csv").iloc[0]
    c = float(frozen_cut().c)
    own = _owners()
    tiles = claim_tiles().set_index("tile_id")
    use = sorted(set(own.values()))
    tdir = out_dir() / "tiles"
    tdir.mkdir(exist_ok=True)
    gt = _grid()
    run_all([tiles.loc[t].to_dict() | {"tile_id": t} for t in use], tuple(gt)[:6], out_dir=tdir, claim_zone=True)
    fl = {t: json.loads((tdir / f"{t}.json").read_text())["flags"] for t in use}
    an = owned_flags(fl, own, ~gt, float(run.t_low))
    del fl
    tree = shapely.STRtree(np.concatenate(list(veto_shapes().values())))
    _, s = strata_grid()
    _, _, name = stratum_namer(s)
    df = assess(an, lambda tid: Path(tiles.loc[tid].path), True, tree, name, gt)
    df["above_t"] = df.score >= run.t
    df["candidate"] = ~df.vetoed & df.valid & df.above_t & (df.x > c)
    df.to_csv(HERE / "p2e_claim_anomalies.csv", index=False)
    print(f"{len(df)} anomalies at t_low {run.t_low}; {int(df.candidate.sum())} candidates at t {run.t}, c {c:.3f}")


def assess(an: list[dict], path_of, claim_zone: bool, tree, name, gt) -> pd.DataFrame:
    """Per anomaly, as p2e_run: veto at the flag, the model-selection window from its tile, x."""
    import shapely

    from akp.stage_p2 import p2e_model_select as ms
    from akp.stage_p2.p2e_anomalies import window

    rows = []
    for tid, g in pd.DataFrame(an, columns=["tile_id", "x", "y", "base", "score"]).groupby("tile_id"):
        with rasterio.Env(GDAL_CACHEMAX=det.GDAL_CACHE_MB), det.open_tile(path_of(tid), claim_zone=claim_zone) as ds:
            for f in g.to_dict("records"):
                vetoed = tree.query(shapely.Point(f["x"], f["y"]), predicate="intersects").size > 0
                w = None if vetoed else window(ds, f)
                gx, gy = ~gt * (f["x"], f["y"])
                rows.append(dict(tile_id=tid, stratum=name(int(np.floor(gy)), int(np.floor(gx))), base=f["base"],
                                 x_m=f["x"], y_m=f["y"], score=f["score"], vetoed=vetoed, valid=w is not None,
                                 x=ms.delta_bic(w, f["base"], det.RES[f["base"]]) / w.size if w is not None else np.nan))
    return pd.DataFrame(rows, columns=["tile_id", "stratum", "base", "x_m", "y_m", "score", "vetoed", "valid", "x"])


@lru_cache(maxsize=1)
def _inject_state():
    cells = np.load(out_dir() / "searched_cells.npy")
    tiles = claim_tiles().set_index("tile_id").path.to_dict()
    return cells, _owners(), tiles, _grid()


def claim_trial(args) -> dict:
    from akp.stage_p2.p2b_explore import VARIANTS

    i, base, height = args
    det.configure(**VARIANTS["V0"])
    return trial_at(i, base, height, *_inject_state(), claim_zone=True)


def trial_at(i, base, height, cells, own, paths, gt, claim_zone: bool, seed: int = CLAIM_SEED) -> dict:
    """One plant at a uniform point of a uniform cell, in its owner tile's window."""
    from akp.stage_p2.p2e_pd import read_window, score_plant

    rng = np.random.default_rng([seed, i])
    rotation = float(rng.uniform(0, 90))
    r, c = (int(v) for v in cells[rng.integers(len(cells))])
    px, py = gt * (c + rng.random(), r + rng.random())
    out = dict(trial=i, base=base, height=height, rotation=round(rotation, 2), cell_row=r, cell_col=c,
               px=round(px, 1), py=round(py, 1), tile_id=own.get((r, c)), miss="")
    if out["tile_id"] is None:
        return out | dict(miss="no owner")
    with rasterio.Env(GDAL_CACHEMAX=det.GDAL_CACHE_MB), det.open_tile(Path(paths[out["tile_id"]]), claim_zone=claim_zone) as ds:
        row, col = ds.index(*Transformer.from_crs("EPSG:3338", ds.crs, always_xy=True).transform(px, py))
        g, n = read_window(ds, col, row)
    if not np.isfinite(g.z[n // 2, n // 2]):
        return out | dict(miss="nodata")
    return out | score_plant(g, n, base, height, rotation)


def inject() -> None:
    import shapely
    from swe.detection import clopper_pearson

    from akp.stage_p2.p2b_injection import HEIGHTS, N_TRIALS, TARGET_BASES
    from akp.stage_p2.p2e_vetoes import veto_shapes
    from akp.stage_p2.p2e_vintage import frozen_cut

    check_freeze()
    run = pd.read_csv(HERE / "p2e_cut.csv").iloc[0]
    t, c = float(run.t), float(frozen_cut().c)
    jobs = [(i, b, h) for i, (b, h) in enumerate((b, h) for b in TARGET_BASES for h in HEIGHTS for _ in range(N_TRIALS))]
    with ProcessPoolExecutor(WORKERS) as ex:
        df = pd.DataFrame(list(ex.map(claim_trial, jobs, chunksize=4)))
    tree = shapely.STRtree(np.concatenate(list(veto_shapes().values())))
    vet = lambda x, y: np.isfinite(x) and tree.query(shapely.Point(x, y), predicate="intersects").size > 0
    own, gt = _owners(), _grid()
    rx, ry = df.get(f"rx_{t:.2f}", np.nan), df.get(f"ry_{t:.2f}", np.nan)
    df["center_vetoed"] = [vet(x, y) for x, y in zip(df.px, df.py)]
    df["rep_vetoed"] = [vet(x, y) for x, y in zip(rx, ry)]
    df["rep_owned"] = [np.isfinite(x) and own.get(tuple(int(np.floor(v)) for v in (~gt * (x, y))[::-1])) == tid
                       for x, y, tid in zip(rx, ry, df.tile_id)]
    df["hit"] = (df.get(f"score_{t:.2f}", np.nan) >= t) & (df.get(f"x_{t:.2f}", np.nan) > c) & df.rep_owned & ~df.rep_vetoed
    df["eligible"] = (df.miss == "") & ~df.center_vetoed  # transfer check 1: owned, valid, unvetoed center
    df.to_csv(HERE / "p2e_claim_trials.csv", index=False)
    rows = []
    for (b, h), g in df.groupby(["base", "height"]):
        for scope, d in (("all", g), ("eligible", g[g.eligible])):
            k, n = int(d.hit.sum()), len(d)
            lo, hi = clopper_pearson(k, n) if n else (np.nan, np.nan)
            rows.append(dict(base=b, height=h, scope=scope, t=t, c=c, n=n, hits=k,
                             p_detect=round(k / n, 3) if n else np.nan, ci_low=round(lo, 3), ci_high=round(hi, 3)))
    pd.DataFrame(rows).to_csv(HERE / "p2e_claim_pd.csv", index=False)
    print(f"{len(df)} trials; misses: {df.miss.value_counts().to_dict()}; wrote p2e_claim_trials.csv, p2e_claim_pd.csv")


def checks() -> None:
    import shapely
    from scipy.stats import gamma, poisson

    from akp.stage_p2.p2b_injection import TARGET_BASES  # noqa: F401  (grid frozen with the runner)
    from akp.stage_p2.p2e_budget import CLAIMED
    from akp.stage_p2.p2e_cut import _logs
    from akp.stage_p2.p2e_pd import summary
    from akp.stage_p2.p2e_vetoes import veto_shapes
    from akp.stage_p2.p2e_vintage import frozen_cut

    check_freeze()
    run = pd.read_csv(HERE / "p2e_cut.csv").iloc[0]
    fz = frozen_cut()
    rows = []
    # 1: P_d on eligible plants against the control route without the blind fraction (third seed)
    pdc = pd.read_csv(HERE / "p2e_claim_pd.csv")
    e = pdc[(pdc.scope == "eligible") & (pdc.base == CLAIMED["base"]) & (pdc.height == CLAIMED["height"])].iloc[0]
    ctl = summary(pd.read_csv(HERE / "p2e_pd_trials.csv", dtype={"tile_id": str}), float(run.t), float(fz.c), 0.0)
    ctl = ctl[(ctl.base == CLAIMED["base"]) & (ctl.height == CLAIMED["height"])].iloc[0]
    rows.append(dict(check="pd_transfer", scope="claimed size", observed=e.p_detect, expected=ctl.p_detect,
                     low=ctl.ci_low, high=ctl.ci_high, fired=not ctl.ci_low <= e.p_detect <= ctl.ci_high))
    # 2: anomaly counts at t_low per stratum and in total, one-sided Poisson
    ca = pd.read_csv(HERE / "p2e_anomalies.csv", dtype={"tile_id": str})
    n_ctl = ca[~ca.vetoed & ca.valid].groupby("stratum").size()
    ctl_land = pd.read_csv(HERE / "p2e_vetoes.csv").set_index("stratum").control_unvetoed_km2.drop("_total")
    claim_land = pd.read_csv(out_dir() / "claim_land.csv").set_index("stratum").unvetoed_km2
    an = pd.read_csv(HERE / "p2e_claim_anomalies.csv", dtype={"tile_id": str})
    use = an[~an.vetoed & an.valid]
    n_obs = use.groupby("stratum").size()
    pred = (n_ctl.reindex(ctl_land.index).fillna(0) * claim_land.reindex(ctl_land.index).fillna(0) / ctl_land).fillna(0)
    for st in list(pred.index) + ["_total"]:
        lam, k = (pred.sum(), len(use)) if st == "_total" else (pred[st], int(n_obs.get(st, 0)))
        p = float(poisson.sf(k - 1, lam)) if lam > 0 else (0.0 if k else 1.0)
        rows.append(dict(check="count_transfer", scope=st, observed=k, expected=round(lam, 2), p_value=p, fired=p < ALPHA))
    # 3: far tail of the claim anomalies against the frozen fit
    u, sg, xi, wu = float(fz.u), float(fz.sigma), float(fz.xi), float(fz.w_u)
    xm = float(use.x.max()) if len(use) else u
    lam = wu * float(np.exp(-_logs(np.array([max(xm - u, 0.0)]), sg, xi)[0]))
    p = float(1 - np.exp(-lam))
    rows.append(dict(check="far_tail", scope="all", observed=xm, expected=round(lam, 4), p_value=p, fired=p < ALPHA))
    for st, g in use.groupby("stratum"):
        ex = g.x[g.x > u] - u
        p = float(gamma.sf(ex.sum(), len(ex), scale=sg)) if len(ex) and xi <= 1e-8 else np.nan
        rows.append(dict(check="far_tail", scope=st, observed=len(ex), p_value=p, fired=bool(p < ALPHA) if p == p else False))
    chk = pd.DataFrame(rows)
    chk.to_csv(HERE / "p2e_claim_checks.csv", index=False)
    # candidates: coarse positions only, with distances to confounds; none is dismissed
    cand = an[an.candidate].copy()
    to_ll = Transformer.from_crs("EPSG:3338", "EPSG:4326", always_xy=True)
    pg = pd.read_csv(HERE / "p2e_pingos.csv")
    pxy = np.column_stack(Transformer.from_crs("EPSG:4326", "EPSG:3338", always_xy=True).transform(pg.lon, pg.lat))
    tree = shapely.STRtree(np.concatenate(list(veto_shapes().values())))
    out = []
    for r in cand.itertuples():
        lon, lat = to_ll.transform(r.x_m, r.y_m)
        pt = shapely.Point(r.x_m, r.y_m)
        out.append(dict(score=r.score, x=r.x, base=r.base, stratum=r.stratum, lon_01=round(lon, 1), lat_01=round(lat, 1),
                        nearest_pingo_km=round(float(np.hypot(*(pxy - (r.x_m, r.y_m)).T).min()) / 1e3, 1),
                        nearest_veto_km=round(float(shapely.distance(pt, tree.geometries[tree.nearest(pt)])) / 1e3, 2),
                        nearest_mine_km=mine_distance_km(r.x_m, r.y_m),
                        budget_label="exceeded or untested" if chk[chk.check != "pd_transfer"].fired.any() else "calibrated"))
    pd.DataFrame(out).to_csv(HERE / "p2e_claim_candidates.csv", index=False)
    print(chk.to_string(index=False))
    print(f"{len(out)} candidates")


@lru_cache(maxsize=1)
def _mines() -> np.ndarray:
    import pyogrio

    d = pyogrio.read_dataframe(data_dir("p2e/inventories") / "ardf" / "ardf.shp", columns=[]).to_crs(3338)
    return np.column_stack([d.geometry.x, d.geometry.y])


def mine_distance_km(x: float, y: float) -> float:
    """Distance to the nearest mine, prospect or occurrence of the USGS Alaska Resource Data File
    (p2e_inventory_sources.py; extracted to $AKP_DATA/p2e/inventories/ardf)."""
    m = _mines()
    return round(float(np.hypot(m[:, 0] - x, m[:, 1] - y).min()) / 1e3, 2)


def selfcheck() -> None:
    """Control tiles and toy tables only: the owner rule, the ownership filter before the
    merge, and the decimated validity against the control run's exact 40 m counts."""
    from akp.stage_p2.p2b_controls_run import tile_path

    toy = pd.DataFrame([dict(row=0, col=0, tile_id="1", km2=0.9, publication_date="2016", file="b"),
                        dict(row=0, col=0, tile_id="2", km2=0.6, publication_date="2018", file="a"),
                        dict(row=0, col=1, tile_id="1", km2=0.3, publication_date="2016", file="b"),
                        dict(row=0, col=1, tile_id="2", km2=0.4, publication_date="2018", file="a"),
                        dict(row=0, col=2, tile_id="3", km2=0.2, publication_date="2020", file="c"),
                        dict(row=1, col=0, tile_id="4", km2=1.0, publication_date="2019", file="z")])
    own = owner_table(toy, {(0, 0), (0, 1), (0, 2)}).set_index(["row", "col"]).tile_id.to_dict()
    assert own == {(0, 0): "2", (0, 1): "2", (0, 2): "3"}, own  # newest full; else most valid; unsearched dropped
    # an unowned higher flag must not absorb an owned one; one feature seen by two tiles counts once
    km = ~rasterio.Affine.scale(1000.0)  # toy 1 km cells
    fl = {"A": [dict(x=200.0, y=500.0, score=0.80, base=100.0)],
          "B": [dict(x=250.0, y=500.0, score=0.95, base=100.0), dict(x=1500.0, y=500.0, score=0.90, base=100.0)]}
    kept = owned_flags(fl, {(0, 0): "A", (0, 1): "B"}, km, 0.76)
    assert [(f["tile_id"], f["score"]) for f in kept] == [("B", 0.90), ("A", 0.80)], kept
    fl2 = {"A": [dict(x=990.0, y=500.0, score=0.85, base=100.0)], "B": [dict(x=1010.0, y=500.0, score=0.80, base=100.0)]}
    assert len(owned_flags(fl2, {(0, 0): "A", (0, 1): "B"}, km, 0.76)) == 1  # merged across tiles
    gt = tuple(_grid())[:6]
    for t in pd.read_csv(HERE / "p2a_controls.csv", dtype={"tile_id": str}).head(3).to_dict("records"):
        exact = {(r, c): a for r, c, a in compact_cells([json.loads(  # checkpoints hold uncompacted blocks
            (data_dir("p2b/control_run/tiles") / f"{t['tile_id']}.json").read_text())["cells"]])}
        approx = {(r, c): a for r, c, a in valid_cells(tile_path(t), gt, claim_zone=False)}
        keys = set(exact) | set(approx)
        e = np.array([exact.get(k, 0.0) for k in keys])
        a = np.array([approx.get(k, 0.0) for k in keys])
        agree = np.mean((e >= OWN_KM2) == (a >= OWN_KM2))
        print(f"tile {t['tile_id']}: total exact {e.sum():.1f} km2, decimated {a.sum():.1f}; full-cell agreement {agree:.3f}")
        assert abs(a.sum() - e.sum()) <= 0.02 * e.sum() and agree >= 0.95
    assert mine_distance_km(*_mines()[0]) == 0.0  # a mapped site is 0 km from itself
    dry_run(tile_id="956")
    print("selfcheck OK")


def dry_run(tile_id: str) -> None:
    """The search and plant paths on one control tile owning its own land cells: the anomalies
    must equal the committed control ones (p2e_anomalies.csv), and plants must run end to end."""
    import shapely

    from akp.stage_p2.p2a_strata import strata_grid
    from akp.stage_p2.p2b_controls_run import stratum_namer, tile_path
    from akp.stage_p2.p2e_vetoes import veto_shapes

    t = pd.read_csv(HERE / "p2a_controls.csv", dtype={"tile_id": str}).set_index("tile_id").loc[tile_id].to_dict()
    t["tile_id"] = tile_id
    gt = _grid()
    _, s = strata_grid()
    _, _, name = stratum_namer(s)
    res = json.loads((data_dir("p2b/control_run/tiles") / f"{tile_id}.json").read_text())
    own = {(int(r), int(c)): tile_id for r, c, _ in compact_cells([res["cells"]]) if name(int(r), int(c)) != "offland"}
    run = pd.read_csv(HERE / "p2e_cut.csv").iloc[0]
    tree = shapely.STRtree(np.concatenate(list(veto_shapes().values())))
    got = assess(owned_flags({tile_id: res["flags"]}, own, ~gt, float(run.t_low)), lambda _: tile_path(t), False,
                 tree, name, gt).sort_values("score", ascending=False, ignore_index=True)
    ref = pd.read_csv(HERE / "p2e_anomalies.csv", dtype={"tile_id": str})
    ref = ref[ref.tile_id == tile_id].sort_values("score", ascending=False, ignore_index=True)
    assert len(got) == len(ref), (len(got), len(ref))
    assert np.allclose(got.score, ref.score) and (got.vetoed == ref.vetoed).all() and (got.valid == ref.valid).all()
    assert np.allclose(got.x.fillna(-9), ref.x.fillna(-9)), "x differs from the control run"
    cells = np.array(sorted(own), dtype=np.int32)
    trials = [trial_at(i, 213.0, 16.0, cells, own, {tile_id: tile_path(t)}, gt, claim_zone=False, seed=1) for i in range(10)]
    found = [r for r in trials if np.isfinite(r.get("score_0.78", np.nan))]
    assert all(r["miss"] in ("", "nodata") for r in trials) and found, trials[:2]
    print(f"dry run on tile {tile_id}: {len(got)} anomalies match the control run; {len(found)} of 10 plants found at 0.78")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for k in ("owners", "search", "inject", "checks", "selfcheck"):
        g.add_argument(f"--{k}", action="store_true")
    a = ap.parse_args()
    {"owners": owners, "search": search, "inject": inject, "checks": checks, "selfcheck": selfcheck}[
        next(k for k in ("owners", "search", "inject", "checks", "selfcheck") if getattr(a, k))]()
