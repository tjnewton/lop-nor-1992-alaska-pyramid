"""P2b calibration, part 1: the detector on every control tile (target-free false alarms).

Runs akp.stage_p2.p2b_detector.tile_flags on each tile of p2a_controls.csv (control tiles
only; the detector refuses other paths). Each tile's result is checkpointed on the drive
($AKP_DATA/p2b/control_run/tiles/<tile_id>.json), so a re-run resumes. Each flag gets the
stratum (p2a_strata, rare strata pooled as in p2a_controls) of its 1 km cell; cells off land
are "offland" and excluded. Control land per stratum is counted from the tiles' valid DTM
cells (40 m blocks), not from bounding boxes.

Writes the flags to $AKP_DATA/p2b/control_run/flags.csv, and commits:
- p2b_control_land.csv: tile_id, land_km2 (on-land valid area; injection draw weights);
- p2b_control_area.csv: stratum, control_km2, searched_km2;
- p2b_control_fa.csv: base, threshold, n_flags (control), expected_fa (post-stratified over the
  searched land, sum over strata of n_s / control_s * searched_s); base "all" is the sum.
Run: python -m akp.stage_p2.p2b_controls_run
"""
from __future__ import annotations

import json
import os
import resource
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import rasterio
from pyproj import Transformer

from akp.paths import data_dir
from akp.stage_p2 import p2b_detector as det
from akp.stage_p2.p2a_controls import MIN_SHARE

HERE = Path(__file__).parent
VBLOCK = 8  # 5 m cells per side of a validity block (40 m)
THRESHOLDS = np.round(np.arange(det.S_MIN, 1.0, 0.01), 2)
# Memory budget (a whole-tile run of 8 workers exhausted 48 GB and crashed the machine,
# 2026-11-11): workers = MEM_FRACTION of RAM / measured per-worker peak, at most MAX_WORKERS;
# no new tile is submitted while available memory is below MIN_AVAILABLE of RAM.
PEAK_GB_PER_WORKER = 3.0  # measured 2.70 GB at BLOCK 2048 with GDAL_CACHEMAX 256 MB (2026-11-11)
MEM_FRACTION = 0.4
MIN_AVAILABLE = 0.25
MAX_WORKERS = 6


def tile_path(t) -> Path:
    return data_dir("p2b/ifsar_controls") / t["project"] / t["subproject"] / t["file"]


def run_tile(t: dict, grid_transform: tuple, out_dir: Path | None = None, claim_zone: bool = False) -> dict:
    """Flags and valid land per 1 km cell of one tile, checkpointed in out_dir (default: the
    control run). A claim tile (claim_zone=True, the frozen claim-zone run only) is read from t["path"]."""
    out = (out_dir or data_dir("p2b/control_run/tiles")) / f"{t['tile_id']}.json"
    if out.exists():
        return json.loads(out.read_text())
    inv = ~rasterio.Affine(*grid_transform)
    fl, cells = [], []
    path = Path(t["path"]) if claim_zone else tile_path(t)
    with rasterio.Env(GDAL_CACHEMAX=det.GDAL_CACHE_MB), det.open_tile(path, claim_zone=claim_zone) as ds:
        to3338 = Transformer.from_crs(ds.crs, "EPSG:3338", always_xy=True)
        for win, core in det.blocks(ds):
            g = det.read_native(ds, win)
            if not np.isfinite(g.z).any():
                continue
            fl += [f for f in det.tile_flags(g) if det.in_core(f, ds, core)]
            # valid land, counted on the core only (40 m blocks) -> 1 km strata cells (EPSG:3338)
            r0, c0 = int(core.row_off - win.row_off), int(core.col_off - win.col_off)
            z = g.z[r0:r0 + int(core.height), c0:c0 + int(core.width)]
            h, w = (z.shape[0] // VBLOCK) * VBLOCK, (z.shape[1] // VBLOCK) * VBLOCK
            counts = np.isfinite(z[:h, :w]).reshape(h // VBLOCK, VBLOCK, w // VBLOCK, VBLOCK).sum(axis=(1, 3))
            r, c = np.nonzero(counts)
            ct = ds.window_transform(core) * rasterio.Affine.scale(VBLOCK, VBLOCK)
            xs, ys = rasterio.transform.xy(ct, r, c)
            gx, gy = inv * to3338.transform(np.asarray(xs), np.asarray(ys))
            area = counts[r, c] * (g.res ** 2) / 1e6  # km2
            cells.append(np.column_stack([np.floor(gy), np.floor(gx), area]))
            del g, z
    res = dict(tile_id=t["tile_id"], flags=fl, cells=compact_cells(cells))
    part = out.with_name(out.name + ".part")
    part.write_text(json.dumps(res))
    os.replace(part, out)  # atomic: a crash never leaves a truncated checkpoint
    return res


def compact_cells(parts) -> list[list[float]]:
    """Sum (row, col, km2) blocks to one entry per 1 km strata cell (about 600 times fewer)."""
    if not len(parts):
        return []
    a = np.concatenate([np.asarray(p, dtype=float).reshape(-1, 3) for p in parts])
    rc, inv = np.unique(a[:, :2].astype(np.int64), axis=0, return_inverse=True)
    km2 = np.bincount(inv.ravel(), weights=a[:, 2])
    return [[int(r), int(c), float(v)] for (r, c), v in zip(rc, km2)]


def run_one(t: dict, grid_transform: tuple, out_dir: Path | None = None, claim_zone: bool = False) -> str:
    """run_tile, returning only the tile id: results stay on disk, not in the parent."""
    if not ((out_dir or data_dir("p2b/control_run/tiles")) / f"{t['tile_id']}.json").exists():
        run_tile(t, grid_transform, out_dir, claim_zone)
    return t["tile_id"]


def run_all(tiles: list[dict], grid_transform: tuple, out_dir: Path | None = None, claim_zone: bool = False) -> None:
    """Checkpoint every tile in a pool sized by the memory budget, with a memory watchdog."""
    total = psutil.virtual_memory().total
    workers = max(1, min(MAX_WORKERS, int(MEM_FRACTION * total / (PEAK_GB_PER_WORKER * 1e9))))
    print(f"{workers} workers (budget {MEM_FRACTION:.0%} of {total / 1e9:.0f} GB, "
          f"{PEAK_GB_PER_WORKER} GB measured peak each)", flush=True)
    pending, queue = set(), list(tiles)
    with ProcessPoolExecutor(workers) as ex:
        while queue or pending:
            low = psutil.virtual_memory().available < MIN_AVAILABLE * total
            if queue and len(pending) < workers and (not low or not pending):
                pending.add(ex.submit(run_one, queue.pop(0), grid_transform, out_dir, claim_zone))
                continue
            done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for d in done:
                d.result()  # surface worker errors
            if low:
                print(f"memory watchdog: available below {MIN_AVAILABLE:.0%}; holding submissions", flush=True)


def stratum_namer(s: np.ndarray):
    """(searched land cells per stratum, rare strata, name(row, col) of a 1 km strata-grid cell)."""
    from akp.stage_p2.p2a_strata import label

    strata = pd.read_csv(HERE / "p2a_strata.csv")
    searched = strata[(strata.unit == "searched") & (strata.stratum != "_total")].set_index("stratum").cells
    rare = set(searched.index[searched / searched.sum() < MIN_SHARE])

    def name(row: int, col: int) -> str:
        if not (0 <= row < s.shape[0] and 0 <= col < s.shape[1]) or s[row, col] < 0:
            return "offland"
        lab = label(int(s[row, col]))
        return "rare" if lab in rare or lab not in searched.index else lab

    return searched, rare, name


def main() -> None:
    from akp.stage_p2.p2a_strata import strata_grid

    transform, s = strata_grid()
    searched, rare, name = stratum_namer(s)

    tiles = pd.read_csv(HERE / "p2a_controls.csv", dtype={"tile_id": str}).to_dict("records")
    run_all(tiles, tuple(transform)[:6])

    # stream the checkpoints in tile order (deterministic sums), one at a time
    inv = ~transform
    area: dict[str, float] = {}
    rows, land = [], []
    for t in sorted(tiles, key=lambda t: int(t["tile_id"])):
        res = json.loads((data_dir("p2b/control_run/tiles") / f"{t['tile_id']}.json").read_text())
        tile_km2 = 0.0
        for r, c, a in compact_cells([res["cells"]]):
            k = name(r, c)
            area[k] = area.get(k, 0.0) + a
            tile_km2 += a if k != "offland" else 0.0
        land.append(dict(tile_id=t["tile_id"], land_km2=round(tile_km2, 3)))
        for f in res["flags"]:
            gx, gy = inv * (f["x"], f["y"])
            rows.append({k: f[k] for k in ("base", "x", "y", "score", "rotation")}
                        | {"tile_id": t["tile_id"], "stratum": name(int(np.floor(gy)), int(np.floor(gx)))})
        del res
    pd.DataFrame(land).to_csv(HERE / "p2b_control_land.csv", index=False)
    print(f"aggregation peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB", flush=True)
    flags = pd.DataFrame(rows)
    flags.to_csv(data_dir("p2b/control_run") / "flags.csv", index=False)

    s_area = searched.groupby(lambda k: "rare" if k in rare else k).sum().astype(float)
    a = pd.DataFrame({"control_km2": pd.Series(area), "searched_km2": s_area}).fillna(0.0)
    a.index.name = "stratum"
    a.round(1).to_csv(HERE / "p2b_control_area.csv")

    on_land = flags[flags.stratum != "offland"]
    weight = (a.searched_km2 / a.control_km2.where(a.control_km2 > 0)).dropna()
    missing = sorted(set(a.index[(a.searched_km2 > 0) & (a.control_km2 <= 0)]) - {"offland"})
    assert not missing, f"strata with searched land but no control land: {missing}"
    fa = []
    for base in det.BASES:
        sub = on_land[on_land.base == base]
        for t in THRESHOLDS:
            hit = sub[sub.score >= t]
            n = hit.groupby("stratum").size()
            fa.append(dict(base=base, threshold=t, n_flags=int(n.sum()),
                           expected_fa=float((n * weight.reindex(n.index).fillna(0)).sum())))
    fa = pd.DataFrame(fa)
    tot = fa.groupby("threshold", as_index=False)[["n_flags", "expected_fa"]].sum().assign(base="all")
    out = pd.concat([fa, tot[fa.columns]], ignore_index=True)
    out["expected_fa"] = out.expected_fa.round(2)
    out.to_csv(HERE / "p2b_control_fa.csv", index=False)
    print(f"{len(land)} tiles; {len(flags)} flags ({(flags.stratum == 'offland').sum()} off land); "
          f"control land {a.control_km2.drop('offland', errors='ignore').sum():.0f} km2")
    print(out[out.base == "all"].iloc[::10].to_string(index=False))


if __name__ == "__main__":
    main()
