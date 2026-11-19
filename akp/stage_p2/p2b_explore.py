"""P2b design revision, exploratory: compare detector variants on control data.

The detector failed its fitness check. Each variant in VARIANTS (rotation
step, curvature smoothing) is run on N_TILES control tiles, drawn by SEED in proportion to
their land (p2b_control_land.csv), and on fresh injections (EXPLORE_SEED, not the seed) at
the CELLS (base, height) grid points. Expected false flags over the searched land are
unstratified here (subset flags per control km2 times the searched km2): fine for ranking
variants, not for the operating point, which the chosen variant gets from a full stratified
re-run. Exploration on control data only; the chosen variant's final P_d comes from another
fresh injection seed.
Per-(variant, tile) flags are checkpointed in $AKP_DATA/p2b/explore/. Writes p2b_explore.csv:
variant, threshold, expected_fa, and P_d at each cell. Run: python -m akp.stage_p2.p2b_explore
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import rasterio

from akp.paths import data_dir
from akp.stage_p2 import p2b_detector as det
from akp.stage_p2 import p2b_injection as inj
from akp.stage_p2.p2b_controls_run import MEM_FRACTION, PEAK_GB_PER_WORKER, tile_path

HERE = Path(__file__).parent
VARIANTS = {
    "V0": dict(rotation_step=15.0, sigma_frac=0.1),  # the design
    "V1": dict(rotation_step=5.0, sigma_frac=0.1),
    "V2": dict(rotation_step=15.0, sigma_frac=0.2),
    "V3": dict(rotation_step=5.0, sigma_frac=0.2),
}
SEED = 20261112
EXPLORE_SEED = 20261113
SMOKE = os.environ.get("AKP_EXPLORE_SMOKE") == "1"  # tiny run to test the plumbing
N_TILES = 2 if SMOKE else 30
CELLS = ((213.0, 16.0), (213.0, 8.0), (100.0, 16.0), (100.0, 8.0))
N_TRIALS = 2 if SMOKE else 50
THRESHOLDS = np.round(np.arange(0.6, 1.0, 0.01), 2)


def tile_flags(args) -> str:
    name, t = args
    det.configure(**VARIANTS[name])
    out = data_dir(f"p2b/explore/{name}") / f"{t['tile_id']}.json"
    control = data_dir("p2b/control_run/tiles") / f"{t['tile_id']}.json"
    if not out.exists() and name == "V0" and control.exists():
        # V0 is the design: its flags for this tile are already in the control run
        fl = [{"base": f["base"], "score": f["score"]} for f in json.loads(control.read_text())["flags"]]
        part = out.with_name(out.name + ".part")
        part.write_text(json.dumps(fl))
        os.replace(part, out)
    if not out.exists():
        fl = []
        with rasterio.Env(GDAL_CACHEMAX=det.GDAL_CACHE_MB), det.open_tile(tile_path(t)) as ds:
            for win, core in det.blocks(ds):
                g = det.read_native(ds, win)
                if np.isfinite(g.z).any():
                    fl += [{"base": f["base"], "score": f["score"]} for f in det.tile_flags(g)
                           if det.in_core(f, ds, core)]
        part = out.with_name(out.name + ".part")
        part.write_text(json.dumps(fl))
        os.replace(part, out)
    return t["tile_id"]


def trial(args) -> dict:
    name, i, base, height = args
    det.configure(**VARIANTS[name])
    inj.SEED = EXPLORE_SEED
    r = inj.trial((i, base, height))
    best = max((r[k] for k in r if k.startswith("s") and k[1:].isdigit() and r[k] == r[k]), default=np.nan)
    return dict(variant=name, base=base, height=height, best=best)


def main() -> None:
    tiles = pd.read_csv(HERE / "p2a_controls.csv", dtype={"tile_id": str})
    land = pd.read_csv(HERE / "p2b_control_land.csv", dtype={"tile_id": str}).set_index("tile_id").land_km2
    w = land.reindex(tiles.tile_id).to_numpy()
    pick = np.random.default_rng(SEED).choice(len(tiles), size=N_TILES, replace=False, p=w / w.sum())
    subset = tiles.iloc[np.sort(pick)].to_dict("records")
    sub_km2 = float(land.reindex([t["tile_id"] for t in subset]).sum())
    searched_km2 = float(pd.read_csv(HERE / "p2b_control_area.csv").searched_km2.sum())
    workers = max(1, min(6, int(MEM_FRACTION * psutil.virtual_memory().total / (PEAK_GB_PER_WORKER * 1e9))))
    rows = []
    for name in VARIANTS:
        with ProcessPoolExecutor(workers) as ex:
            list(ex.map(tile_flags, [(name, t) for t in subset]))
        scores = np.array([f["score"] for t in subset for f in json.loads(
            (data_dir(f"p2b/explore/{name}") / f"{t['tile_id']}.json").read_text())])
        jobs = [(name, k * 1000 + i, b, h) for k, (b, h) in enumerate(CELLS) for i in range(N_TRIALS)]
        with ProcessPoolExecutor(4) as ex:
            tr = pd.DataFrame(list(ex.map(trial, jobs, chunksize=4)))
        for t in THRESHOLDS:
            row = dict(variant=name, threshold=t,
                       expected_fa=round(float((scores >= t).sum()) / sub_km2 * searched_km2, 1))
            for b, h in CELLS:
                s = tr[(tr.base == b) & (tr.height == h)].best
                row[f"pd_{int(b)}_{int(h)}"] = round(float((s >= t).mean()), 3)
            rows.append(row)
        print(name, "done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / ("p2b_explore_smoke.csv" if SMOKE else "p2b_explore.csv"), index=False)
    at = out[out.expected_fa <= 100].sort_values("threshold").groupby("variant").head(1)
    print(f"subset {N_TILES} tiles, {sub_km2:.0f} km2; at the lowest threshold with expected_fa <= 100:")
    print(at.to_string(index=False))


if __name__ == "__main__":
    main()
