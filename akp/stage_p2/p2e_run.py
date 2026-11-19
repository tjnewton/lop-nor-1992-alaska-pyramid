"""P2e on the control tiles: anomalies, vetoes, x, and the one-method route's cut.

Reads the P2b control flags of the chosen variant (`--flags`, its control run's flags.csv) and,
from the operating threshold t (`--t`) downward in steps of 0.01, merges each tile's flags into
anomalies (p2e_anomalies.merge), dropping those inside a veto (p2e_vetoes.veto_shapes) and
those whose window is not fully valid, until the count reaches p2e_cut.N_TAIL: that threshold
is t_low. Each anomaly gets x = delta_bic per window pixel. p2e_cut.cut sets c at t_low, and
p2e_cut.split_half runs the falsifiers. Writes p2e_anomalies.csv (the t_low anomalies) and
p2e_cut.csv (one row: t, t_low, counts, the fit, c, E point and bound, the counting bound, the
control candidates at t, and the split-half results).
Run: python -m akp.stage_p2.p2e_run --t T [--flags PATH] [--tag SUFFIX]
"""
from __future__ import annotations

import argparse
import resource
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import shapely
from scipy.stats import chi2

from akp.paths import data_dir
from akp.stage_p2 import p2b_detector as det
from akp.stage_p2 import p2e_cut as pc
from akp.stage_p2.p2b_controls_run import tile_path
from akp.stage_p2.p2e_anomalies import merge, window
from akp.stage_p2.p2e_model_select import delta_bic

HERE = Path(__file__).parent
STEP = 0.01


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--flags", type=Path, default=None)
    ap.add_argument("--tag", default="", help="output suffix, e.g. _t073 for a second operating point")
    a = ap.parse_args()
    flags = pd.read_csv(a.flags or data_dir("p2b/control_run") / "flags.csv", dtype={"tile_id": str})
    flags = flags[flags.stratum != "offland"]
    tiles = pd.read_csv(HERE / "p2a_controls.csv", dtype={"tile_id": str}).set_index("tile_id")
    from akp.stage_p2.p2e_vetoes import veto_shapes

    tree = shapely.STRtree(np.concatenate(list(veto_shapes().values())))
    seen: dict[tuple, dict] = {}  # anomaly key -> its row (veto, validity, x), computed once

    def assess(t_id: str, ds, f: dict) -> dict:
        key = (t_id, f["base"], f["x"], f["y"])
        if key not in seen:
            vetoed = tree.query(shapely.Point(f["x"], f["y"]), predicate="intersects").size > 0
            w = None if vetoed else window(ds, f)
            x = delta_bic(w, f["base"], det.RES[f["base"]]) / w.size if w is not None else np.nan
            seen[key] = dict(tile_id=t_id, stratum=f["stratum"], base=f["base"], x_m=f["x"], y_m=f["y"],
                             score=f["score"], vetoed=vetoed, valid=w is not None, x=x)
        return seen[key]

    t = a.t
    while True:
        rows = []
        for t_id, g in flags[flags.score >= t].groupby("tile_id"):
            with rasterio.Env(GDAL_CACHEMAX=det.GDAL_CACHE_MB), det.open_tile(tile_path(tiles.loc[t_id].to_dict())) as ds:
                rows += [assess(t_id, ds, f) for f in merge(g.to_dict("records"))]
        an = pd.DataFrame(rows)
        usable = an[~an.vetoed & an.valid]
        print(f"t={t:.2f}: {len(an)} anomalies, {len(usable)} unvetoed with a valid window", flush=True)
        if len(usable) >= pc.N_TAIL or t - STEP < det.S_MIN:
            break
        t = round(t - STEP, 2)
    an["above_t"] = an.score >= a.t
    an.to_csv(HERE / f"p2e_anomalies{a.tag}.csv", index=False)

    land = pd.read_csv(HERE / "p2e_tile_land.csv", dtype={"tile_id": str})
    searched = pc._searched()
    w = pc.weights(land, searched)
    x, wv = usable.x.to_numpy(), usable.stratum.map(w).fillna(0.0).to_numpy()
    rng = np.random.default_rng(pc.SEED)
    c = pc.cut(x, wv, usable.tile_id.to_numpy(), rng)
    sh = pc.split_half(usable[["tile_id", "stratum", "x"]], land, searched, rng)
    at_t = usable[usable.score >= a.t]
    n_cand = int((at_t.x > c["c"]).sum())
    ratio = searched.sum() / land.unvetoed_km2.sum()
    out = dict(t=a.t, t_low=t, n_anomalies=len(an), n_usable=len(usable), n_usable_at_t=len(at_t),
               base_mix_top=";".join(f"{int(b)}:{n}" for b, n in usable.nlargest(c.get("k", 0), "x").base.value_counts().items()),
               **{k: v for k, v in c.items() if k != "boots"},
               control_candidates_at_t=n_cand,
               counting_bound=0.5 * chi2.ppf(0.95, 2 * n_cand + 2) * ratio,
               **{f"split_{k}": v for k, v in sh.items()})
    pd.DataFrame([out]).to_csv(HERE / f"p2e_cut{a.tag}.csv", index=False)
    print(pd.Series(out).to_string())
    print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
