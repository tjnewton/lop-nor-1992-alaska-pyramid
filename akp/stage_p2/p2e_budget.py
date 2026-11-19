"""What the one-method route through P2b would gain from a larger budget share.

Recomputes the cut at the first operating point from the committed anomalies
(p2e_anomalies.csv; same estimator, seed and bootstrap as p2e_run) at each share in SHARES, and
reads P_d at the claimed base and the largest relief from the committed trials of both seeds
(p2e_pd.summary). The share in force is the one fixed in advance.
Writes p2e_budget.csv. Run: python -m akp.stage_p2.p2e_budget
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from akp.stage_p2 import p2e_cut as pc
from akp.stage_p2.p2e_pd import summary

HERE = Path(__file__).parent
SHARES = (0.04, 0.06, 0.08, 0.10)  # Share; plus P2c's, plus the two-method route's, the whole budget
CLAIMED = dict(base=213.0, height=16.0)


def main() -> None:
    run = pd.read_csv(HERE / "p2e_cut.csv").iloc[0]
    an = pd.read_csv(HERE / "p2e_anomalies.csv", dtype={"tile_id": str})
    u = an[~an.vetoed & an.valid]
    w = pc.weights(pd.read_csv(HERE / "p2e_tile_land.csv", dtype={"tile_id": str}), pc._searched())
    x, wv, tile = u.x.to_numpy(), u.stratum.map(w).fillna(0.0).to_numpy(), u.tile_id.to_numpy()
    boots = pc.boot_fits(x, wv, tile, np.random.default_rng(pc.SEED))  # p2e_run's first draw from SEED
    v = pd.read_csv(HERE / "p2e_vetoes.csv").set_index("stratum").loc["_total"]
    blind = float(1 - v.searched_unvetoed_km2 / v.searched_km2)
    trials = {s: pd.read_csv(HERE / f"p2e_pd_trials{tag}.csv", dtype={"tile_id": str})
              for s, tag in (("third", ""), ("fourth", "_t2"))}
    x_max = float(u[u.score >= run.t].x.max())
    rows = []
    for share in SHARES:
        c = pc.cut(x, wv, tile, None, budget=share, boots=boots)
        if share == pc.BUDGET:  # self-check: reproduces the committed cut
            assert np.isclose(c["c"], run.c), (c["c"], run.c)
        row = dict(t=run.t, share=share, c=c["c"], e_point=c["e_point"], e_bound=c["e_bound"],
                   control_max=x_max, control_candidates=int((u[u.score >= run.t].x > c["c"]).sum()))
        for s, df in trials.items():
            p = summary(df, float(run.t), c["c"], blind)
            row[f"p_detect_{s}"] = float(p[(p.base == CLAIMED["base"]) & (p.height == CLAIMED["height"])].p_detect.iloc[0])
        rows.append(row)
    out = pd.DataFrame(rows)
    assert out.c.is_monotonic_decreasing  # more budget, lower cut
    out.to_csv(HERE / "p2e_budget.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
