"""P3 the joint magnetic-and-conductive lever (exploratory; control data).

A body both magnetic and conductive would show in both sensors of a helicopter EM-plus-magnetics
survey. For every open member with both null outputs, a coincidence is a declustered magnetic cluster
scoring at least t_mag with a declustered EM cluster scoring at least t_em within COINCIDE_M. The
density of coincidences per km2 of flown area, for the threshold pairs PAIRS, is compared with the
budget density of p3_budget_density (helicopter class). Writes p3_joint.csv (t_mag, t_em, km2,
coincidences, per_km2). Run: python -m akp.stage_p3.p3_joint
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from akp.paths import data_dir
from akp.stage_p3.p3_null_rates import decluster, read_peaks

HERE = Path(__file__).parent
COINCIDE_M = 250.0
PAIRS = ((10.0, 5.0), (20.0, 10.0), (50.0, 25.0), (100.0, 50.0))


def main() -> None:
    ch = pd.read_csv(HERE / "p3_channels.csv").query("reader_ok == 'yes'")
    km2, counts = 0.0, {p: 0 for p in PAIRS}
    for r in ch.itertuples():
        d = data_dir("p3/g1b") / Path(r.package).stem / Path(r.member).stem
        fm, fe, fa = d / "null_mag.csv.gz", d / "null_em.csv.gz", d / "area.csv.gz"
        if not (fm.exists() and fe.exists() and fa.exists()):
            continue
        m, e = read_peaks(fm), read_peaks(fe)
        if m.empty or e.empty:
            continue
        m, e = decluster(m), decluster(e)
        km2 += float(pd.read_csv(fa).km2.sum())
        for tm, te in PAIRS:
            mm, ee = m[m.score >= tm], e[e.score >= te]
            if len(mm) and len(ee):
                hits = cKDTree(ee[["x", "y"]].to_numpy()).query_ball_point(mm[["x", "y"]].to_numpy(), COINCIDE_M)
                counts[(tm, te)] += int(sum(1 for h in hits if h))
    out = pd.DataFrame([dict(t_mag=tm, t_em=te, km2=round(km2, 1), coincidences=c, per_km2=round(c / km2, 4) if km2 else np.nan)
                        for (tm, te), c in counts.items()])
    out.to_csv(HERE / "p3_joint.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
