"""P3 the budget argument without a tail model.

Per system class (p3_null_rates.band) and sensor:
- budget_density = the Track A budget share / the class's Track A line-km.
  Any cut must hold false candidates per line-km below this;
- observed null density = declustered null clusters (p3_null_rates.decluster) scoring at least t,
  per open line-km, for t in THRESHOLDS (which span the planted-body scores);
- ratio = observed / budget density;
- certify_line_km = 3 / budget_density, the open control line-km that would be needed just to certify the
  budget by counting with zero exceedances (Poisson 95% upper bound of about 3 events).
Writes p3_budget_density.csv. Run: python -m akp.stage_p3.p3_budget_density
Exposure is line-km (p3_null_rates): every density here is per line-km.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from akp.stage_p3.p3_member_heights import load, member_class
from akp.stage_p3.p3_null_rates import BUDGET, load_null, track_a_area

HERE = Path(__file__).parent
THRESHOLDS = (10.0, 20.0, 50.0, 100.0)
ZERO_COUNT_95 = 2.996  # Poisson upper 95% bound on the mean when no event is seen


def main() -> None:
    ch = pd.read_csv(HERE / "p3_channels.csv").query("reader_ok == 'yes'")
    inv = pd.read_csv(HERE / "p3_survey_inventory.csv")
    hmap = inv.set_index("survey_id")
    rows, mh = [], load()
    for sensor in ("mag", "em"):
        c2 = ch.assign(cls=[member_class(mh, hmap, r.package, r.member, r.survey_id, sensor) for r in ch.itertuples()]).dropna(subset=["cls"])
        clusters, areas = load_null(c2, sensor)
        ta = track_a_area(inv, sensor)
        ta_total = float(ta.line_km.sum()) if len(ta) else 0.0
        for cl, cc in clusters.groupby("cls"):
            ctl_km2 = float(areas[areas.cls == cl].line_km.sum())
            ta_km2 = float(ta[ta.cls == cl].line_km.sum()) if len(ta) else 0.0
            if ta_km2 == 0 or ctl_km2 == 0:
                continue
            budget = BUDGET * ta_km2 / ta_total
            bd = budget / ta_km2
            for t in THRESHOLDS:
                dens = float((cc.score >= t).sum() / ctl_km2)
                rows.append(dict(cls=cl, threshold=t, control_line_km=round(ctl_km2, 1), track_a_line_km=round(ta_km2, 1),
                                 budget=round(budget, 5), budget_density=float(f"{bd:.4g}"), observed_density=round(dens, 4),
                                 ratio=float(f"{dens / bd:.4g}"), certify_line_km=float(f"{ZERO_COUNT_95 / bd:.4g}"),
                                 expected_false_in_track_a=round(dens * ta_km2, 1)))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "p3_budget_density.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
