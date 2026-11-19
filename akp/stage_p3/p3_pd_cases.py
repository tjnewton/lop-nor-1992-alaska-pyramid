"""P4 M0: detection probability of the full-prior plants on the open packages, by magnetization case.

Input: the trials of p3_g1b_run (trials.csv.gz per open member; plants drawn from the full prior),
p3_null_rates.csv (the Track A budget cut per system class) and p3_lookalike.csv (the lead
threshold per system class: the pooled median planted score of T2_high, or of T3_central for EM).
For each system class, target class and case (magnetic plants with q = 0 are "built", with q > 0
"monolith"; EM plants have one case, "all"):
- plants, and the median planted and unplanted scores at the plant;
- p_budget: the share with score >= the budget cut and unplanted score below it ('s hit rule);
- p_lead: the same at the lead threshold; p_lead_at_or_above: score >= the lead threshold alone.
Writes p3_pd_cases.csv. Run: python -m akp.stage_p3.p3_pd_cases
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from akp.paths import data_dir
from akp.stage_p3.p3_lookalike import SENSOR
from akp.stage_p3.p3_member_heights import load, member_class
from akp.stage_p3.p3_null_rates import read_peaks

HERE = Path(__file__).parent
LEAD_CLASS = {"mag": "T2_high", "em": "T3_central"}


def main() -> None:
    ch = pd.read_csv(HERE / "p3_channels.csv").query("reader_ok == 'yes'")
    inv = pd.read_csv(HERE / "p3_survey_inventory.csv").set_index("survey_id")
    cuts = pd.read_csv(HERE / "p3_null_rates.csv").query("track == 'A'").set_index("cls").c
    look = pd.read_csv(HERE / "p3_lookalike.csv").query("plant_score == 'q50_pooled' and geo == 'all'")
    parts, mh = [], load()
    for r in ch.itertuples():
        t = read_peaks(data_dir("p3/g1b") / Path(r.package).stem / Path(r.member).stem / "trials.csv.gz")
        if len(t):
            t["system"] = [member_class(mh, inv, r.package, r.member, r.survey_id, SENSOR[c]) for c in t.cls]
            parts.append(t)
    tr = pd.concat(parts).dropna(subset=["system"])
    tr["case"] = np.where(tr.cls.map(SENSOR) == "em", "all", np.where(tr.get("q", 0).fillna(0) > 0, "monolith", "built"))
    rows = []
    for (system, cls, case), g in tr.groupby(["system", "cls", "case"]):
        lk = look[(look.cls == LEAD_CLASS[SENSOR[cls]]) & (look.system == system)]
        lead = float(lk.threshold.iloc[0]) if len(lk) else np.nan
        cut = float(cuts.get(system, np.nan))
        s, s0 = g.score.fillna(-np.inf), g.score_bg.fillna(-np.inf)

        def hit(t, s=s, s0=s0):
            return round(float(((s >= t) & (s0 < t)).mean()), 4) if np.isfinite(t) else np.nan

        rows.append(dict(system=system, cls=cls, case=case, plants=len(g), score_median=round(float(g.score.median()), 2),
                         score_bg_median=round(float(g.score_bg.median()), 2), budget_cut=round(cut, 2), p_budget=hit(cut),
                         lead_threshold=lead, p_lead=hit(lead),
                         p_lead_at_or_above=round(float((s >= lead).mean()), 4) if np.isfinite(lead) else np.nan))
    out = pd.DataFrame(rows)
    assert (out.p_lead <= out.p_lead_at_or_above + 1e-9).all()
    out.to_csv(HERE / "p3_pd_cases.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
