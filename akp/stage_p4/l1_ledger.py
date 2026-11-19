"""P4 L1: the detectability ledger. What each sensor could and could not say about each target class.

One row per target class and sensor, assembled from committed tables only (no data are read):
- sees_it: whether the sensor responds to the class at all. Magnetics and helicopter EM: "yes" when
  every contrast of the class passes the forward bound of p3_g1a_decision.csv on some platform,
  "partly" when only some do, "no" when none does. Gravity and drape respond to any dense or hollow
  body ("yes", by physics; their limits are coverage and size).
- prior_under_data: the share of the cover-weighted Track A prior under data of that sensor
  (magnetics and EM: flown_prior_mass of p3_trackA_hide.csv; gravity: the searched mass of
  p3_gravity_closure.csv at the most generous threshold, since a station's reach is its coverage).
- searched_at_budget and searched_at_lead: the searched mass of p3_trackA_hide.csv at the budget cut
  and at the lead threshold (magnetics: the built case; both contrasts given as a range).
- leads and expected_lookalikes: Track A leads and the open-data expectation (p3_trackA_leads.csv).
- pd_terrain_at_drape_bound: for the drape row, the claim-zone terrain P_d (p2e_claim_pd.csv, all
  terrain) at the relief grid value nearest the sand-fill drape bound (p3_surface_bounds.csv), for
  the two larger bases.
- seismic stations (S9): the simulated body is stone at the most extreme contrast, so the
  stone classes "respond" and the hollow class is "not simulated". prior_under_data is the share of the
  cover-weighted prior within the largest range of s9_sensitivity.csv that has generous attenuation,
  of any receiver, with position uncertainty added: the most generous exposure that is physical.
  searched_at_budget is zero: no per-station test could be posed, so nothing is searched.
Classes: T1 felsic stone, T2 mafic stone, T3 conductor in frozen ground, T4 resistive body in
conductive ground, T5 hollow. Writes l1_ledger.csv.
Run: python -m akp.stage_p4.l1_ledger
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
P3, P2 = HERE.parent / "stage_p3", HERE.parent / "stage_p2"
CLASSES = {"T1": "felsic stone", "T2": "mafic stone", "T3": "conductor in frozen ground", "T4": "resistive body in conductive ground", "T5": "hollow"}
GRAVITY = {"T1": "T1_felsic_stone", "T2": "T2_mafic_stone", "T5": "T5_hollow"}
SEISMIC = {"T1": "yes", "T2": "yes", "T5": "not simulated"}  # S9: the simulated body is stone


def main() -> None:
    dec = pd.read_csv(P3 / "p3_g1a_decision.csv")
    hide = pd.read_csv(P3 / "p3_trackA_hide.csv").query("prior == 'W_cover'").set_index(["cls", "at"])
    leads = pd.read_csv(P3 / "p3_trackA_leads.csv").query("geo == 'all'")
    grav = pd.read_csv(P3 / "p3_gravity_closure.csv")
    drape = pd.read_csv(P3 / "p3_surface_bounds.csv").query("signature == 'drape' and quantity == 'max_relief'")
    pd2 = pd.read_csv(P2 / "p2e_claim_pd.csv").query("scope == 'all'")
    sand = float(drape[drape.case.str.startswith("sand")].value.max())
    relief = float(pd2.height.iloc[(pd2.height - sand).abs().argmin()])
    s9 = pd.read_csv(HERE / "s9_sensitivity.csv").query("cls == 'all' and uncertainty == 'added' and attenuation == 'generous'")
    seis_exposure = float(s9.loc[s9.range_km.idxmax(), "cover"])
    rows = []

    def sees(cls: str, sensor: str) -> str:
        d = dec[(dec.cls == cls) & (dec.sensor == sensor)]
        if d.empty:
            return "no"
        ok = d.groupby("contrast").passes.any()
        return "yes" if ok.all() else "partly" if ok.any() else "no"

    for cls, name in CLASSES.items():
        mag = sees(cls, "magnetics")
        m = dict(target=cls, description=name, sensor="magnetics", sees_it=mag)
        if cls == "T2":
            lo, hi = (float(hide.at[(f"T2_{c}_built", "lead_threshold"), "searched_mass"]) for c in ("low", "high"))
            m.update(prior_under_data=float(hide.at[("T2_high_built", "budget_cut"), "flown_prior_mass"]),
                     searched_at_budget=float(hide.at[("T2_high_built", "budget_cut"), "searched_mass"]),
                     searched_at_lead_low=lo, searched_at_lead_high=hi,
                     leads=int(leads[leads.system.str.startswith("mag")].leads.sum()),
                     expected_lookalikes=round(float(leads[leads.system.str.startswith("mag")].expected_lookalikes.sum()), 1))
        rows.append(m)
        em = sees(cls, "helicopter_fdem")
        e = dict(target=cls, description=name, sensor="helicopter EM", sees_it=em)
        if cls == "T3":
            e.update(prior_under_data=float(hide.at[("T3_central", "budget_cut"), "flown_prior_mass"]),
                     searched_at_budget=float(hide.at[("T3_central", "budget_cut"), "searched_mass"]),
                     searched_at_lead_low=float(hide.at[("T3_central", "lead_threshold"), "searched_mass"]),
                     searched_at_lead_high=float(hide.at[("T3_opt", "lead_threshold"), "searched_mass"]),
                     leads=int(leads[leads.system.str.startswith("em")].leads.sum()),
                     expected_lookalikes=round(float(leads[leads.system.str.startswith("em")].expected_lookalikes.sum()), 1))
        rows.append(e)
        if cls in GRAVITY:
            g = grav[grav.target == GRAVITY[cls]]
            rows.append(dict(target=cls, description=name, sensor="gravity stations", sees_it="yes",
                             prior_under_data=float(g.searched_mass_W_cover.max()), searched_at_budget=0.0,
                             peak_mgal=float(g.peak_mgal.iloc[0])))
        if cls in ("T1", "T2", "T5"):
            p = pd2[(pd2.height == relief) & pd2.base.isin([213.0, 400.0])].sort_values("base")
            rows.append(dict(target=cls, description=name, sensor="terrain drape", sees_it="yes", searched_at_budget=np.nan,
                             drape_bound_sand_m=round(sand, 2), relief_tested_m=relief,
                             pd_terrain_at_drape_bound_low=float(p.p_detect.min()), pd_terrain_at_drape_bound_high=float(p.p_detect.max())))
        if cls in SEISMIC:
            rows.append(dict(target=cls, description=name, sensor="seismic stations", sees_it=SEISMIC[cls],
                             prior_under_data=seis_exposure if SEISMIC[cls] == "yes" else np.nan,
                             searched_at_budget=0.0 if SEISMIC[cls] == "yes" else np.nan))
    out = pd.DataFrame(rows)
    assert (out.searched_at_budget.dropna() == 0).all(), "the ledger assumes nothing is searched at the budget cut"
    out.to_csv(HERE / "l1_ledger.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
