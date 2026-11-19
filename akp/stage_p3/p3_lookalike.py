"""P3 output (b) context: how many natural compact sources look like a planted body.

For each target class (T2_low, T2_high, T3_opt, T3_central), system class (p3_null_rates.band,
by sensor height) and geology class (the geo part of the P2a stratum: bedrock, glacial, surficial,
other):
- the planted-body score quantiles (Q50, Q25) from the trials of p3_g1b_run, misses counted as 0;
- the density of declustered null clusters (p3_null_rates.decluster) scoring at least that high,
  per 1000 line-km flown (the exposure: counts scale with line length, not with the area between
  lines), with the control line-km and the cluster count behind it;
- the expected number of such look-alikes over the class's Track A line-km in that geology class
  (p3_null_rates.track_a_area).
Rows with plant_score "q50_pooled" use one threshold per target and system class, the median planted
score pooled over geology classes, and give the null density at it per geology class and over all of
them (geo "all"). That threshold is the lead threshold, and those densities are the open-data
side of the transfer comparison.
This is descriptive. It says how many natural sources any follow-up of a planted-like anomaly
would face; it is not a gate. Writes p3_lookalike.csv. Run: python -m akp.stage_p3.p3_lookalike
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from akp.paths import data_dir
from akp.stage_p3.p3_g1b_run import spacing
from akp.stage_p3.p3_member_heights import load, member_class
from akp.stage_p3.p3_null_rates import decluster, read_peaks, track_a_area

HERE = Path(__file__).parent
SENSOR = {"T2_low": "mag", "T2_high": "mag", "T3_opt": "em", "T3_central": "em"}


def geo(stratum: str) -> str:
    parts = str(stratum).split("|")
    return parts[1] if len(parts) == 3 else "off_land"


def tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(trials, declustered null clusters, flown area with line_km, inventory by survey id) of every open
    member that has trials, each with its package, its system class and its geology class."""
    ch = pd.read_csv(HERE / "p3_channels.csv").query("reader_ok == 'yes'")
    inv = pd.read_csv(HERE / "p3_survey_inventory.csv").set_index("survey_id")
    trials, nulls, areas, mh = [], [], [], load()
    for r in ch.itertuples():
        d = data_dir("p3/g1b") / Path(r.package).stem / Path(r.member).stem
        if not (d / "trials.csv.gz").exists():
            continue
        sys_cls = {sensor: member_class(mh, inv, r.package, r.member, r.survey_id, sensor) for sensor in ("mag", "em")}
        t = read_peaks(d / "trials.csv.gz")
        if len(t):
            t["system"] = t.cls.map(lambda c: sys_cls[SENSOR[c]])
            trials.append(t.assign(package=r.package))
        for sensor in ("mag", "em"):
            f = d / f"null_{sensor}.csv.gz"
            if sys_cls[sensor] and f.exists():
                pk = read_peaks(f)
                if len(pk):
                    nulls.append(decluster(pk).assign(system=sys_cls[sensor], package=r.package))
        if (d / "area.csv.gz").exists():
            a = pd.read_csv(d / "area.csv.gz").assign(line_km=lambda x: x.km2 * 1000.0 / spacing(r), package=r.package)
            for sensor in ("mag", "em"):
                if sys_cls[sensor]:
                    areas.append(a.assign(system=sys_cls[sensor]))
    tr, nu, ar = pd.concat(trials), pd.concat(nulls), pd.concat(areas)
    for d in (tr, nu, ar):
        d["geo"] = d.stratum.map(geo)
    return tr, nu, ar, inv


def main() -> None:
    tr, nu, ar, inv = tables()
    ta = pd.concat([track_a_area(inv.reset_index(), s) for s in ("mag", "em")])
    ta = ta.rename(columns={"cls": "system"}).assign(geo=lambda x: x.stratum.map(geo)).groupby(["system", "geo"]).line_km.sum()
    rows = []

    def add(cls, system, g, plants, qn, thr):
        def sel(d):
            return d[(d.system == system) & ((d.geo == g) | (g == "all"))]

        line_km = float(sel(ar).line_km.sum())
        n = int((sel(nu).score >= thr).sum())
        ta_km = float(ta[ta.index.get_level_values("system") == system].sum() if g == "all" else ta.get((system, g), 0.0))
        dens = n / line_km * 1000 if line_km > 0 and thr > 0 else np.nan
        rows.append(dict(cls=cls, system=system, geo=g, plants=plants, plant_score=qn, threshold=round(thr, 2),
                         control_line_km=round(line_km, 1), control_clusters=n,
                         lookalikes_per_1000_line_km=round(dens, 2) if np.isfinite(dens) else np.nan,
                         track_a_line_km=round(ta_km, 1),
                         lookalikes_in_track_a=round(dens * ta_km / 1000, 1) if np.isfinite(dens) else np.nan))

    tr = tr.dropna(subset=["system"])
    for (cls, system, g), t in tr.groupby(["cls", "system", "geo"]):
        for qn, q in (("q50", 0.5), ("q25", 0.25)):
            add(cls, system, g, len(t), qn, float(t.score.fillna(0.0).quantile(q)))
    for (cls, system), t in tr.groupby(["cls", "system"]):
        thr = float(t.score.fillna(0.0).quantile(0.5))
        for g in sorted(set(ar[ar.system == system].geo)) + ["all"]:
            add(cls, system, g, len(t) if g == "all" else int((t.geo == g).sum()), "q50_pooled", thr)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "p3_lookalike.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
