"""P4 S10: the census rows for the simulations at the generous end.

Question: for each new simulated case, at what range does the scattered wave reach the clutter levels
of the table, how much of the prior lies within that range of any receiver, and does the
reopening test fire on those rows? And how does the simulated largest base compare with the
two base factors of the census?

Rules implemented here, none changed:
- The computation (s9_census.sensitivity), the four clutter levels, the four priors, exposure
  with position uncertainty, the reopening test and its threshold;
- "Census": the committed table stays. Rows for the new cases are added beside it by
  the same computation and the reopening test is applied to them. The ratio of R3 to the production
  run is set beside the base factors of the census;
- "Census": the unsmoothed share, no attenuation and the generous Q. The outcome of the
  test is reported; what follows was fixed in advance;
- A case simulated at the production base gets four variants of the body: as simulated; with
  the base effect R3 measured (primary); with each of the two assumed factors. A case
  simulated at the largest base takes no factor. The reopening test is applied to every variant.

Inputs:
- a8_generous.csv and a8_generous_attenuation.csv (the new cases, in the columns of a8_production.csv
  and a8_attenuation.csv); whatever cases they hold are taken and named in s10_checks.csv;
- a8_generous_shares.csv and a8_smoothing.csv: the same unsmoothed share by range (forward azimuth),
  to six decimals, with the edge-wave flag of the new cases;
- a8_generous.CASES and a8_production.BODY: the base of each case's body, by import (the tables do
  not hold it);
- everything s9_census reads: the station lists, the prior, the clutter tables;
- the committed s9_sensitivity.csv, s9_detection_range.csv and s9_checks.csv, for the self-check.

Method:
- s9_census is used by import and not changed. Its functions read fixed file names and module
  constants, so they are wrapped two ways. `read_as_production` points s9_census.STAGE_A at a
  temporary folder of links in which the given tables stand under the production names.
  `case_rows` sets s9_census.CASE, BAND, F_HZ and MAX_BASE_M for the call. All are put back afterwards.
- Band of a case: the band with the largest table-read range against the cleanest twentieth, elastic
  (s9_census.detection_ranges, the rule by which the production case and band were named).
  Bands that tie are all computed.
- Rows of a case: s9_census.sensitivity gives the range from a power law fitted between FIT_KM and
  extended beyond the last receiver, as for the production case. Kept: the attenuations of ATTENUATIONS.
- Base ratio: the share of BASE_CASE over the share of the production case (s9_census.CASE, the same
  ground) at each range and band, beside the ratio of the bases and its square. The measured base
  effect of a band is the median of that ratio inside FIT_KM, over the rows without the edge-wave flag.
- Body variants, column `body_variant`. The base of a case is read from its spec. At the
  largest base: `as_simulated` only. At the production base: `as_simulated`; `assumed_linear` and
  `assumed_squared`, the two rows s9_census.sensitivity makes from its own constants; `measured_base`,
  the same linear row with s9_census.MAX_BASE_M set so that its factor is the measured base effect.
  The effect is taken in the case's own band. If that is not the census band, `note` says so. If
  BASE_CASE has no unflagged ratio in that band inside FIT_KM, the variant is left out and `note`
  says "not available". `primary` is true for `measured_base` where it exists, else for `as_simulated`.
- Beside each range, the table's own reading: the largest tabulated range at which the share,
  scaled and attenuated as the row is, is at least the clutter level (factor and level as the table
  prints them). `at_least` is true
  when the share is still at or above the level at the last receiver. The range of such a row lies
  beyond the model and comes from the fit alone.
- Exposure: s9_census.receivers and s9_census.distances, on the sub-cell points of s9_census.main.
  main writes the committed tables and so cannot be called. Its lines that build the points and the
  exposure are repeated here, in chunks of cells to bound memory.
- Reopening test, per case and body variant: the exposure of all receivers, with position
  uncertainty, at the largest range among the rows with the generous Q, under each prior, against
  REOPEN.

Outputs, beside this module: s10_sensitivity.csv (the columns of s9_sensitivity.csv with `body` as
`body_variant`, and case, band, the number of flagged tabulated rows, the table reading, `primary`
and `note`); s10_reopening.csv (one row per case, body variant and prior); s10_base_ratio.csv;
s10_checks.csv.

Run: PYTHONPATH=. python scripts/mem_guard.py --gb 2 --every 0.5 -- python -m akp.stage_p4.s10_census_generous

Self-check (asserted on every run, before any new row): on the production case this module's rows
equal the committed s9_sensitivity.csv exactly; its table reading equals s9_detection_range.csv for
the elastic and the attenuated table; its reopening test returns the exposures and the outcome in
s9_checks.csv. This proves that the wrapping, the chunked points and the inputs give the committed
numbers. It does not test s9_census.sensitivity itself, which is the code that runs.
For each case at the production base, a factor of exactly one gives the rows of `as_simulated`. This
proves that the measured variant differs from the simulated one by its factor alone.
"""
from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from affine import Affine

from akp.stage_a import a8_generous as gen
from akp.stage_a.a8_attenuation import CENTER_HZ
from akp.stage_p4 import s9_census as s9

HERE = Path(__file__).parent
STAGE_A = s9.STAGE_A
VARIANT = {"as_simulated": "as_simulated", "largest_base_linear": "assumed_linear", "largest_base_squared": "assumed_squared"}  # s9_census's bodies, the fixed names
SAME = ["clutter_level", "attenuation", "range_km", "table_read_km", "at_least"]  # what a factor of one must leave as simulated
ATTENUATIONS = ("none", "generous")
Q = {"none": np.inf, "generous": s9.Q_GENEROUS, "upper_sediment": s9.Q_UPPER}  # the attenuation names of s9_census.sensitivity
BASE_CASE = "R3"  # Base MAX_BASE_M in the production ground, at the production depth and contrast
CHUNK = 2000  # cells per pass of the distance search (memory: measured peak falls with the chunk)


@contextmanager
def read_as_production(table: str, attenuation: str):
    """Inside, s9_census reads `table` as a8_production.csv and `attenuation` as a8_attenuation.csv."""
    with tempfile.TemporaryDirectory() as tmp:
        for name, src in (("a8_production.csv", table), ("a8_attenuation.csv", attenuation),
                          ("a5_dprk_site_spread.csv",) * 2, ("a5_lopnor_repeatable.csv",) * 2):
            (Path(tmp) / name).symlink_to(STAGE_A / src)
        with mock.patch.object(s9, "STAGE_A", Path(tmp)):
            yield


def forward(shares: str) -> pd.DataFrame:
    s = pd.read_csv(STAGE_A / shares)
    return s[s.azimuth == "forward"]


def case_rows(case: str, band: str, g: pd.DataFrame, base_m: float = s9.MAX_BASE_M) -> tuple[pd.DataFrame, dict]:
    """s9_census.sensitivity for one case and band, with the table's own reading of `g` (its shares by range) beside each row."""
    f = CENTER_HZ[band]
    with mock.patch.multiple(s9, CASE=case, BAND=band, F_HZ=f, MAX_BASE_M=base_m):
        sens, info = s9.sensitivity()
    read = []
    for row in sens.to_dict("records"):
        share = g.share * row["body_factor"] * np.exp(-np.pi * f * g.range_m / (Q[row["attenuation"]] * info["rayleigh_speed_ms"]))
        ok = g.range_m[share >= row["clutter"]]
        read.append(dict(table_read_km=float(ok.max()) / 1000 if len(ok) else 0.0, last_receiver_km=float(g.range_m.max()) / 1000,
                         at_least=bool(len(ok) and ok.max() == g.range_m.max())))
    head = pd.DataFrame(dict(case=case, band=band, flagged_rows=int(g.edge_flag.sum()) if "edge_flag" in g else 0), index=sens.index)
    return pd.concat([head, sens, pd.DataFrame(read)], axis=1), info


def scaled(case: str, band: str, g: pd.DataFrame, k: float) -> pd.DataFrame:
    """The rows of a case with its share times k: s9_census's row for the ratio of bases, with the largest base set to k times the simulated one."""
    rows = case_rows(case, band, g, s9.SIM_BASE_M * k)[0]
    return rows[rows.body == "largest_base_linear"].assign(body="measured_base")


def points() -> tuple[dict, dict, float]:
    """Distances (km) of the sub-cell points of Track A to the nearest receiver by class, as s9_census.distances
    returns them; the points' masses under the four priors; the area of a point (km2). The lines of s9_census.main."""
    pr, cv = np.load(s9.prior_dir() / "p3_prior.npz"), np.load(s9.prior_dir() / "p3_cover.npz")
    ta = pr["track_a"]
    tf = Affine(*pr["transform"])
    rows, cols = np.nonzero(pr["land"])
    rows, cols = rows[ta], cols[ta]
    off = (np.arange(s9.SUB) + 0.5) / s9.SUB
    ox, oy = np.meshgrid(off, off)
    rec = s9.receivers()[0]
    recs = {"passive": rec[rec.cls == "passive"], "all": rec}
    n = s9.SUB**2
    dist = {cls: (np.empty(len(rows) * n, np.float32), np.empty(len(rows) * n, np.float32)) for cls in recs}
    for i in range(0, len(rows), CHUNK):
        c, r = cols[i:i + CHUNK], rows[i:i + CHUNK]
        px, py = tf * ((c[:, None] + ox.ravel()[None, :]).ravel(), (r[:, None] + oy.ravel()[None, :]).ravel())
        for cls, g in recs.items():
            for out, d in zip(dist[cls], s9.distances(px, py, g)):
                out[i * n:(i + len(c)) * n] = d
    mass = {"prior": pr["mass"][ta], "cover": cv["mass_cover"][ta], "equal": pr["mass_equal"][ta], "equal_cover": cv["mass_equal_cover"][ta]}
    return dist, {k: np.repeat(v / n, n) for k, v in mass.items()}, (abs(tf.a) / s9.SUB / 1000) ** 2


def reopening(rows: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """The fixed test for each group of `rows`: all receivers, uncertainty added, at the group's largest range with the generous Q."""
    g = rows[(rows.attenuation == "generous") & (rows.cls == "all") & (rows.uncertainty == "added")]
    top = g.loc[g.groupby(by).range_km.idxmax()].drop(columns=["attenuation", "cls", "uncertainty", "area_km2", "flagged_rows", "last_receiver_km"])
    t = top.melt(id_vars=[c for c in top.columns if c not in s9.PRIORS], value_vars=list(s9.PRIORS), var_name="prior", value_name="exposure")
    return t.assign(threshold=s9.REOPEN, excess=(t.exposure - s9.REOPEN).round(6), fires=t.exposure > s9.REOPEN).sort_values(by, kind="stable")


def pick(d: pd.DataFrame, name: str) -> pd.DataFrame:
    return d[d.primary] if name == "primary" else d[d.body_variant == name]


def main() -> None:
    dist, pmass, area = points()

    def exposed(sens: pd.DataFrame) -> pd.DataFrame:  # the exposure of s9_census.main, in its order of rows
        out = []
        for row in sens.to_dict("records"):
            for cls, pair in dist.items():
                for name, d in zip(("no", "added"), pair):
                    w = d <= row["range_km"]
                    out.append(dict(**row, cls=cls, uncertainty=name, area_km2=round(float(w.sum()) * area, 2),
                                    **{k: round(float(v[w].sum()), 6) for k, v in pmass.items()}))
        return pd.DataFrame(out)

    # self-check: the production case, against the committed tables
    old = forward("a8_smoothing.csv")
    with read_as_production("a8_production.csv", "a8_attenuation.csv"):
        sens, info0 = case_rows(s9.CASE, s9.BAND, old[(old.case == s9.CASE) & (old.band == s9.BAND)])
    prod = exposed(sens)
    committed = pd.read_csv(HERE / "s9_sensitivity.csv")
    pd.testing.assert_frame_equal(prod[committed.columns], committed, check_exact=True)
    det = pd.read_csv(HERE / "s9_detection_range.csv")
    det = det[(det.case == s9.CASE) & (det.band == s9.BAND) & (det.clutter_level == "p05")].set_index("table").range_m / 1000
    d83 = sens[(sens.clutter_level == "cleanest_twentieth") & (sens.body == "as_simulated")].set_index("attenuation").table_read_km
    assert d83["none"] == det["elastic"] and d83["upper_sediment"] == det["attenuated"], "the table reading differs from s9_detection_range.csv"
    was = pd.read_csv(HERE / "s9_checks.csv").set_index("check").value
    test = reopening(prod, ["case"])
    assert test.exposure.tolist() == [float(was[f"reopen_test_exposure_{k}"]) for k in s9.PRIORS], "the reopening exposures differ from s9_checks.csv"
    assert str(bool(test.fires.any())) == was["reopened"] and float(test.range_km.iloc[0]) == float(was["largest_range_generous_attenuation_km"])

    # the simulated largest base beside the base factors of the census, and the base effect it measures by band
    new = forward("a8_generous_shares.csv")
    k = s9.MAX_BASE_M / s9.SIM_BASE_M
    m = new[new.case == BASE_CASE].merge(old[old.case == s9.CASE], on=["band", "range_m"], suffixes=("", "_production"))
    ratio = pd.DataFrame(dict(case=BASE_CASE, production_case=s9.CASE, band=m.band, range_m=m.range_m, share=m.share, share_production=m.share_production,
                              ratio=(m.share / m.share_production).round(4), base_factor_linear=round(k, 3), base_factor_squared=round(k**2, 3), edge_flag=m.edge_flag))
    ratio.to_csv(HERE / "s10_base_ratio.csv", index=False)
    fit = ratio[ratio.range_m.between(s9.FIT_KM[0] * 1000, s9.FIT_KM[1] * 1000) & ~ratio.edge_flag]
    effect = fit.groupby("band").ratio.median().to_dict()

    # the new cases
    parts, info, failed, used, bases, notes = [], {}, {}, {}, {}, {}
    with read_as_production("a8_generous.csv", "a8_generous_attenuation.csv"):
        det = s9.detection_ranges().query("table == 'elastic' and clutter_level == 'p05'")
        top = det[det.range_m == det.groupby("case").range_m.transform("max")]
        for case, band in zip(top.case, top.band):
            key, g = f"{case}_{band}", new[(new.case == case) & (new.band == band)]
            try:
                rows, info[key] = case_rows(case, band, g)
            except (ValueError, TypeError) as e:  # s9_census's fit or root search cannot take this case: named in the checks
                failed[key] = f"{type(e).__name__}: {e}"
                continue
            rows, note, bases[case] = rows.assign(body=rows.body.map(VARIANT)), "", gen.CASES[case].get("body", gen.prod.BODY)["base"]
            if bases[case] == s9.MAX_BASE_M:  # Simulated at the largest base, no factor
                rows = rows[rows.body == "as_simulated"]
            else:
                assert bases[case] == s9.SIM_BASE_M, f"{case}: its base is neither the production base nor the largest (rules on these two)"
                assert scaled(case, band, g, 1.0)[SAME].reset_index(drop=True).equals(rows[rows.body == "as_simulated"][SAME].reset_index(drop=True)), "a factor of one changes the rows"
                if band in effect:
                    rows, used[key] = pd.concat([rows, scaled(case, band, g, effect[band])], ignore_index=True), float(effect[band])
                    note = "" if band == s9.BAND else f"base effect measured in this case's band, not in the census band {s9.BAND}"
                else:
                    note = f"measured_base not available: {BASE_CASE} has no ratio without the edge-wave flag in this band inside the fit range"
            primary, notes[key] = "measured_base" if key in used else "as_simulated", note
            parts.append(rows[rows.attenuation.isin(ATTENUATIONS)].assign(primary=lambda d, primary=primary: d.body == primary, note=note).rename(columns={"body": "body_variant"}))
    out = exposed(pd.concat(parts, ignore_index=True))
    out.to_csv(HERE / "s10_sensitivity.csv", index=False)
    re = reopening(out, ["case", "body_variant"])
    re.to_csv(HERE / "s10_reopening.csv", index=False)

    wide = out[(out.cls == "all") & (out.uncertainty == "added")]
    checks = [dict(check="cases_in_tables", value=" ".join(sorted(new.case.unique()))), dict(check="cases_with_rows", value=" ".join(sorted(out.case.unique()))),
              dict(check="cases_at_production_base", value=" ".join(c for c, b in bases.items() if b == s9.SIM_BASE_M), detail="four body variants"),
              dict(check="cases_at_largest_base", value=" ".join(c for c, b in bases.items() if b == s9.MAX_BASE_M), detail="as simulated only"),
              dict(check="rows", value=len(out)), dict(check="committed_rows_reproduced", value=len(prod)),
              dict(check="reopen_threshold", value=s9.REOPEN), dict(check="any_reopening_row_fires", value=bool(re.fires.any())),
              dict(check="reopening_rows_that_fire", value=int(re.fires.sum()), detail=" ".join(f"{c}:{v}:{p}" for c, v, p in zip(re.case[re.fires], re.body_variant[re.fires], re.prior[re.fires])))]
    checks += [dict(check=f"not_computed_{key}", value=why, detail="s9_census.sensitivity raised this") for key, why in failed.items()]
    checks += [dict(check=f"measured_base_effect_{key}", value=v, detail=f"median ratio of {BASE_CASE} to {s9.CASE} in the case's band, {s9.FIT_KM[0]} to {s9.FIT_KM[1]} km, unflagged rows")
               for key, v in used.items()]
    checks += [dict(check=f"measured_base_note_{key}", value=note) for key, note in notes.items() if note]
    for name in [*dict.fromkeys(out.body_variant), "primary"]:
        t = pick(re, name)
        checks.append(dict(check=f"reopening_fires_{name}", value=bool(t.fires.any()), detail=" ".join(f"{c}:{p}" for c, p in zip(t.case[t.fires], t.prior[t.fires]))))
        for att in ATTENUATIONS:
            g = pick(wide[wide.attenuation == att], name)
            for p in s9.PRIORS:
                r = g.loc[g[p].idxmax()]
                checks.append(dict(check=f"largest_exposure_{name}_{att}_{p}", value=r[p], detail=f"{r.case} {r.band} {r.body_variant} {r.clutter_level} at {r.range_km} km"))
    for key, v in info.items():
        checks += [dict(check=f"{name}_{key}", value=value) for name, value in v.items()]
    if len(ratio):
        assert all(v["rayleigh_speed_ms"] == info0["rayleigh_speed_ms"] for key, v in info.items() if key.startswith(BASE_CASE)), "the base case is not in the production ground"
        assert gen.CASES[BASE_CASE].get("body", gen.prod.BODY)["base"] == s9.MAX_BASE_M, "the base case does not have the largest base"
        checks += [dict(check="base_factor_linear", value=round(k, 3)), dict(check="base_factor_squared", value=round(k**2, 3))]
        fit = fit[fit.band == s9.BAND].ratio
        checks.append(dict(check=f"base_ratio_{s9.BAND}_fit_window_min_median_max", value=f"{fit.min():.3f} {fit.median():.3f} {fit.max():.3f}",
                           detail=f"{BASE_CASE} over {s9.CASE}, {s9.FIT_KM[0]} to {s9.FIT_KM[1]} km, unflagged rows"))
        for band, g in ratio.groupby("band"):
            checks.append(dict(check=f"base_ratio_{band}_min_median_max", value=f"{g.ratio.min():.3f} {g.ratio.median():.3f} {g.ratio.max():.3f}",
                               detail=f"all ranges; {int(g.edge_flag.sum())} of {len(g)} rows carry the edge-wave flag"))
    checks = pd.DataFrame(checks).reindex(columns=["check", "value", "detail"])
    checks.to_csv(HERE / "s10_checks.csv", index=False)
    print(checks.to_string(index=False))
    print(wide.drop(columns=["cls", "uncertainty", "body_factor", "flagged_rows"]).to_string(index=False))
    print(re.to_string(index=False))


if __name__ == "__main__":
    main()
