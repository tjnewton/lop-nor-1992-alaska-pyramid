"""A5 late windows, as measured: every description of a5_late_windows that the graded-sediment case sets
through the upper end of its allowance for the source-edge artifact, recomputed with that case as measured.

Why: in a5_late_windows the graded-sediment case (R5) enters ranges and bounds by the upper end of its bracket.
That allowance assumes the artifact scatters as the plane wave does. The paper gives the value as measured beside
every number that the allowance sets. This module gives them, from the committed tables, with no new measurement.

Inputs (committed tables only; no data drive): a5_late_records.csv and a5_late_levels.csv (as written by
a5_late_windows), the simulation tables through a5_late_windows.simulated, and akp/stage_p4/s11_*.csv.
Method: the functions of a5_late_windows (starts, bound_at, by_band, thresholds, crossing), with every simulated
ratio replaced by its value as measured (ratio_measured: the larger of the two measures, with no allowance).
Output: a5_late_as_measured_checks.csv (check, value, note), in the same terms as a5_late_checks.csv.
Self-check: with the allowance put back, the module reproduces the committed a5_late_checks.csv values it shadows.
Run: PYTHONPATH=. python -m akp.stage_a.a5_late_as_measured
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from akp.stage_a import a5_late_windows as late

HERE = Path(__file__).parent
S11 = HERE.parent / "stage_p4"
FAR = "km > 20 and version not in ['disc_denali', 'belt_denali', 'valdez_town']"  # the versions whose nearest station is far (main text, question two)


def describe(sim: pd.DataFrame, rec: pd.DataFrame, lv: pd.DataFrame) -> dict:
    pairs, slow = late.bound_pairs(), late.slowest()
    st = late.starts(sim)
    reads = pd.DataFrame([dict(version=p.version, km=p.km, kind=p.kind, bound=late.bound_at(p.km * 1e3, st)[0]) for p in pairs.itertuples()])
    bb, _ = late.by_band(pairs, lv, rec, sim, slow)
    thw, th = late.thresholds(bb, float(st.k.max()))
    tw, to = thw[thw.kind != "open waveform"], th[th.kind == "open waveform"]
    names = sorted(set(pairs.station) | set(pd.read_csv(S11 / "s11_stations_inside.csv").code))
    cross = late.crossing(sim, lv, rec, [c for c in names if c in set(rec.sta)])
    med = cross[cross.level == "pooled median, way valid"]
    s11 = pd.read_csv(S11 / "s11_ranges.csv").set_index("quantity")
    return {
        "bound_at_committed_start_range": late.bound_at(float(s11.range_m["bound_start"]), st)[0],
        "bound_min": float(reads.bound.min()),
        "bound_max": float(reads.bound.max()),
        "bound_far_min": float(reads.query(FAR).bound.min()),
        "bound_far_max": float(reads.query(FAR).bound.max()),
        "open_waveform_bound_min": float(to.bound_band.min()),
        "open_waveform_bound_max": float(to.bound_band.max()),
        "open_waveform_q_least_min": float(to.q_least.min()),
        "crossing_committed_median_max_km": float(med.range_m.max()) / 1e3,
        "crossing_high_snr_median_max_km": float(cross[cross.level == "pooled median, way high_snr"].range_m.max()) / 1e3,
        "crossing_p05_spreading_range_min_km": float(cross[cross.level.str.startswith("pooled fifth")].spreading_range_m.min()) / 1e3,
        "crossing_p05_spreading_range_max_km": float(cross[cross.level.str.startswith("pooled fifth")].spreading_range_m.max()) / 1e3,
        "crossing_col1992_spreading_range_max_km": float(cross[cross.level == "1992 record at IU.COL"].spreading_range_m.max()) / 1e3,
        "threshold_q_least_min": float(tw.q_least.min()),
        "threshold_speed_at_least_q": float(tw.speed_below_kms[tw.q_least.idxmin()]),
        "threshold_q_least_for_faster_than_1_kms": float(tw[tw.speed_below_kms >= 1.0].q_least.min()),
        "threshold_speed_below_kms_max": float(tw.speed_below_kms.max()),
    }


def main() -> None:
    rec = pd.read_csv(HERE / "a5_late_records.csv")
    lv = pd.read_csv(HERE / "a5_late_levels.csv")
    sim = late.simulated()
    # self-check: with the allowance, the committed descriptions come back
    committed = pd.read_csv(HERE / "a5_late_checks.csv").set_index("check").value
    upper = describe(sim, rec, lv)
    for k in ("bound_at_committed_start_range", "bound_min", "bound_max", "crossing_committed_median_max_km", "threshold_q_least_min",
              "threshold_q_least_for_faster_than_1_kms", "threshold_speed_below_kms_max", "open_waveform_bound_min", "open_waveform_bound_max",
              "open_waveform_q_least_min"):
        tol = 1e-3 * abs(upper[k]) if "q_least" in k else 1e-4 * max(1.0, abs(upper[k]))  # the committed tables are rounded to 6 decimals; Q = pi f t / ln(bound / level)
        assert abs(upper[k] - float(committed[k])) <= tol, (k, upper[k], committed[k])  # magnifies that rounding where the bound is close to the level
    far = pd.read_csv(HERE / "a5_late_readings.csv").query(FAR).bound
    assert abs(upper["bound_far_min"] - far.min()) <= 1e-6 and abs(upper["bound_far_max"] - far.max()) <= 1e-6, "far bounds as in a5_late_readings.csv"
    meas = describe(sim.assign(ratio=sim.ratio_measured), rec, lv)
    rows = [dict(check=k, value=v, note="graded-sediment case as measured (no allowance)") for k, v in meas.items()]
    rows += [dict(check=f"{k}_with_allowance", value=v, note="as in a5_late_checks.csv or a5_late_crossing.csv") for k, v in upper.items()]
    pd.DataFrame(rows).to_csv(HERE / "a5_late_as_measured_checks.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
