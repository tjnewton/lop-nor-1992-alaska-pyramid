"""A5 late windows against a measured regional Q: would a wave with the attenuation measured in the region stand above the 1992 record?

Why: the thresholds of a5_late_windows give, for each bound, band and window, the least Q with which the bound still
reaches the coda amplitude of the 1992 record at IU.COL. The paper had compared them only with the near-surface Q of its
sources (at most 100). For a fast crustal wave the relevant value is the regional Q of crustal waves: McNamara (2000),
Lg in south-central Alaska, Q(f) = 220 (+-30) f**0.66 (+-0.09) for 0.75 to 12 Hz.

Method: for every row of a5_late_threshold_windows.csv (bound, band, window, least Q, the speed below which the wave must
travel), the regional Q at the lower edge of the band, as the thresholds themselves take the frequency, with the central
values and with the two ends of the stated uncertainties taken together (lowest and highest Q). A row "stands" if the
regional Q is at or above its least Q. Rows are counted for waves faster than 1 km/s, the crustal waves of the argument.

Output: a5_late_regional_q.csv (per row) and a5_late_regional_q_checks.csv. Run: PYTHONPATH=. python akp/stage_a/a5_late_regional_q.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
Q0, ETA, DQ0, DETA = 220.0, 0.66, 30.0, 0.09  # McNamara (2000), Geophysical Research Letters 27(23), 3949-3952
LOW = {"0.5-1": 0.5, "1-2": 1.0, "2-4": 2.0}  # Hz, the lower edge of each band, as in a5_late_windows.thresholds


def regional_q(f: float, q0: float = Q0, eta: float = ETA) -> float:
    return q0 * f ** eta


def main() -> None:
    t = pd.read_csv(HERE / "a5_late_threshold_windows.csv", dtype={"lag": str})
    t = t[t.kind != "open waveform"].copy()
    f = t.band.map(LOW)
    t["q_regional"] = [regional_q(x) for x in f]
    # f >= 1 at the bands that count, so the lowest Q takes both lower ends and the highest both upper ends; at 0.5 Hz the
    # exponent works the other way, so each row takes the extreme of the four combinations
    combos = [(Q0 + a, ETA + b) for a in (-DQ0, DQ0) for b in (-DETA, DETA)]
    t["q_regional_low"] = [min(regional_q(x, q, e) for q, e in combos) for x in f]
    t["q_regional_high"] = [max(regional_q(x, q, e) for q, e in combos) for x in f]
    t["stands"] = t.q_regional >= t.q_least
    t["stands_high"] = t.q_regional_high >= t.q_least
    t["stands_low"] = t.q_regional_low >= t.q_least
    t["margin"] = t.q_regional / t.q_least - 1  # how far the regional Q clears the least Q (negative: it does not)
    t.round(4).to_csv(HERE / "a5_late_regional_q.csv", index=False)
    fast = t[t.speed_below_kms >= 1.0]
    def sts(g): return " ".join(sorted(set(g.station)))
    checks = [dict(check="regional_q_at_1_hz", value=regional_q(1.0)), dict(check="regional_q_at_2_hz", value=regional_q(2.0)),
              dict(check="fast_rows", value=len(fast)),
              dict(check="fast_rows_standing_central", value=int(fast.stands.sum()), note=sts(fast[fast.stands])),
              dict(check="fast_rows_standing_high", value=int(fast.stands_high.sum()), note=sts(fast[fast.stands_high])),
              dict(check="fast_rows_standing_low", value=int(fast.stands_low.sum()), note=sts(fast[fast.stands_low])),
              dict(check="fast_standing_margin_percent", value=float(100 * fast[fast.stands].margin.max()), note="the largest margin among the rows that stand"),
              dict(check="fast_standing_q_least", value=float(fast[fast.stands].q_least.min()), note=""),
              dict(check="fast_standing_q_regional", value=float(fast[fast.stands].q_regional.max()), note=""),
              dict(check="fast_bounds_standing_central", value=int(fast[fast.stands].groupby(["version", "kind"]).ngroups)),
              dict(check="fast_bounds_standing_high", value=int(fast[fast.stands_high].groupby(["version", "kind"]).ngroups)),
              dict(check="bounds", value=int(t.groupby(["version", "kind"]).ngroups))]
    pd.DataFrame(checks).to_csv(HERE / "a5_late_regional_q_checks.csv", index=False)
    print(pd.DataFrame(checks).to_string(index=False))


def _selfcheck() -> None:
    assert abs(regional_q(1.0) - 220.0) < 1e-9 and abs(regional_q(2.0) - 220.0 * 2 ** 0.66) < 1e-9


if __name__ == "__main__":
    _selfcheck()
    main()
