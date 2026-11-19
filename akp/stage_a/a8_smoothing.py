"""A8 smoothing check: the scattered share measured the way the clutter is measured.

The production table (a8_production.csv) gives the scattered share as a peak of an unsmoothed envelope
over the unsmoothed peak of the incident P. The clutter of Section 4 (a2_wavefield.py, used by
a5_dprk_network.py) is taken from a Hilbert envelope smoothed by a moving average of a2.SMOOTH_S, over
the peak of that smoothed envelope within a2.PEAK of the predicted P. This module recomputes the share
with A2's envelope on both sides, from the saved traces of the production runs. No new simulation.

Rules (fixed before any number):
- envelope: |band_envelope| of a8_production (same filter), then A2's moving average of a2.SMOOTH_S
  ("same" convolution, as A2);
- P peak: the largest smoothed value of the twin run's vertical trace within a2.PEAK of the incident
  arrival at the receiver;
- scattered peak: the largest smoothed value of the scattered vertical trace (body less twin) within
  the production measurement's slow window, unchanged;
- the unsmoothed share is recomputed beside it and must equal the committed table (REPRO_TOL), or the
  module stops.
Outputs: a8_smoothing.csv (case, azimuth, range_m, band; share and share_smoothed; factor = their
ratio; p_peak_kept and scattered_peak_kept, the smoothed peak over the unsmoothed peak of each) and
a8_smoothing_checks.csv (the statements against the clutter of a5_dprk_site_spread.csv).
Self-check: smoothing leaves a constant envelope unchanged away from the ends and lowers the peak of a
packet shorter than the window.
Run: PYTHONPATH=. python scripts/mem_guard.py --frac 0.40 --every 0.5 -- python -m akp.stage_a.a8_smoothing
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from akp.stage_a import a2_wavefield as a2
from akp.stage_a import a8_production as prod

HERE = Path(__file__).parent
REPRO_TOL = 1e-9  # relative: the recomputed unsmoothed share against the committed one
FAR_M = 1000.0  # The statement concerns ranges of 1 km or more
CLUTTER_LAG = "5-15"  # s after P: the first coda window of Section 4, the one the share is set against


def clutter() -> pd.DataFrame:
    """The clutter at in-box sites by band (median, p05 and the rest), over the three later tests."""
    return pd.read_csv(HERE / "a5_dprk_site_spread.csv").query("box == 'mckinley_nome' and event == 'all' and lag == @CLUTTER_LAG").set_index("band")


def smooth(env: np.ndarray, dt: float) -> np.ndarray:
    """A2's moving average of SMOOTH_S along the time axis (axis 0)."""
    n = max(1, int(round(a2.SMOOTH_S / dt)))
    k = np.ones(n) / n
    return np.apply_along_axis(lambda x: np.convolve(x, k, "same"), 0, env)


def measure(case: str, spec=None, runs=None) -> list[dict]:
    """spec and runs = (twin, body) as a8_production.measure takes them: a case outside the production list."""
    s = prod.setup(case, spec)
    twin, body = runs or (prod.load(case, False), prod.load(case, True))
    dt, nt = s["dt"], s["nt"]
    t = np.arange(nt) * dt
    n = len(s["ranges"])
    sc = body["vz"].astype(float) - twin["vz"].astype(float)
    rows = []
    for band in s["bands"]:
        period = 1 / np.mean(prod.BANDS[band])
        inc, ver = np.abs(prod.band_envelope(twin["vz"].astype(float), band, dt)), np.abs(prod.band_envelope(sc, band, dt))
        inc_s, ver_s = smooth(inc, dt), smooth(ver, dt)
        for a, (az, (ux, _)) in enumerate(prod.AZIMUTHS.items()):
            for i, r in enumerate(s["ranges"]):
                j = a * n + i
                t_inc = s["t_surf"] + s["p"] * r * ux
                lo, hi = s["t0"] + r / s["vs_fast"] - period / 2, s["t0"] + r / (0.85 * s["vs_slow"]) + 1.5 * period  # the slow window
                p_raw = prod.peak(inc[:, j], t, t_inc - 1 / prod.F0 - period, t_inc + 1 / prod.F0 + period)
                p_smooth = prod.peak(inc_s[:, j], t, t_inc + a2.PEAK[0], t_inc + a2.PEAK[1])
                v_raw, v_smooth = prod.peak(ver[:, j], t, lo, hi), prod.peak(ver_s[:, j], t, lo, hi)
                rows.append(dict(case=case, azimuth=az, range_m=float(r), band=band, share=v_raw / p_raw, share_smoothed=v_smooth / p_smooth,
                                 factor=(v_smooth / p_smooth) / (v_raw / p_raw), p_peak_kept=p_smooth / p_raw, scattered_peak_kept=v_smooth / v_raw))
    return rows


def main() -> None:
    # self-check of the smoothing
    dt = 0.01
    flat = smooth(np.ones((1000, 1)), dt)
    assert abs(flat[500, 0] - 1.0) < 1e-12, "a constant envelope must stay constant"
    spike = np.zeros((1000, 1))
    spike[480:520] = 1.0  # a packet of 0.4 s under a window of 1 s
    assert abs(smooth(spike, dt).max() - 0.4) < 1e-9, "a short packet's peak must fall by its length over the window"

    out = pd.DataFrame([row for case in prod.CASES for row in measure(case)])
    com = pd.read_csv(HERE / "a8_production.csv").query("window == 'slow' and component == 'vertical'")
    m = out.merge(com[["case", "azimuth", "range_m", "band", "numerical"]], on=["case", "azimuth", "range_m", "band"], validate="one_to_one")
    assert len(m) == len(out) == len(com), "every committed row must be recomputed"
    repro = float((m.share / m.numerical - 1).abs().max())
    assert repro <= REPRO_TOL, f"the unsmoothed share does not reproduce the committed table ({repro:.2e})"
    out.round(6).to_csv(HERE / "a8_smoothing.csv", index=False)

    clut = clutter()
    f = out[out.azimuth == "forward"]
    checks = [dict(check="rows", value=len(out)), dict(check="unsmoothed_share_max_relative_difference_from_committed", value=repro),
              dict(check="factor_min", value=round(float(out.factor.min()), 4)), dict(check="factor_median", value=round(float(out.factor.median()), 4)),
              dict(check="factor_max", value=round(float(out.factor.max()), 4))]
    worst = None
    for (case, band), g in f.groupby(["case", "band"]):
        far = g[g.range_m >= FAR_M]
        med, p05 = float(clut.loc[band, "median"]), float(clut.loc[band, "p05"])
        over = float((far.share_smoothed / med).max())
        worst = over if worst is None else max(worst, over)
        reach = g[g.share_smoothed >= p05].range_m
        reach0 = g[g.share >= p05].range_m
        for rng in (1000.0, 3000.0):
            row = g[g.range_m == rng].iloc[0]
            checks += [dict(check=f"share_smoothed_{case}_{band}_{rng:.0f}m", value=round(float(row.share_smoothed), 4)),
                       dict(check=f"factor_{case}_{band}_{rng:.0f}m", value=round(float(row.factor), 3))]
        checks += [dict(check=f"p_peak_kept_{case}_{band}", value=round(float(g.p_peak_kept.median()), 3)),
                   dict(check=f"smoothed_share_over_median_clutter_max_beyond_1km_{case}_{band}", value=round(over, 3)),
                   dict(check=f"range_reaching_cleanest_twentieth_km_smoothed_{case}_{band}", value=float(reach.max()) / 1000 if len(reach) else 0.0),
                   dict(check=f"range_reaching_cleanest_twentieth_km_committed_{case}_{band}", value=float(reach0.max()) / 1000 if len(reach0) else 0.0)]
    checks += [dict(check="smoothed_share_over_median_clutter_max_beyond_1km", value=round(worst, 3)),
               dict(check="statement_below_median_beyond_1km_holds_with_smoothing", value=bool(worst < 1.0))]
    pd.DataFrame(checks).to_csv(HERE / "a8_smoothing_checks.csv", index=False)
    print(pd.DataFrame(checks).to_string(index=False))


if __name__ == "__main__":
    main()
