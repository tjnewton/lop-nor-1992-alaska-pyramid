"""P3 a falsifier on the quiet open basins (Track B only; control data).

For each package in PACKAGES (open magnetics-only surveys over sediment basins):
1. Statistic, template-free: the peak |residual| (p3_mag_detect.regional_residual) per traverse line
   segment, at local maxima at least PEAK_SEP_M apart, edges trimmed (EDGE_M), declustered within
   DECLUSTER_M (p3_null_rates.decluster). Units are nT.
2. Tail: generalized Pareto fits (scipy, shape free in sign) to the top TAIL_FRACS of the cluster
   amplitudes; a negative shape gives a bounded tail. The shape falls as the threshold rises, so every
   threshold is tried and the smallest resulting cut is used (the choice most favorable to detection).
3. Cut: the smallest c whose expected count of null clusters above c over this survey's own flown
   area is at most this survey's share of the Track B budget. For an unbounded tail the cut is beyond the data.
4. Plants: N_PLANTS T2 bodies (p3_mag_inject; induced; base 300 m, top 38.1 m) per contrast, at the
   survey's own sensor heights and line spacing, placed as in p3_mag_trials. A plant is detected if the
   largest |residual| within HIT_R_M of it reaches c on the planted data and not on the unplanted data.
Writes p3_quiet_falsifier.csv (package, quantity, value). If P_d > 0 for some survey, the fixed "no
candidate can arise in Track B" is false there. Run: python -m akp.stage_p3.p3_quiet_falsifier
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import genpareto

from akp.stage_p3 import p3_mag_detect as det
from akp.stage_p3.p3_g1b_run import cache_dir, frame, spacing
from akp.stage_p3.p3_mag_inject import FT_M, plant_lines
from akp.stage_p3.p3_null_rates import BUDGET, decluster

HERE = Path(__file__).parent
PACKAGES = ("holitnamag", "bethel", "lower_ykd")
PEAK_SEP_M = 300.0
DECLUSTER_M = 300.0
TAIL_FRACS = (0.25, 0.10, 0.05, 0.02, 0.01)  # tail thresholds; the smallest resulting cut is used
N_PLANTS = 100
HIT_R_M = 250.0
DKS = {"T2_low": 0.0241, "T2_high": 0.0691}
SEED = 20261113


def amp_peaks(df: pd.DataFrame, mag: str):
    rows, km = [], 0.0
    keep = set(det.survey_lines(df))
    for ln, line in df[(df.kind == "line") & df.line.isin(keep)].groupby("line"):
        for grid, v, x, y in det.segments(line, mag):
            a = det.trim_edges(np.abs(det.regional_residual(v)))
            km += (grid[-1] - grid[0]) / 1e3
            pk, _ = find_peaks(np.where(np.isfinite(a), a, 0.0), height=1.0, distance=int(PEAK_SEP_M / det.DS_M))
            rows += [dict(line=ln, x=x[i], y=y[i], score=a[i]) for i in pk]
    return pd.DataFrame(rows), km


def max_amp_near(df, lines, mag, x0, y0) -> float:
    best = np.nan
    for ln in lines:
        line = df[(df.line == ln) & (df.kind == "line")]
        for grid, v, x, y in det.segments(line, mag):
            near = np.hypot(x - x0, y - y0) <= HIT_R_M
            if near.any():
                a = det.trim_edges(np.abs(det.regional_residual(v)))[near]
                if np.isfinite(a).any():
                    best = np.nanmax([best, float(np.nanmax(a))])
    return best


def main() -> None:
    ch = pd.read_csv(HERE / "p3_channels.csv").query("reader_ok == 'yes'")
    open_km2 = 0.0  # all open flown area, for this survey's budget share
    from akp.paths import data_dir
    for r in ch.itertuples():
        a = data_dir("p3/g1b") / Path(r.package).stem / Path(r.member).stem / "area.csv.gz"
        if a.exists():
            open_km2 += float(pd.read_csv(a).km2.sum())
    rows = []
    rng = np.random.default_rng(SEED)
    for key in PACKAGES:
        sel = ch[ch.package.str.contains(key)]
        if sel.empty:
            continue
        row = sel.iloc[0]
        mag, alt = row.mag.strip().upper(), row.alt.strip().upper()
        feet = str(row.alt_units).lower().startswith("f")
        sp = spacing(row)
        dfs = [frame(p, row) for p in sorted(cache_dir(row).glob("chunk*.npz"))]
        peaks, km = zip(*[amp_peaks(d, mag) for d in dfs])
        cl = decluster(pd.concat(peaks)) if sum(map(len, peaks)) else pd.DataFrame(columns=["score"])
        km2 = sum(km) * sp / 1e3
        x = np.sort(cl.score.to_numpy())[::-1]
        share = BUDGET * km2 / open_km2
        cuts = {}
        for frac in TAIL_FRACS:
            k = int(round(frac * len(x)))
            u = float(x[k])
            xi, _, sig = genpareto.fit(x[:k] - u, floc=0.0)
            endpoint = u + sig / -xi if xi < 0 else np.inf
            lo, hi = u, (endpoint if np.isfinite(endpoint) else u + 1e6 * sig)
            expected = lambda c: k * genpareto.sf(c - u, xi, scale=sig)
            if expected(hi) > share:
                cut = np.inf
            else:
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if expected(mid) > share else (lo, mid)
                cut = hi
            cuts[frac] = cut
            rows += [dict(package=row.package, quantity=f"tail_xi_top{frac:g}", value=round(float(xi), 4)),
                     dict(package=row.package, quantity=f"cut_nT_top{frac:g}", value=round(float(cut), 2))]
        cut = min(cuts.values())  # the tail choice most favorable to detection
        rows += [dict(package=row.package, quantity=q, value=v) for q, v in
                 (("clusters", len(cl)), ("flown_km2", round(km2, 1)), ("max_amp_nT", round(float(x[0]), 2)),
                  ("budget_share", round(share, 6)), ("cut_nT_min", round(float(cut), 2)))]
        for cname, dk in DKS.items():
            hits, amps = [], []
            for _ in range(N_PLANTS):
                d = dfs[int(rng.integers(len(dfs)))]
                keep = set(det.survey_lines(d))
                lines = d[(d.kind == "line") & d[mag].notna() & d.line.isin(keep)].reset_index(drop=True)
                if len(lines) < 3:
                    continue
                i = int(rng.integers(1, len(lines) - 1))
                a, b = lines.iloc[i - 1], lines.iloc[i + 1]
                hd = np.arctan2(b.Y - a.Y, b.X - a.X)
                uu = float(rng.uniform(-sp / 2, sp / 2))
                x0, y0 = lines.X.iloc[i] - uu * np.sin(hd), lines.Y.iloc[i] + uu * np.cos(hd)
                out, near = plant_lines(d, mag, alt, feet, x0, y0, 300.0, 38.1, float(rng.uniform(0, 90)), dk)
                touched = [ln for ln in d.line.iloc[near].unique() if ln in keep]
                s = max_amp_near(out, touched, mag, x0, y0) if touched else np.nan
                s0 = max_amp_near(d, touched, mag, x0, y0) if touched else np.nan
                hits.append(bool(np.isfinite(cut) and np.isfinite(s) and s >= cut and not (np.isfinite(s0) and s0 >= cut)))
                amps.append(s)
            rows.append(dict(package=row.package, quantity=f"pd_{cname}", value=round(float(np.mean(hits)) if hits else np.nan, 4)))
            rows.append(dict(package=row.package, quantity=f"plants_{cname}", value=len(hits)))
            a = np.array([v for v in amps if np.isfinite(v)])
            for q in (0.5, 0.9, 1.0):
                rows.append(dict(package=row.package, quantity=f"plant_amp_nT_q{q:g}_{cname}", value=round(float(np.quantile(a, q)), 2) if a.size else np.nan))
        print(pd.DataFrame(rows).tail(10).to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv(HERE / "p3_quiet_falsifier.csv", index=False)


if __name__ == "__main__":
    main()
