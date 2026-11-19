"""P3 M3/G1b EM: T3 injection trials on one open (control) package, for the multi-channel detector.

scale multiplies the modeled change (the median 3D/1D ratio gives the central case).
Plants are drawn as in p3_mag_trials: a random traverse-line sample, shifted across the local
heading by u ~ U(-spacing/2, spacing/2). Each plant gets the coplanar in-phase and quadrature
change of p3_em_model.delta_ppm at every sample within reach, in the survey frame, at each
sample's own bird height. The body is axis-aligned in that frame. The touched traverse lines are
rescored with p3_em_detect, and the trial keeps the largest score within HIT_R_M of the plant.
Misses are kept: no valid sample nearby gives NaN. trials() returns one row per plant.
Self-check: python -m akp.stage_p3.p3_em_trials (a small Iron Creek run).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from akp.stage_p3 import p3_em_detect as edet
from akp.stage_p3 import p3_mag_detect as mdet
from akp.stage_p3.p3_em_model import delta_ppm
from akp.stage_p3.p3_mag_inject import clean_heights
from akp.stage_p3.p3_mag_trials import HIT_R_M, draw


def plant(df: pd.DataFrame, chans: list[str], freqs, sep, alt: str, alt_in_feet: bool, x0, y0, base, top, rho_b, rho_h, scale=1.0):
    reach = 0.5 * base * np.sqrt(2) + 150.0
    near = np.flatnonzero(np.hypot(df.X.to_numpy() - x0, df.Y.to_numpy() - y0) <= reach)
    out = df.copy()
    if near.size:
        h = clean_heights(df[alt].to_numpy()[near] * (mdet.FT_M if alt_in_feet else 1.0))
        d = scale * delta_ppm(df.X.to_numpy()[near] - x0, df.Y.to_numpy()[near] - y0, h, base, top, rho_b, rho_h, freqs, sep)
        for k, c in enumerate(chans):  # chans ordered f0 in-phase, f0 quadrature, f1 in-phase, ...
            col = out.columns.get_loc(c)
            out.iloc[near, col] = out.iloc[near, col].to_numpy() + d[k // 2, k % 2]
    return out, near


def max_score_near(df, lines, chans, freqs, sep, alt, alt_in_feet, x0, y0) -> float:
    best = np.nan
    for ln in lines:
        line = df[(df.line == ln) & (df.kind == "line")]
        h = float(np.nanmedian(line[alt])) * (mdet.FT_M if alt_in_feet else 1.0)
        tbank = edet.templates(round(h / 5.0) * 5.0, tuple(freqs), float(sep))
        for grid, vv, x, y in edet.segments(line, chans):
            near = np.hypot(x - x0, y - y0) <= HIT_R_M
            if near.any():
                sc, _ = edet.score(vv, tbank)
                sc = mdet.trim_edges(sc)
                if not np.isfinite(sc[near]).any():
                    continue
                best = float(np.max(sc[near])) if not np.isfinite(best) else max(best, float(np.max(sc[near])))
    return best


def trials(df, chans, freqs, sep, alt, alt_in_feet, spacing, n, seed, base, top, rho_b, rho_h, scale=1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    keep = set(mdet.survey_lines(df))
    lines = df[(df.kind == "line") & df.line.isin(keep) & df[chans].notna().all(axis=1)].reset_index(drop=True)
    rows = []
    if len(lines) < 3:  # too few traverse samples in this chunk to place a plant
        return pd.DataFrame(rows)
    for _ in range(n):
        i = int(rng.integers(1, len(lines) - 1))
        a, b = lines.iloc[i - 1], lines.iloc[i + 1]
        hd = np.arctan2(b.Y - a.Y, b.X - a.X)
        u = float(rng.uniform(-spacing / 2, spacing / 2))
        x0, y0 = lines.X.iloc[i] - u * np.sin(hd), lines.Y.iloc[i] + u * np.cos(hd)
        b, t = draw(rng, base), draw(rng, top)  # a (low, high) range is drawn per plant
        out, near = plant(df, chans, freqs, sep, alt, alt_in_feet, x0, y0, b, t, rho_b, rho_h, scale)
        touched = [ln for ln in df.line.iloc[near].unique() if ln in keep] if near.size else []
        s = max_score_near(out, touched, chans, freqs, sep, alt, alt_in_feet, x0, y0) if touched else np.nan
        s_bg = max_score_near(df, touched, chans, freqs, sep, alt, alt_in_feet, x0, y0) if touched else np.nan
        rows.append(dict(x0=x0, y0=y0, u=u, base=b, top=t, score=s, score_bg=s_bg, n_samples=int(near.size)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import time

    from akp.paths import data_dir, data_glob
    from akp.stage_p3.p3_lines import read_package

    pkg = [p for p in data_glob(data_dir("p3/lines/open"), "*.zip") if "iron-creek" in p.name][0]
    _, df = next(read_package(pkg))
    chans = ["CPI900", "CPQ900", "CPI7200", "CPQ7200", "CPI56K", "CPQ56K"]
    t0 = time.time()
    tr = trials(df, chans, (900.0, 7200.0, 56000.0), 8.0, "ALTR", True, 400.0, 8, 1, 300.0, 38.1, 10.0, 1000.0)
    print(tr.round(2).to_string(index=False))
    print(f"{time.time() - t0:.0f} s")
    assert tr.score.notna().sum() >= 6
    print("p3_em_trials self-check OK")
