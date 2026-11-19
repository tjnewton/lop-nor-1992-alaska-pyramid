"""P3 M3/G1b magnetics: injection trials on one open (control) package.

Plants are uniform over the flown area. Each plant:
- picks a random line sample (uniform along the lines);
- shifts it across the local heading by u ~ U(-spacing/2, spacing/2);
- takes a free turn, a base and a top depth drawn uniformly from the given ranges (a single value
  is used as is), and the class's magnetization: induced dk, and for every second plant on average
  a remanence Q * dk in a uniform random direction, Q uniform in q_range. Plants with q = 0 model a
  built body, whose blocks' remanence averages out; plants with q > 0 model a monolith.
Its response is added (p3_mag_inject), the touched traverse lines (ties dropped, as in p3_mag_detect) are rescored (p3_mag_detect: segments, the
regional, the template bank, local noise), and the trial records the largest score within HIT_R_M
of the plant, and score_bg, the same maximum on the unplanted data (a hit must not be
background; P_d at a cut t counts score >= t with score_bg < t). Misses are kept: a plant with no valid samples nearby gets score NaN (counted as a
miss). The same seed gives the same plants. trials() returns one row per plant: x0, y0, turn, u,
score, n_samples. P_d at a cut t is mean(score >= t), with NaN as a miss.
Self-check: python -m akp.stage_p3.p3_mag_trials (a small run on Iron Creek, induced T2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from akp.stage_p3 import p3_mag_detect as det
from akp.stage_p3.p3_mag_inject import FT_M, plant_lines

HIT_R_M = 250.0


def max_score_near(df: pd.DataFrame, lines, mag: str, alt: str, alt_in_feet: bool, x0: float, y0: float) -> float:
    best = np.nan
    for ln in lines:
        line = df[(df.line == ln) & (df.kind == "line")]
        h = float(np.nanmedian(line[alt])) * (FT_M if alt_in_feet else 1.0)
        tbank = det.templates(h, det.line_heading(line))
        for grid, v, x, y in det.segments(line, mag):
            near = np.hypot(x - x0, y - y0) <= HIT_R_M
            if not near.any():
                continue
            sc, _ = det.score(det.regional_residual(v), tbank)
            sc = det.trim_edges(sc)
            if not np.isfinite(sc[near]).any():
                continue
            m = float(np.max(sc[near]))
            best = m if not np.isfinite(best) else max(best, m)
    return best


def draw(rng, v) -> float:
    """A value, or a uniform draw from a (low, high) range."""
    return float(rng.uniform(*v)) if isinstance(v, tuple) else float(v)


def trials(df: pd.DataFrame, mag: str, alt: str, alt_in_feet: bool, spacing: float, n: int, seed: int,
           base, top, dk: float, q_range: tuple[float, float] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    keep = set(det.survey_lines(df))
    lines = df[(df.kind == "line") & df[mag].notna() & df.line.isin(keep)].reset_index(drop=True)
    rows = []
    if len(lines) < 3:  # too few traverse samples in this chunk to place a plant
        return pd.DataFrame(rows)
    for _ in range(n):
        i = int(rng.integers(1, len(lines) - 1))
        a, b = lines.iloc[i - 1], lines.iloc[i + 1]
        hd = np.arctan2(b.Y - a.Y, b.X - a.X)
        u = float(rng.uniform(-spacing / 2, spacing / 2))
        x0, y0 = lines.X.iloc[i] - u * np.sin(hd), lines.Y.iloc[i] + u * np.cos(hd)
        turn = float(rng.uniform(0, 90))
        rem = (float(np.degrees(np.arcsin(rng.uniform(-1, 1)))), float(rng.uniform(0, 360)))
        b, t = draw(rng, base), draw(rng, top)
        q = float(rng.uniform(*q_range)) if q_range is not None and rng.random() < 0.5 else 0.0
        out, near = plant_lines(df, mag, alt, alt_in_feet, x0, y0, b, t, turn, dk, q, rem)
        touched = [ln for ln in df.line.iloc[near].unique() if ln in keep] if near.size else []
        s = max_score_near(out, touched, mag, alt, alt_in_feet, x0, y0) if touched else np.nan
        s_bg = max_score_near(df, touched, mag, alt, alt_in_feet, x0, y0) if touched else np.nan
        rows.append(dict(x0=x0, y0=y0, turn=turn, u=u, base=b, top=t, q=q, rem_inc=rem[0], rem_dec=rem[1], score=s, score_bg=s_bg,
                         n_samples=int(near.size)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import time

    from akp.paths import data_dir, data_glob
    from akp.stage_p3.p3_lines import read_package

    pkg = [p for p in data_glob(data_dir("p3/lines/open"), "*.zip") if "iron-creek" in p.name][0]
    _, df = next(read_package(pkg))
    t0 = time.time()
    tr = trials(df, "MAGIGRF", "ALTR", True, 400.0, 12, 1, 300.0, 45.72, 0.03)
    print(tr.round(2).to_string(index=False))
    print(f"{time.time() - t0:.0f} s; median score {tr.score.median():.1f}")
    assert tr.score.notna().sum() >= 8  # plants near segment ends are unscored (EDGE_M)
    print("p3_mag_trials self-check OK")
