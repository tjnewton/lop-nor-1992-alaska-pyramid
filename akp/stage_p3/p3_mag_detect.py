"""P3 M4 magnetics: along-line matched-filter detector.

Per line:
1. Order samples by fiducial and compute the along-track distance from X, Y. Resample the
   IGRF-removed field to DS_M spacing, breaking segments at gaps longer than GAP_M.
2. Remove the regional field: a short median despike, then a zero-phase low-pass at REGIONAL_M
   (regional_residual). Scores within EDGE_M of a segment end are dropped.
3. Correlate the residual with each template, each normalized to unit L2 norm.
4. Divide by the local noise: the robust standard deviation (1.4826 MAD) of the residual in the
   annulus EXCL_M to NOISE_WIN_M on either side, floored at SIGMA_FLOOR_NT (the 0.1 nT
   instrument floor let quiet stretches give runaway scores). The score at a position is the largest normalized correlation over the templates,
   in units of local noise. The sign is kept only through the template.
Templates: the vector bank of p3_mag_templates (magnetization directions incl. induced, line-heading
bins, bases, offsets) when built, else the induced p3_mag_forward profiles; resampled to DS_M, for
the clearance nearest the line's median sensor height and the line's heading bin. Tie lines (more
than 45 degrees off the dominant heading) are not scored. Peaks are local maxima of the score above T_LOW, at least
PEAK_SEP_M apart. Returns one row per peak: line, s (along-track m), x, y, score, template.
Self-check: python -m akp.stage_p3.p3_mag_detect (an open control package).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import fftconvolve, find_peaks

from functools import lru_cache

from akp.paths import data_dir
from akp.stage_p3.p3_mag_forward import CLEARANCES_M, DX, NPTS, OFFSETS_M, WHITE_NT
from akp.stage_p3.p3_mag_templates import HEADINGS_DEG

DS_M = 10.0
GAP_M = 60.0
REGIONAL_M = 3000.0
NOISE_WIN_M = 1500.0
EXCL_M = 400.0  # half-width excluded from the noise annulus: a template's support
SIGMA_FLOOR_NT = 1.0  # clutter floor: regional-survey precision of 1-5 nT (p3_PROPERTIES, doi:10.1046/j.1365-2478.1998.00080.x)
TEMPLATE_OFFSETS_M = (0.0, 100.0, 200.0)
T_LOW = 4.0
PEAK_SEP_M = 300.0
DESPIKE_M = 50.0  # short median before the regional low-pass (spikes only)
EDGE_M = 750.0  # scores within this of a segment end are dropped (padding effects)
TEMPLATE_HALF_MAX_M = 1500.0  # templates are capped at +-this, inside the regional window
BANK = "templates_vector.npz"  # set to templates_vector_depths.npz for the depth-classified statistic
TARGET_TOP = 38  # the target depth class of the depth bank (p3_mag_templates.TOP_M, rounded)
FT_M = 0.3048


def _prep(prof: np.ndarray) -> np.ndarray:
    """Resample to DS_M, trim symmetrically about the apex (x = 0) so that the template's center is the
    apex (fftconvolve "same" then scores at the apex), make it zero-mean (a step or offset scores ~0),
    and normalize to unit L2 norm."""
    x = (np.arange(NPTS) - NPTS // 2) * DX
    xs = np.arange(x[0], x[-1] + 1e-9, DS_M)
    t = np.interp(xs, x, prof)
    t = t - np.median(t)
    c0 = int(np.argmin(np.abs(xs)))
    keep = np.flatnonzero(np.abs(t) > 0.01 * np.abs(t).max())
    half = int(min(max(c0 - keep[0], keep[-1] - c0), TEMPLATE_HALF_MAX_M / DS_M))
    t = t[max(0, c0 - half):c0 + half + 1]
    t = t - t.mean()
    return t / np.linalg.norm(t)


@lru_cache(maxsize=64)
def _bank(c: float, hd: float) -> dict[str, np.ndarray]:
    vec = data_dir("p3/mag_forward") / BANK
    if vec.exists():
        bank = np.load(vec)
        return {k: _prep(bank[k]) for k in bank.files if k.startswith(f"hd{hd:.0f}_") and k.endswith(f"_h{c:.0f}")}
    prof = np.load(data_dir("p3/mag_forward") / "profiles.npz")
    return {f"{k}_o{off:.0f}": _prep(prof[k][i]) for k in prof.files if k.endswith(f"_h{c:.0f}")
            for i, off in enumerate(OFFSETS_M) if off in TEMPLATE_OFFSETS_M}


def templates(clearance: float, heading: float = 90.0) -> dict[str, np.ndarray]:
    """Unit-norm templates for the nearest modeled clearance and heading bin (vector bank if built).

    heading is the flight azimuth in [0, 360). The bank is modeled for azimuths in [0, 180) flown
    toward +x; a line flown the other way gets each template reversed, so that its along-track
    samples and the template run the same way.
    """
    c = min(CLEARANCES_M, key=lambda v: abs(v - clearance))
    h180 = heading % 180
    hd = min(HEADINGS_DEG, key=lambda v: min(abs(v - h180), 180 - abs(v - h180)))
    bank = _bank(c, hd)
    reverse = (heading % 360) >= 180 if abs(hd - h180) <= 90 else (heading % 360) < 180
    return {k: v[::-1].copy() for k, v in bank.items()} if reverse else bank


def line_heading(line: pd.DataFrame) -> float:
    """Flight azimuth in [0, 360) from the first to the last sample (in fiducial order when FID exists)."""
    d = line.sort_values("FID") if "FID" in line else line
    xy = d[["X", "Y"]].dropna().to_numpy()
    dx, dy = xy[-1] - xy[0]
    return float(np.degrees(np.arctan2(dx, dy)) % 360)


def survey_lines(df: pd.DataFrame) -> list:
    """Traverse lines: those within 45 degrees of the length-weighted dominant heading (ties dropped)."""
    hd, w, ids = [], [], []
    for ln, line in df[df.kind == "line"].groupby("line"):
        xy = line[["X", "Y"]].dropna().to_numpy()
        if len(xy) < 2:
            continue
        hd.append(line_heading(line) % 180); w.append(np.hypot(*(xy[-1] - xy[0]))); ids.append(ln)
    if not hd:
        return []
    hd, w = np.array(hd), np.array(w)
    order = np.argsort(hd)
    dom = hd[order][np.searchsorted(np.cumsum(w[order]), w.sum() / 2)]
    diff = np.minimum(np.abs(hd - dom), 180 - np.abs(hd - dom))
    return [ln for ln, d in zip(ids, diff) if d <= 45.0]


def segments(line: pd.DataFrame, mag: str):
    d = line.sort_values("FID") if "FID" in line else line
    xy = d[["X", "Y"]].to_numpy()
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    v = d[mag].to_numpy()
    ok = np.isfinite(v) & np.isfinite(s)
    s, v, xy = s[ok], v[ok], xy[ok]
    if s.size < 2:
        return
    breaks = np.flatnonzero(np.diff(s) > GAP_M) + 1
    for a, b in zip(np.r_[0, breaks], np.r_[breaks, s.size]):
        if s[b - 1] - s[a] < 2 * NOISE_WIN_M:
            continue
        grid = np.arange(s[a], s[b - 1], DS_M)
        yield grid, np.interp(grid, s[a:b], v[a:b]), np.interp(grid, s[a:b], xy[a:b, 0]), np.interp(grid, s[a:b], xy[a:b, 1])


def regional_residual(v: np.ndarray) -> np.ndarray:
    """v minus its regional: a short median despike, then a zero-phase Butterworth low-pass at REGIONAL_M.

    A running median over REGIONAL_M (the earlier regional) equals the signal on any monotone
    stretch, so the flanks of real anomalies vanished from the residual. A linear
    low-pass removes only wavelengths longer than REGIONAL_M."""
    from scipy.signal import butter, sosfiltfilt

    d = median_filter(v, size=int(DESPIKE_M / DS_M) | 1, mode="nearest")
    sos = butter(4, 1.0 / REGIONAL_M, btype="low", fs=1.0 / DS_M, output="sos")
    pad = min(len(d) - 1, int(REGIONAL_M / DS_M))
    return v - sosfiltfilt(sos, d, padtype="even", padlen=pad)


def trim_edges(sc: np.ndarray) -> np.ndarray:
    e = int(EDGE_M / DS_M)
    out = sc.copy()
    out[:e], out[-e:] = -np.inf, -np.inf
    return out


def annulus_sigma(resid: np.ndarray, half_win: int, half_excl: int) -> np.ndarray:
    """Local noise at i: 1.4826 x the MAD of the residual values in [i - W, i - E] and [i + E, i + W],
    about their own median. The anomaly at i is excluded, so a body does not inflate its own noise. The deviations are taken about the window's own median, not a pointwise running
    median: on a smooth monotonic stretch a running median equals the signal, which gave zero noise
    and runaway scores."""
    pad = np.pad(resid.astype(float), half_win, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half_win + 1).copy()
    win[:, half_win - half_excl:half_win + half_excl + 1] = np.nan
    med = np.nanmedian(win, axis=1)
    return 1.4826 * np.nanmedian(np.abs(win - med[:, None]), axis=1)


def score(resid: np.ndarray, tbank: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    w = int(NOISE_WIN_M / DS_M)
    sigma = annulus_sigma(resid, w, int(EXCL_M / DS_M))
    best, which = np.full(resid.size, -np.inf), np.full(resid.size, -1)
    for k, t in enumerate(tbank.values()):
        c = fftconvolve(resid, t[::-1], mode="same") / np.maximum(sigma, SIGMA_FLOOR_NT)
        better = c > best
        best[better], which[better] = c[better], k
    if BANK.endswith("_depths.npz"):
        # depth-classified statistic: a position scores only where its best template over all
        # depth classes has the target depth; elsewhere it is set to 0
        target = np.array([f"_t{TARGET_TOP}_" in k for k in tbank])
        best = np.where(target[which], best, 0.0)
    return best, which


def detect(df: pd.DataFrame, mag: str, alt: str, alt_in_feet: bool) -> pd.DataFrame:
    rows = []
    keep = set(survey_lines(df))
    for ln, line in df[df.kind == "line"].groupby("line"):
        if ln not in keep:
            continue
        h = float(np.nanmedian(line[alt])) * (FT_M if alt_in_feet else 1.0)
        tbank = templates(h, line_heading(line))
        names = list(tbank)
        for grid, v, x, y in segments(line, mag):
            resid = regional_residual(v)
            sc, which = score(resid, tbank)
            sc = trim_edges(sc)
            pk, _ = find_peaks(sc, height=T_LOW, distance=int(PEAK_SEP_M / DS_M))
            rows += [dict(line=ln, s=grid[i], x=x[i], y=y[i], score=sc[i], template=names[which[i]], clearance_m=h) for i in pk]
    return pd.DataFrame(rows)


def alignment_check() -> pd.DataFrame:
    """Noise-free plants on straight synthetic lines (both flight directions, four headings): the score
    peak must sit within 150 m of the apex (offset and remanent templates shift their own peak
    along track by up to about that; the hit radius is 250 m). Small white noise keeps the local sigma finite."""
    from akp.stage_p3.p3_mag_inject import response

    rng = np.random.default_rng(0)
    rows = []
    for az in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0):
        s = np.arange(-4000.0, 4000.0, 5.0)
        ux, uy = np.sin(np.radians(az)), np.cos(np.radians(az))
        x, y = s * ux, s * uy
        tmi = response(x, y, np.full(s.size, 30.0), 300.0, 38.1, 0.0, 0.03) + rng.normal(0, 0.5, s.size)
        line = pd.DataFrame(dict(line=1.0, kind="line", X=x, Y=y, FID=np.arange(s.size), MAG=tmi, ALT=30.0))
        pk = detect(line, "MAG", "ALT", False)
        best = pk.loc[pk.score.idxmax()]
        rows.append(dict(azimuth=az, peak_offset_m=float(np.hypot(best.x, best.y)), score=float(best.score)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import time

    al = alignment_check()
    print(al.round(1).to_string(index=False))
    assert (al.peak_offset_m <= 150.0).all(), "template alignment"

    from akp.paths import data_glob
    from akp.stage_p3.p3_lines import read_package

    pkg = [p for p in data_glob(data_dir("p3/lines/open"), "*.zip") if "iron-creek" in p.name][0]
    _, df = next(read_package(pkg))
    t0 = time.time()
    peaks = detect(df, "MAGIGRF", "ALTR", alt_in_feet=True)
    print(f"{pkg.name}: {df.line.nunique()} lines, {len(peaks)} peaks above {T_LOW} in {time.time() - t0:.0f} s")
    print(peaks.score.describe().to_string())
    assert len(peaks) and (peaks.score >= T_LOW).all()
    print("p3_mag_detect self-check OK")
