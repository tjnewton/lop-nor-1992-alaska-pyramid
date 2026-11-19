"""P3 M4 EM: multi-channel along-line matched filter for T3, a conductor in resistive ground.

Per traverse line (ties dropped, as in p3_mag_detect):
1. Resample every coplanar in-phase and quadrature channel to DS_M along track. Remove the regional
   per channel: despike, then a zero-phase low-pass at REGIONAL_M. Edges are trimmed as in p3_mag_detect.
2. Each channel k gets its local robust noise sigma_k: 1.4826 MAD in the annulus
   p3_mag_detect.EXCL_M to NOISE_WIN_M (the anomaly itself excluded), floored at FLOOR_PPM.
3. Templates come from p3_em_model.delta_ppm along a straight profile:
   - bases BASES_M, the middle top depth, hosts HOSTS;
   - line offsets OFFSETS_M;
   - the line's median bird height and the survey's frequencies and coil separation.
   They keep the relative amplitude across channels, and each channel is made zero-mean.
4. The score is the multi-channel matched filter in independent noise:
       sum_k (r_k / sigma_k^2) * t_k   /   sqrt(sum_k ||t_k||^2 / sigma_k^2).
   It is in units of noise. The largest over templates is kept.
Peaks above T_LOW, at least PEAK_SEP_M apart, are returned as in p3_mag_detect.
Self-check: python -m akp.stage_p3.p3_em_detect (the Iron Creek control, DIGHEM coplanar channels).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import fftconvolve, find_peaks

from akp.stage_p3 import p3_mag_detect as mdet
from akp.stage_p3.p3_em_model import delta_ppm

DS_M = mdet.DS_M
REGIONAL_M = 2000.0
NOISE_WIN_M = 1500.0
FLOOR_PPM = 1.0
BASES_M = (200.0, 300.0, 400.0)
TOP_M = 38.1
HOSTS = (1000.0, 3000.0)
BODY = 10.0
OFFSETS_M = (0.0, 100.0, 200.0)
T_LOW = 4.0
PEAK_SEP_M = 300.0
HALF_LEN_M = 600.0


@lru_cache(maxsize=32)
def templates(h: float, freqs: tuple, sep: float) -> dict[str, np.ndarray]:
    """Per template, an array (n_channel = 2 x n_freq, n_samples) of ppm changes along +x."""
    x = np.arange(-HALF_LEN_M, HALF_LEN_M + 1e-9, DS_M)
    out = {}
    for base in BASES_M:
        for host in HOSTS:
            for off in OFFSETS_M:
                d = delta_ppm(x, np.full(x.size, off), np.full(x.size, h), base, TOP_M, BODY, host, freqs, sep)
                t = d.reshape(2 * len(freqs), x.size)  # rows: f0 in-phase, f0 quadrature, f1 in-phase, ...
                t = t - t.mean(axis=1, keepdims=True)  # zero-mean per channel: steps and offsets score ~0
                if np.abs(t).max() > 0:
                    out[f"b{base:.0f}_h{host:.0f}_o{off:.0f}"] = t
    return out


def segments(line: pd.DataFrame, chans: list[str]):
    d = line.sort_values("FID") if "FID" in line else line
    xy = d[["X", "Y"]].to_numpy()
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    v = d[chans].to_numpy()
    ok = np.isfinite(v).all(axis=1) & np.isfinite(s)
    s, v, xy = s[ok], v[ok], xy[ok]
    if s.size < 2:
        return
    breaks = np.flatnonzero(np.diff(s) > mdet.GAP_M) + 1
    for a, b in zip(np.r_[0, breaks], np.r_[breaks, s.size]):
        if s[b - 1] - s[a] < 2 * NOISE_WIN_M:
            continue
        grid = np.arange(s[a], s[b - 1], DS_M)
        vv = np.column_stack([np.interp(grid, s[a:b], v[a:b, k]) for k in range(v.shape[1])])
        yield grid, vv, np.interp(grid, s[a:b], xy[a:b, 0]), np.interp(grid, s[a:b], xy[a:b, 1])


def _resid(v: np.ndarray) -> np.ndarray:
    """EM regional removal with the EM regional wavelength: despike, then zero-phase low-pass."""
    from scipy.signal import butter, sosfiltfilt

    d = median_filter(v, size=int(mdet.DESPIKE_M / DS_M) | 1, mode="nearest")
    sos = butter(4, 1.0 / REGIONAL_M, btype="low", fs=1.0 / DS_M, output="sos")
    return v - sosfiltfilt(sos, d, padtype="even", padlen=min(len(d) - 1, int(REGIONAL_M / DS_M)))


def score(vv: np.ndarray, tbank: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    w = int(NOISE_WIN_M / DS_M)
    resid = np.column_stack([_resid(vv[:, c]) for c in range(vv.shape[1])])
    excl = int(mdet.EXCL_M / DS_M)
    sig = np.column_stack([mdet.annulus_sigma(resid[:, c], w, excl) for c in range(resid.shape[1])])
    sig2 = np.maximum(sig, FLOOR_PPM) ** 2
    best, which = np.full(len(vv), -np.inf), np.full(len(vv), -1)
    for k, t in enumerate(tbank.values()):
        num = sum(fftconvolve(resid[:, c] / sig2[:, c], t[c][::-1], mode="same") for c in range(t.shape[0]))
        den = np.sqrt(sum(np.sum(t[c] ** 2) / sig2[:, c] for c in range(t.shape[0])))
        sc = num / den
        better = sc > best
        best[better], which[better] = sc[better], k
    return best, which


def detect(df: pd.DataFrame, chans: list[str], freqs: tuple, sep: float, alt: str, alt_in_feet: bool) -> pd.DataFrame:
    rows = []
    keep = set(mdet.survey_lines(df))
    for ln, line in df[(df.kind == "line") & df.line.isin(keep)].groupby("line"):
        h = float(np.nanmedian(line[alt])) * (mdet.FT_M if alt_in_feet else 1.0)
        tbank = templates(round(h / 5.0) * 5.0, tuple(freqs), float(sep))
        names = list(tbank)
        for grid, vv, x, y in segments(line, chans):
            sc, which = score(vv, tbank)
            sc = mdet.trim_edges(sc)
            pk, _ = find_peaks(sc, height=T_LOW, distance=int(PEAK_SEP_M / DS_M))
            rows += [dict(line=ln, s=grid[i], x=x[i], y=y[i], score=sc[i], template=names[which[i]], clearance_m=h) for i in pk]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import time

    from akp.paths import data_dir, data_glob
    from akp.stage_p3.p3_lines import read_package

    pkg = [p for p in data_glob(data_dir("p3/lines/open"), "*.zip") if "iron-creek" in p.name][0]
    _, df = next(read_package(pkg))
    chans = ["CPI900", "CPQ900", "CPI7200", "CPQ7200", "CPI56K", "CPQ56K"]
    t0 = time.time()
    peaks = detect(df, chans, (900.0, 7200.0, 56000.0), 8.0, "ALTR", True)
    print(f"{len(peaks)} EM peaks above {T_LOW} in {time.time() - t0:.0f} s")
    print(peaks.score.describe().round(2).to_string())
    assert len(peaks)
    print("p3_em_detect self-check OK")
