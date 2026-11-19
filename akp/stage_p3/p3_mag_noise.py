"""P3 M1: along-profile magnetic clutter spectrum on a control survey, for the G1a bound.

Control survey: DGGS GPR 2020-15, the Alaska Highway corridor (2005), fetched in setup by
akp.setup.pr_e_airborne. It lies outside Track A, so reading it before the P3 freeze is allowed. Only its grids are held, so this is a grid-based bound: the 25 m grid of the final
residual field ("sim_magtf", IGRF removed) is read along rows (E-W) and columns (N-S).
Every STEP-th row and column is cut into valid runs, then into SEG-cell segments with half
overlap. Each segment is linearly detrended and gets a Welch periodogram (Hann). Writes
p3_mag_noise.csv: direction, freq_cpm, wavelength_m, and the median and 90th-percentile one-sided
PSD over segments (nT^2 m). The median is divided by ln 2: a single periodogram bin is chi-squared
with two degrees of freedom, whose median is ln 2 times its mean. Also writes p3_mag_noise_checks.csv: segment counts, the RMS of the
residual band-passed to BAND_M per segment (median, p90), and the bird height (altbird grid:
median, p10, p90 over valid cells). G1b replaces this bound with injection into line data.
Run: python -m akp.stage_p3.p3_mag_noise
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.signal import butter, detrend, sosfiltfilt, welch

from akp.paths import data_dir

HERE = Path(__file__).parent
GRIDS = "probes/airborne_access/AKHighway/extracted/gpr2020_015_akhwy_2005_grids_ermapper/grids_ermapper"
DX = 25.0  # m, the grid cell (ERS header)
STEP = 10  # every 10th row and column
SEG = 256  # 6.4 km segments
BAND_M = (150.0, 3000.0)  # wavelengths a 200-400 m body at these depths occupies


def runs(v: np.ndarray, n: int):
    ok = np.isfinite(v)
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        for s in range(a, b - n + 1, n // 2):
            yield v[s:s + n]


def read(name: str) -> np.ndarray:
    with rasterio.open(data_dir(GRIDS) / f"alaska_highway_corridor_{name}.ers") as r:
        a = r.read(1).astype(np.float64)
        nodata = r.nodata
    a[(a < -1e30) | (a == nodata)] = np.nan
    return a


def main() -> None:
    tmf = read("sim_magtf")
    sos = butter(4, [1 / BAND_M[1], 1 / BAND_M[0]], btype="band", fs=1 / DX, output="sos")
    psd_rows, checks = [], []
    for direction, profiles in (("rows_ew", tmf[::STEP, :]), ("cols_ns", tmf[:, ::STEP].T)):
        p, rms = [], []
        for prof in profiles:
            for seg in runs(prof, SEG):
                seg = detrend(seg)
                f, pxx = welch(seg, fs=1 / DX, window="hann", nperseg=SEG, scaling="density")
                p.append(pxx)
                rms.append(np.sqrt(np.mean(sosfiltfilt(sos, seg)[SEG // 8:-SEG // 8] ** 2)))
        p = np.array(p)
        for i in range(1, f.size):
            psd_rows.append(dict(direction=direction, freq_cpm=f[i], wavelength_m=1 / f[i],
                                 psd_median=float(np.median(p[:, i]) / np.log(2)), psd_p90=float(np.percentile(p[:, i], 90))))
        checks += [dict(check="segments", scope=direction, value=len(p)),
                   dict(check="band_rms_nT_median", scope=direction, value=round(float(np.median(rms)), 3)),
                   dict(check="band_rms_nT_p90", scope=direction, value=round(float(np.percentile(rms, 90)), 3))]
    del tmf
    alt = read("altbird")[::STEP, ::STEP]
    alt = alt[np.isfinite(alt)]
    checks += [dict(check=f"bird_height_m_{k}", scope="altbird", value=round(float(np.percentile(alt, q)), 1))
               for k, q in (("p10", 10), ("median", 50), ("p90", 90))]
    pd.DataFrame(psd_rows).to_csv(HERE / "p3_mag_noise.csv", index=False)
    out = pd.DataFrame(checks)
    out.to_csv(HERE / "p3_mag_noise_checks.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
