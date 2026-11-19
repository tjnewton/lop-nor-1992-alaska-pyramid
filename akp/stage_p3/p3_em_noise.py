"""P3 M1: along-profile EM clutter spectrum on the control survey, in log10 apparent resistivity.

Same control survey, sampling and segmenting as p3_mag_noise: DGGS GPR 2020-15, which lies
outside Track A. It reads the 25 m apparent-resistivity grids, coplanar at 400, 1800, 8200, 40000
and 140000 Hz and coaxial at 3300 Hz, takes log10, and computes the Welch PSD of linearly
detrended SEG-cell segments along rows and columns. Writes p3_em_noise.csv: freq_hz, direction,
freq_cpm, wavelength_m, and the median (divided by ln 2, as in p3_mag_noise) and 90th-percentile
one-sided PSD (log10(ohm m)^2 m).
Grid-based, so a bound; G1b replaces it with line-data injection.
Run: python -m akp.stage_p3.p3_em_noise
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend, welch

from akp.stage_p3.p3_mag_noise import DX, SEG, STEP, read, runs

HERE = Path(__file__).parent
GRIDS_HZ = {400: "res400hz", 1800: "res1800hz", 3300: "res3300hz", 8200: "res8200hz", 40000: "res40khz", 140000: "res140khz"}


def main() -> None:
    rows = []
    for hz, name in GRIDS_HZ.items():
        a = read(name)
        a = np.log10(np.where(a > 0, a, np.nan))
        for direction, profiles in (("rows_ew", a[::STEP, :]), ("cols_ns", a[:, ::STEP].T)):
            p = []
            for prof in profiles:
                for seg in runs(prof, SEG):
                    f, pxx = welch(detrend(seg), fs=1 / DX, window="hann", nperseg=SEG, scaling="density")
                    p.append(pxx)
            p = np.array(p)
            for i in range(1, f.size):
                rows.append(dict(freq_hz=hz, direction=direction, freq_cpm=f[i], wavelength_m=1 / f[i],
                                 psd_median=float(np.median(p[:, i]) / np.log(2)), psd_p90=float(np.percentile(p[:, i], 90)),
                                 segments=len(p)))
        del a
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "p3_em_noise.csv", index=False)
    print(out.groupby(["freq_hz", "direction"]).agg(segments=("segments", "first"), psd_med_band=("psd_median", "median")).to_string())


if __name__ == "__main__":
    main()
