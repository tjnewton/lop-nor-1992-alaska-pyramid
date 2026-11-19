"""A3b-2 run-2 diagnosis: how much of the weak-check error is the full-record Hanning taper?

Run 2 took each spectral amplitude from the whole 4 s record multiplied by one Hanning window,
so each arrival was weighted by the window value at its arrival time. The direct wave (the
normalizer) and the scattered P and S arrive at different times, so the ratio is biased by
w(t_scattered) / w(t_direct). This writer predicts that bias from run 2's settings (hard-coded
here: the run-3 writer changed them) and sets it beside the observed C for the weak check.
Writes a3b2_run2_diagnosis.csv. Run: python -m akp.stage_a.a3b2_run2_diagnosis
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
VP, VP_VS, Z_BODY, Z_SRC, T_END = 2000.0, 1.8, 1500.0, 2700.0, 4.0  # run-2 settings


def hann(t):
    return 0.5 * (1 - np.cos(2 * np.pi * t / T_END))


def main() -> None:
    df = pd.read_csv(HERE / "a3b2_run2_calibration.csv")
    df = df[df.label == "weak_check"].copy()
    t_direct = 1.5 / df.f0_hz + (Z_SRC - Z_BODY) / VP  # Ricker delay + source-to-receiver-line time
    speed = np.where(df["mode"] == "PP", VP, VP / VP_VS)
    df["t_direct_s"] = t_direct
    df["t_scattered_s"] = t_direct + df.offset_m / speed
    df["hann_bias"] = hann(df.t_scattered_s) / hann(df.t_direct_s)
    df["C_over_bias"] = df.C / df.hann_bias
    out = df[["mode", "offset_m", "C", "t_direct_s", "t_scattered_s", "hann_bias", "C_over_bias"]]
    out.round(4).to_csv(HERE / "a3b2_run2_diagnosis.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
