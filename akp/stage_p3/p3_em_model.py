"""P3 EM model for injection and templates: the change in coplanar ppm caused by the pyramid.

It is the footprint-weighted 1D model of p3_em_forward (same layering, linear mixing in
conductivity, footprint disc of radius r_fp; a bound, not a 3D model), generalized to any survey's
frequencies and coil separation. Each receiver gets its own sounding at its own bird height. The
layered models are cached by (rounded height, footprint shares), so receivers far from the body
reuse the background. delta_ppm returns an array of shape (n_freq, 2 = in-phase/quadrature, n_rx):
the change from the host-only model.
Self-check: python -m akp.stage_p3.p3_em_model (directly over the apex, the change is largest, and it
fades with offset; a resistive host gives a stronger conductor response than a conductive one).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from simpeg import maps
from simpeg.electromagnetics import frequency_domain as fdem

from akp.stage_p3.p3_em_forward import DZ, footprint_share
from akp.stage_p3.p3_mag_forward import HEIGHT_M

H_STEP_M = 2.0  # bird heights are rounded to this for caching


@lru_cache(maxsize=256)
def _sim(h: float, freqs: tuple, sep: float, top: float):
    srcs = []
    for f in freqs:
        rx = [fdem.receivers.PointMagneticFieldSecondary(np.array([[sep, 0.0, h]]), orientation="z",
                                                         data_type="ppm", component=c) for c in ("real", "imag")]
        srcs.append(fdem.sources.MagDipole(rx, frequency=f, location=np.array([0.0, 0.0, h]), orientation="z"))
    n_l = int(np.ceil(HEIGHT_M / DZ))
    thick = np.r_[top, np.full(n_l, DZ)]
    return fdem.Simulation1DLayered(survey=fdem.survey.Survey(srcs), thicknesses=thick,
                                    sigmaMap=maps.IdentityMap(nP=n_l + 2)), n_l


def delta_ppm(xr, yr, hr, base, top, rho_b, rho_h, freqs, sep, r_fp=150.0) -> np.ndarray:
    freqs = tuple(float(f) for f in freqs)
    out = np.zeros((len(freqs), 2, len(xr)))
    reach = 0.5 * base * np.sqrt(2) + r_fp
    for j, (x, y, h) in enumerate(zip(xr, yr, hr)):
        if np.hypot(x, y) > reach or not np.isfinite(h):
            continue
        hh = float(max(H_STEP_M, round(h / H_STEP_M) * H_STEP_M))
        sim, n_l = _sim(hh, freqs, float(sep), float(top))
        zc = top + (np.arange(n_l) + 0.5) * DZ
        hw = 0.5 * base * (zc - top) / HEIGHT_M
        f = footprint_share(x, y, hw, r_fp)
        if not f.any():
            continue
        bg = np.full(n_l + 2, 1.0 / rho_h)
        sig = np.r_[1.0 / rho_h, 1.0 / rho_h + f * (1.0 / rho_b - 1.0 / rho_h), 1.0 / rho_h]
        out[:, :, j] = (sim.dpred(sig) - sim.dpred(bg)).reshape(len(freqs), 2)
    return out


if __name__ == "__main__":
    freqs, sep = (900.0, 7200.0, 56000.0), 8.0
    x = np.array([0.0, 100.0, 200.0, 400.0])
    d = delta_ppm(x, np.zeros(4), np.full(4, 30.0), 300.0, 38.1, 10.0, 1000.0, freqs, sep)
    amp = np.sqrt((d ** 2).sum(axis=(0, 1)))
    print("ppm change vs offset:", amp.round(2))
    assert amp[0] > amp[1] > amp[2] and amp[3] == 0.0
    d2 = delta_ppm(x[:1], np.zeros(1), np.full(1, 30.0), 300.0, 38.1, 10.0, 100.0, freqs, sep)
    print("conductive host:", np.sqrt((d2 ** 2).sum()).round(2))
    print("p3_em_model self-check OK")
