"""P3 M1 and G1a (EM): footprint-weighted 1D helicopter FDEM response of the target, and its SNR bound.

System: horizontal coplanar coils, separation SEP_M, at FREQS_HZ (the DGGS DIGHEM/RESOLVE coplanar set
of the control survey, p3_em_noise), bird at CLEARANCES_M. Target: the p3_mag_forward pyramid. Classes:
- T3: a conductive body (BODY_T3) in frozen, resistive ground (HOSTS_T3);
- T4: a resistive body (BODY_T4) in thawed ground (HOSTS_T4).
Each sounding is a 1D layered model. Down to the apex it is host; below, in DZ layers to the base,
the conductivity is host + f(z) (body - host), where f is the share of a footprint disc of radius
R_FP_M covered by the body's square section at that depth; below the base it is host. Linear mixing
in conductivity and a hard footprint make this a bound, not a 3D model.

Apparent resistivity per frequency comes from a half-space lookup, fitting in-phase and quadrature
(ppm) at the known bird height. The anomaly is the change in log10 apparent resistivity from the
host-only model, along a profile at offsets OFFSETS_M. Its matched-filter SNR uses the control
survey's along-profile PSD at that frequency (p3_em_noise, both directions; median and p90), plus
a white floor FLOOR_LOG10 per sample, over wavelengths >= p3_mag_forward.LAMBDA_MIN_M. G1a uses cols_ns.
The instrument limit is the white-noise SNR of the ppm anomaly (in-phase and quadrature) at
NOISE_PPM per sample. Each frequency's SNR is the smaller of the two (whichever noise dominates).
Where the host is above RHO_LIMIT_OHM_M, apparent resistivity saturates and only the instrument
limit applies. The
statistics are the best single frequency (conservative) and the root-sum-square over frequencies
(optimistic: frequencies share clutter). z, the threshold, is the magnetic one (same budget and
tests). Writes p3_em_forward.csv. Run: python -m akp.stage_p3.p3_em_forward
"""
from __future__ import annotations

import itertools
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from simpeg import maps
from simpeg.electromagnetics import frequency_domain as fdem

from akp.stage_p3.p3_mag_forward import BASES_M, DX, HEIGHT_M, NPTS, OFFSETS_M, TOPS_M, snr, z_threshold

HERE = Path(__file__).parent
FREQS_HZ = (400.0, 1800.0, 8200.0, 40000.0, 140000.0)
SEP_M = 7.9  # coplanar coil separation, a typical DIGHEM/RESOLVE value (input; p3_PROPERTIES)
CLEARANCES_M = (30.0, 45.0)
BODY_T3, HOSTS_T3 = 10.0, (1000.0, 3000.0)  # ohm m
BODY_T4, HOSTS_T4 = 3000.0, (100.0, 300.0)
R_FP_M = (100.0, 150.0)  # footprint radius: "less than 200 m in general" (p3_PROPERTIES)
DZ = 6.0
RHO_GRID = np.logspace(0.0, 4.5, 4501)  # step 0.001 in log10
FLOOR_LOG10 = 0.01  # per-sample white floor in log10 apparent resistivity (input: about 2%)
# Instrument noise per frequency, ppm per 25 m sample, with no averaging gain: the RESOLVE totals
# including drift (doi:10.1190/1.2187752, Table 1, to be checked in p3_PROPERTIES), nearest band.
NOISE_PPM = {400.0: 8.0, 1800.0: 8.75, 8200.0: 16.0, 40000.0: 29.0, 140000.0: 38.5}
# Above these host resistivities the apparent resistivity at that frequency saturates
# (doi:10.1190/1.1440817, p. 14: about 1000 ohm m at 900 Hz), so the clutter spectrum is not usable.
RHO_LIMIT_OHM_M = {400.0: 1000.0, 1800.0: 1000.0}


def survey(h: float):
    srcs = []
    for f in FREQS_HZ:
        rx = [fdem.receivers.PointMagneticFieldSecondary(np.array([[SEP_M, 0.0, h]]), orientation="z",
                                                         data_type="ppm", component=c) for c in ("real", "imag")]
        srcs.append(fdem.sources.MagDipole(rx, frequency=f, location=np.array([0.0, 0.0, h]), orientation="z"))
    return fdem.survey.Survey(srcs)


def halfspace_table(h: float) -> np.ndarray:
    sim = fdem.Simulation1DLayered(survey=survey(h), thicknesses=np.array([]), sigmaMap=maps.IdentityMap(nP=1))
    return np.array([sim.dpred(np.array([1.0 / r])) for r in RHO_GRID]).reshape(RHO_GRID.size, len(FREQS_HZ), 2)


def apparent(d: np.ndarray, table: np.ndarray) -> np.ndarray:
    d = d.reshape(len(FREQS_HZ), 2)
    return np.array([RHO_GRID[np.argmin(((table[:, i, :] - d[i]) ** 2).sum(axis=1))] for i in range(len(FREQS_HZ))])


def footprint_share(x: float, y: float, hw: np.ndarray, r_fp: float) -> np.ndarray:
    g = np.arange(-r_fp, r_fp + 1e-9, 4.0)
    gx, gy = np.meshgrid(g, g)
    disc = gx ** 2 + gy ** 2 <= r_fp ** 2
    px, py = (gx + x)[disc], (gy + y)[disc]
    return np.array([np.mean((np.abs(px) <= w) & (np.abs(py) <= w)) for w in hw])


def case_rows(noise, z, clr, table, sv, cls, rho_b, rho_h, base, top, r_fp) -> list[dict]:
    """Rows of one class x host x geometry x footprint case at clearance clr (all offsets, both directions, both stats)."""
    x_all = (np.arange(NPTS) - NPTS // 2) * DX
    rows = []
    n_l = int(np.ceil(HEIGHT_M / DZ))
    thick = np.r_[top, np.full(n_l, DZ)]
    zc = top + (np.arange(n_l) + 0.5) * DZ
    hw = 0.5 * base * (zc - top) / HEIGHT_M
    sim = fdem.Simulation1DLayered(survey=sv, thicknesses=thick, sigmaMap=maps.IdentityMap(nP=n_l + 2))
    d_bg = sim.dpred(np.full(n_l + 2, 1.0 / rho_h))
    bg = apparent(d_bg, table)
    for off in OFFSETS_M:
        prof = np.zeros((len(FREQS_HZ), NPTS))
        dppm = np.zeros((len(FREQS_HZ), 2, NPTS))
        for j, x in enumerate(x_all):
            if abs(x) > 0.5 * base + r_fp or off > 0.5 * base + r_fp:
                continue
            f = footprint_share(x, off, hw, r_fp)
            if not f.any():
                continue
            sig = np.r_[1.0 / rho_h, 1.0 / rho_h + f * (1.0 / rho_b - 1.0 / rho_h), 1.0 / rho_h]
            d = sim.dpred(sig)
            prof[:, j] = np.log10(apparent(d, table)) - np.log10(bg)
            dppm[:, :, j] = (d - d_bg).reshape(len(FREQS_HZ), 2)
        snr_instr = [float(np.sqrt(np.sum(dppm[i] ** 2)) / NOISE_PPM[hz]) for i, hz in enumerate(FREQS_HZ)]
        for direction, stat in itertools.product(("rows_ew", "cols_ns"), ("median", "p90")):
            snrs = []
            for i, hz in enumerate(FREQS_HZ):
                nz = noise[(noise.freq_hz == hz) & (noise.direction == direction)].sort_values("freq_cpm")
                clutter = snr(prof[i], nz[f"psd_{stat}"].to_numpy() + FLOOR_LOG10 ** 2 * 2 * DX)
                if hz in RHO_LIMIT_OHM_M and rho_h >= RHO_LIMIT_OHM_M[hz]:
                    clutter = np.inf  # apparent resistivity saturates; the instrument limit governs
                snrs.append(min(clutter, snr_instr[i]))
            rows.append(dict(cls=cls, rho_body=rho_b, rho_host=rho_h, base_m=base, top_m=round(top, 2), clearance_m=clr,
                             r_fp_m=r_fp, offset_m=off, noise=f"{direction}_{stat}", z=round(z, 3),
                             peak_dlog10=round(float(np.abs(prof).max()), 4),
                             peak_dppm=round(float(np.abs(dppm).max()), 3),
                             snr_best=round(max(snrs), 3), snr_rss=round(float(np.sqrt(np.sum(np.square(snrs)))), 3),
                             best_freq_hz=FREQS_HZ[int(np.argmax(snrs))]))
    return rows


@lru_cache(maxsize=256)
def snr_offsets_em(rho_h: float, base: float, top: float, clearance: float, r_fp: float = 150.0,
                   noise_key: str = "cols_ns_median") -> np.ndarray:
    """T3 best-frequency SNR for each of OFFSETS_M at this exact bird clearance (no grid rounding)."""
    noise = pd.read_csv(HERE / "p3_em_noise.csv")
    rows = case_rows(noise, np.nan, float(clearance), halfspace_table(float(clearance)), survey(float(clearance)),
                     "T3", BODY_T3, float(rho_h), float(base), float(top), float(r_fp))
    return np.array([r["snr_best"] for r in rows if r["noise"] == noise_key])


def main() -> None:
    noise = pd.read_csv(HERE / "p3_em_noise.csv")
    track_a = float(pd.read_csv(HERE / "p3_prior.csv").query("check == 'track_a_land_km2'").value.iloc[0])
    z = z_threshold(track_a, 400.0)
    cases = [("T3", BODY_T3, h) for h in HOSTS_T3] + [("T4", BODY_T4, h) for h in HOSTS_T4]
    rows = []
    for clr in CLEARANCES_M:
        table = halfspace_table(clr)
        sv = survey(clr)
        for (cls, rho_b, rho_h), base, top, r_fp in itertools.product(cases, BASES_M, TOPS_M, R_FP_M):
            rows += case_rows(noise, z, clr, table, sv, cls, rho_b, rho_h, base, top, r_fp)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "p3_em_forward.csv", index=False)
    s = out[(out.noise == "cols_ns_median") & (out.r_fp_m == 150.0) & (out.offset_m.isin([0, 100, 200]))]
    print(s.pivot_table(index=["cls", "rho_host", "base_m", "top_m", "clearance_m"], columns="offset_m", values="snr_best").round(1).to_string())
    print("z =", round(z, 3))


if __name__ == "__main__":
    main()
