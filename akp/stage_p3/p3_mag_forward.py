"""P3 M1 and G1a (magnetics): forward profiles of the target, and the signal-to-noise bound.

Target: a square pyramid, height HEIGHT_M (the stated 550 ft), base BASES_M,
top TOPS_M down (the stated 150 and 100 ft), turned 0 or 45 degrees to the flight line. Induced
magnetization only, in the field FIELD (an approximate interior-Alaska IGRF; an input, not a cited
value). Built from 10 m x 10 m x 6 m cells (a 5 m check is asserted). SimPEG integral TMI at
CLEARANCES_M above flat ground, along a profile at lateral offsets OFFSETS_M from the apex, sampled
at the control grid's 25 m. The reference contrast is DK_REF SI; the TMI anomaly is linear in it.

Signal-to-noise: the matched filter in stationary colored noise, SNR^2 = (dx/N) sum_k |FFT_k|^2 / S(f_k),
with two-sided S = P/2 and P the one-sided clutter PSD of p3_mag_noise.csv (median, and 90th
percentile, of the control survey; a white instrument floor WHITE_NT per sample is added). DC, and
wavelengths longer than the segment or shorter than LAMBDA_MIN_M, are dropped: the control grid's
clutter is suppressed at short wavelengths by gridding (its cross-line PSD falls steeply below the
400 m line spacing), so the filter must not use that band. The along-line direction of the control
survey is N-S (cols_ns): its PSD falls far less below the line spacing. G1a uses cols_ns. The detection threshold z comes from the
family-wise budget: N tests = Track A land (p3_prior.csv) x line-km per km2 (1000/spacing) x
tests per line-km (1000/TEST_STEP_M), alpha = BUDGET/N, z = Phi^-1(1 - alpha). P_d = Phi(SNR - z),
so P_d >= G1_FLOOR (0.5) needs SNR >= z. dk_req is the contrast that gives SNR = z on one line.
This is a bound: it assumes Gaussian stationary clutter and the true template, and G1b replaces it.
Writes p3_mag_forward.csv and $AKP_DATA/p3/mag_forward/profiles.npz. Run: python -m akp.stage_p3.p3_mag_forward
"""
from __future__ import annotations

import itertools
from functools import lru_cache
from pathlib import Path

import discretize
import numpy as np
import pandas as pd
from scipy.stats import norm
from simpeg import maps
from simpeg.potential_fields import magnetics as mag

from akp.paths import data_dir

HERE = Path(__file__).parent
HEIGHT_M = 167.64  # 550 ft (C3, C4, C7)
TOPS_M = (30.48, 45.72)  # 100 ft (C7) and 150 ft (C3, C4)
BASES_M = (200.0, 300.0, 400.0)
TURNS_DEG = (0.0, 45.0)
CLEARANCES_M = (30.0, 45.0, 60.0, 100.0, 150.0, 300.0)
OFFSETS_M = (0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0)
FIELD = dict(amplitude=56000.0, inclination=77.0, declination=16.0)  # approximate IGRF, interior Alaska (input)
DK_REF = 0.01  # SI
DX = 25.0  # m, as the control grid
NPTS = 256  # 6.4 km profile, as the noise segments
WHITE_NT = 0.1  # per-sample instrument floor (nT)
LAMBDA_MIN_M = 150.0  # the matched filter uses wavelengths >= this: gridding suppresses shorter clutter
SPACINGS_M = (400.0, 800.0)
TEST_STEP_M = 100.0
BUDGET = 0.1  # Family-wise false-alarm budget
G1_FLOOR = 0.5


def pyramid_cells(mesh, base: float, top: float, turn: float) -> np.ndarray:
    cc = mesh.cell_centers
    c, s = np.cos(np.radians(turn)), np.sin(np.radians(turn))
    u, v = c * cc[:, 0] + s * cc[:, 1], -s * cc[:, 0] + c * cc[:, 1]
    below = -top - cc[:, 2]
    half = 0.5 * base * below / HEIGHT_M
    return (below >= 0) & (below <= HEIGHT_M) & (np.abs(u) <= half) & (np.abs(v) <= half)


def forward(base: float, top: float, turn: float, clearance: float, cell: float = 10.0) -> np.ndarray:
    """TMI (nT) at DK_REF, shape (len(OFFSETS_M), NPTS)."""
    half_w = 0.5 * base * np.sqrt(2) + 2 * cell
    n_xy = int(np.ceil(2 * half_w / cell))
    dz = 0.6 * cell
    n_z = int(np.ceil((top + HEIGHT_M + dz) / dz))
    mesh = discretize.TensorMesh([[(cell, n_xy)], [(cell, n_xy)], [(dz, n_z)]],
                                 origin=[-n_xy * cell / 2, -n_xy * cell / 2, -n_z * dz])
    act = pyramid_cells(mesh, base, top, turn)
    x = (np.arange(NPTS) - NPTS // 2) * DX
    loc = np.vstack([np.c_[x, np.full(NPTS, d), np.full(NPTS, clearance)] for d in OFFSETS_M])
    rx = mag.receivers.Point(loc, components="tmi")
    src = mag.sources.UniformBackgroundField(receiver_list=[rx], **FIELD)
    sim = mag.simulation.Simulation3DIntegral(mesh=mesh, survey=mag.survey.Survey(src),
                                              chiMap=maps.IdentityMap(nP=int(act.sum())), active_cells=act,
                                              store_sensitivities="forward_only")
    return sim.dpred(np.full(int(act.sum()), DK_REF)).reshape(len(OFFSETS_M), NPTS)


def snr(profile: np.ndarray, psd: np.ndarray, f_max: float = 1.0 / LAMBDA_MIN_M) -> float:
    """Matched-filter SNR of one profile in noise of one-sided PSD psd (on rfft bins 1..), bins f <= f_max only."""
    spec = np.fft.rfft(profile - profile[[0, -1]].mean())
    k = np.arange(1, spec.size)
    k = k[np.fft.rfftfreq(NPTS, DX)[k] <= f_max]
    weight = np.where((k == spec.size - 1) & (NPTS % 2 == 0), 1.0, 2.0)  # two-sided sum from the one-sided bins
    return float(np.sqrt(DX / NPTS * np.sum(weight * np.abs(spec[k]) ** 2 / (psd[k - 1] / 2))))


def z_threshold(track_a_km2: float, spacing: float) -> float:
    n = track_a_km2 * (1000.0 / spacing) * (1000.0 / TEST_STEP_M)
    return float(norm.isf(BUDGET / n))


def main() -> None:
    noise = pd.read_csv(HERE / "p3_mag_noise.csv")
    track_a = float(pd.read_csv(HERE / "p3_prior.csv").query("check == 'track_a_land_km2'").value.iloc[0])
    white = WHITE_NT ** 2 * 2 * DX
    psds = {}
    for direction in ("rows_ew", "cols_ns"):
        d = noise[noise.direction == direction].sort_values("freq_cpm")
        assert len(d) == NPTS // 2
        psds[f"{direction}_median"] = d.psd_median.to_numpy() + white
        psds[f"{direction}_p90"] = d.psd_p90.to_numpy() + white
    psds["white_only"] = np.full(NPTS // 2, white)
    zs = {sp: z_threshold(track_a, sp) for sp in SPACINGS_M}
    coarse = forward(300.0, TOPS_M[1], 0.0, 45.0)
    fine = forward(300.0, TOPS_M[1], 0.0, 45.0, cell=5.0)
    rel = float(np.abs(coarse - fine).max() / np.abs(fine).max())
    assert rel < 0.05, f"mesh not converged: {rel:.3f}"
    rows, prof = [], {}
    for base, top, turn, clr in itertools.product(BASES_M, TOPS_M, TURNS_DEG, CLEARANCES_M):
        p = forward(base, top, turn, clr)
        prof[f"b{base:.0f}_t{top:.0f}_r{turn:.0f}_h{clr:.0f}"] = p
        for i, off in enumerate(OFFSETS_M):
            for nk, psd in psds.items():
                s = snr(p[i], psd)
                for sp, z in zs.items():
                    rows.append(dict(base_m=base, top_m=round(top, 2), turn_deg=turn, clearance_m=clr, offset_m=off,
                                     noise=nk, spacing_m=sp, z=round(z, 3), peak_nT_ref=round(float(np.abs(p[i]).max()), 3),
                                     snr_ref=round(s, 3), dk_req_si=float(DK_REF * z / s)))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "p3_mag_forward.csv", index=False)
    np.savez_compressed(data_dir("p3/mag_forward") / "profiles.npz", **prof)
    print(f"mesh check: max relative difference 10 m vs 5 m cells = {rel:.4f}")
    print("z thresholds:", {k: round(v, 3) for k, v in zs.items()})
    s = out[(out.turn_deg == 0) & (out.spacing_m == 400) & (out.offset_m.isin([0, 200]))
            & (out.noise.isin(["cols_ns_median", "cols_ns_p90", "white_only"]))]
    print(s.pivot_table(index=["base_m", "top_m", "clearance_m", "offset_m"], columns="noise", values="dk_req_si").round(4).to_string())


@lru_cache(maxsize=256)
def _psds() -> dict:
    noise = pd.read_csv(HERE / "p3_mag_noise.csv")
    white = WHITE_NT ** 2 * 2 * DX
    out = {"white_only": np.full(NPTS // 2, white)}
    for direction in ("rows_ew", "cols_ns"):
        d = noise[noise.direction == direction].sort_values("freq_cpm")
        out[f"{direction}_median"] = d.psd_median.to_numpy() + white
        out[f"{direction}_p90"] = d.psd_p90.to_numpy() + white
    return out


@lru_cache(maxsize=512)
def snr_offsets(base: float, top: float, turn: float, clearance: float, noise: str = "cols_ns_median") -> np.ndarray:
    """Matched-filter SNR at DK_REF for each of OFFSETS_M, modeled at this exact clearance (no grid rounding)."""
    p = forward(base, top, turn, float(clearance))
    return np.array([snr(p[i], _psds()[noise]) for i in range(len(OFFSETS_M))])


if __name__ == "__main__":
    main()
