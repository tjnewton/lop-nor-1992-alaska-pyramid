"""A8 production (P4 item S5): the eight simulations, their measurement and the analytic chain beside it.

Decisions fixed before any production run (2026-11-13); not changed after results:
 1. Run list: four pairs, each a run with the body and its no-body twin: thawed host vp 1200 m/s;
    thawed host vp 850 m/s; a frozen layer over each. The twins count against the fixed cap: eight
    simulations in total, no more. A simulation that fails for a numerical reason is fixed and rerun
    once, and the rerun is logged (RERUNS).
 2. Body: the square inclusion of base 300 m, height 168 m, apex 38 m below the surface, at +100% in
    vp, vs and density over the host (the stage's dense, stiff bracket).
 3. Frozen layer (inputs, not results): 30 m thick, vp 3000 m/s, vp/vs 1.8, density the host's (the
    stage has no density for frozen ground). a3_0_PROPERTIES.md cites 2740-3401 m/s for frozen sand
    and gravel and a thinner layer at one site; 30 m is chosen so the grid resolves it.
 4. Incidence: an upgoing plane P wave with the stage's teleseismic ray parameter. The stage stores
    no constant; `ray_parameter` computes it the stage's way (iasp91 P, surface source, as
    a3b_dense_array.py) from the committed 1992 origin (setup/pr_b_isc_event.csv) to the station
    whose 1992 record the stage uses (a2_wavefield.py's coordinates).
 5. Receivers: on the surface every 100 m out to 3 km from the point above the apex, along three
    azimuths: forward and backward in the incidence plane, and perpendicular to it. Three components.
 6. Measurement, per case, azimuth, range and band (the stage's bands 0.5-1, 1-2 and 2-4 Hz):
    scattered field = run with the body less its twin. In each of two time windows: the peak
    envelope of the scattered vertical and of the scattered radial component, each as a share of the
    peak envelope of the incident vertical motion at the same receiver in the twin run, band by
    band. With r the horizontal range, T the center period of the band, t0 the time the incident P
    reaches the apex, vp_fast and vs_fast the largest speeds in the model and vs_slow the smallest
    shear speed: the fast window runs from t0 + r / vp_fast - T / 2 to t0 + r / vs_fast - T / 2
    (scattered P and head waves); the slow window from there to t0 + r / (0.85 vs_slow) + 1.5 T
    (scattered S and Rayleigh waves). In the slow window also the horizontal-to-vertical ratio and
    the phase lag between radial and vertical, as evidence for or against a Rayleigh wave; no label
    is forced.
 7. Beside each numerical value, the analytic chain's value for the same case and range, from the
    stage's own functions (a3_detectability.py and the P-to-Rg estimate of a3b4_rg.py), and the ratio
    numerical over analytic. The chain has no value for the layered cases: left empty.
 8. Resolution: the points per wavelength that passed gate three (15.4) for the slowest Rayleigh
    wave at the top of the highest band measured. The 850 m/s cases cannot meet it at 4 Hz within
    8 GB (7 m cells over this box are about 100 million points), so they are measured up to 2 Hz only.

Choices of mine inside those decisions, also fixed before any run:
 - Wavelet: Ricker, peak 2 Hz (its spectrum at 0.5 and 4 Hz is 16% and 20% of its peak, so it carries
   all three bands). Band filters: Butterworth, order 4, zero phase. Envelope: analytic signal.
 - T = 1 / (arithmetic center frequency of the band): 0.75, 1.5 and 3 Hz.
 - The speeds in the windows include the body's (decision 6 says "in the model"). In a thawed host
   the body's S speed exceeds the host's P speed, so the fast window closes before the host's P wave
   arrives. A third window is therefore reported beside the two, labeled `host_p` and not part of
   decision 6: from t0 + r / vp_host - T / 2 to t0 + r / vs_host - T / 2.
 - Incident peak: the peak envelope of the twin's vertical motion within 1 / f0 + T of the predicted
   arrival of the plane wave at that receiver.
 - Rayleigh evidence: h_over_v = peak envelope radial / vertical in the slow window; phase_lag_deg =
   phase of the radial relative to the vertical analytic signal, envelope-weighted over the window.
   The same statistic on gate three's pure Rayleigh wave is logged in the checks as the yardstick.
   Radial is outward from the body. The vertical sample is h / 2 below the surface, the horizontal
   ones h / 2 to the side (the staggered grid).
 - Analytic values. Slow window, vertical: the Born P-to-Rg estimate, E / sqrt(r) with E from
   a3b4_rg.rg_3d_E (linearized perturbations, no strong-contrast factor, as the stage defines it for
   the dense body). Fast and host_p windows, radial: the Born body-wave share of a3_detectability
   under the geometry rule (75-90 degrees), k^2 V |p F| / (4 pi r), times the A3b-1 geometry factor
   for the dense body (a3b_factors.csv), as A3 applies it. Both at the band's upper frequency (the
   stage's convention; columns analytic, ratio) and at its center frequency (analytic_center,
   ratio_center). Other rows have no chain value.
 - Grid: 10 m cells in all four cases (the 850 m/s cases keep 10 m so that the 30 m layer and the
   38 m cover are three cells or more). The problem is symmetric about the incidence plane, so half
   of it is computed (mirror plane y = 0, a8_elastic3d). Box: 3300 m either way along x, 3300 m
   along y, 1000 m deep; sponges outside. The sheet is 400 m deep.

Outputs: a8_production.csv (one row per case, azimuth, range, band, window, component) and
a8_production_checks.csv. Traces: $AKP_DATA/a8/production. The measurement reruns from saved traces.
Run: OMP_NUM_THREADS=6 PYTHONPATH=. python scripts/mem_guard.py --gb 8 --every 0.5 -- \
       python -m akp.stage_a.a8_production            (runs what is not archived, then measures)
     python -m akp.stage_a.a8_production --measure    (saved traces only)
     AKP_A8_SMOKE=1 ...                               (a miniature box to test the pipeline; outputs to $AKP_DATA)
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from akp.stage_a import a8_elastic3d as fd  # first: sets the Devito environment

HERE = Path(__file__).parent
SMOKE = os.environ.get("AKP_A8_SMOKE") == "1"
VP_VS, RHO = 1.8, 2000.0
BODY = dict(base=300.0, height=168.0, apex_depth=38.0)
LAYER = dict(thickness=30.0, vp=3000.0)
CASES = {  # name: (thawed host vp, frozen layer, bands measured)
    "thawed_1200": (1200.0, False, ("0.5-1", "1-2", "2-4")),
    "thawed_850": (850.0, False, ("0.5-1", "1-2")),
    "frozen_1200": (1200.0, True, ("0.5-1", "1-2", "2-4")),
    "frozen_850": (850.0, True, ("0.5-1", "1-2")),
}
BANDS = {"0.5-1": (0.5, 1.0), "1-2": (1.0, 2.0), "2-4": (2.0, 4.0)}
H, F0, PPW_RAYLEIGH = 10.0, 2.0, 15.4
RMAX, DR, MARGIN, DEPTH, Z_SHEET = (400.0, 100.0, 200.0, 400.0, 300.0) if SMOKE else (3000.0, 100.0, 300.0, 1000.0, 400.0)
AZIMUTHS = {"forward": (1.0, 0.0), "backward": (-1.0, 0.0), "perpendicular": (0.0, 1.0)}
RERUNS: dict[str, str] = {}  # "case_run": what failed and what was fixed (decision 1)
COLS = ["case", "azimuth", "range_m", "band", "window", "component", "numerical", "analytic", "ratio",
        "analytic_center", "ratio_center", "h_over_v", "phase_lag_deg", "t_lo_s", "t_hi_s"]


@lru_cache(maxsize=1)  # the travel-time model is slow to build, and a8_generous sets a case up many times to size its box
def ray_parameter() -> float:
    """iasp91 P ray parameter (s/m) for the 1992 test at the stage's station, computed as the stage does."""
    import pandas as pd
    from obspy.geodetics import locations2degrees
    from obspy.taup import TauPyModel
    ev = pd.read_csv(HERE.parent / "setup" / "pr_b_isc_event.csv").iloc[0]
    station = (64.873566, -147.861772)  # a2_wavefield.records(): the station of the stage's 1992 record
    dist = locations2degrees(float(ev["latitude"]), float(ev["longitude"]), *station)
    return float(TauPyModel("iasp91").get_travel_times(0.0, dist, ["P"])[0].ray_param / 6371000.0)


def setup(case, spec=None):
    """Everything a run and its measurement need: model, grid, timing, receivers.

    spec: a case that is not in CASES (a8_generous), as dict(host, bands) with any of
    stone, body, h, margin; a key left out takes the production value. No frozen layer.
    """
    if spec is None:
        vp, frozen, bands = CASES[case]
        spec = dict(host=(vp, vp / VP_VS, RHO))
    else:
        frozen, bands = False, spec["bands"]
    host = spec["host"]
    vp = host[0]
    stone = spec.get("stone", tuple(2.0 * q for q in host))
    body, h = spec.get("body", BODY), spec.get("h", H)
    layer = (LAYER["vp"], LAYER["vp"] / VP_VS, RHO)
    p = ray_parameter()
    half = RMAX + spec.get("margin", MARGIN)
    g = fd.make_grid(lo=(-half, 0.0, 0.0), hi=(half, half, DEPTH), h=h, free_surface=True, sym_y=True, nbl=fd.NBL)
    speeds = [host, stone] + ([layer] if frozen else [])
    vp_fast, vs_fast, vs_slow = max(s[0] for s in speeds), max(s[1] for s in speeds), min(s[1] for s in speeds)
    dt = fd.time_step(g, vp_fast)
    q_host = np.sqrt(1 / vp**2 - p**2)  # vertical slowness
    t_c = 1 / F0 + (half + g["nbl"] * h) * p + 0.05  # wavelet center at the sheet under x = 0
    t0 = t_c + (Z_SHEET - body["apex_depth"]) * q_host
    t_surf = t_c + (Z_SHEET - LAYER["thickness"] * frozen) * q_host \
        + LAYER["thickness"] * frozen * np.sqrt(1 / LAYER["vp"] ** 2 - p**2)  # wavelet center at the surface, x = 0
    t_low = 1 / np.mean(BANDS["0.5-1"])
    nt = int((t0 + RMAX / (0.85 * vs_slow) + 1.5 * t_low + 1.5) / dt) + 2
    ranges = np.arange(DR, RMAX + 1e-6, DR)
    nodes = fd.snap(g, np.vstack([np.column_stack([r * ux, r * uy, 0 * r]) for ux, uy in AZIMUTHS.values()
                                  for r in [ranges]]))
    from akp.stage_a.a3b4_rg import rayleigh_speed
    return dict(case=case, host=host, stone=stone, layer=layer if frozen else None, bands=bands, p=p, g=g, dt=dt, nt=nt,
                t_c=t_c, t0=t0, t_surf=t_surf, vp_fast=vp_fast, vs_fast=vs_fast, vs_slow=vs_slow, ranges=ranges,
                nodes=nodes, c_r=rayleigh_speed(host[0], host[1]), body=body, h=h, half=half)


def run_case(case, with_body, spec=None):
    """One simulation: the case with the body, or its twin."""
    s = setup(case, spec)
    host, layer = s["host"], s["layer"]

    def profile(z):
        if layer is None:
            return host
        top = z < LAYER["thickness"]
        return tuple(np.where(top, a, b) for a, b in zip(layer, host))
    body = fd.square_body(x=0.0, y=0.0, **s["body"]) if with_body else None
    par = fd.effective_params(s["g"], profile, body, s["stone"] if with_body else None)
    half = s["half"]
    out = fd.simulate(s["g"], par, dt=s["dt"], nt=s["nt"], v_ref=host[0], rec=fd.native(s["nodes"], s["h"]),
                      plane=dict(z=Z_SHEET, angle_deg=float(np.degrees(np.arcsin(s["p"] * host[0]))), vp=host[0],
                                 f0=F0, t0=s["t_c"], flat=(half, half)))
    return {k: (v.astype(np.float32) if isinstance(v, np.ndarray) else v) for k, v in out.items()}


def archive(case, with_body):
    from akp.paths import data_dir
    d = data_dir("a8") / ("smoke" if SMOKE else "production")
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{case}_{'body' if with_body else 'twin'}.npz"


def load(case, with_body):
    z = np.load(archive(case, with_body))
    return {k: (z[k] if z[k].ndim else z[k].item()) for k in z.files}


def run_missing(cases):
    """Run, one at a time and each in a fresh process, the simulations that have no archive yet."""
    import multiprocessing as mp
    for case in cases:
        for with_body in (False, True):
            path = archive(case, with_body)
            if path.exists():
                continue
            with mp.get_context("spawn").Pool(1) as pool:
                out = pool.apply(run_case, (case, with_body))
            np.savez_compressed(path, **out)
            print(f"{path.stem}: {out['npts'] / 1e6:.1f}M points, {out['nt']} steps, {out['seconds']:.0f} s, "
                  f"{out['mpts_per_s']:.0f} Mpts/s, peak rss {out['rss_gb']:.2f} GB", flush=True)


# ---------------------------------------------------------------- measurement
def band_envelope(x, band, dt):
    """Analytic signal of the zero-phase band-passed traces (time along axis 0)."""
    from scipy.signal import butter, hilbert, sosfiltfilt
    sos = butter(4, BANDS[band], btype="band", fs=1 / dt, output="sos")
    return hilbert(sosfiltfilt(sos, x, axis=0), axis=0)


def peak(env, t, lo, hi):
    sel = (t >= lo) & (t <= hi)
    return float(np.abs(env[sel]).max()) if sel.any() else float("nan")


def rayleigh_evidence(rad, ver, t, lo, hi):
    """(h_over_v, phase_lag_deg) of radial against vertical analytic signals in a window."""
    sel = (t >= lo) & (t <= hi)
    if not sel.any():
        return float("nan"), float("nan")
    return (float(np.abs(rad[sel]).max() / np.abs(ver[sel]).max()),
            float(np.degrees(np.angle(np.sum(rad[sel] * np.conj(ver[sel]))))))


def analytic_chain(host, band, body=BODY):
    """The stage's Born values for the body in a thawed host: (E_rg, B_body) at the band's upper and center frequency.

    Slow-window vertical share = E_rg / sqrt(r); body-wave radial share = B_body / r (r in m).
    """
    import pandas as pd

    from akp.stage_a import a3_detectability as a3
    from akp.stage_a import a3b4_rg as rg
    vp, vs, rho = host
    fac = float(pd.read_csv(HERE / "a3b_factors.csv").set_index("contrast").loc["dense_stiff", "factor_geo"])
    out = {}
    for label, f in (("top", BANDS[band][1]), ("center", float(np.mean(BANDS[band])))):
        lamb = rg.Lamb(vp=vp, vs=vs, rho=rho, freq=f)
        step = min(2 * np.pi / lamb.k / 10, body["base"] / 8, body["height"] / 8)
        xyz, dv = rg.square_body_voxels(body["base"], body["height"], body["apex_depth"], step)
        e_rg = rg.rg_3d_E(lamb, xyz, dv, *rg.perturbations(vp, vs, rho, 1.0)) * rg.qs_factor("dense_stiff")
        pgeo, _, vol, k = a3.max_pattern(f, vp, body["base"], body["height"], 1.0, theta_range=a3.GEO_THETA)
        out[label] = (float(e_rg), float(k**2 * vol * pgeo * fac / (4 * np.pi)))
    return out


def measure(case, spec=None, runs=None):
    """Rows of a8_production.csv and check rows for one case, from its two archived runs (or runs = (twin, body))."""
    s = setup(case, spec)
    twin, body = runs or (load(case, False), load(case, True))
    dt, nt = s["dt"], s["nt"]
    assert twin["dt"] == body["dt"] == dt and twin["nt"] == body["nt"] == nt
    t = np.arange(nt) * dt
    n = len(s["ranges"])
    sc = {c: body[c].astype(float) - twin[c].astype(float) for c in ("vx", "vy", "vz")}
    dense = s["stone"] == tuple(2.0 * q for q in s["host"])  # the chain has a value for the stage's dense, stiff body only
    rows, checks = [], []
    for band in s["bands"]:
        T = 1 / np.mean(BANDS[band])
        chain = analytic_chain(s["host"], band, s["body"]) if s["layer"] is None and dense else None
        inc = band_envelope(twin["vz"].astype(float), band, dt)
        env = {c: band_envelope(sc[c], band, dt) for c in sc}
        inc_peaks = []
        for a, (az, (ux, uy)) in enumerate(AZIMUTHS.items()):
            for i, r in enumerate(s["ranges"]):
                j = a * n + i
                t_inc = s["t_surf"] + s["p"] * r * ux
                inc_peak = peak(inc[:, j], t, t_inc - 1 / F0 - T, t_inc + 1 / F0 + T)
                inc_peaks.append((inc_peak, float(np.abs(inc[:, j]).max())))
                ver = env["vz"][:, j]
                rad = (ux * env["vx"][:, j]) if ux else env["vy"][:, j]
                t_split = s["t0"] + r / s["vs_fast"] - T / 2
                windows = {"fast": (s["t0"] + r / s["vp_fast"] - T / 2, t_split),
                           "slow": (t_split, s["t0"] + r / (0.85 * s["vs_slow"]) + 1.5 * T),
                           "host_p": (s["t0"] + r / s["host"][0] - T / 2, s["t0"] + r / s["host"][1] - T / 2)}
                for win, (lo, hi) in windows.items():
                    hv, lag = rayleigh_evidence(rad, ver, t, lo, hi) if win == "slow" else ("", "")
                    for comp, tr in (("vertical", ver), ("radial", rad)):
                        num = peak(tr, t, lo, hi) / inc_peak
                        ana = ana_c = ""
                        if chain is not None and win == "slow" and comp == "vertical":
                            ana, ana_c = chain["top"][0] / np.sqrt(r), chain["center"][0] / np.sqrt(r)
                        if chain is not None and win in ("fast", "host_p") and comp == "radial":
                            ana, ana_c = chain["top"][1] / r, chain["center"][1] / r
                        rows.append(dict(case=case, azimuth=az, range_m=float(r), band=band, window=win, component=comp,
                                         numerical=num, analytic=ana, ratio=num / ana if ana != "" else "",
                                         analytic_center=ana_c, ratio_center=num / ana_c if ana_c != "" else "",
                                         h_over_v=hv, phase_lag_deg=lag, t_lo_s=lo, t_hi_s=hi))
        ip = np.array(inc_peaks)
        checks += [dict(check=f"{case}_incident_peak_min_over_max_{band}", value=float(ip[:, 0].min() / ip[:, 0].max()),
                        detail="flatness of the incident vertical peak over the 90 receivers (edge waves of the sheet lower it)"),
                   dict(check=f"{case}_incident_window_peak_over_record_peak_min_{band}", value=float((ip[:, 0] / ip[:, 1]).min()),
                        detail="1 when nothing in the twin's record exceeds the incident pulse")]
    v_inc = float(np.abs(twin["vz"]).max())
    for name, run in (("twin", twin), ("body", body)):
        late = max(float(np.abs(run[c][-int(0.5 / dt):]).max()) for c in ("vx", "vy", "vz")) / v_inc
        checks += [dict(check=f"{case}_{name}_run_seconds", value=round(float(run["seconds"]), 1),
                        detail=f"{float(run['mpts_per_s']):.0f} Mpts/s"),
                   dict(check=f"{case}_{name}_peak_rss_gb", value=round(float(run["rss_gb"]), 2), detail="limit 8"),
                   dict(check=f"{case}_{name}_field_left_over_incident_peak", value=float(run["vmax_end"]) / v_inc,
                        detail="largest |v| anywhere in the box at the last step over the incident vertical peak at the surface: "
                               "the energy has left the box when this is small"),
                   dict(check=f"{case}_{name}_receivers_last_half_second_over_incident_peak", value=late, detail="")]
    f_top = BANDS[s["bands"][-1]][1]
    checks += [dict(check=f"{case}_grid_shape", value="x".join(str(v) for v in s["g"]["shape"]),
                    detail=f"h {s['h']} m, sponge {s['g']['nbl']} cells, mirror plane y = 0; {np.prod(s['g']['shape']) / 1e6:.1f}M points"),
               dict(check=f"{case}_time_step_s", value=float(dt), detail=f"{nt} steps"),
               dict(check=f"{case}_points_per_rayleigh_wavelength_at_top", value=round(s["c_r"] / f_top / s["h"], 1),
                    detail=f"host Rayleigh speed {s['c_r']:.0f} m/s at {f_top} Hz; rule {PPW_RAYLEIGH} (gate three)"),
               dict(check=f"{case}_bands_measured", value=" ".join(s["bands"]),
                    detail="" if len(s["bands"]) == 3 else "2-4 Hz not measured: 10 m cells give under 15.4 points per Rayleigh "
                                                           "wavelength at 4 Hz, and 7 m cells exceed 8 GB (decision 8)"),
               dict(check=f"{case}_incidence_deg", value=float(np.degrees(np.arcsin(s["p"] * s["host"][0]))), detail="in the thawed host"),
               dict(check=f"{case}_t0_s", value=float(s["t0"]), detail="incident P at the apex"),
               dict(check=f"{case}_window_speeds", value=f"{s['vp_fast']:.0f} {s['vs_fast']:.0f} {s['vs_slow']:.0f}",
                    detail="vp_fast vs_fast vs_slow (m/s)")]
    return rows, checks


def rayleigh_yardstick():
    """The Rayleigh evidence statistic on gate three's pure Rayleigh wave (2400 m), per band it carries."""
    from akp.stage_a import a8_gates as gates
    s = gates.surface_setup()
    z = np.load(gates.archive("surface", 2, "half_space"))
    t = np.arange(s["nt"]) * s["dt"]
    j, r = 1, gates.RANGES[1]
    lo, hi = s["t0"] + r / s["c_r"] - 0.6, s["t0"] + r / s["c_r"] + 0.6
    out = []
    for band in ("1-2", "2-4"):
        rad, ver = (band_envelope(z[c][:, j].astype(float), band, s["dt"]) for c in ("vx", "vz"))
        hv, lag = rayleigh_evidence(rad, ver, t, lo, hi)
        out.append(dict(check=f"rayleigh_yardstick_h_over_v_{band}", value=hv,
                        detail=f"gate three's Rayleigh wave at {r:g} m, same statistic; phase lag {lag:.1f} deg"))
        out.append(dict(check=f"rayleigh_yardstick_phase_lag_deg_{band}", value=lag, detail="radial outward against vertical (z down)"))
    return out


def write(rows, checks, cases):
    import pandas as pd
    from akp.paths import data_dir
    out = (data_dir("a8") / "smoke") if SMOKE else HERE
    pd.DataFrame(rows).reindex(columns=COLS).to_csv(out / "a8_production.csv", index=False)
    p = ray_parameter()
    common = [dict(check="ray_parameter_s_per_km", value=p * 1e3,
                   detail="iasp91 P, surface source, 1992 origin to the stage's station; computed, the stage stores no constant"),
              dict(check="wavelet", value=f"Ricker {F0} Hz", detail="spectrum at 0.5 and 4 Hz: 16% and 20% of its peak"),
              dict(check="simulations_archived", value=sum(archive(c, b).exists() for c in CASES for b in (False, True)),
                   detail="cap 8"),
              dict(check="reruns", value=len(RERUNS), detail="; ".join(f"{k}: {v}" for k, v in RERUNS.items()) or "none"),
              dict(check="sponge_caveat", value="", detail="a wave grazing a sponge is reflected from its inner edge (gate three, attempt 1: "
                   "9-45% of the direct Rayleigh wave); here such reflections of the scattered field come from the far side "
                   "and end sponges and arrive late in the slow window, after its peak")]
    try:
        common += rayleigh_yardstick()
    except FileNotFoundError:
        common.append(dict(check="rayleigh_yardstick", value="", detail="gate three's archive not found"))
    pd.DataFrame(common + checks).reindex(columns=["check", "value", "detail"]).to_csv(out / "a8_production_checks.csv", index=False)
    print(f"wrote {len(rows)} rows for {cases} to {out}")


def main(cases, measure_only=False):
    if not measure_only:
        run_missing(cases)
    rows, checks = [], []
    for case in cases:
        if archive(case, False).exists() and archive(case, True).exists():
            r, c = measure(case)
            rows += r
            checks += c
    write(rows, checks, cases)


def _selfcheck() -> None:
    """No simulation: the windows, the filters, the evidence statistic and the chain's scaling."""
    s = setup("thawed_1200")
    assert s["vp_fast"] == 2400.0 and abs(s["vs_slow"] - 1200 / 1.8) < 1e-9 and len(s["nodes"]) == 3 * len(s["ranges"])
    assert 0.055e-3 < s["p"] < 0.065e-3, s["p"]  # s/m: a teleseismic P near 65 degrees
    assert setup("frozen_850")["vp_fast"] == 3000.0 and s["t_surf"] > s["t0"] > s["t_c"]
    dt = 2e-3
    t = np.arange(6000) * dt
    pulse = np.exp(-((t - 6.0) / 1.2) ** 2)
    ver = band_envelope(pulse * np.cos(2 * np.pi * 1.5 * t), "1-2", dt)
    rad = band_envelope(0.7 * pulse * np.cos(2 * np.pi * 1.5 * t - np.pi / 2), "1-2", dt)  # radial a quarter cycle late
    hv, lag = rayleigh_evidence(rad, ver, t, 3.0, 9.0)
    assert abs(hv - 0.7) < 0.01 and abs(lag + 90.0) < 1.0, (hv, lag)  # a lag is negative
    assert abs(peak(ver, t, 3.0, 9.0) - 1.0) < 0.02 and np.isnan(peak(ver, t, 20.0, 21.0))
    assert peak(band_envelope(pulse * np.cos(2 * np.pi * 1.5 * t), "2-4", dt), t, 3.0, 9.0) < 0.05  # out of band
    c1, c2 = analytic_chain(s["host"], "1-2"), analytic_chain(s["host"], "2-4")
    assert all(v > 0 for v in c1["top"] + c1["center"] + c2["top"])
    assert abs(c1["top"][0] / c2["center"][0] - 1) > 0.01  # different frequencies give different values
    print("a8_production selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        names = [a for a in sys.argv[1:] if a in CASES] or list(CASES)
        main(names, measure_only="--measure" in sys.argv)
