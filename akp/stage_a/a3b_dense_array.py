"""A3b-3: dense-array upper bound with real ZE 2019 teleseism data.

Per grid cell (body, contrast, apex depth, host vp, band):
- Traces (vertical DPZ, in-box nodes, a3b_fetch_ze.py) are band-passed and normalized by each node's
  own direct-P envelope peak (the gain cancels; counts are fine).
- A synthetic scattered arrival at node i is that node's own direct-P waveform scaled by
  A_i = Born P->P amplitude (swe.scattering, with the body's voxel form factor and the A3b-1
  quasi-static factor) and delayed by tau_i (plane-wave incidence + scattering path).
- Incidence is treated as vertical for the form factor (the teleseismic tilt is a few degrees);
  delays use the full plane-wave slowness.
- Detector (knows the true position): stack over nodes within DETECT_KM of the body, aligned on tau_i
  and weighted by A_i/||A||; statistic = RMS of the stack in STACK_WIN around zero lag.
- Threshold = FA_PCT percentile of the statistic at target-free random positions (N_NULL);
  P_d from swe.detection.injection_recovery with random positions (N_TRIALS).
Writes a3b3_pd.csv and a3b3_checks.csv. Run: python -m akp.stage_a.a3b_dense_array
"""
from __future__ import annotations

import csv
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime as T
from obspy import read
from obspy.geodetics import gps2dist_azimuth, locations2degrees
from obspy.taup import TauPyModel
from pyproj import Geod

from akp.paths import data_dir
from akp.stage_a import a3_detectability as a3
from swe.detection import clopper_pearson, injection_recovery
from swe.scattering import born_point_scattering, voxel_form_factor

HERE = Path(__file__).parent
GEOD = Geod(ellps="WGS84")
BODIES = [(100.0, 50.0), (230.0, 168.0), (500.0, 300.0)]
APEX_DEPTHS = [30.0, 200.0]
HOST_VP = [1200.0, 2000.0, 3000.0]
BANDS = {"1-2": (1.0, 2.0), "2-4": (2.0, 4.0)}
DETECT_KM = 3.0
STACK_WIN = (-0.5, 1.5)
FA_PCT = 99.0
N_NULL = 500
N_TRIALS = 200
OFFSET_KM = 1.0  # body placed within this distance of a randomly chosen node (inside the footprint)
P_WIN = (-3.0, 10.0)  # direct-P envelope-peak search window around the predicted P
WAVELET = (-2.0, 6.0)  # direct-P wavelet around its envelope peak
N_VOX = 24


class Array:
    """Band-limited, normalized node traces and geometry for one band."""

    def __init__(self, band: tuple[float, float]):
        fz = pd.read_csv(HERE / "a3b_fetch_ze.csv")
        fz = fz[fz.bytes > 0].reset_index(drop=True)
        d = data_dir("a3b/ze2019_event")
        self.lat, self.lon, self.data, self.wavelet, self.ipk = [], [], [], [], []
        fs = None
        for _, r in fz.iterrows():
            st = read(str(d / f"ZE.{r.station}.DPZ.mseed")).merge(method=0)
            if len(st) != 1 or np.ma.isMaskedArray(st[0].data):
                continue
            tr = st[0]
            tr.detrend("demean").taper(0.02).filter("bandpass", freqmin=band[0], freqmax=band[1], corners=4, zerophase=True)
            fs = tr.stats.sampling_rate
            t = tr.times() + (tr.stats.starttime - T(r.p_time))
            x = tr.data.astype(float)
            env = np.abs(x)
            win = (t >= P_WIN[0]) & (t <= P_WIN[1])
            ipk = int(np.flatnonzero(win)[np.argmax(env[win])])
            x = x / env[ipk]
            w = np.zeros_like(x)
            i0, i1 = ipk + int(WAVELET[0] * fs), ipk + int(WAVELET[1] * fs)
            w[i0:i1] = x[i0:i1]
            self.lat.append(r.lat)
            self.lon.append(r.lon)
            self.data.append(x)
            self.wavelet.append(w)
            self.ipk.append(ipk)
        self.fs = fs
        self.lat, self.lon = np.array(self.lat), np.array(self.lon)
        n = min(len(x) for x in self.data)
        self.data = np.array([x[:n] for x in self.data])
        self.wavelet = np.array([w[:n] for w in self.wavelet])
        self.ipk = np.array(self.ipk)
        ev = pd.read_csv(HERE.parent / "setup" / "pr_h_events.csv")
        e = ev[ev.kind == "ze2019_teleseism"].iloc[0]
        clat, clon = self.lat.mean(), self.lon.mean()
        dist = locations2degrees(e.lat, e.lon, clat, clon)
        arr = TauPyModel("iasp91").get_travel_times(0.0, dist, ["P"])[0]
        self.p_s_per_m = arr.ray_param / 6371000.0  # s/rad -> s/m
        self.prop_az = (gps2dist_azimuth(clat, clon, e.lat, e.lon)[1] + 180.0) % 360.0  # wave travels away from event


def geometry(arr: Array, blat, blon, apex_depth, height, vp):
    """Per-node (index, R, theta, tau, q-vector) for nodes within DETECT_KM of the body."""
    az, _, dist = GEOD.inv(np.full(len(arr.lat), blon), np.full(len(arr.lat), blat), arr.lon, arr.lat)
    near = dist <= DETECT_KM * 1000.0
    idx = np.flatnonzero(near)
    zc = apex_depth + 0.75 * height  # centroid depth of an apex-up square-based body
    xe, yn = dist[idx] * np.sin(np.radians(az[idx])), dist[idx] * np.cos(np.radians(az[idx]))
    rvec = np.column_stack([xe, yn, np.full(len(idx), zc)])  # body -> node (z up)
    R = np.linalg.norm(rvec, axis=1)
    p = arr.p_s_per_m
    eta = np.sqrt(max(1.0 / vp**2 - p**2, 0.0))
    s = np.array([np.sin(np.radians(arr.prop_az)), np.cos(np.radians(arr.prop_az))])
    e_hat = np.array([p * vp * s[0], p * vp * s[1], eta * vp])  # unit incident direction (upgoing)
    theta = np.arccos(np.clip((rvec / R[:, None]) @ e_hat, -1.0, 1.0))
    # incident arrival at body relative to node i, then scattered path: tau_i = t_body + R/v - t_node
    horiz = np.column_stack([xe, yn])  # node position relative to body
    tau = -(horiz @ (p * s)) - eta * zc + R / vp
    return idx, R, theta, tau, e_hat


def amplitudes(R, theta, e_hat, *, freq, vp, base, height, rel, qs_factor):
    k = 2 * np.pi * freq / vp
    inside, bounds = a3.square_based_body(base, height)
    rhat = np.column_stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)])  # angle only enters via theta
    q = k * (rhat - np.array([0.0, 0.0, 1.0]))
    f, vol = voxel_form_factor(inside, bounds=bounds, q_vector=q, n=N_VOX)
    vs = vp / a3.VP_VS
    out, far = [], []
    for Ri, th, fi in zip(R, theta, np.atleast_1d(f)):
        res = born_point_scattering(frequency=freq, volume=vol, distance=float(Ri), theta=float(th), vp=vp, vs=vs,
                                    density=a3.DENSITY, dvp=rel * vp, dvs=rel * vs, drho=rel * a3.DENSITY,
                                    form_factor=complex(fi))
        out.append(np.real(res.amplitude) * qs_factor)
        far.append(res.far_field)
    return np.array(out), np.array(far)


def statistic(arr: Array, idx, tau, weights, inject_amp=None):
    """Weighted delay-and-sum RMS around the predicted scattered arrival."""
    fs = arr.fs
    i0, i1 = int(STACK_WIN[0] * fs), int(STACK_WIN[1] * fs)
    n = i1 - i0
    stack = np.zeros(n)
    for j, (i, tj, wj) in enumerate(zip(idx, tau, weights)):
        x = arr.data[i]
        shift = int(round(tj * fs))
        if inject_amp is not None and inject_amp[j] != 0:
            x = x + inject_amp[j] * np.roll(arr.wavelet[i], shift)
        c = arr.ipk[i] + shift
        if c + i0 < 0 or c + i1 > len(x):
            continue
        stack += wj * x[c + i0 : c + i1]
    return float(np.sqrt(np.mean(stack**2)))


def random_position(arr: Array, rng):
    i = rng.integers(len(arr.lat))
    az, r = rng.uniform(0, 360), OFFSET_KM * 1000.0 * np.sqrt(rng.uniform())
    lon, lat, _ = GEOD.fwd(arr.lon[i], arr.lat[i], az, r)
    return lat, lon


def main() -> None:
    fac = pd.read_csv(HERE / "a3b_factors.csv").set_index("contrast").factor
    rows, checks = [], []
    for band, (fmin, fmax) in BANDS.items():
        arr = Array((fmin, fmax))
        for (base, height), cname, apex, vp in product(BODIES, a3.CONTRASTS, APEX_DEPTHS, HOST_VP):
            rel = a3.CONTRASTS[cname]
            kw = dict(freq=fmax, vp=vp, base=base, height=height, rel=rel, qs_factor=float(fac[cname]))

            def setup(rng):
                while True:
                    blat, blon = random_position(arr, rng)
                    idx, R, theta, tau, e_hat = geometry(arr, blat, blon, apex, height, vp)
                    if len(idx) >= 2:
                        amp, far = amplitudes(R, theta, e_hat, **kw)
                        w = amp / np.linalg.norm(amp) if np.linalg.norm(amp) > 0 else np.ones_like(amp) / np.sqrt(len(amp))
                        return idx, tau, w, amp, far

            rng = np.random.default_rng([7, int(vp), int(base), int(apex), len(band)])
            null, near_frac = [], []
            for _ in range(N_NULL):
                idx, tau, w, amp, far = setup(rng)
                null.append(statistic(arr, idx, tau, w))
                near_frac.append(1.0 - float(far.mean()))
            thr = float(np.percentile(null, FA_PCT))

            def make_trial(rng, params):
                idx, tau, w, amp, far = setup(rng)
                return statistic(arr, idx, tau, w, inject_amp=amp * params["scale"])

            def detect(data, params):
                return data > thr

            df = injection_recovery(make_trial=make_trial, detect=detect, grid=[{"scale": 1.0}],
                                    n_trials=N_TRIALS, random_state=11)
            r = df.iloc[0]
            rows.append(dict(band=band, base_m=base, height_m=height, contrast=cname, apex_depth_m=apex, host_vp=vp,
                             threshold=round(thr, 5), p_detect=r.p_detect, ci_low=round(r.ci_low, 4),
                             ci_high=round(r.ci_high, 4), n_trials=int(r.n_trials),
                             near_field_node_frac=round(float(np.mean(near_frac)), 3)))
            if band == "2-4" and (base, height) == (230.0, 168.0) and cname == "cavity" and apex == 30.0 and vp == 1200.0:
                sanity = injection_recovery(make_trial=make_trial, detect=detect,
                                            grid=[{"scale": 100.0}, {"scale": 0.0}], n_trials=N_NULL, random_state=13)
                p100, p0 = sanity.p_detect.iloc[0], sanity.p_detect.iloc[1]
                k0 = int(sanity.n_detected.iloc[1])
                lo, hi = clopper_pearson(k0, N_NULL)
                fa = 1.0 - FA_PCT / 100.0
                checks.append(dict(check="sanity_scale100_pd", value=round(p100, 4), passed=bool(p100 >= 0.95)))
                checks.append(dict(check="sanity_zero_pd", value=round(p0, 4), passed=bool(lo <= fa <= hi)))
    pd_df = pd.DataFrame(rows)
    pd_df.to_csv(HERE / "a3b3_pd.csv", index=False)
    checks.append(dict(check="n_nodes_used", value=len(Array(BANDS["2-4"]).lat), passed=""))
    checks.append(dict(check="gate_pass", value=int(all(c["passed"] for c in checks if c["passed"] != "")), passed=""))
    for vp in HOST_VP:
        sub = pd_df[pd_df.host_vp == vp]
        checks.append(dict(check=f"max_pd_vp_{int(vp)}", value=float(sub.p_detect.max()), passed=""))
        checks.append(dict(check=f"min_pd_vp_{int(vp)}", value=float(sub.p_detect.min()), passed=""))
    with (HERE / "a3b3_checks.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["check", "value", "passed"])
        w.writeheader()
        w.writerows(checks)
    print(pd_df.groupby(["band", "host_vp"]).p_detect.describe()[["min", "50%", "max"]])
    for c in checks:
        print(c)


if __name__ == "__main__":
    main()
