"""A2: observed teleseismic P wavefield and P-coda clutter at Alaskan sites.

Records (open data only):
- test1992:   IU.COL, the only open Alaskan recording of the 1992-05-21 test;
- explosion:  the 2017 DPRK test at IU.COLA (about 4 km from COL) and at the 1992 in-box sites
              ANM, PMR, TTA (pr_h_events.csv);
- analog_eq:  the 2016 Xinjiang earthquake (near the 1992 path) at the same stations;
- teleseism:  shallow (<= 70 km) M>=6.3 events at 55-75 deg from COLA, within +-20 deg of the 1992
              back-azimuth, 2000-2024 (event-to-event spread at the same site).
Responses come through swe.io.get_response_files / load_response; raw miniSEED and StationXML are
cached under $AKP_DATA/a2. Writes a2_levels.csv, a2_clutter.csv, a2_checks.csv.
Run: python -m akp.stage_a.a2_wavefield
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime as T
from obspy import read
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.geodetics import gps2dist_azimuth, locations2degrees
from obspy.taup import TauPyModel

from akp.paths import data_dir
from swe.io import catalog_to_dataframe, get_events, get_response_files, load_response

HERE = Path(__file__).parent
SETUP = HERE.parent / "setup"
ES = "https://service.earthscope.org"
BANDS = [(0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0)]
LAGS = [(5.0, 15.0), (15.0, 30.0), (30.0, 60.0)]
NOISE = (-280.0, -20.0)
PEAK = (-3.0, 10.0)
SMOOTH_S = 1.0
PRE, POST = 300.0, 600.0
F1_PCT = (5.0, 95.0)
F2_FACTOR = 2.0
TELE = dict(start=T("2000-01-01"), end=T("2025-01-01"), mmin=6.3, dmin=55.0, dmax=75.0, depth_max=70.0, baz_tol=20.0)
TAUP = TauPyModel("iasp91")


def records() -> list[dict]:
    """(kind, event name, origin, lat, lon, depth_km, net, sta) for every record A2 measures."""
    isc = pd.read_csv(SETUP / "pr_b_isc_event.csv").iloc[0]
    o92 = dict(name="lopnor_1992", time=T(isc["origin_time"]), lat=float(isc["latitude"]),
               lon=float(isc["longitude"]), depth_km=0.0)
    recs = [dict(kind="test1992", **o92, net="IU", sta="COL")]
    ev = pd.read_csv(SETUP / "pr_h_events.csv")
    for _, e in ev[ev.kind == "analog"].iterrows():
        kind = "explosion" if "explosion" in e["name"] else "analog_eq"
        for net, sta in (("IU", "COLA"), ("AK", "ANM"), ("AT", "PMR"), ("AT", "TTA")):
            recs.append(dict(kind=kind, name=e["name"], time=T(e["time"]), lat=e.lat, lon=e.lon, depth_km=np.nan,
                             net=net, sta=sta))
    cola = (64.873566, -147.861772)
    baz92 = gps2dist_azimuth(*cola, o92["lat"], o92["lon"])[1]
    cat = catalog_to_dataframe(get_events(starttime=TELE["start"], endtime=TELE["end"], minmagnitude=TELE["mmin"],
                                          latitude=cola[0], longitude=cola[1],
                                          minradius=TELE["dmin"], maxradius=TELE["dmax"]))
    cat["baz"] = [gps2dist_azimuth(*cola, a, o)[1] for a, o in zip(cat.latitude, cat.longitude)]
    sel = cat[(np.abs(((cat.baz - baz92) + 180) % 360 - 180) <= TELE["baz_tol"]) & (cat.depth_km <= TELE["depth_max"])]
    for _, e in sel.sort_values("time").iterrows():
        recs.append(dict(kind="teleseism", name=e.time[:16], time=T(e.time), lat=e.latitude, lon=e.longitude,
                         depth_km=e.depth_km, net="IU", sta="COLA"))
    return recs


def fetch(c: Client, r: dict, wf_dir: Path, resp_dir: Path, sta_xy: dict) -> tuple:
    """Raw vertical trace around predicted P, its response inventory, and the P time."""
    lat, lon = sta_xy[(r["net"], r["sta"])]
    dist = locations2degrees(r["lat"], r["lon"], lat, lon)
    depth = 0.0 if np.isnan(r["depth_km"]) else float(r["depth_km"])
    tp = r["time"] + TAUP.get_travel_times(depth, dist, ["P"])[0].time
    path = wf_dir / f"{r['net']}.{r['sta']}.{r['name'].replace(':', '')}.mseed"
    if not path.exists():
        c.get_waveforms(r["net"], r["sta"], "*", "BHZ", tp - PRE, tp + POST, filename=str(path))
    st = read(str(path))
    locs = sorted({tr.stats.location for tr in st})
    st = st.select(location=locs[0])  # one sensor: the lowest location code ("" or "00" first)
    tr = st[0]
    try:
        inv = load_response(tr, resp_dir)
    except FileNotFoundError:
        get_response_files(r["net"], [r["sta"]], "BHZ", tp - PRE, tp + POST, output_dir=resp_dir, client=c)
        inv = load_response(tr, resp_dir)
    return st, inv, tp, dist


def clipped(data: np.ndarray, run: int = 3) -> bool:
    """True if the absolute maximum repeats on >= `run` consecutive samples (a clipping plateau)."""
    a = np.abs(data)
    hit = np.flatnonzero(a == a.max())
    return bool(len(hit) >= run and np.any(np.convolve(np.diff(hit) == 1, np.ones(run - 1), "valid") == run - 1))


def envelopes(tr, fmin, fmax):
    """(times relative to start, Hilbert envelope smoothed SMOOTH_S, RMS envelope in SMOOTH_S windows)."""
    from scipy.signal import hilbert
    f = tr.copy().filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
    x = f.data.astype(np.float64)
    n = max(1, int(round(SMOOTH_S * f.stats.sampling_rate)))
    k = np.ones(n) / n
    hil = np.convolve(np.abs(hilbert(x)), k, "same")
    rms = np.sqrt(np.convolve(x**2, k, "same")) * np.sqrt(2.0)  # sqrt(2): RMS -> envelope of a sinusoid
    return f.times(), hil, rms


def main() -> None:
    root = data_dir("a2")
    wf_dir, resp_dir = data_dir("a2/wf"), data_dir("a2/resp")
    c = Client(base_url=ES, timeout=600)
    recs = records()
    sta_xy = {}
    for net, sta in sorted({(r["net"], r["sta"]) for r in recs}):
        inv = c.get_stations(network=net, station=sta, level="station")
        s = inv[0][0]
        sta_xy[(net, sta)] = (s.latitude, s.longitude)
    levels, clutter = [], []
    for r in recs:
        try:
            st, inv, tp, dist = fetch(c, r, wf_dir, resp_dir, sta_xy)
        except (FDSNNoDataException, FileNotFoundError, IndexError) as exc:
            levels.append(dict(kind=r["kind"], event=r["name"], net=r["net"], sta=r["sta"], status=f"no data: {type(exc).__name__}"))
            continue
        gaps = len(st.get_gaps())
        raw = st.copy().merge(method=0)[0]
        is_clipped = clipped(raw.data.astype(np.float64))
        tr = raw.copy()
        if np.ma.isMaskedArray(tr.data):
            tr.data = tr.data.filled(0)
        nyq = tr.stats.sampling_rate / 2.0
        tr.detrend("demean").taper(0.02)
        tr.remove_response(inventory=inv, output="VEL", pre_filt=(0.2, 0.3, 0.9 * nyq, 0.98 * nyq), water_level=None)
        base = dict(kind=r["kind"], event=r["name"], net=r["net"], sta=r["sta"], loc=raw.stats.location,
                    fs=tr.stats.sampling_rate, dist_deg=round(dist, 2), depth_km=r["depth_km"])
        for fmin, fmax in BANDS:
            if fmax >= 0.95 * nyq:
                continue
            t, hil, rms = envelopes(tr, fmin, fmax)
            rel = t - (tp - tr.stats.starttime)
            win = (rel >= PEAK[0]) & (rel <= PEAK[1])
            if not win.any():
                continue
            ipk = np.flatnonzero(win)[np.argmax(hil[win])]
            tpk = rel[ipk]
            for est, env in (("hilbert", hil), ("rms", rms)):
                peak = env[ipk] if est == "hilbert" else env[win].max()
                nz = env[(rel >= NOISE[0]) & (rel <= NOISE[1])]
                levels.append(dict(base, status="ok", band=f"{fmin:g}-{fmax:g}", estimator=est, p_peak_vel=peak,
                                   noise_vel=float(np.median(nz)), noise_ratio=float(np.median(nz) / peak),
                                   p_peak_lag_s=round(float(tpk), 2), gaps=gaps, clipped=is_clipped))
                for a, b in LAGS:
                    w = env[(rel >= tpk + a) & (rel <= tpk + b)]
                    clutter.append(dict(base, band=f"{fmin:g}-{fmax:g}", estimator=est, lag=f"{a:g}-{b:g}",
                                        ratio=float(np.median(w) / peak) if w.size else np.nan))
    lv, cl = pd.DataFrame(levels), pd.DataFrame(clutter)
    lv.to_csv(HERE / "a2_levels.csv", index=False)
    cl.to_csv(HERE / "a2_clutter.csv", index=False)

    # gate
    h = cl[cl.estimator == "hilbert"]
    t92 = h[h.kind == "test1992"].set_index(["band", "lag"]).ratio
    tele = h[(h.kind == "teleseism") & (h.sta == "COLA")]
    outside, cells = [], 0
    for (band, lag), v in t92.items():
        ref = tele[(tele.band == band) & (tele.lag == lag)].ratio.dropna()
        if len(ref) < 5:
            continue
        cells += 1
        lo, hi = np.percentile(ref, F1_PCT)
        if not lo <= v <= hi:
            outside.append(f"{band}Hz/{lag}s:{v:.3f} vs [{lo:.3f},{hi:.3f}]")
    r92 = cl[cl.kind == "test1992"].pivot_table(index=["band", "lag"], columns="estimator", values="ratio")
    fac = np.maximum(r92.hilbert / r92.rms, r92.rms / r92.hilbert)
    l92 = lv[(lv.kind == "test1992") & (lv.status == "ok")]
    checks = [
        dict(check="F1_outside_cells", value=len(outside), of=cells, triggered=bool(cells and len(outside) >= cells / 2),
             detail="; ".join(outside)),
        dict(check="F2_max_estimator_factor", value=round(float(fac.max()), 3), of="", triggered=bool(fac.max() > F2_FACTOR),
             detail=f"cell {fac.idxmax()}"),
        dict(check="F3_gaps_or_clipping", value=int(l92.gaps.max() > 0 or l92.clipped.any()), of="",
             triggered=bool(l92.gaps.max() > 0 or l92.clipped.any()),
             detail=f"gaps={int(l92.gaps.max())} clipped={bool(l92.clipped.any())}"),
        dict(check="n_teleseisms_cola", value=int(tele.event.nunique()), of="", triggered="", detail=""),
    ]
    checks.append(dict(check="gate_pass", value=int(not any(ch["triggered"] is True for ch in checks)), of="", triggered="", detail=""))
    with (HERE / "a2_checks.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(checks[0]))
        w.writeheader()
        w.writerows(checks)
    print(lv.groupby(["kind", "status"]).size().to_dict())
    for ch in checks:
        print({k: str(v)[:200] for k, v in ch.items()})


if __name__ == "__main__":
    x = np.zeros(100); x[50:53] = 7.0
    assert clipped(x) and not clipped(np.sin(np.linspace(0, 20, 500)))
    main()
