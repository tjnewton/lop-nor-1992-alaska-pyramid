"""A6: pre-check for P4 item S4 (coda f-k on the open small arrays) and scoping for S8 (blast Rg).

Question: is fundamental-mode Rg unaliased on any open Alaskan small
array in a band where the teleseismic P coda has signal? And (S8): do open catalogs hold blasts
whose Rg could cross an array or node line on frozen ground?

Inputs (all open, no account; nothing identifying is sent):
- EarthScope FDSN station metadata, network IM (ILAR, Burnt Mountain, Beaver Creek, Indian
  Mountain), channel SHZ, and network ZE 2019 (nodes; the 15xx cluster is listed as a dense
  reference geometry). StationXML cached under $AKP_DATA/a6.
- USGS FDSN event catalog: the DPRK tests of 2016-2017 (type "nuclear explosion"), M>=6
  teleseisms since the youngest array opened, and events typed "explosion" in interior Alaska.
  CSV cached under $AKP_DATA/a6, so a later catalog revision cannot change the numbers silently.
- EarthScope waveforms: SHZ, predicted P (iasp91) -120 s to +300 s, every array element, three
  DPRK tests; and two blast windows for the S8 look. Raw miniSEED cached under $AKP_DATA/a6.

Method:
- Geometry (k_sidelobe_*: smallest wavenumber f ds, cycles/km, at which a sidelobe reaches that
  level; Rg at f, c is free of such sidelobes while 2 f / c stays below it): aperture (largest pair distance), smallest pair distance, median nearest-neighbor
  distance, from WGS84 geodesics; local x (east), y (north) about the centroid.
- Array response |mean_i exp(2 pi i f ds.r_i)|^2 on a polar grid of slowness difference ds.
  Along each azimuth: half-width = first ds where the response falls below 0.5 (-3 dB);
  alias = first ds beyond that where it is back at or above 0.5. s_alias is the smallest alias
  over azimuth: the unaliased slowness radius. Rg at phase velocity c is called unaliased when
  s_alias > 2/c (two Rg arrivals from any two azimuths cannot be confused), and unaliased against
  steep P when s_alias > 1/c + S_P. sl_max is the largest response outside the main lobe within
  2/c, so another threshold can be applied.
- Resolution: slowness = the larger half-width; azimuth = 2 asin(half-width c) (full width, 360
  when the half-width reaches the Rg slowness); cross-range cell at range R = R x azimuth width;
  radial cell = c x window, window = aperture/c + N_CYC/f (array transit plus N_CYC cycles): a
  formula for a sliding-window plane-wave f-k, not a measurement. fraunhofer_km = 2 D^2 f / c is
  the range below which the plane-wave assumption fails.
- SNR: per element, zero-phase band-pass; noise = rms over P-100..P-15 s; t_pk = time of the
  largest array-median 1 s smoothed amplitude within +-10 s of predicted P; P = rms over
  t_pk-2..t_pk+3; coda = rms over t_pk+5..+30 and +30..+60. Amplitude ratios, median over elements.
- S8: counts of cataloged explosions near the ZE 2019 node line during its deployment and near
  ILAR, with distances to the nearest and farthest element. A look, not the measurement: for one
  blast per array, 1-2 Hz rms in the group-velocity window r/U_FAST..r/U_SLOW over pre-origin
  noise, and the group velocity of the envelope peak. For the ILAR blast, plane-wave semblance
  over slowness in the P window (control), the Rg window and a pre-origin noise window: is Rg coherent across the array and
  from the blast's direction?

Outputs: a6_array_precheck.csv (one table; `section` says which columns apply):
  element | geometry | response | snr | teleseism | s8_events | s8_look | rg_beam
Run: PYTHONPATH=. conda run -n swe-akp python -m akp.stage_a.a6_array_precheck
"""
from __future__ import annotations

import re
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime as T
from obspy import read, read_inventory
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel
from pyproj import Geod

from akp.paths import data_dir

HERE = Path(__file__).parent
GEOD = Geod(ellps="WGS84")
ES = "https://service.earthscope.org"
USGS = "https://earthquake.usgs.gov/fdsnws/event/1/query"

ARRAYS = {  # name -> station-code pattern (network IM, SHZ). IL31 (broadband at IL01) and the ILAR beam are left out
    "ILAR": r"IL(0[1-9]|1[0-9])", "BM": r"BM0[1-5]", "BC": r"BC0[1-5]", "IM": r"IM0[1-5]"}
ZE_SPAN = (T("2019-02-11"), T("2019-03-27"))  # ZE 2019 deployment (pr_c_ze2019)
ZE_DENSE = r"15\d\d"  # dense cluster of the ZE 2019 nodes: reference geometry only (no DPRK test in its epoch)
FREQS = (0.5, 1.0, 2.0, 4.0)  # Hz (task input)
C_RG = (1.5, 3.2)  # km/s, Rg phase velocity range (task input)
S_P = 0.08  # s/km, upper teleseismic P slowness (about 30 deg; input)
HALF = 0.5  # -3 dB response level that defines main-lobe width and an alias (input)
N_CYC = 3  # cycles in the f-k window beyond the array transit time (input)
RANGES = (10, 20, 40)  # km (task input)
BANDS = ((0.5, 1.0), (1.0, 2.0), (2.0, 4.0))  # Hz (task input)
PRE, POST = 120.0, 300.0  # s around predicted P
NOISE = (-100.0, -15.0)  # s from predicted P
P_WIN, CODA1, CODA2 = (-2.0, 3.0), (5.0, 30.0), (30.0, 60.0)  # s from t_pk
TELE_FROM, TELE_TO = "2006-07-01", "2025-12-31"  # all four arrays open .. fixed end for reproducibility
BLAST_BOX = dict(minlatitude=62.0, maxlatitude=66.5, minlongitude=-152.0, maxlongitude=-143.0)
BLAST_FROM, BLAST_TO = "2005-01-01", "2025-12-31"
NEAR_KM = 50.0  # an explosion counts for an array or line if within this of its nearest element (input)
LOOK_KM = 100.0  # the S8 look samples nodes out to this distance (input)
U_BEAM = 1.2  # km/s, slow end of the Rg beam window (input)
FROZEN_MONTHS = (11, 12, 1, 2, 3, 4)  # season label only; ground state must be documented separately
U_FAST, U_SLOW = 3.2, 1.0  # km/s, group-velocity window for the S8 look (input)
LOOK_NODES = 16  # ZE nodes sampled across distance for the S8 look (input)


def km(lat1, lon1, lat2, lon2):
    return GEOD.inv(lon1, lat1, lon2, lat2)[2] / 1000.0


def local_xy(lat, lon):
    """East, north (km) of each point about the centroid."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    la0, lo0 = lat.mean(), lon.mean()
    az, _, d = GEOD.inv(np.full_like(lon, lo0), np.full_like(lat, la0), lon, lat)
    return d / 1000.0 * np.sin(np.deg2rad(az)), d / 1000.0 * np.cos(np.deg2rad(az))


def geometry(x, y):
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    np.fill_diagonal(d, 0.0)
    return dict(n_elem=len(x), aperture_km=d.max(), min_spacing_km=nn.min(), median_nn_spacing_km=np.median(nn))


def response(x, y, f, s_max, ds=0.001, n_az=180):
    """(radii, azimuths deg, |array response|^2 [az, radius]); symmetric in ds, so 180 deg suffice."""
    r = np.arange(0.0, s_max + ds, ds)
    az = np.arange(n_az) * 180.0 / n_az
    sx, sy = np.outer(np.sin(np.deg2rad(az)), r), np.outer(np.cos(np.deg2rad(az)), r)
    acc = np.zeros(sx.shape, complex)
    for xi, yi in zip(x, y):
        acc += np.exp(2j * np.pi * f * (sx * xi + sy * yi))
    return r, az, np.abs(acc / len(x)) ** 2


def lobes(r, p):
    """Per azimuth: main-lobe half-width, first alias radius (inf if none), index where the main lobe ends."""
    below = p < HALF
    has = below.any(axis=1)
    i_hw = np.where(has, below.argmax(axis=1), len(r) - 1)
    hw = np.where(has, r[i_hw], np.inf)
    alias = np.full(len(p), np.inf)
    i_min = np.zeros(len(p), int)
    for k in range(len(p)):
        back = np.nonzero(~below[k, i_hw[k]:])[0]
        if has[k] and len(back):
            alias[k] = r[i_hw[k] + back[0]]
        rising = np.nonzero(np.diff(p[k]) > 0)[0]  # first local minimum: end of the main lobe
        i_min[k] = rising[0] if len(rising) else len(r) - 1
    return hw, alias, i_min


def response_rows(name, x, y, aperture):
    """Response rows per frequency and Rg velocity, plus the sidelobe summary by wavenumber.

    The response depends on f and ds only through k = f ds (cycles/km), so it is computed once at 1 Hz.
    """
    k, _, p = response(x, y, 1.0, max(FREQS) * 2.0 / min(C_RG) + 0.01)
    hw, alias, i_min = lobes(k, p)
    side = np.where(np.arange(len(k))[None, :] >= i_min[:, None], p, 0.0).max(axis=0)  # worst azimuth, outside the main lobe

    def first(level):
        hit = np.nonzero(side >= level)[0]
        return float(k[hit[0]]) if len(hit) else np.inf
    summary = {"k_sidelobe_0.25_cpkm": first(0.25), "k_sidelobe_0.5_cpkm": first(HALF)}
    rows = []
    for f in FREQS:
        s_alias = float(alias.min()) / f
        for c in C_RG:
            s = 1.0 / c
            h = hw.max() / f
            width = 360.0 if h >= s else float(np.rad2deg(2 * np.arcsin(h / s)))
            win = aperture / c + N_CYC / f
            row = dict(section="response", array=name, freq_hz=f, c_kms=c, hw_min_skm=hw.min() / f, hw_max_skm=h,
                       s_alias_skm=s_alias, sl_max=float(side[k <= 2 * s * f].max()),
                       rg_unaliased=s_alias > 2 * s, rg_unaliased_vs_p=s_alias > s + S_P,
                       az_width_deg=width, fk_window_s=win, radial_cell_km=c * win,
                       fraunhofer_km=2 * aperture ** 2 * f / c)
            for rng in RANGES:
                row[f"cross_cell_{rng}km"] = np.nan if width >= 360 else rng * np.deg2rad(width)
            rows.append(row)
    return summary, rows


def beam(st, x, y, s_max=0.8, ds=0.02):
    """Plane-wave semblance of unit-energy traces over a slowness grid: (peak, |s| at peak, back-azimuth deg).

    x, y follow the trace order. Semblance is 1 for identical delayed traces and about 1/N for incoherent ones.
    simplification: plane wave; inside the Fraunhofer range a curved front loses some semblance, so this is a lower bound.
    """
    n = min(tr.stats.npts for tr in st)
    d = np.array([tr.data[:n].astype(float) for tr in st])
    t0 = np.array([tr.stats.starttime - st[0].stats.starttime for tr in st])
    d = (d - d.mean(axis=1, keepdims=True)) * np.hanning(n)
    d /= np.sqrt((d ** 2).sum(axis=1, keepdims=True))
    f = np.fft.rfftfreq(n, st[0].stats.delta)
    spec = np.fft.rfft(d, axis=1) * np.exp(-2j * np.pi * f[None, :] * t0[:, None])
    g = np.arange(-s_max, s_max + ds / 2, ds)
    best = (0.0, 0.0, 0.0)
    for sx in g:
        for sy in g:
            b = (spec * np.exp(2j * np.pi * f[None, :] * (sx * x + sy * y)[:, None])).sum(axis=0)
            sem = float((np.abs(b) ** 2).sum() / (len(st) * (np.abs(spec) ** 2).sum()))
            if sem > best[0]:
                best = (sem, float(np.hypot(sx, sy)), float(np.rad2deg(np.arctan2(-sx, -sy)) % 360))
    return best


def usgs(cache: Path, name: str, **params) -> pd.DataFrame:
    path = cache / f"usgs_{name}.csv"
    if not path.exists():
        url = f"{USGS}?{urllib.parse.urlencode(dict(format='csv', orderby='time-asc', **params))}"
        with urllib.request.urlopen(url, timeout=300) as r:
            path.write_bytes(r.read())
    df = pd.read_csv(path)
    df["t"] = pd.to_datetime(df.time, utc=True)
    return df


def waveforms(c, cache: Path, tag: str, net: str, stations: list[str], cha: str, t1, t2):
    """Cached miniSEED for one window; None if the server has no data."""
    path = cache / f"{tag}.mseed"
    if not path.exists():
        try:
            c.get_waveforms_bulk([(net, s, "*", cha, t1, t2) for s in stations], filename=str(path))
        except FDSNNoDataException:
            return None
    st = read(str(path))
    st.merge(method=1, fill_value=0)
    return st


def rms(tr, t1, t2):
    d = tr.slice(t1, t2).data
    return float(np.sqrt(np.mean(d.astype(float) ** 2))) if len(d) else np.nan


def band(st, lo, hi):
    s = st.copy()
    s.detrend("demean")
    s.taper(0.02)
    s.filter("bandpass", freqmin=lo, freqmax=hi, corners=4, zerophase=True)
    return s


def snr_rows(name, ev_name, st, tp, n_meta, extra):
    full = [tr for tr in st if tr.stats.npts >= 0.9 * (PRE + POST) * tr.stats.sampling_rate]
    rows = []
    if not full:
        return [dict(section="snr", array=name, event=ev_name, n_elem=n_meta, n_data=0, **extra)]
    ref = band(st.__class__(full), 1.0, 2.0)
    n = min(tr.stats.npts for tr in ref)
    k = int(ref[0].stats.sampling_rate)
    env = np.median([np.convolve(np.abs(tr.data[:n]), np.ones(k) / k, "same") for tr in ref], axis=0)
    t = ref[0].times(reftime=tp)[:n]  # s from predicted P (elements share the request window)
    near = np.abs(t) <= 10.0
    t_pk = tp + float(t[near][int(np.argmax(env[near]))])
    for lo, hi in BANDS:
        b = band(st.__class__(full), lo, hi)
        noise = np.array([rms(tr, tp + NOISE[0], tp + NOISE[1]) for tr in b])
        def med(w):
            return float(np.median(np.array([rms(tr, t_pk + w[0], t_pk + w[1]) for tr in b]) / noise))
        rows.append(dict(section="snr", array=name, event=ev_name, band_hz=f"{lo:g}-{hi:g}", n_elem=n_meta,
                         n_data=len(full), t_pk_minus_pred_s=float(t_pk - tp), p_snr=med(P_WIN),
                         coda_snr_5_30=med(CODA1), coda_snr_30_60=med(CODA2), **extra))
    return rows


def look_rows(tag, st, origin, dist):
    """S8 look: 1-2 Hz rms in the Rg group-velocity window over pre-origin noise, per station."""
    rows = []
    for tr in band(st, 1.0, 2.0):
        r = dist.get(tr.stats.station)
        if r is None or tr.stats.npts < 100:
            continue
        noise = rms(tr, origin - 50, origin - 5)
        w = tr.slice(origin + r / U_FAST, origin + r / U_SLOW)
        if not len(w.data) or not noise:
            continue
        k = int(tr.stats.sampling_rate)
        env = np.convolve(np.abs(w.data.astype(float)), np.ones(k) / k, "same")
        t_peak = r / U_FAST + int(np.argmax(env)) / tr.stats.sampling_rate
        rows.append(dict(section="s8_look", array=tag, station=tr.stats.station, dist_km=r,
                         rg_window_snr_1_2=rms(tr, origin + r / U_FAST, origin + r / U_SLOW) / noise,
                         body_window_snr_1_2=rms(tr, origin + r / 6.5, origin + r / U_FAST) / noise,
                         u_peak_kms=r / t_peak))
    return rows


def main() -> None:
    cache = data_dir("a6")
    c = Client(base_url=ES, timeout=600)
    taup = TauPyModel("iasp91")
    rows = []

    # --- metadata: IM arrays and the ZE 2019 nodes
    inv_path = cache / "IM_arrays.xml"
    if not inv_path.exists():
        c.get_stations(network="IM", station="IL*,BM0*,BC0*,IM0*", channel="SHZ", level="channel",
                       filename=str(inv_path))
    ze_path = cache / "ZE2019.xml"
    if not ze_path.exists():
        c.get_stations(network="ZE", starttime=ZE_SPAN[0], endtime=ZE_SPAN[1], channel="DPZ", level="channel",
                       filename=str(ze_path))
    im = read_inventory(str(inv_path))[0]
    ze = read_inventory(str(ze_path))[0]
    elems: dict[str, pd.DataFrame] = {}
    for name, pat in ARRAYS.items():
        e = []
        for sta in im:
            if not re.fullmatch(pat, sta.code):
                continue
            ch = sorted(sta.channels, key=lambda q: q.start_date)
            e.append(dict(section="element", array=name, station=sta.code, lat=sta.latitude, lon=sta.longitude,
                          elev_m=sta.elevation, start=str(ch[0].start_date)[:10],
                          end="open" if ch[-1].end_date is None else str(ch[-1].end_date)[:10],
                          channel=ch[-1].code, sample_rate=ch[-1].sample_rate,
                          sensor=ch[-1].sensor.description if ch[-1].sensor else ""))
        elems[name] = pd.DataFrame(e)
    nodes = pd.DataFrame([dict(station=s.code, lat=s.latitude, lon=s.longitude,
                               sample_rate=s.channels[0].sample_rate,
                               sensor=s.channels[0].sensor.description if s.channels[0].sensor else "") for s in ze])
    dense = nodes[nodes.station.str.fullmatch(ZE_DENSE)]
    elems["ZE19_dense"] = dense.assign(section="element", array="ZE19_dense", start=str(ZE_SPAN[0])[:10],
                                       end=str(ZE_SPAN[1])[:10], channel="DPZ")
    line = nodes[~nodes.station.str.fullmatch(ZE_DENSE)].reset_index(drop=True)

    # --- geometry and array response
    for name, e in elems.items():
        if name != "ZE19_dense":  # 90 node rows would swamp the table; its geometry row carries the summary
            rows += e.to_dict("records")
        x, y = local_xy(e.lat, e.lon)
        g = geometry(x, y)
        summary, resp = response_rows(name, x, y, g["aperture_km"])
        rows.append(dict(section="geometry", array=name, lat=e.lat.mean(), lon=e.lon.mean(),
                         sample_rate=e.sample_rate.iloc[0], sensor=e.sensor.iloc[0], start=e.start.max(),
                         end=e.end.iloc[0], **g, **summary))
        rows += resp

    # --- SNR on the DPRK tests
    dprk = usgs(cache, "dprk", starttime="2016-01-01", endtime="2017-12-31", minlatitude=40, maxlatitude=43,
                minlongitude=127, maxlongitude=131, eventtype="nuclear explosion")
    for _, ev in dprk.iterrows():
        o = T(ev.time)
        for name in ARRAYS:
            e = elems[name]
            d = locations2degrees(ev.latitude, ev.longitude, e.lat.mean(), e.lon.mean())
            arr = {a.name: a.time for a in reversed(taup.get_travel_times(0.0, d, ["P", "PcP", "PP"]))}
            tp = o + arr["P"]
            extra = dict(mag=ev.mag, dist_deg=d, pcp_minus_p_s=arr["PcP"] - arr["P"], pp_minus_p_s=arr["PP"] - arr["P"])
            ev_name = f"dprk_{o.date}"
            st = waveforms(c, cache, f"{ev_name}_{name}", "IM", list(e.station), "SHZ", tp - PRE, tp + POST)
            rows += (snr_rows(name, ev_name, st, tp, len(e), extra) if st is not None else
                     [dict(section="snr", array=name, event=ev_name, n_elem=len(e), n_data=0, **extra)])

    # --- teleseisms that could add events (counts only)
    tele = usgs(cache, "tele_m6", starttime=TELE_FROM, endtime=TELE_TO, minmagnitude=6.0, eventtype="earthquake")
    il = elems["ILAR"]
    tele["deg"] = [locations2degrees(a, b, il.lat.mean(), il.lon.mean()) for a, b in zip(tele.latitude, tele.longitude)]
    tele = tele[(tele.deg >= 30) & (tele.deg <= 95)]
    for label, m in (("deep_ge300km_m6plus", tele.depth >= 300), ("shallow_le50km_m6.0-6.5", (tele.depth <= 50) & (tele.mag <= 6.5)),
                     ("shallow_le50km_m_gt6.5", (tele.depth <= 50) & (tele.mag > 6.5)), ("all_m6plus_30-95deg", tele.depth >= 0)):
        rows.append(dict(section="teleseism", array="ILAR", event=label, n_events=int(m.sum())))

    # --- S8: cataloged explosions near the node line and near ILAR
    blasts = usgs(cache, "explosions", starttime=BLAST_FROM, endtime=BLAST_TO, eventtype="explosion", **BLAST_BOX)

    def near_far(lat, lon, e):
        d = GEOD.inv(np.full(len(e), lon), np.full(len(e), lat), e.lon.to_numpy(), e.lat.to_numpy())[2] / 1000.0
        return float(d.min()), float(d.max())

    in_ze = blasts[(blasts.t >= str(ZE_SPAN[0])) & (blasts.t < str(ZE_SPAN[1]))]
    sets = {"ZE19_line": (in_ze, line), "ILAR": (blasts, il), "ILAR_in_ZE_span": (in_ze, il)}
    picked = {}
    for tag, (cat, e) in sets.items():
        nf = np.array([near_far(a, b, e) for a, b in zip(cat.latitude, cat.longitude)]).reshape(-1, 2)
        sel = cat[nf[:, 0] <= NEAR_KM].assign(near=nf[nf[:, 0] <= NEAR_KM, 0], far=nf[nf[:, 0] <= NEAR_KM, 1])
        frozen = sel.t.dt.month.isin(FROZEN_MONTHS)
        rows.append(dict(section="s8_events", array=tag, n_events=len(sel), n_frozen_months=int(frozen.sum()),
                         n_other_months=int((~frozen).sum()), start=str(sel.t.min())[:10], end=str(sel.t.max())[:10],
                         near_km_min=sel.near.min(), near_km_median=sel.near.median(),
                         far_km_median=sel.far.median(), far_km_max=sel.far.max(),
                         mag_median=sel.mag.median(), mag_max=sel.mag.max()))
        if len(sel):
            picked[tag] = big = sel.sort_values("mag").iloc[-1]
            d = np.array([km(big.latitude, big.longitude, a, q) for a, q in zip(e.lat, e.lon)])
            rows[-1] |= {"n_elem_le50km": int((d <= NEAR_KM).sum()), "n_elem_le100km": int((d <= LOOK_KM).sum())}  # of the largest event

    # --- S8 look: the largest blast near the node line, and the largest near ILAR in the same weeks
    b = picked.get("ZE19_line")
    if b is not None:
        o = T(b.time)
        d = pd.Series([km(b.latitude, b.longitude, a, q) for a, q in zip(line.lat, line.lon)], index=line.station)
        d = d[d <= LOOK_KM].sort_values()
        pick = d.iloc[np.unique(np.linspace(0, len(d) - 1, LOOK_NODES).astype(int))]
        st = waveforms(c, cache, f"blast_{o.date}_ZE", "ZE", list(pick.index), "DPZ", o - 60, o + LOOK_KM / U_SLOW + 20)
        if st is not None:
            pos = line.set_index("station")
            rows += [r | dict(event=f"blast_{o.date}", mag=b.mag, lat=pos.lat[r["station"]], lon=pos.lon[r["station"]])
                     for r in look_rows("ZE19_line", st, o, pick.to_dict())]
    b = picked.get("ILAR_in_ZE_span")
    if b is not None:
        o = T(b.time)
        d = {s: km(b.latitude, b.longitude, a, q) for s, a, q in zip(il.station, il.lat, il.lon)}
        st = waveforms(c, cache, f"blast_{o.date}_ILAR", "IM", list(il.station), "SHZ", o - 60, o + max(d.values()) / U_SLOW + 20)
        if st is not None:
            rows += [r | dict(event=f"blast_{o.date}", mag=b.mag) for r in look_rows("ILAR", st, o, d)]
            # is the blast's Rg coherent across ILAR, from the right direction? (P window = control of the beam)
            baz, _, rc = GEOD.inv(il.lon.mean(), il.lat.mean(), b.longitude, b.latitude)
            rc /= 1000.0
            xs, ys = local_xy(il.lat, il.lon)
            xy = {sta: (a, q) for sta, a, q in zip(il.station, xs, ys)}
            rg = (rc / U_FAST - 2, rc / U_BEAM + 3)
            for win, (t1, t2) in (("P", (rc / 6.5 - 1, rc / 6.5 + 4)), ("Rg", rg),
                                  ("noise", (-25 - (rg[1] - rg[0]), -25.0))):  # same length as Rg, before the origin
                for lo, hi in BANDS[:2]:
                    w = band(st, lo, hi).trim(o + t1, o + t2)
                    w = w.__class__([tr for tr in w if tr.stats.station in xy and tr.stats.npts > 10])
                    sem, slow, bz = beam(w, np.array([xy[tr.stats.station][0] for tr in w]),
                                         np.array([xy[tr.stats.station][1] for tr in w]))
                    rows.append(dict(section="rg_beam", array="ILAR", event=f"blast_{o.date}", band_hz=f"{lo:g}-{hi:g}",
                                     window=win, n_data=len(w), dist_km=rc, semblance=sem, semblance_incoherent=1 / len(w),
                                     slow_peak_skm=slow, baz_peak_deg=bz, baz_true_deg=baz % 360))

    lead = ["section", "array", "event", "station", "freq_hz", "c_kms", "band_hz"]
    out = pd.DataFrame(rows)
    out = out[lead + [k for k in out.columns if k not in lead]]
    out.to_csv(HERE / "a6_array_precheck.csv", index=False, float_format="%.7g")
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
    for sec in ("geometry", "response", "snr", "teleseism", "s8_events", "s8_look", "rg_beam"):
        print(f"\n== {sec}")
        print(out[out.section == sec].dropna(axis=1, how="all").drop(columns="section").to_string(index=False))


def self_check() -> None:
    f, d = 1.0, 2.0
    # two elements d apart along x: response cos^2(pi f ds_x d); half-width 1/(4 f d), full alias at 1/(f d)
    r, az, p = response(np.array([-d / 2, d / 2]), np.zeros(2), f, 1.0)
    east = int(np.argmin(np.abs(az - 90.0)))
    assert np.allclose(p[east], np.cos(np.pi * f * r * d) ** 2, atol=1e-9)
    hw, alias, _ = lobes(r, p)
    assert abs(hw[east] - 1 / (4 * f * d)) < 2e-3
    assert abs(alias[east] - 3 / (4 * f * d)) < 2e-3  # back above -3 dB on the way up to the grating lobe at 1/(f d)
    assert abs(p[east][int(np.argmin(np.abs(r - 1 / (f * d))))] - 1.0) < 1e-6
    assert np.allclose(p[0], 1.0)  # no resolution across the pair (north)
    # uniform line of 8 at spacing d: first null at 1/(N f d), grating lobe at 1/(f d)
    x = (np.arange(8) - 3.5) * d
    r, az, p = response(x, np.zeros(8), f, 1.0)
    assert p[east][int(np.argmin(np.abs(r - 1 / (8 * f * d))))] < 1e-4
    assert abs(lobes(r, p)[1].min() - 1 / (f * d)) < 0.06  # alias radius just inside the grating lobe
    # geometry on a 3-4-5 triangle (km), through the geodesic projection
    la = 64.0 + np.array([0.0, 0.0, 4.0]) / 111.5
    lo = -147.0 + np.array([0.0, 3.0, 0.0]) / (111.3 * np.cos(np.deg2rad(64.0)))
    g = geometry(*local_xy(la, lo))
    assert abs(g["aperture_km"] - 5.0) < 0.05 and abs(g["min_spacing_km"] - 3.0) < 0.05
    assert abs(km(0, 0, 0, 1) - 111.32) < 0.05
    # beam: a 1.5 Hz wavelet crossing 7 scattered elements at 0.4 s/km toward azimuth 60 deg (back-azimuth 240)
    from obspy import Stream, Trace
    rng = np.random.default_rng(0)
    ex, ey = rng.uniform(-4, 4, 7), rng.uniform(-4, 4, 7)
    sx, sy = 0.4 * np.sin(np.deg2rad(60.0)), 0.4 * np.cos(np.deg2rad(60.0))
    tt = np.arange(0, 30, 0.05)
    def wave(t):
        return np.exp(-((t - 15) / 1.5) ** 2) * np.cos(2 * np.pi * 1.5 * t)
    sem, slow, bz = beam(Stream([Trace(wave(tt - (sx * a + sy * q)), dict(delta=0.05)) for a, q in zip(ex, ey)]), ex, ey)
    assert sem > 0.95 and abs(slow - 0.4) < 0.03 and abs(bz - 240.0) < 4.0, (sem, slow, bz)


if __name__ == "__main__":
    self_check()
    main()
