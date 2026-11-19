"""A5 late windows: the coda level out to 570 s after the P peak, counted three ways,
and the far-field bound set against it over the whole range of delays it could arrive at.

Why: the bounds beyond the simulated range (s11_versions) were timed with the Rayleigh speed of a soft
half-space and compared with coda levels measured to 60 s only. A surface wave over tens of kilometers can
travel several times faster or as slowly, so the bound has to be set against every window it could arrive in.

Inputs (cached; no network): the records of a5_dprk_network (three later explosions, stations inside the two
boxes) and the IU.COL record of the 1992 explosion (a5_lopnor_repeats), read with their own functions;
a5_dprk_site_spread.csv and a2_clutter.csv (to reproduce); the simulated amplitude ratios (a8_smoothing.csv,
a8_generous_shares.csv, a8_gradient_shares.csv); the stations and distances of
akp/stage_p4/s11_versions.csv; a1_stations.csv (station positions, for the predicted phases).

Method (fixed before any window after 60 s was measured):
- measure(): a5_lopnor_repeats.measure with the list of windows as an argument. Same filter, envelope,
  smoothing, P-peak rule and validity rule (a median above SNR_FACTOR times the pre-P level). A window is
  measured only if the trace covers all of it. Two levels per window: "as measured" (the whole window) and
  "clean" (the part before the record's first predicted later phase of PHASES, less GUARD; none if less
  than CLEAN_MIN is left). The clean level is the coda of P alone; later phases only add to it.
- LAGS: the three committed windows, then LATE.
- three ways of counting, per box, band, window and level: "valid" (usable records, valid values: the
  committed rule); "all" (usable records, no validity rule); "high_snr" (usable records with P peak >=
  SNR_HIGH times the pre-P level, no validity rule). Each with its count, the count of values under
  SNR_FACTOR times their record's pre-P level, and whether the median or the fifth percentile is read from
  such a value (then it is an upper limit on the coda). The 1992 record at IU.COL, as measured, beside them.
- reproduction: way "valid", as measured, on the committed windows must equal a5_dprk_site_spread.csv, and
  the 1992 values must equal a2_clutter.csv, each within REPRO_TOL, or stop.
- the bound at range r: the largest over the simulated cases of the case's largest (amplitude ratio times
  sqrt(receiver range)) over every receiver at or beyond START_M (every band and line, the larger of the two
  measures, flagged rows included), over sqrt(r). Spreading alone.
- delay: from zero to r over the slowest tabled Rayleigh speed of the simulated grounds, widened at both
  ends by r times MOVEOUT (P between body and station) and by a2.PEAK (the lag of the P peak). A bound whose
  range passes the last window cannot take the first or second reading.
- readings per bound, against the clean levels of the Denali-Nome box (every spanned window, band and way):
  "below p05" if a level exists, is not an upper limit and the bound is below the fifth percentile, and
  below every usable clean value of its own station where it is among the reference records; "below median"
  likewise with the median; else "not settled by amplitude". Beside them, not a reading: the end of the
  last window up to which each condition holds from the first on. A version takes its lower reading.
- crossing(): the largest simulated range at which any case's ratio is at or above a level of the first
  window: the pooled median and fifth percentile by each way, the 1992 record, and each named station's own
  median over its usable records. "at_least" marks a range that is a case's last receiver; spreading_range_m
  is then the range at which spreading alone, from that receiver, brings the ratio down to the level.
- attenuation(): exp(-pi f t / Q) at the start of each late window, at the lower edge of each band, for
  a8_attenuation.Q_VALUES. Arithmetic only: it enters no bound and no reading.

- descriptions added after the readings were seen; they carry no reading: the least value of each
  cell beside the bound (a5_late_bounds.csv, columns min and below_min); and by_band(): for every bound, and
  for the one open waveform (IU.COL) at its distance from each version, the bound of each band on its own
  (the same formula over the rows of that band) against the clean levels of way "valid" and against the 1992
  record at IU.COL as measured, window by window, with the end of the last window up to which each holds.

- thresholds() (a description): for every bound and band, the first window
  in which the band's own bound is at or above the 1992 record's level; the speed below which a wave must travel to
  arrive there (distance over the window's start); and the least Q with which exp(-pi f t / Q), at the lower edge of
  the band and the start of a window, still leaves the bound at or above that level.

Outputs beside this file: a5_late_records.csv, a5_late_levels.csv, a5_late_phases.csv, a5_late_bounds.csv,
a5_late_readings.csv, a5_late_versions.csv, a5_late_valdez.csv, a5_late_crossing.csv,
a5_late_attenuation.csv, a5_late_by_band.csv, a5_late_by_band_summary.csv, a5_late_thresholds.csv, a5_late_threshold_windows.csv,\na5_late_checks.csv.
Self-check: selfcheck() (window spanning, the bound formula, the measure on a synthetic trace, the flag for a
statistic read from a noise value).
Run: PYTHONPATH=. python scripts/mem_guard.py --frac 0.40 --every 0.5 -- python -m akp.stage_a.a5_late_windows
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime as T
from obspy.core.util.obspy_types import ObsPyException
from obspy.geodetics import locations2degrees

from akp.paths import data_dir
from akp.stage_a import a2_wavefield as a2
from akp.stage_a import a2b_explosions as a2b
from akp.stage_a import a5_dprk_network as net
from akp.stage_a import a5_lopnor_repeats as a5
from akp.stage_a.a8_attenuation import Q_VALUES

HERE = Path(__file__).parent
S11 = HERE.parent / "stage_p4"
LATE = [(60.0, 120.0), (120.0, 240.0), (240.0, 480.0), (480.0, 570.0)]  # s after the P peak
LAGS = list(a2.LAGS) + LATE
BANDS = a2.BANDS[:3]  # the bands of the simulations
SNR_HIGH = 20.0  # Way "high_snr"
PHASES = ("PcP", "PP", "ScP", "PcS", "S", "PKiKP")
GUARD, CLEAN_MIN = 5.0, 10.0  # s: left out before the first later phase; least clean part of a window
START_M = 1000.0  # Receivers at or beyond this range may set the bound
MOVEOUT = 0.065  # s/km, P between body and station
NEAR_M = 900.0  # the simulated receiver nearest the distance of the station at the town of Valdez (0.88 km)
REPRO_TOL = 1e-6
BOX = "mckinley_nome"
COL1992 = "lopnor_1992-05-21"
WAYS = ("valid", "all", "high_snr")
ORDER = ("below p05", "below median", "not settled by amplitude")  # best first


def lab(w) -> str:
    return f"{w[0]:g}-{w[1]:g}"


@functools.lru_cache(maxsize=None)
def phase_delays(dist_deg: float) -> tuple:
    """((phase, seconds after P), ...) for a surface source at a distance rounded to 0.01 degree (iasp91)."""
    arr = a2.TAUP.get_travel_times(source_depth_in_km=0.0, distance_in_degree=dist_deg, phase_list=("P",) + PHASES)
    first = {}
    for a in arr:
        first.setdefault(a.name, a.time)
    return tuple((p, first[p] - first["P"]) for p in PHASES if p in first)


def first_phase(dist_deg: float) -> float:
    return min((d for _, d in phase_delays(round(dist_deg, 2)) if d > 0), default=np.inf)


def measure(tr, tp: T, t_first: float = np.inf, lags=LAGS) -> list[dict]:
    """Per band and window: the level as measured and the clean level (before t_first - GUARD after P)."""
    rows = []
    nyq, fs = tr.stats.sampling_rate / 2.0, tr.stats.sampling_rate
    for fmin, fmax in BANDS:
        if fmax >= 0.95 * nyq:
            continue
        t, hil, _ = a2.envelopes(tr, fmin, fmax)
        rel = t - (tp - tr.stats.starttime)
        win = (rel >= a2.PEAK[0]) & (rel <= a2.PEAK[1])
        nz = hil[(rel >= a2.NOISE[0]) & (rel <= a2.NOISE[1])]
        if not win.any() or not nz.size:
            continue
        ipk = np.flatnonzero(win)[np.argmax(hil[win])]
        tpk, peak, noise = rel[ipk], hil[ipk], float(np.median(nz))
        for a, b in lags:
            sel = (rel >= tpk + a) & (rel <= tpk + b)
            if sel.sum() < (b - a) * fs - 1:  # the trace must cover the whole window
                continue
            med = float(np.median(hil[sel]))
            c = hil[sel & (rel < t_first - GUARD)]
            clean = float(np.median(c)) if c.size >= CLEAN_MIN * fs else np.nan
            rows.append(dict(band=lab((fmin, fmax)), lag=lab((a, b)), ratio=med / peak, ratio_clean=clean / peak,
                             noise_ratio=noise / peak, snr=peak / noise, valid=bool(med > a2b.SNR_FACTOR * noise),
                             valid_clean=bool(clean > a2b.SNR_FACTOR * noise)))
    return rows


def network_records() -> pd.DataFrame:
    """Every record of a5_dprk_network, measured in LAGS, with the usable flag of that module."""
    wf_dir = data_dir("a5/wf")
    ev = a5.explosions("dprk", ("USGS",), starttime=T("2006-01-01"), endtime=T("2018-01-01"), latitude=a2b.SITE[0],
                       longitude=a2b.SITE[1], maxradius=0.5, minmagnitude=4.0)
    ev = ev[ev.time.str[:10].isin(net.DATES)].copy()
    ev["name"] = "dprk_" + ev.time.str[:10]
    out = []
    for e in ev.itertuples():
        rows, invs = net.sites(e.name, T(e.time))
        for r in rows:
            r["tp"], r["dist_deg"] = a5.p_time(T(e.time), e.latitude, e.longitude, r["lat"], r["lon"])
        st = a5.waveforms(wf_dir / f"{e.name}.mseed", [(r["net"], r["sta"], r["loc"], r["cha"], r["tp"] - a2.PRE,
                                                         r["tp"] + a2.POST) for r in rows])
        for r in rows:
            sel = st.select(network=r["net"], station=r["sta"], location=r["loc"], channel=r["cha"]) if st else []
            if not len(sel):
                continue
            try:
                tr, gaps, is_clipped, covered = a5.velocity(sel, invs[r["box"]], r["tp"])
            except (ObsPyException, ValueError, IndexError):
                continue
            t1 = first_phase(r["dist_deg"])
            base = dict(event=e.name, net=r["net"], sta=r["sta"], box=r["box"], lat=r["lat"], lon=r["lon"],
                        dist_deg=round(r["dist_deg"], 2), first_phase_s=round(t1, 1))
            out += [dict(base, **x, usable=bool(covered and not gaps and not is_clipped and x["snr"] >= a5.SNR_SEEN))
                    for x in measure(tr, r["tp"], t1)]
    return pd.DataFrame(out)


def col1992() -> pd.DataFrame:
    """The vertical record of the 1992 explosion at IU.COL, by the path of a5_lopnor_repeats; as measured."""
    ev = a5.explosions("lopnor", a5.SOURCES, starttime=a5.SPAN[0], endtime=a5.SPAN[1], latitude=a5.SITE[0],
                       longitude=a5.SITE[1], maxradius=a5.RADIUS_DEG)
    e = ev[ev.time.str[:10] == COL1992[-10:]].iloc[0]
    inv = a5.inventory(data_dir("a5/sta") / "lopnor_college.xml", network=a5.NETS, station=a5.STAS, channel=a5.CHANS,
                       starttime=a5.SPAN[0], endtime=a5.SPAN[1])
    origin = T(e.time)
    sta = next(s for n in inv.select(time=origin, network="IU", station="COL") for s in n)
    tp, dist = a5.p_time(origin, e.latitude, e.longitude, sta.latitude, sta.longitude)
    st = a5.waveforms(data_dir("a5/wf") / f"IU.COL.{COL1992}.mseed", [("IU", "COL", "*", a5.CHANS, tp - a2.PRE, tp + a2.POST)])
    tr, gaps, is_clipped, covered = a5.velocity(st.select(channel="BHZ"), inv, tp)
    return pd.DataFrame([dict(event=COL1992, net="IU", sta="COL", box="col1992", lat=sta.latitude, lon=sta.longitude,
                              dist_deg=round(dist, 2), first_phase_s=round(first_phase(dist), 1), **x,
                              usable=bool(covered and not gaps and not is_clipped)) for x in measure(tr, tp)])


def read_from_noise(vals: np.ndarray, ok: np.ndarray, q: float) -> bool:
    """Whether the q-th quantile (linear interpolation) is read from a value that is not above the noise rule."""
    order = np.argsort(vals)
    pos = (len(vals) - 1) * q
    return not (ok[order][int(np.floor(pos))] and ok[order][int(np.ceil(pos))])


def levels(rec: pd.DataFrame) -> pd.DataFrame:
    """Per box, band, window, level kind and way: counts, least value, fifth percentile, median and noise flags."""
    ref = rec[rec.box == "col1992"].set_index(["band", "lag"])
    rows = []
    for (box, band, lag), g in rec[rec.usable & (rec.box != "col1992")].groupby(["box", "band", "lag"], sort=False):
        for kind, col, vcol in (("as measured", "ratio", "valid"), ("clean", "ratio_clean", "valid_clean")):
            h = g[g[col].notna()]
            for way, s in dict(valid=h[h[vcol]], all=h, high_snr=h[h.snr >= SNR_HIGH]).items():
                v, ok = s[col].to_numpy(), s[vcol].to_numpy()
                row = dict(box=box, band=band, lag=lag, level=kind, way=way, n_usable=len(g), n=len(v), n_under_noise_rule=int((~ok).sum()))
                if len(v):
                    row.update(min=v.min(), p05=np.percentile(v, 5), median=np.median(v),
                               p05_upper_limit=read_from_noise(v, ok, 0.05), median_upper_limit=read_from_noise(v, ok, 0.5))
                rows.append(dict(row, col1992=float(ref.ratio.get((band, lag), np.nan)),
                                 col1992_valid=bool(ref.valid.get((band, lag), False)),
                                 col1992_noise_ratio=float(ref.noise_ratio.get((band, lag), np.nan))))
    return pd.DataFrame(rows)


def holding(delay: float) -> str:
    """The window that holds a delay after P, the P-peak lag ignored (it is at most a2.PEAK)."""
    return next((lab(w) for w in LAGS if w[0] <= delay <= w[1]), "before the first" if delay < LAGS[0][0] else "after the last")


def phases(rec: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """For the reference records: how many have each phase in each window. For the 1992 explosion: the delay
    and the window of each phase at each station of the bounds and at IU.COL."""
    use = rec[rec.usable & (rec.box == BOX)].drop_duplicates(["event", "net", "sta"])
    hold = pd.DataFrame([dict(phase=p, lag=holding(d)) for dist in use.dist_deg for p, d in phase_delays(round(dist, 2))])
    rows = [dict(kind="reference records", station="", dist_deg="", phase=p, lag=lag, value=len(g), unit=f"records of {len(use)}")
            for (p, lag), g in hold.groupby(["phase", "lag"])]
    rows.append(dict(kind="reference records", station="", dist_deg=f"{use.dist_deg.min():g} to {use.dist_deg.max():g}", phase="first later phase",
                     lag="", value=f"{use.first_phase_s.min():g} to {use.first_phase_s.max():g}", unit="s after P"))
    sta = pd.read_csv(HERE / "a1_stations.csv").drop_duplicates("code").set_index("code")
    ev = pd.read_csv(HERE.parent / "setup" / "pr_b_isc_event.csv").iloc[0]
    for code in sorted(set(pairs.station) | {"COL"}):
        dist = round(locations2degrees(ev.latitude, ev.longitude, sta.lat[code], sta.lon[code]), 2)
        rows += [dict(kind="1992 explosion", station=code, dist_deg=dist, phase=p, lag=holding(d), value=round(d, 1), unit="s after P")
                 for p, d in phase_delays(dist)]
    return pd.DataFrame(rows)


def simulated() -> pd.DataFrame:
    """Every simulated row: case, azimuth, range_m, band, ratio (the larger of the two measures; for the further
    case the upper end of its bracket) and ratio_measured (the larger of the two measures as measured)."""
    d = pd.concat([pd.read_csv(HERE / f) for f in ("a8_smoothing.csv", "a8_generous_shares.csv")], ignore_index=True)
    d = d.assign(ratio=d[["share", "share_smoothed"]].max(axis=1), ratio_measured=lambda x: x.ratio)
    # The further case enters by the upper end of its bracket for the source-edge artifact; as measured beside it
    g = pd.read_csv(HERE / "a8_gradient_shares.csv")
    g = g.assign(ratio=g[["share_high", "share_smoothed_high"]].max(axis=1), ratio_measured=g[["share", "share_smoothed"]].max(axis=1))
    return pd.concat([d, g], ignore_index=True)[["case", "azimuth", "range_m", "band", "ratio", "ratio_measured"]]


def starts(sim: pd.DataFrame) -> pd.DataFrame:
    """Per case: the row at or beyond START_M with the largest ratio times sqrt(range); k is that product."""
    far = sim[sim.range_m >= START_M].assign(k=lambda d: d.ratio * np.sqrt(d.range_m))
    return far.loc[far.groupby("case").k.idxmax()].reset_index(drop=True)


def bound_at(m: float, st: pd.DataFrame) -> tuple[float, str]:
    assert m >= st.range_m.max(), "the bound is carried outward only"
    i = st.k.idxmax()
    return float(st.k[i] / np.sqrt(m)), f"{st.case[i]} {st.band[i]} Hz {st.azimuth[i]} at {st.range_m[i]:g} m"


def slowest() -> float:
    """The slowest tabled Rayleigh speed of the simulated grounds, m/s."""
    return float(pd.concat([pd.read_csv(HERE / f) for f in ("a8_attenuation.csv", "a8_generous_attenuation.csv")]).rayleigh_speed_ms.min())


def spanned(km: float, slow: float) -> tuple[list, bool, float]:
    """(windows met by the widened delay range, whether it passes the last window, its upper end after the peak)."""
    lo = -km * MOVEOUT - a2.PEAK[1]  # delay after the peak = delay after P less the peak's lag
    hi = km * 1e3 / slow + km * MOVEOUT - a2.PEAK[0]
    return [lab(w) for w in LAGS if lo <= w[1] and hi >= w[0]], bool(hi > LAGS[-1][1]), hi


def bound_pairs() -> pd.DataFrame:
    """The (version, kind, station, km) of every bound that s11_versions carries outward."""
    v = pd.read_csv(S11 / "s11_versions.csv")
    rows = []
    for kind in ("recorded", "operating"):
        g = v[v[f"{kind}_bound_spreading"].notna()]
        rows += [dict(version=r.version, kind=kind, station=r[f"{kind}_code"], km=r[f"{kind}_km"]) for _, r in g.iterrows()]
    return pd.DataFrame(rows)


def through(g: pd.DataFrame, col: str) -> float:
    """The end (s) of the last window up to which every cell passes `col`, from the first window on."""
    end = 0.0
    for w in LAGS:
        c = g[g.lag == lab(w)]
        if c.empty or not c[col].all():
            break
        end = w[1]
    return end


def compare(pairs: pd.DataFrame, lv: pd.DataFrame, rec: pd.DataFrame, st: pd.DataFrame, slow: float):
    lev = lv[(lv.box == BOX) & (lv.level == "clean")].set_index(["band", "lag", "way"])
    own = rec[rec.usable & (rec.box != "col1992") & rec.ratio_clean.notna()]
    c92 = rec[rec.box == "col1992"].set_index(["band", "lag"]).ratio
    rows, reads = [], []
    for p in pairs.itertuples():
        b, by = bound_at(p.km * 1e3, st)
        wins, after, hi = spanned(p.km, slow)
        mine = own[own.sta == p.station]
        cells = []
        for w in wins:
            for band in map(lab, BANDS):
                s = mine[(mine.band == band) & (mine.lag == w)].ratio_clean
                own_ok = bool((b < s).all())  # true when the station has no value there
                for way in WAYS:
                    L = lev.loc[(band, w, way)] if (band, w, way) in lev.index else None
                    has = L is not None and L.n > 0
                    cells.append(dict(version=p.version, kind=p.kind, station=p.station, km=p.km, bound=b, band=band, lag=w, way=way,
                                      n=int(L.n) if has else 0, median=L["median"] if has else np.nan, p05=L.p05 if has else np.nan,
                                      min=L["min"] if has else np.nan, below_min=bool(has and b < L["min"]),
                                      median_upper_limit=bool(L.median_upper_limit) if has else None,
                                      p05_upper_limit=bool(L.p05_upper_limit) if has else None,
                                      own_station_n=len(s), own_station_min=float(s.min()) if len(s) else np.nan,
                                      col1992=float(c92.get((band, w), np.nan)),
                                      pass_median=bool(has and not L.median_upper_limit and b < L["median"]),
                                      pass_p05=bool(has and not L.p05_upper_limit and b < L.p05 and own_ok)))
        g = pd.DataFrame(cells)
        rows.append(g)
        reading = ORDER[2] if after else ORDER[0] if g.pass_p05.all() else ORDER[1] if g.pass_median.all() else ORDER[2]
        why = []
        if after:
            why.append(f"the delay range passes {LAGS[-1][1]:g} s")
        no = sorted(set(g.lag[g.n == 0]), key=lambda x: float(x.split("-")[0]))
        if no:
            why.append("no clean level in " + ", ".join(no) + " s")
        lim = g[(g.n > 0) & (g.median_upper_limit == True)]  # noqa: E712
        if len(lim):
            why.append("median read from a noise value in " + ", ".join(sorted({f"{r.band} Hz {r.lag} s ({r.way})" for r in lim.itertuples()})))
        over = g[(g.n > 0) & (g.bound >= g["median"])]
        if len(over):
            why.append("at or above the median in " + ", ".join(sorted({f"{r.band} Hz {r.lag} s ({r.way})" for r in over.itertuples()})))
        c = g.drop_duplicates(["band", "lag"])
        reads.append(dict(version=p.version, kind=p.kind, station=p.station, km=p.km, bound=b, set_by=by, delay_max_s=round(hi, 1),
                          windows=" ".join(wins), passes_last_window=after, reading=reading, why="; ".join(why),
                          median_holds_through_s=through(g, "pass_median"), p05_holds_through_s=through(g, "pass_p05"),
                          own_station_values=int(c.own_station_n.sum()),
                          below_col1992_in_every_spanned_window=bool((c.bound < c.col1992).all()),
                          not_below_col1992=", ".join(f"{r.band} Hz {r.lag} s" for r in c[~(c.bound < c.col1992)].itertuples())))
    reads = pd.DataFrame(reads)
    vers = reads.assign(rank=reads.reading.map(ORDER.index)).sort_values("rank").groupby("version", sort=False).agg(
        reading=("reading", "last"), bounds=("kind", "count"), stations=("station", lambda s: " ".join(s)),
        median_holds_through_s=("median_holds_through_s", "min"), p05_holds_through_s=("p05_holds_through_s", "min")).reset_index()
    return pd.concat(rows, ignore_index=True), reads, vers


def by_band(pairs: pd.DataFrame, lv: pd.DataFrame, rec: pd.DataFrame, sim: pd.DataFrame, slow: float):
    """A description: each band's own bound against the clean levels (way valid) and the 1992 record."""
    v = pd.read_csv(S11 / "s11_versions.csv")
    far = sim[sim.range_m >= START_M].assign(k=lambda d: d.ratio * np.sqrt(d.range_m))
    k = far.groupby("band").k.max()
    openw = v[v.open_km.notna() & (v.open_km * 1e3 >= far.range_m.max())]
    allp = pd.concat([pairs, pd.DataFrame(dict(version=openw.version, kind="open waveform", station="COL", km=openw.open_km))], ignore_index=True)
    lev = lv[(lv.box == BOX) & (lv.level == "clean") & (lv.way == "valid")].set_index(["band", "lag"])
    c92 = rec[rec.box == "col1992"].set_index(["band", "lag"]).ratio
    rows = []
    for p in allp.itertuples():
        wins, after, hi = spanned(p.km, slow)
        for band in map(lab, BANDS):
            b = float(k[band] / np.sqrt(p.km * 1e3))
            for w in wins:
                has = (band, w) in lev.index and lev.n[(band, w)] > 0
                rows.append(dict(version=p.version, kind=p.kind, station=p.station, km=p.km, band=band, bound_band=b, lag=w, delay_max_s=round(hi, 1),
                                 n=int(lev.n[(band, w)]) if has else 0, median=lev["median"][(band, w)] if has else np.nan,
                                 p05=lev.p05[(band, w)] if has else np.nan, col1992=float(c92[(band, w)]),
                                 below_median=bool(has and b < lev["median"][(band, w)]), below_p05=bool(has and b < lev.p05[(band, w)]),
                                 below_col1992=bool(b < c92[(band, w)])))
    d = pd.DataFrame(rows)
    summ = [dict(version=ver, kind=kind, station=sta, km=km, band=band, bound_band=g.bound_band.iloc[0], delay_max_s=g.delay_max_s.iloc[0],
                 below_median_through_s=through(g, "below_median"), below_p05_through_s=through(g, "below_p05"),
                 below_col1992_through_s=through(g, "below_col1992"), last_window_end_s=float(g.lag.iloc[-1].split("-")[1]))
            for (ver, kind, sta, km, band), g in d.groupby(["version", "kind", "station", "km", "band"], sort=False)]
    return d, pd.DataFrame(summ)


def thresholds(bb: pd.DataFrame, meas_k: float):
    """A description: what a wave must be for a band's bound to stand at or above the 1992 record's level.
    Per window in which the bound is at or above that level: the least travel time of a wave that arrives in it (the window's
    start, less the lag by which the P peak may precede P, less the travel of P between body and station), the speed below
    which the wave must travel (distance over that time), and the least Q that leaves the bound at or above the level
    (exp(-pi f t / Q) at the lower edge of the band and that time). Speed and Q of one row belong together.
    Returns (rows per window, one row per bound and band: its window of least Q), with the bound as measured beside."""
    low = {lab(b): b[0] for b in BANDS}
    up = bb[bb.bound_band >= bb.col1992].copy()
    t0 = up.lag.str.split("-").str[0].astype(float)
    up["travel_s"] = t0 + a2.PEAK[0] - up.km * MOVEOUT
    up = up[up.travel_s > 0]
    up["speed_below_kms"] = up.km / up.travel_s
    up["q_least"] = np.pi * up.band.map(low) * up.travel_s / np.log(up.bound_band / up.col1992).replace(0.0, np.nan)
    up = up.dropna(subset="q_least")
    key = ["version", "kind", "station", "km", "band"]
    one = bb.drop_duplicates(key)[key + ["bound_band"]].merge(
        up.loc[up.groupby(key).q_least.idxmin(), key + ["lag", "travel_s", "speed_below_kms", "q_least"]], on=key, how="left")
    one["windows_above"] = one.merge(up.groupby(key).size().rename("n").reset_index(), on=key, how="left").n.fillna(0).astype(int)
    one["bound_all_bands_as_measured"] = meas_k / np.sqrt(one.km * 1e3)
    return up[key + ["bound_band", "lag", "col1992", "travel_s", "speed_below_kms", "q_least"]], one


def valdez(reads: pd.DataFrame, lv: pd.DataFrame) -> pd.DataFrame:
    """The bound of the Valdez version against the least and the median clean value of the Valdez box, way 'all'."""
    r = reads[reads.version == "valdez_town"].iloc[0]
    L = lv[(lv.box == "valdez") & (lv.level == "clean") & (lv.way == "all") & lv.lag.isin(r.windows.split())]
    return L.assign(station=r.station, km=r.km, bound=r.bound, below_min=r.bound < L["min"], below_median=r.bound < L["median"])[
        ["station", "km", "bound", "band", "lag", "n", "n_under_noise_rule", "min", "median", "median_upper_limit", "below_min", "below_median"]]


def crossing(sim: pd.DataFrame, lv: pd.DataFrame, rec: pd.DataFrame, names: list) -> pd.DataFrame:
    """The largest simulated range at which any case's ratio is at or above each level of the first window."""
    lag0 = lab(LAGS[0])
    last = sim.groupby("case").range_m.max()
    rows = []

    def add(level_name, band, level, n):
        over = sim[(sim.band == band) & (sim.ratio >= level)]
        if over.empty:
            rows.append(dict(level=level_name, band=band, value=level, n=n, range_m=0.0, at_least=False, case="", spreading_range_m=np.nan))
            return
        top = over.loc[over.range_m.idxmax()]
        reach = over[over.range_m == over.case.map(last)]
        # where a case is still at or above the level at its last receiver: the range at which spreading alone brings it down
        spread = float((reach.range_m * (reach.ratio / level) ** 2).max()) if len(reach) else np.nan
        rows.append(dict(level=level_name, band=band, value=level, n=n, range_m=float(top.range_m), at_least=bool(len(reach)),
                         case=" ".join(sorted(set(reach.case))) if len(reach) else str(top.case), spreading_range_m=spread))

    pooled = lv[(lv.box == BOX) & (lv.lag == lag0) & (lv.level == "as measured")].set_index(["band", "way"])
    for band in map(lab, BANDS):
        for way in WAYS:
            add(f"pooled median, way {way}", band, float(pooled["median"][(band, way)]), int(pooled.n[(band, way)]))
            add(f"pooled fifth percentile, way {way}", band, float(pooled.p05[(band, way)]), int(pooled.n[(band, way)]))
        add("1992 record at IU.COL", band, float(pooled.col1992[(band, "valid")]), 1)
        for code in names:
            g = rec[(rec.sta == code) & rec.usable & (rec.band == band) & (rec.lag == lag0)]
            if len(g):
                add(f"station {code}, median of its usable records", band, float(g.ratio.median()), len(g))
    return pd.DataFrame(rows)


def attenuation() -> pd.DataFrame:
    """exp(-pi f t / Q): what a wave arriving t seconds after P has lost, whatever its speed and path length."""
    return pd.DataFrame([dict(delay_s=w[0], hz=b[0], q=q, factor=float(np.exp(-np.pi * b[0] * w[0] / q)))
                         for w in LATE for b in BANDS for q in Q_VALUES])


def selfcheck() -> None:
    wins, after, hi = spanned(160.0, 285.8)
    assert wins == [lab(w) for w in LAGS] and after and abs(hi - (160e3 / 285.8 + 160 * MOVEOUT + 3)) < 1e-9, "to 573 s: past the last window"
    wins, after, _ = spanned(7.2, 285.8)
    assert wins == [lab(w) for w in LAGS[:2]] and not after, "to 29 s"
    st = starts(pd.DataFrame(dict(case=["a", "a", "b", "a"], azimuth="forward", band="1-2", range_m=[2800.0, 3000.0, 1500.0, 500.0],
                                  ratio=[0.12, 0.1, 0.2, 0.9])))
    b, by = bound_at(12000.0, st)
    assert by.startswith("b ") and abs(b - 0.2 * np.sqrt(1500 / 12000)) < 1e-12, "the largest ratio times root range wins"
    assert float(st.k[st.case == "a"].iloc[0]) == 0.12 * np.sqrt(2800), "not the last receiver, and not one inside START_M"
    v, ok = np.array([0.3, 0.1, 0.2, 0.4, 0.5]), np.array([True, False, True, True, True])
    assert read_from_noise(v, ok, 0.05) and not read_from_noise(v, ok, 0.5), "the lowest value is noise; the median is not"
    from obspy import Trace
    fs, n = 20.0, int(900 * 20)
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 1.5 * t) * np.where(t < 300, 0.01, np.where(t < 302, 1.0, np.where(t < 400, 0.2, 0.6)))  # noise, P, coda, a later phase
    m = pd.DataFrame(measure(Trace(x, dict(sampling_rate=fs, starttime=T(0))), T(300), t_first=100.0))
    c = m[m.band == "1-2"].set_index("lag")
    assert len(c) == len(LAGS) and abs(c.ratio["30-60"] - 0.2) < 0.02 and abs(c.ratio["120-240"] - 0.6) < 0.05, "as measured"
    assert abs(c.ratio_clean["60-120"] - 0.2) < 0.02 and c.ratio_clean[["120-240", "240-480", "480-570"]].isna().all(), "clean: before the later phase only"


def main() -> None:
    selfcheck()
    rec = pd.concat([network_records(), col1992()], ignore_index=True)
    lv = levels(rec)
    committed = [lab(w) for w in a2.LAGS]

    # reproduction of the committed tables on the committed windows
    old = pd.read_csv(HERE / "a5_dprk_site_spread.csv", dtype={"band": str, "lag": str}).query("event == 'all'")
    mine = lv[(lv.way == "valid") & (lv.level == "as measured") & lv.lag.isin(committed)]
    m = mine.merge(old, on=["box", "band", "lag"], suffixes=("", "_old"))
    assert len(m) == len(mine) == len(old[old.band.isin(mine.band)]), "every committed cell is matched"
    worst = float(max((m["median"] / m.median_old - 1).abs().max(), (m.p05 / m.p05_old - 1).abs().max()))
    assert worst <= REPRO_TOL and (m.n == m.n_valid).all() and (m.n_usable == m.n_usable_old).all(), f"site spread not reproduced ({worst:.2e})"
    ref = pd.read_csv(HERE / "a2_clutter.csv", dtype={"band": str, "lag": str})
    ref = ref[(ref.kind == "test1992") & (ref.estimator == "hilbert")].set_index(["band", "lag"]).ratio
    c92 = rec[rec.box == "col1992"].set_index(["band", "lag"]).ratio
    w92 = float(max(abs(c92[k] / ref[k] - 1) for k in c92.index if k in ref.index))
    assert w92 <= REPRO_TOL, f"the 1992 record is not reproduced ({w92:.2e})"
    early = rec[(rec.box != "col1992") & rec.lag.isin(committed)]
    same = float((early.ratio_clean / early.ratio - 1).abs().max())

    pairs = bound_pairs()
    sim = simulated()
    st, slow = starts(sim), slowest()
    cells, reads, vers = compare(pairs, lv, rec, st, slow)
    names = sorted(set(pairs.station) | set(pd.read_csv(S11 / "s11_stations_inside.csv").code))
    here = [c for c in names if c in set(rec[rec.box != "col1992"].sta)]  # The 1992 record is not a reference record
    cross = crossing(sim, lv, rec, here)
    s11 = pd.read_csv(S11 / "s11_ranges.csv").set_index("quantity")
    b3, by3 = bound_at(float(s11.range_m["bound_start"]), st)
    med_km = float(cross[cross.level == "pooled median, way valid"].range_m.max()) / 1e3
    medians = lv[(lv.box == BOX) & (lv.lag == lab(LAGS[0])) & (lv.level == "as measured") & (lv.way == "valid")].set_index("band")["median"]
    med_rows = sim[sim.ratio >= sim.band.map(medians)]
    med_meas = sim[sim.ratio_measured >= sim.band.map(medians)]
    assert abs(med_km * 1e3 - float(s11.range_m["range_median_coda"])) < 1e-6, "the crossing range at the committed median is the committed one"

    bb, bs = by_band(pairs, lv, rec, sim, slow)
    meas = sim.assign(ratio=sim.ratio_measured)
    thw, th = thresholds(bb, float(starts(meas).k.max()))
    tn, tw = th[th.kind != "open waveform"], thw[thw.kind != "open waveform"]  # the stations of the bounds; the open waveform apart
    to = th[th.kind == "open waveform"]
    easy = tn.loc[tn.q_least.idxmin()]
    near_m = sim[sim.range_m == NEAR_M]
    way_med = lv[(lv.box == BOX) & (lv.lag == lab(LAGS[0])) & (lv.level == "as measured")].pivot(index="band", columns="way", values="median")
    beyond = sim[sim.range_m >= START_M]
    uni = sim[sim.case != "R5"]
    clean_end = rec[rec.usable & (rec.box == BOX)].first_phase_s - GUARD
    checks = [dict(check="site_spread_max_relative_difference", value=worst, note="way valid, as measured, committed windows, median and p05"),
              dict(check="col1992_max_relative_difference", value=w92, note="against a2_clutter.csv, committed windows"),
              dict(check="clean_equals_as_measured_committed_windows_max_relative_difference", value=same, note="reference records; above zero where a record's first later phase falls inside the third window"),
              dict(check="records_measured", value=len(rec.drop_duplicates(["event", "net", "sta"])), note="network records and the 1992 record"),
              dict(check="simulated_cases", value=sim.case.nunique(), note=" ".join(sim.case.unique())),
              dict(check="slowest_rayleigh_speed_ms", value=slow, note="a8_attenuation.csv and a8_generous_attenuation.csv"),
              dict(check="bound_at_committed_start_range", value=b3, note=f"set by {by3}"),
              dict(check="committed_start", value=float(s11.ratio["bound_start"]), note=f"s11_ranges.csv bound_start, {s11.case['bound_start']} {s11.band['bound_start']} Hz; last receiver, unflagged rows only"),
              dict(check="crossing_committed_median_max_km", value=med_km, note="equals s11_ranges.csv range_median_coda"),
              *[dict(check=f"crossing_committed_median_{name}_km", value=float(x.range_m.max()) / 1e3 if len(x) else 0.0, note=note) for name, x, note in (
                  ("uniform_ground_cases", med_rows[med_rows.case != "R5"], "the eight cases in uniform ground"),
                  ("further_case_as_measured", med_meas[med_meas.case == "R5"], "R5, the larger of its two measures as measured"),
                  ("further_case_upper", med_rows[med_rows.case == "R5"], "R5, the upper end of its bracket"))],
              dict(check="bounds", value=len(pairs), note="pairs of version and kind with a bound in s11_versions.csv"),
              *[dict(check=f"bounds_reading_{k.replace(' ', '_')}", value=int((reads.reading == k).sum()), note="") for k in ORDER],
              dict(check="bounds_with_no_clean_level_in_a_spanned_window", value=int(reads.why.str.contains("no clean level").sum()), note="readings whose delay range meets a window with no level of the coda of P alone"),
              dict(check="clean_level_ends_s_min", value=float(clean_end.min()), note="first later phase less the guard, s after P, reference records of the box"),
              dict(check="clean_level_ends_s_max", value=float(clean_end.max()), note=""),
              dict(check="bound_at_committed_start_range_as_measured", value=bound_at(float(s11.range_m["bound_start"]), starts(meas))[0],
                   note=f"set by {bound_at(float(s11.range_m['bound_start']), starts(meas))[1]}; every case by its ratio as measured"),
              dict(check="crossing_high_snr_median_uniform_ground_cases_km", value=float(uni[uni.ratio >= uni.band.map(way_med.high_snr)].range_m.max()) / 1e3, note="the eight cases in uniform ground against the median of way high_snr"),
              dict(check="largest_ratio_beyond_start_over_lowest_way_median_as_measured", value=float((beyond.ratio_measured / beyond.band.map(way_med.min(axis=1))).max()), note="every case as measured, at or beyond START_M, against the lowest of the three medians of its band"),
              dict(check="largest_ratio_beyond_start_over_lowest_way_median_with_allowance", value=float((beyond.ratio / beyond.band.map(way_med.min(axis=1))).max()), note="the further case by the upper end of its bracket"),
              dict(check="threshold_bound_bands", value=len(tn), note="pairs of bound and band, the stations of the bounds"),
              dict(check="threshold_bound_bands_never_above_1992", value=int((tn.windows_above == 0).sum()), note="the band's own bound is below the 1992 record's level in every window it could arrive in"),
              dict(check="threshold_speed_below_kms_max", value=float(tw.speed_below_kms.max()), note="the fastest wave that could arrive in a window where its bound is at or above the 1992 record's level"),
              dict(check="threshold_q_at_fastest", value=float(tw.q_least[tw.speed_below_kms.idxmax()]), note="the least Q for that same bound, band and window"),
              dict(check="threshold_q_least_min", value=float(easy.q_least), note=f"the least Q over every bound, band and window: {easy.station} {easy.band} Hz, window {easy.lag}"),
              dict(check="threshold_speed_at_least_q", value=float(easy.speed_below_kms), note="the speed below which the wave of that same row must travel"),
              dict(check="threshold_q_least_for_faster_than_1_kms", value=float(tw[tw.speed_below_kms >= 1.0].q_least.min()), note="the least Q among the rows a wave faster than 1 km/s could arrive in"),
              dict(check="threshold_q_least_for_faster_than_2_kms", value=float(tw[tw.speed_below_kms >= 2.0].q_least.min()), note="likewise for 2 km/s"),
              dict(check="open_waveform_bound_min", value=float(to.bound_band.min()), note="IU.COL at its distance from each version; band's own bound"),
              dict(check="open_waveform_bound_max", value=float(to.bound_band.max()), note=""),
              dict(check="open_waveform_q_least_min", value=float(to.q_least.min()), note="as threshold_q_least_min, for IU.COL"),
              *[dict(check=f"largest_ratio_as_measured_at_{NEAR_M:g}m_{band}", value=float(g.ratio_measured.max()), note=f"every case and line, as measured; set by {g.case[g.ratio_measured.idxmax()]}")
                for band, g in near_m.groupby("band")],
              dict(check="bounds_passing_last_window", value=int(reads.passes_last_window.sum()), note=f"widened delay range passes {LAGS[-1][1]:g} s"),
              dict(check="bound_min", value=float(reads.bound.min()), note=""), dict(check="bound_max", value=float(reads.bound.max()), note=""),
              dict(check="named_stations_among_reference_records", value=len(here), note=" ".join(here)),
              *[dict(check=f"largest_ratio_at_{NEAR_M:g}m_{band}", value=float(g.ratio.max()), note=f"every case and line; set by {g.case[g.ratio.idxmax()]}")
                for band, g in sim[sim.range_m == NEAR_M].groupby("band")]]

    rec.round(6).to_csv(HERE / "a5_late_records.csv", index=False)
    lv.round(6).to_csv(HERE / "a5_late_levels.csv", index=False)
    phases(rec, pairs).to_csv(HERE / "a5_late_phases.csv", index=False)
    cells.round(6).to_csv(HERE / "a5_late_bounds.csv", index=False)
    reads.round(6).to_csv(HERE / "a5_late_readings.csv", index=False)
    vers.to_csv(HERE / "a5_late_versions.csv", index=False)
    valdez(reads, lv).round(6).to_csv(HERE / "a5_late_valdez.csv", index=False)
    cross.round(6).to_csv(HERE / "a5_late_crossing.csv", index=False)
    attenuation().round(6).to_csv(HERE / "a5_late_attenuation.csv", index=False)
    bb.round(6).to_csv(HERE / "a5_late_by_band.csv", index=False)
    bs.round(6).to_csv(HERE / "a5_late_by_band_summary.csv", index=False)
    th.round(6).to_csv(HERE / "a5_late_thresholds.csv", index=False)
    thw.round(6).to_csv(HERE / "a5_late_threshold_windows.csv", index=False)
    pd.DataFrame(checks).to_csv(HERE / "a5_late_checks.csv", index=False)
    print(pd.DataFrame(checks).to_string(index=False))
    print(reads[["version", "kind", "station", "km", "bound", "delay_max_s", "reading", "median_holds_through_s", "p05_holds_through_s"]].to_string(index=False))


if __name__ == "__main__":
    main()
