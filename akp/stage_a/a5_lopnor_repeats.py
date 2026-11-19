"""A5a (P4 item S2): repeat Lop Nor tests at College. How much of the P coda at COL repeats?

Inputs (open data only):
- catalog: every event typed "nuclear explosion" within RADIUS_DEG of the Lop Nor anchor in SPAN, from the
  ISC FDSN event service (USGS if ISC does not answer); the catalog is recorded per event (column source);
- waveforms: BH? at COL / COLA from EarthScope for A2's window (a2.PRE before to a2.POST after iasp91 P).
  Networks IU and DW: DW.COL is the same station's code before March 1991 (rows carry net; IU alone is the
  literal S2 set). Responses come from the same response-level StationXML.
Catalog (QuakeML), StationXML and raw miniSEED are cached under $AKP_DATA/a5. A definitive "no data" answer
is cached as an empty .nodata file, so a re-run needs no network and gives the same rows.

Method (fixed before results):
- per event, channel and band: P peak, pre-event noise and clutter ratio exactly as A2 (a2.envelopes, Hilbert
  estimator as in A2b, a2.BANDS / LAGS / NOISE / PEAK); snr = P peak / noise. A clutter cell is valid by A2b's
  rule: coda envelope above SNR_FACTOR x the noise envelope.
- usable record (per band): window covered, no gaps, not clipped, snr >= SNR_SEEN (A1's screening value). A
  record without a pre-event noise window (a triggered recording) cannot be measured: status "not covered".
- K1 (the plan's kill): fewer than MIN_EVENTS events whose vertical record is usable in the lowest band and has
  a valid first clutter cell. Then the checks are written and no correlation is computed.
- repeatability: every pair of events on the same station and channel, band-passed with A2's filter and
  resampled to FS_CC. Alignment on P: the lag (|lag| <= ALIGN_MAX_S about the iasp91 prediction) that
  maximizes the normalized cross-correlation of the ALIGN_BAND records over A2's P window; one lag per pair,
  used for every band. Then the zero-lag normalized correlation in the P window and in each A2 lag window
  (lapse time counted from the earlier event's P peak in that band). Baseline: the same statistic in windows of
  the same length tiled over the pre-event noise window. cc_ceiling is what two records with these
  coda-to-noise ratios would reach if their coda were identical.
- split: if two records share a common part and differ in the rest, their correlation is the common share of
  the coda power. repeat_frac = median cc / median ceiling over valid pairs, clipped to 0-1. cc_p_median, the
  P-window correlation of the same pairs, says how alike the sources themselves were.

Outputs beside this file: a5_lopnor_levels.csv, a5_lopnor_clutter.csv, a5_lopnor_corr.csv (one row per pair,
band and window), a5_lopnor_repeatable.csv (per station, channel, band and window), a5_lopnor_checks.csv.
Run: python -m akp.stage_a.a5_lopnor_repeats
"""
from __future__ import annotations

import csv
import functools
import itertools
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime as T
from obspy import read, read_events, read_inventory
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.core.util.obspy_types import ObsPyException
from obspy.geodetics import locations2degrees
from scipy.signal import resample_poly

from akp.paths import data_dir, data_glob
from akp.stage_a import a2_wavefield as a2
from akp.stage_a.a1_stations import SNR_SEEN
from akp.stage_a.a2b_explosions import SNR_FACTOR
from swe.io import catalog_to_dataframe, get_events

HERE = Path(__file__).parent
SITE = (41.6, 88.8)  # approximate test-site center: search anchor only; parameters come from the catalog
RADIUS_DEG = 1.5
SPAN = (T("1990-01-01"), T("1997-01-01"))
SOURCES = ("ISC", "USGS")  # ISC first: A2 took the 1992 origin from the ISC
NETS, STAS, CHANS = "IU,DW", "COL,COLA", "BH?"
MIN_EVENTS = 4  # plan S2 kill criterion (input)
ALIGN_BAND = (0.5, 4.0)
ALIGN_MAX_S = 3.0
FS_CC = 80.0
GRID = (-290.0, 80.0)  # seconds about predicted P held for correlation
LOW = f"{a2.BANDS[0][0]:g}-{a2.BANDS[0][1]:g}"
LAG0 = f"{a2.LAGS[0][0]:g}-{a2.LAGS[0][1]:g}"
KEY = ["event", "sta", "cha", "band"]


@functools.cache
def client() -> Client:
    """EarthScope FDSN client, made only on a cache miss (a cached run opens no connection)."""
    return Client(base_url=a2.ES, timeout=600)


def explosions(tag: str, sources: tuple, **query) -> pd.DataFrame:
    """Events typed 'nuclear explosion' from the first catalog that answers (cached as QuakeML)."""
    for src in sources:
        path = data_dir("a5/cat") / f"{tag}_{src}.xml"
        if not path.exists():
            try:
                cat = get_events(client=src, **query)
            except Exception as exc:  # service down or unreachable: try the next catalog
                print(f"catalog {src}: {type(exc).__name__}")
                continue
            if not len(cat):
                continue
            cat.write(str(path), format="QUAKEML")
        cat = read_events(str(path))
        df = catalog_to_dataframe(cat)
        df["mb"] = [next((m.mag for m in ev.magnitudes if (m.magnitude_type or "").lower() == "mb"), np.nan)
                    for ev in cat]
        df["source"] = src
        return df[[ev.event_type == "nuclear explosion" for ev in cat]].sort_values("time").reset_index(drop=True)
    raise RuntimeError(f"no catalog answered for {tag}")


def inventory(path: Path, **query):
    """Response-level StationXML for the query, cached at `path`; None if the service has none."""
    nodata = path.with_suffix(".nodata")
    if nodata.exists():
        return None
    if not path.exists():
        try:
            client().get_stations(level="response", filename=str(path), **query)
        except FDSNNoDataException:
            nodata.touch()
            return None
    return read_inventory(str(path))


def waveforms(path: Path, bulk: list[tuple]):
    """Raw miniSEED for the request lines, cached at `path`; None on a definitive no-data answer."""
    nodata = path.with_suffix(".nodata")
    if nodata.exists():
        return None
    if not path.exists():
        try:
            client().get_waveforms_bulk(bulk, filename=str(path))
        except FDSNNoDataException:
            nodata.touch()
            return None
    return read(str(path))


def p_time(origin: T, elat: float, elon: float, slat: float, slon: float) -> tuple:
    """(iasp91 P arrival for a surface source, distance in degrees), as A2."""
    dist = locations2degrees(elat, elon, slat, slon)
    return origin + a2.TAUP.get_travel_times(0.0, dist, ["P"])[0].time, dist


def velocity(st, inv, tp: T) -> tuple:
    """One channel's stream -> (velocity trace, gaps, clipped, covered), processed exactly as A2."""
    st = st.select(location=sorted({tr.stats.location for tr in st})[0])  # one sensor: lowest location code
    gaps = len(st.get_gaps())
    raw = st.copy().merge(method=0)[0]
    is_clipped = a2.clipped(raw.data.astype(np.float64))
    tr = raw.copy()
    if np.ma.isMaskedArray(tr.data):
        tr.data = tr.data.filled(0)
    nyq = tr.stats.sampling_rate / 2.0
    tr.detrend("demean").taper(0.02)
    tr.remove_response(inventory=inv, output="VEL", pre_filt=(0.2, 0.3, 0.9 * nyq, 0.98 * nyq), water_level=None)
    covered = bool(tr.stats.starttime <= tp + GRID[0] and tr.stats.endtime >= tp + GRID[1])
    return tr, gaps, is_clipped, covered


def measure(tr, tp: T) -> tuple[list, list]:
    """A2's level (per band) and clutter (per band x lag) cells, Hilbert estimator, with A2b's validity flag."""
    levels, clutter = [], []
    nyq = tr.stats.sampling_rate / 2.0
    for fmin, fmax in a2.BANDS:
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
        band = f"{fmin:g}-{fmax:g}"
        levels.append(dict(band=band, p_peak_vel=peak, noise_vel=noise, noise_ratio=noise / peak, snr=peak / noise,
                           p_peak_lag_s=round(float(tpk), 2)))
        for a, b in a2.LAGS:
            w = hil[(rel >= tpk + a) & (rel <= tpk + b)]
            med = float(np.median(w)) if w.size else np.nan
            clutter.append(dict(band=band, lag=f"{a:g}-{b:g}", ratio=med / peak, noise_ratio=noise / peak,
                                valid=bool(med > SNR_FACTOR * noise)))
    return levels, clutter


def on_grid(tr, tp: T, band: tuple, fs: float):
    """Band-passed (A2's filter) samples at rate `fs` over GRID about `tp` (nearest sample); None if short."""
    f = tr.copy().filter("bandpass", freqmin=band[0], freqmax=band[1], corners=4, zerophase=True)
    r = Fraction(fs / f.stats.sampling_rate).limit_denominator(1000)
    x = resample_poly(f.data.astype(np.float64), r.numerator, r.denominator)
    i0 = int(round((tp + GRID[0] - f.stats.starttime) * fs))
    n = int(round((GRID[1] - GRID[0]) * fs))
    return x[i0:i0 + n] if i0 >= 0 and i0 + n <= len(x) else None


def ix(t: float, fs: float) -> int:
    """Sample index on GRID of time `t` (seconds about predicted P)."""
    return int(round((t - GRID[0]) * fs))


def max_ncc(X: np.ndarray, i0: int, n: int, maxlag: int) -> tuple:
    """Normalized cross-correlation of row i's window [i0, i0+n) with row j's window shifted by k samples,
    maximized over |k| <= maxlag. Returns (maximum, k at the maximum), each rows x rows. Signed, not absolute."""
    def unit(k: int) -> np.ndarray:
        w = X[:, i0 + k:i0 + k + n]
        w = w - w.mean(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return w / np.linalg.norm(w, axis=1, keepdims=True)
    a = unit(0)
    best = np.full((len(X), len(X)), -np.inf)
    lag = np.zeros(best.shape, dtype=int)
    for k in range(-maxlag, maxlag + 1):
        c = a @ unit(k).T
        up = c > best
        best[up], lag[up] = c[up], k
    best[np.isinf(best)] = np.nan
    return best, lag


def write_checks(path: Path, checks: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(checks[0]))
        w.writeheader()
        w.writerows(checks)
    for ch in checks:
        print({k: str(v)[:300] for k, v in ch.items()})


def correlate(recs: dict, lv: pd.DataFrame, cl: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Pairwise P-aligned correlation per band and window -> (pair rows, noise values of the valid pairs)."""
    level = lv[lv.status == "ok"].set_index(KEY)
    cell = cl.set_index(KEY + ["lag"])
    n_p = int(round((a2.PEAK[1] - a2.PEAK[0]) * FS_CC))
    rows, pooled = [], {}
    for sta, cha in sorted({k[:2] for k in recs}):
        names = sorted(k[2] for k in recs if k[:2] == (sta, cha))  # names sort by date
        grid = {(n, b): on_grid(*recs[(sta, cha, n)][1:], b, FS_CC) for n in names for b in [ALIGN_BAND] + a2.BANDS}
        for a, b in itertools.combinations(names, 2):
            if grid[(a, ALIGN_BAND)] is None or grid[(b, ALIGN_BAND)] is None:
                continue
            best, lag = max_ncc(np.vstack([grid[(a, ALIGN_BAND)], grid[(b, ALIGN_BAND)]]), ix(a2.PEAK[0], FS_CC), n_p,
                                int(ALIGN_MAX_S * FS_CC))
            shift = int(lag[0, 1])
            for band in a2.BANDS:
                lab = f"{band[0]:g}-{band[1]:g}"
                ka, kb = (a, sta, cha, lab), (b, sta, cha, lab)
                if ka not in level.index or kb not in level.index:
                    continue
                X = np.vstack([grid[(a, band)], np.roll(grid[(b, band)], -shift)])

                def cc(t: float, length: float) -> float:
                    return float(max_ncc(X, ix(t, FS_CC), int(round(length * FS_CC)), 0)[0][0, 1])

                both = bool(level.usable[ka] and level.usable[kb])
                t_pk = float(level.p_peak_lag_s[ka])
                windows = [("P", a2.PEAK[0], a2.PEAK[1] - a2.PEAK[0])]
                windows += [(f"{p:g}-{q:g}", t_pk + p, q - p) for p, q in a2.LAGS]
                for name, t, length in windows:
                    noise = [cc(a2.NOISE[0] + m * length, length)
                             for m in range(int((a2.NOISE[1] - a2.NOISE[0]) // length))]
                    valid, ceiling = both, np.nan
                    if name != "P":
                        ca, cb = cell.loc[ka + (name,)], cell.loc[kb + (name,)]
                        valid = bool(both and ca.valid and cb.valid)
                        ceiling = float(np.sqrt(max(0.0, 1 - (ca.noise_ratio / ca.ratio) ** 2)
                                                * max(0.0, 1 - (cb.noise_ratio / cb.ratio) ** 2)))
                    rows.append(dict(sta=sta, cha=cha, event_a=a, event_b=b, net_a=recs[(sta, cha, a)][0],
                                     net_b=recs[(sta, cha, b)][0], band=lab, lag=name, align_s=shift / FS_CC,
                                     cc_align=float(best[0, 1]), cc=cc(t, length),
                                     noise_cc_p95=float(np.percentile(noise, 95)), cc_ceiling=ceiling, valid=valid))
                    if valid:
                        pooled.setdefault((sta, cha, lab, name), []).extend(noise)
    return pd.DataFrame(rows), pooled


def repeatable(pairs: pd.DataFrame, pooled: dict) -> pd.DataFrame:
    """Per station, channel, band and window over valid pairs: correlation, noise baseline, repeatable share."""
    rows = []
    cells = [(f"{lo:g}-{hi:g}", lag) for lo, hi in a2.BANDS for lag in ["P"] + [f"{p:g}-{q:g}" for p, q in a2.LAGS]]
    valid = pairs[pairs.valid]
    for (sta, cha), (band, lag) in itertools.product(sorted(set(zip(valid.sta, valid.cha))), cells):
        g = valid[(valid.sta == sta) & (valid.cha == cha) & (valid.band == band) & (valid.lag == lag)]
        if g.empty:
            continue
        p = pairs[(pairs.sta == sta) & (pairs.cha == cha) & (pairs.band == band) & (pairs.lag == "P")]
        p = p.merge(g[["event_a", "event_b"]], on=["event_a", "event_b"])
        ceiling = float(g.cc_ceiling.median())
        noise_p95 = float(np.percentile(pooled[(sta, cha, band, lag)], 95))
        rows.append(dict(sta=sta, cha=cha, band=band, lag=lag, n_pairs=len(g),
                         n_events=len(set(g.event_a) | set(g.event_b)), cc_median=float(g.cc.median()),
                         cc_q25=float(g.cc.quantile(0.25)), cc_q75=float(g.cc.quantile(0.75)),
                         cc_min=float(g.cc.min()), cc_max=float(g.cc.max()),
                         noise_cc_p95=noise_p95, frac_above_noise=float((g.cc > noise_p95).mean()),
                         cc_p_median=float(p.cc.median()), cc_ceiling_median=ceiling,
                         repeat_frac=np.nan if lag == "P" else float(np.clip(g.cc.median() / ceiling, 0, 1))))
    out = pd.DataFrame(rows)
    out["above_noise"] = out.cc_median > out.noise_cc_p95
    return out


def main() -> None:
    wf_dir, sta_dir = data_dir("a5/wf"), data_dir("a5/sta")
    ev = explosions("lopnor", SOURCES, starttime=SPAN[0], endtime=SPAN[1], latitude=SITE[0], longitude=SITE[1],
                    maxradius=RADIUS_DEG)
    ev["name"] = "lopnor_" + ev.time.str[:10]
    inv = inventory(sta_dir / "lopnor_college.xml", network=NETS, station=STAS, channel=CHANS, starttime=SPAN[0],
                    endtime=SPAN[1])
    levels, clutter, recs = [], [], {}
    for e in ev.itertuples():
        origin = T(e.time)
        ev_base = dict(event=e.name, source=e.source, origin_time=e.time, mb=e.mb)
        here = [(net.code, sta) for net in inv.select(time=origin) for sta in net]
        if not here:
            levels.append(dict(ev_base, status="no station epoch"))
        for net, sta in here:
            tp, dist = p_time(origin, e.latitude, e.longitude, sta.latitude, sta.longitude)
            st = waveforms(wf_dir / f"{net}.{sta.code}.{e.name}.mseed",
                           [(net, sta.code, "*", CHANS, tp - a2.PRE, tp + a2.POST)])
            if st is None:
                levels.append(dict(ev_base, net=net, sta=sta.code, status="no data"))
                continue
            for cha in sorted({tr.stats.channel for tr in st}):
                base = dict(ev_base, net=net, sta=sta.code, cha=cha)
                try:
                    tr, gaps, is_clipped, covered = velocity(st.select(channel=cha), inv, tp)
                except (ObsPyException, ValueError, IndexError) as exc:
                    levels.append(dict(base, status=f"failed: {type(exc).__name__}"))
                    continue
                base.update(loc=tr.stats.location, fs=tr.stats.sampling_rate, dist_deg=round(dist, 2))
                lev, clu = measure(tr, tp)
                if not lev:
                    levels.append(dict(base, status="not covered"))
                    continue
                levels += [dict(base, status="ok", **r, gaps=gaps, clipped=is_clipped, covered=covered,
                                usable=bool(covered and not gaps and not is_clipped and r["snr"] >= SNR_SEEN))
                           for r in lev]
                clutter += [dict(base, **r) for r in clu]
                recs[(sta.code, cha, e.name)] = (net, tr, tp)
    lv, cl = pd.DataFrame(levels), pd.DataFrame(clutter)
    lv.to_csv(HERE / "a5_lopnor_levels.csv", index=False)
    cl.to_csv(HERE / "a5_lopnor_clutter.csv", index=False)

    # cached case: the 1992 record must reproduce A2's clutter cells
    ref = pd.read_csv(HERE / "a2_clutter.csv", dtype={"band": str, "lag": str})
    ref = ref[(ref.kind == "test1992") & (ref.estimator == "hilbert")].set_index(["band", "lag"]).ratio
    name92 = "lopnor_" + pd.read_csv(a2.SETUP / "pr_b_isc_event.csv").origin_time[0][:10]
    new = cl[(cl.event == name92) & (cl.net == "IU") & (cl.cha == "BHZ")].set_index(["band", "lag"]).ratio
    a2_diff = float((new / ref - 1).abs().max()) if len(new) else np.nan

    ok = lv[lv.status == "ok"]
    z = ok[ok.cha.str.endswith("Z") & (ok.band == LOW)]
    first = cl[(cl.band == LOW) & (cl.lag == LAG0) & cl.valid][["event", "sta", "cha"]]
    usable = sorted(z[z.usable].merge(first, on=["event", "sta", "cha"]).event.unique())
    usable_iu = sorted(set(usable) & set(z[z.net == "IU"].event))
    failed = lv[lv.status != "ok"]
    size = sum(p.stat().st_size for d in (wf_dir, sta_dir, data_dir("a5/cat")) for p in data_glob(d, "*lopnor*"))
    checks = [
        dict(check="catalog_explosions", value=len(ev), of="", triggered="",
             detail="; ".join(f"{n} mb{m:g} {s}" for n, m, s in zip(ev.name, ev.mb, ev.source))),
        dict(check="events_with_vertical_record", value=int(z.event.nunique()), of=len(ev), triggered="",
             detail="; ".join(f"{e}:{n}.{s}" for e, n, s in zip(z.event, z.net, z.sta))),
        dict(check="records_not_measured", value=len(failed), of="", triggered="",
             detail="; ".join(sorted({f"{e}:{s}" for e, s in zip(failed.event, failed.status)}))),
        dict(check="events_p_snr_ok_low_band", value=int(z[z.usable].event.nunique()), of=int(z.event.nunique()),
             triggered="", detail=f"band {LOW} Hz, snr >= {SNR_SEEN:g}, covered, no gaps, not clipped; "
             + "; ".join(f"{e}:{s:.1f}" for e, s in zip(z.event, z.snr))),
        dict(check="K1_usable_events_low_band", value=len(usable), of=MIN_EVENTS, triggered=len(usable) < MIN_EVENTS,
             detail=f"also a valid {LAG0} s clutter cell; " + "; ".join(usable)),
        dict(check="usable_events_low_band_IU_only", value=len(usable_iu), of=MIN_EVENTS, triggered="",
             detail="; ".join(usable_iu)),
        dict(check="reproduces_a2_1992_cells", value=a2_diff, of=len(ref), triggered=not a2_diff < 1e-3,
             detail="max relative difference from a2_clutter.csv, kind=test1992, hilbert"),
        dict(check="cache_bytes", value=size, of="", triggered="", detail="$AKP_DATA/a5: this script's catalog, StationXML and miniSEED"),
    ]
    out_corr, out_rep = HERE / "a5_lopnor_corr.csv", HERE / "a5_lopnor_repeatable.csv"
    if checks[4]["triggered"]:
        out_corr.unlink(missing_ok=True)
        out_rep.unlink(missing_ok=True)
    else:
        pairs, pooled = correlate(recs, lv, cl)
        rep = repeatable(pairs, pooled)
        pairs.to_csv(out_corr, index=False)
        rep.to_csv(out_rep, index=False)
        zp = pairs[pairs.cha.str.endswith("Z") & pairs.valid & (pairs.band == LOW) & (pairs.lag == LAG0)]
        checks.append(dict(check="valid_vertical_pairs_low_band", value=len(zp), of="", triggered="",
                           detail=f"band {LOW} Hz, lag {LAG0} s"))
        checks.append(dict(check="align_lag_abs_max_s", value=float(pairs.align_s.abs().max()), of=ALIGN_MAX_S,
                           triggered=bool(pairs.align_s.abs().max() >= ALIGN_MAX_S),
                           detail="a lag at the search limit means the alignment did not converge"))
        print(rep.to_string())
    checks.append(dict(check="gate_pass", value=int(not any(ch["triggered"] is True for ch in checks)), of="",
                       triggered="", detail=""))
    write_checks(HERE / "a5_lopnor_checks.csv", checks)
    print(lv.groupby("status").size().to_dict())


def selfcheck() -> None:
    """Two records sharing half their power, offset by 7 samples: the lag and the common share come back."""
    rng = np.random.default_rng(0)
    s = rng.standard_normal(6000)
    a, b = s + rng.standard_normal(6000), np.roll(s, 7) + rng.standard_normal(6000)
    best, lag = max_ncc(np.vstack([a, b]), 1000, 4000, 20)
    assert lag[0, 1] == 7 and abs(best[0, 1] - 0.5) < 0.05, (lag[0, 1], best[0, 1])
    assert abs(max_ncc(np.vstack([a, np.roll(b, -7)]), 1000, 4000, 0)[0][0, 1] - best[0, 1]) < 1e-12
    assert abs(max_ncc(np.vstack([a, rng.standard_normal(6000)]), 1000, 4000, 0)[0][0, 1]) < 0.06
    assert ix(GRID[0], FS_CC) == 0 and ix(0.0, 40.0) == 11600


if __name__ == "__main__":
    selfcheck()
    main()
