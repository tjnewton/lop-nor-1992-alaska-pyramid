"""A5b (P4 item S3): the DPRK tests on the dense Alaska network inside the claim boxes.

Inputs (open data only):
- catalog: A2b's USGS query for the test site, kept to events typed "nuclear explosion" on DATES;
- stations: every station of NETS inside a claim box (akp.zones.BOXES) operating at the origin time, one
  vertical channel each (first of CHANNELS present, lowest location code);
- waveforms: one bulk request per event to EarthScope, A2's window about iasp91 P at each station; responses
  from the same response-level StationXML.
Everything is cached under $AKP_DATA/a5 by a5_lopnor_repeats' readers, so a re-run needs no network. The
download is sized first and refused above MAX_BYTES.

Method (fixed before results):
- per station, event and band: P peak, noise, clutter ratio, validity and the usable flag, exactly as
  a5_lopnor_repeats (A2's measures, A2b's validity rule, A1's SNR screen).
- site spread: per box, event, band and lag, the distribution of valid clutter ratios over usable sites, beside
  the single COL 1992 value from a2_clutter.csv.
- coherence against separation: per event and band, usable records band-passed with A2's filter, resampled to
  FS_CC and aligned on predicted P. For each station pair, the maximum of the signed normalized
  cross-correlation within |lag| <= LAG_MAX_S, in A2's P window and in each A2 lag window (counted from the
  event's median P-peak time over usable stations). Baseline: the same statistic in N_NOISE windows of the same
  length tiled from the start of the pre-event noise window.
- curve: valid pairs binned by separation (BINS_KM), plus the bin "nn": each station's nearest usable
  neighbor. noise_cc_p95 pools the noise windows of the same pairs.
- lag_abs_median_s (a diagnostic added after the first run, not part of the rule): the median |lag| at the
  maximum. Unrelated records give a lag uniform over the search window, median LAG_MAX_S / 2; records that
  share a wavefront give a lag near zero. It separates coherence from a baseline that sits low only because
  noise is broader-band than coda.
- the plan's rule: coherence is at the noise level in a band x lag cell when the nearest-neighbor median does
  not exceed that noise 95th percentile (all events pooled). If no coda cell is above it, back-projection is
  ruled out and the curve is the result. The P window is the positive control.

Outputs beside this file: a5_dprk_levels.csv, a5_dprk_clutter.csv, a5_dprk_site_spread.csv, a5_dprk_pairs.csv
(pairs of usable records), a5_dprk_coherence.csv (the curve), a5_dprk_checks.csv.
Run: python -m akp.stage_a.a5_dprk_network
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime as T
from obspy.core.util.obspy_types import ObsPyException
from obspy.geodetics import gps2dist_azimuth

from akp.paths import data_dir, data_glob
from akp.stage_a import a2_wavefield as a2
from akp.stage_a import a2b_explosions as a2b
from akp.stage_a import a5_lopnor_repeats as a5
from akp.zones import BOXES

HERE = Path(__file__).parent
DATES = ("2016-01-06", "2016-09-09", "2017-09-03")
NETS = "TA,AK,XV,IM,IU,II,AT"
CHANNELS = ("BHZ", "HHZ", "SHZ", "EHZ")  # the first present at a station is used
FS_CC = 40.0
LAG_MAX_S = 2.0
N_NOISE = 8
BINS_KM = (0, 2, 5, 10, 20, 50, 100, 200, 400, 800, 1600)
MAX_BYTES = 2e9
LAGS = [f"{p:g}-{q:g}" for p, q in a2.LAGS]
NOISE_COLS = [f"n{k}" for k in range(N_NOISE)]


def sites(name: str, origin: T) -> tuple[list, dict]:
    """One vertical channel per station inside a claim box at `origin`, and the inventory per box."""
    rows, invs = [], {}
    for box, bounds in BOXES.items():
        inv = a5.inventory(data_dir("a5/sta") / f"{name}_{box}.xml", network=NETS, channel=",".join(CHANNELS),
                           starttime=origin, endtime=origin + 3600, **bounds)
        if inv is None:
            continue
        invs[box] = inv
        for net in inv.select(time=origin):
            for sta in net:
                code = next(c for c in CHANNELS if any(ch.code == c for ch in sta))
                ch = min((ch for ch in sta if ch.code == code), key=lambda ch: ch.location_code)
                rows.append(dict(net=net.code, sta=sta.code, loc=ch.location_code, cha=code, fs=ch.sample_rate,
                                 lat=sta.latitude, lon=sta.longitude, box=box))
    return sorted(rows, key=lambda r: (r["net"], r["sta"])), invs


def pair_table(event: str, band: str, ids: np.ndarray, X: np.ndarray, t_pk: float, valid: dict,
               dist: np.ndarray) -> pd.DataFrame:
    """Every station pair of one event and band: max correlation per window, its noise windows, flags."""
    n = len(ids)
    iu = np.triu_indices(n, 1)
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    nn = np.zeros((n, n), dtype=bool)
    nn[np.arange(n), d.argmin(axis=1)] = True
    nn |= nn.T
    maxlag = int(LAG_MAX_S * FS_CC)
    windows = [("P", a2.PEAK[0], a2.PEAK[1] - a2.PEAK[0], np.ones(n, dtype=bool))]
    windows += [(lab, t_pk + p, q - p, valid[lab]) for lab, (p, q) in zip(LAGS, a2.LAGS)]
    frames = []
    for name, t, length, v in windows:
        m = int(round(length * FS_CC))
        cc, lag = a5.max_ncc(X, a5.ix(t, FS_CC), m, maxlag)
        noise = {col: a5.max_ncc(X, a5.ix(a2.NOISE[0] + k * length, FS_CC), m, maxlag)[0][iu]
                 for k, col in enumerate(NOISE_COLS)}
        frames.append(pd.DataFrame(dict(event=event, band=band, lag=name, a=ids[iu[0]], b=ids[iu[1]],
                                        dist_km=dist[iu], cc=cc[iu], lag_s=lag[iu] / FS_CC, nn=nn[iu],
                                        valid=(v[:, None] & v[None, :])[iu], **noise)))
    return pd.concat(frames, ignore_index=True)


def curve(pairs: pd.DataFrame) -> pd.DataFrame:
    """Valid pairs by event (and 'all'), band, window and separation bin: correlation beside its noise baseline."""
    p = pairs[pairs.valid]
    rows = []
    for (event, band, lag), g in pd.concat([p, p.assign(event="all")]).groupby(["event", "band", "lag"], sort=False):
        sets = [(f"{lo:g}-{hi:g}", g[(g.dist_km >= lo) & (g.dist_km < hi)]) for lo, hi in zip(BINS_KM, BINS_KM[1:])]
        for label, s in sets + [("nn", g[g.nn])]:
            if s.empty:
                continue
            noise = s[NOISE_COLS].to_numpy().ravel()
            p95 = float(np.nanpercentile(noise, 95))
            rows.append(dict(event=event, band=band, lag=lag, dist_bin=label, n_pairs=len(s),
                             dist_median_km=float(s.dist_km.median()), cc_median=float(s.cc.median()),
                             cc_q25=float(s.cc.quantile(0.25)), cc_q75=float(s.cc.quantile(0.75)),
                             noise_cc_median=float(np.nanmedian(noise)), noise_cc_p95=p95,
                             frac_above_noise=float((s.cc > p95).mean()),
                             lag_abs_median_s=float(s.lag_s.abs().median())))
    return pd.DataFrame(rows)


def spread(lv: pd.DataFrame, cl: pd.DataFrame) -> pd.DataFrame:
    """Valid clutter ratios over usable sites per box, event (and 'all'), band and lag, beside COL 1992."""
    ref = pd.read_csv(HERE / "a2_clutter.csv", dtype={"band": str, "lag": str})
    ref = ref[(ref.kind == "test1992") & (ref.estimator == "hilbert")].set_index(["band", "lag"]).ratio
    c = cl.merge(lv[lv.usable][["event", "net", "sta", "band"]], on=["event", "net", "sta", "band"])
    rows = []
    for (box, event, band, lag), g in pd.concat([c, c.assign(event="all")]).groupby(["box", "event", "band", "lag"],
                                                                                   sort=False):
        r, col = g[g.valid].ratio, float(ref.get((band, lag), np.nan))
        if r.empty:
            continue
        q = np.percentile(r, [0, 5, 25, 50, 75, 95, 100])
        rows.append(dict(box=box, event=event, band=band, lag=lag, n_usable=len(g), n_valid=len(r), min=q[0], p05=q[1],
                         q25=q[2], median=q[3], q75=q[4], p95=q[5], max=q[6], col1992=col,
                         frac_below_col1992=float((r < col).mean())))
    return pd.DataFrame(rows)


def main() -> None:
    wf_dir = data_dir("a5/wf")
    ev = a5.explosions("dprk", ("USGS",), starttime=T("2006-01-01"), endtime=T("2018-01-01"), latitude=a2b.SITE[0],
                       longitude=a2b.SITE[1], maxradius=0.5, minmagnitude=4.0)
    ev = ev[ev.time.str[:10].isin(DATES)].copy()
    ev["name"] = "dprk_" + ev.time.str[:10]
    plan = []
    for e in ev.itertuples():
        rows, invs = sites(e.name, T(e.time))
        for r in rows:
            r["tp"], r["dist_deg"] = a5.p_time(T(e.time), e.latitude, e.longitude, r["lat"], r["lon"])
        plan.append((e, rows, invs))
    est = sum(r["fs"] for _, rows, _ in plan for r in rows) * (a2.PRE + a2.POST) * 4  # 4 bytes a sample: upper bound
    if est > MAX_BYTES:
        raise SystemExit(f"download would be about {est / 1e9:.1f} GB (limit {MAX_BYTES / 1e9:g} GB): nothing fetched")

    levels, clutter, tables = [], [], []
    for e, rows, invs in plan:
        st = a5.waveforms(wf_dir / f"{e.name}.mseed", [(r["net"], r["sta"], r["loc"], r["cha"], r["tp"] - a2.PRE,
                                                         r["tp"] + a2.POST) for r in rows])
        grids, n0 = {}, len(levels)
        for r in rows:
            sid = f"{r['net']}.{r['sta']}"
            base = dict(event=e.name, source=e.source, mb=e.mb, **{k: r[k] for k in ("net", "sta", "loc", "cha", "box")},
                        lat=r["lat"], lon=r["lon"], dist_deg=round(r["dist_deg"], 2))
            sel = st.select(network=r["net"], station=r["sta"], location=r["loc"], channel=r["cha"]) if st else []
            if not len(sel):
                levels.append(dict(base, status="no data"))
                continue
            try:
                tr, gaps, is_clipped, covered = a5.velocity(sel, invs[r["box"]], r["tp"])
            except (ObsPyException, ValueError, IndexError) as exc:
                levels.append(dict(base, status=f"failed: {type(exc).__name__}"))
                continue
            lev, clu = a5.measure(tr, r["tp"])
            if not lev:
                levels.append(dict(base, status="not covered"))
                continue
            base["fs"] = tr.stats.sampling_rate
            levels += [dict(base, status="ok", **x, gaps=gaps, clipped=is_clipped, covered=covered,
                            usable=bool(covered and not gaps and not is_clipped and x["snr"] >= a5.SNR_SEEN))
                       for x in lev]
            clutter += [dict(base, **x) for x in clu]
            grids[sid] = (tr, r)
        new = pd.DataFrame(levels[n0:])
        cells = pd.DataFrame(clutter)
        for band in a2.BANDS:
            lab = f"{band[0]:g}-{band[1]:g}"
            use = new[(new.status == "ok") & (new.band == lab) & new.usable.eq(True)] if "band" in new else new[:0]
            if len(use) < 2:
                continue
            ids = (use.net + "." + use.sta).to_numpy()
            X = np.vstack([a5.on_grid(grids[i][0], grids[i][1]["tp"], band, FS_CC) for i in ids])
            c = cells[(cells.event == e.name) & (cells.band == lab)]
            c = c.assign(sid=c.net + "." + c.sta).pivot(index="sid", columns="lag", values="valid").reindex(ids)
            valid = {lag: c[lag].to_numpy(dtype=bool) for lag in LAGS}
            xy = [(grids[i][1]["lat"], grids[i][1]["lon"]) for i in ids]
            dist = np.array([[gps2dist_azimuth(*p, *q)[0] / 1000.0 if p != q else 0.0 for q in xy] for p in xy])
            tables.append(pair_table(e.name, lab, ids, X, float(use.p_peak_lag_s.median()), valid, dist))
    lv, cl, pairs = pd.DataFrame(levels), pd.DataFrame(clutter), pd.concat(tables, ignore_index=True)
    lv["usable"] = lv.usable.eq(True)  # records that were not measured are not usable
    cv, sp = curve(pairs), spread(lv, cl)
    lv.to_csv(HERE / "a5_dprk_levels.csv", index=False)
    cl.to_csv(HERE / "a5_dprk_clutter.csv", index=False)
    sp.to_csv(HERE / "a5_dprk_site_spread.csv", index=False)
    cv.to_csv(HERE / "a5_dprk_coherence.csv", index=False)
    out = pairs.drop(columns=NOISE_COLS).assign(cc_noise_median=pairs[NOISE_COLS].median(axis=1),
                                                cc_noise_max=pairs[NOISE_COLS].max(axis=1))
    out.round(dict(dist_km=2, cc=4, lag_s=3, cc_noise_median=4, cc_noise_max=4)).to_csv(HERE / "a5_dprk_pairs.csv",
                                                                                     index=False)

    ok = lv[lv.status == "ok"]
    checks = [
        dict(check="catalog_explosions", value=len(ev), of=len(DATES), triggered=len(ev) != len(DATES),
             detail="; ".join(f"{n} mb{m:g} {s}" for n, m, s in zip(ev.name, ev.mb, ev.source))),
        dict(check="download_estimate_bytes", value=int(est), of=int(MAX_BYTES), triggered=False,
             detail="4 bytes per sample over every requested window (upper bound)"),
    ]
    for e, rows, _ in plan:
        bad = lv[(lv.event == e.name) & (lv.status != "ok")]
        got = ok[ok.event == e.name]
        checks.append(dict(check=f"stations_measured_{e.name}", value=int((got.net + "." + got.sta).nunique()),
                           of=len(rows), triggered="",
                           detail="; ".join(f"{n}.{s}:{x}" for n, s, x in zip(bad.net, bad.sta, bad.status))))
    for lab in ok.band.unique():
        b, c = ok[ok.band == lab], cl[cl.band == lab].merge(ok[ok.usable & (ok.band == lab)][["event", "net", "sta"]])
        checks.append(dict(check=f"usable_records_{lab}", value=int(b.usable.sum()), of=len(b), triggered="",
                           detail=f"snr >= {a5.SNR_SEEN:g}, covered, no gaps, not clipped; valid clutter cells of "
                           "the usable: " + "; ".join(f"{lag}s:{int(c[c.lag == lag].valid.sum())}" for lag in LAGS)))
    nn = cv[(cv.event == "all") & (cv.dist_bin == "nn")]

    def cells_above(sel: pd.DataFrame) -> tuple[int, str]:
        return int((sel.cc_median > sel.noise_cc_p95).sum()), "; ".join(
            f"{r.band}Hz/{r.lag}:{r.cc_median:.3f} vs {r.noise_cc_p95:.3f} (n={r.n_pairs}, {r.dist_median_km:.0f} km, "
            f"|lag| {r.lag_abs_median_s:.2f} s)"
            for r in sel.itertuples())

    n_coda, d_coda = cells_above(nn[nn.lag != "P"])
    n_p, d_p = cells_above(nn[nn.lag == "P"])
    closest = cv[(cv.event == "all") & (cv.dist_bin == f"{BINS_KM[0]:g}-{BINS_KM[1]:g}") & (cv.lag != "P")]
    n_close, d_close = cells_above(closest)
    size = sum(p.stat().st_size for d in (wf_dir, data_dir("a5/sta"), data_dir("a5/cat")) for p in data_glob(d, "*dprk*"))
    checks += [
        dict(check="nn_p_cells_above_noise", value=n_p, of=int((nn.lag == "P").sum()), triggered=n_p == 0,
             detail="positive control, nearest-neighbor median vs noise p95: " + d_p),
        dict(check="nn_coda_cells_above_noise", value=n_coda, of=int((nn.lag != "P").sum()), triggered="",
             detail="nearest-neighbor median vs noise p95: " + d_coda),
        dict(check="closest_bin_coda_cells_above_noise", value=n_close, of=len(closest), triggered="",
             detail="reported beside the rule, the shortest separation bin: " + d_close),
        dict(check="back_projection_ruled_out", value=int(n_coda == 0), of="", triggered="",
             detail="plan S3 rule: nearest-pair coda coherence at the noise level in every cell"),
        dict(check="cache_bytes", value=size, of="", triggered="", detail="$AKP_DATA/a5: this script's catalog, StationXML and miniSEED"),
    ]
    checks.append(dict(check="gate_pass", value=int(not any(ch["triggered"] is True for ch in checks)), of="",
                       triggered="", detail=""))
    a5.write_checks(HERE / "a5_dprk_checks.csv", checks)
    print(lv.groupby(["event", "status"]).size().to_dict())
    print(cv[cv.event == "all"].to_string())


def selfcheck() -> None:
    """Three synthetic sites: A and B share a signal after P (B 0.75 s late), C is noise. The pair A-B must come
    out coherent and above its noise baseline, A-C and B-C must not; nearest neighbors are A-B and B-C."""
    rng = np.random.default_rng(0)
    n = int(round((a5.GRID[1] - a5.GRID[0]) * FS_CC))
    s = rng.standard_normal(n) * (np.arange(n) >= a5.ix(-5.0, FS_CC))
    X = np.vstack([s, np.roll(s, 30), np.zeros(n)]) + 0.2 * rng.standard_normal((3, n))
    dist = np.array([[0.0, 1.0, 50.0], [1.0, 0.0, 49.0], [50.0, 49.0, 0.0]])
    p = pair_table("syn", "1-2", np.array(["A", "B", "C"]), X, 2.0, {lag: np.ones(3, dtype=bool) for lag in LAGS}, dist)
    ab, ac = p[(p.a == "A") & (p.b == "B")], p[(p.a == "A") & (p.b == "C")]
    assert (ab.cc > 0.9).all() and (ab.lag_s == 0.75).all() and (ac.cc < 0.3).all(), (ab.cc.min(), ac.cc.max())
    assert ab.nn.all() and not ac.nn.any() and p[(p.a == "B") & (p.b == "C")].nn.all()
    c = curve(p).set_index(["event", "lag", "dist_bin"])
    near, far = c.loc[("all", LAGS[0], "0-2")], c.loc[("all", LAGS[0], "20-50")]
    assert near.cc_median > near.noise_cc_p95 > far.cc_median and near.n_pairs == far.n_pairs == 1


if __name__ == "__main__":
    selfcheck()
    main()
