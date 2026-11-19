"""P4 S9: is any exposed station quiet enough, after the P pulse, for the most favorable simulated body to stand as high as its record?

Inputs:
- a5_dprk_levels.csv: S3's records of the three Korean tests (usable flag and P-peak lag per band);
- the cached catalog, StationXML and miniSEED S3 used ($AKP_DATA/a5), through S3's own readers
  (a5_lopnor_repeats, a5_dprk_network). A file missing from the cache stops the run: nothing is fetched;
- s9_receivers.csv and s9_checks.csv (the census), and the Track A cells of p3_prior.npz;
- a8_production.csv and the saved traces of the eight simulations ($AKP_DATA/a8/production).
Rules fixed in advance, with the fitted law of the census. Every value those rules fix is a constant below.

Method:
- pulse end, per band: an ideal impulse through A2's filter and A2's one-second smoothing
  (a2_wavefield.envelopes, the Hilbert envelope S3 measured with); the last time the response is at or
  above PULSE_LEVEL of its peak, plus PULSE_ALLOW_S. Asserted longer than the fixed rule (two periods of the
  band's lowest frequency);
- stations: S3 stations with a usable record in BANDS. Exposed: a passive receiver of the census carries
  the station code within MATCH_KM of S3's position, and stands on Track A or within the largest range
  of the census rows with generous attenuation. Control: farther than CONTROL_KM from Track A (nearest
  cell center of the prior grid, as the census computes it). Other stations are not measured;
- records: usable (S3's flag, per band) and a committed P-peak lag of LAG_LIMIT_S or less;
- envelope: a8_production.band_envelope (zero-phase band-pass, analytic signal, no smoothing) of the
  velocity record (S3's processing), as a share of its P peak: the largest value within P_SEARCH_S of
  the committed P-peak time;
- bins: [k, k + 1) s after the P peak, k from the first whole second after the pulse end to LAST_LAG_S;
- reference share of bin k: A (SPEED_KMS k)^b, with A and b the fit of s9_census.sensitivity() (the
  most exposed case, body as simulated, no attenuation) and the range the one a wave of SPEED_KMS
  reaches at the start of the bin;
- statistic of a record: the smallest over bins of (bin peak / reference share). No station is ranked
  or flagged: the tables hold the statistic, and the summary holds counts and percentiles.

Gate, run before any record is read: the same two functions (`shares`, `window_peak`) on the
saved vertical trace of each run with the body, the twin's trace giving the P peak, in the committed
slow window. The result must be within GATE_TOL of the committed share (forward direction) at every
range where the twin's own envelope in that window is below that share. If it is not, the run stops.

Pulse table (committed table and the simulations' own timing; no trace is needed): for each ground
case, direction, band and range, the start of the slow window after the incident P at the receiver,
against the pulse end. Time origin: a8_production.csv gives t_lo_s in simulation time, where the
wavelet center of the plane wave leaves the source sheet under x = 0 at t_c (a8_elastic3d: sheet
delay t0 + x sin(angle) / vp) and reaches the surface at x = 0 at t_surf. At a receiver at range r in
the direction with unit vector (ux, uy) the incident P peaks at t_surf + p r ux (p the ray parameter;
a8_production.measure's t_inc), so the start after P is t_lo_s - t_surf - p r ux. The gate records how
far the twin's envelope peak lies from that time.

Choices not fixed by each is the option least favorable to "the record is quiet enough", or
does not change a bin:
- the sampling rate of the ideal impulse: the response is computed at every sampling rate of S3's
  usable records and the longest pulse is used (the rates differ by under 0.1 s, inside one bin);
- A2 has two envelopes; the pulse uses the smoothed Hilbert envelope, which S3's measures use;
- bins are counted from the time of the P peak found in the unsmoothed envelope, not from the
  committed (smoothed) time: the rule defines the P peak as that largest value;
- "to 15 s": the last bin is [14, 15);
- the code match uses the station code without the network, because the census joins the codes at
  one position and the registry carries no network;
- a station is a network and station code; a station S3 lists at two positions is treated per record;
- control stations need no census match (the census table stops 50 km from Track A);
- the gate runs on all four ground cases in the bands of the check that were simulated, not on one
  trace; the P-peak time given to the code is the predicted arrival of the plane wave at the
  receiver; a gate with no counted range fails;
- the pulse table also gives the lowest band, by the same rule, though the check does not use it.

Outputs beside this file: s9_station_gate.csv, s9_pulse_table.csv, s9_station_records.csv (one row per
station, event and band), s9_station_check.csv (per group and band), s9_station_checks.csv.
Self-check: a test trace with a packet of known share at a known lag gives that share in that bin
within SELF_TOL, with a committed P-peak time that is off by a second.
Run: PYTHONPATH=. python scripts/mem_guard.py --frac 0.40 --every 0.5 -- python -m akp.stage_p4.s9_station_check
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from affine import Affine
from obspy import Trace
from obspy import UTCDateTime as T
from scipy.spatial import cKDTree

from akp.paths import data_dir
from akp.stage_a import a2_wavefield as a2
from akp.stage_a import a5_dprk_network as s3
from akp.stage_a import a5_lopnor_repeats as a5
from akp.stage_a import a8_production as a8
from akp.stage_p3.p3_prior import out_dir as prior_dir
from akp.stage_p4.s9_census import BAND, CASE, TO_AK, sensitivity
from akp.zones import BOXES

HERE = Path(__file__).parent
STAGE_A = HERE.parent / "stage_a"
BANDS = ("1-2", "2-4")  # The two highest bands with a simulation
MATCH_KM = 1.0  # Station match tolerance
CONTROL_KM = 22.0  # Control distance from Track A
LAG_LIMIT_S = 5.0  # P-peak lag limit
P_SEARCH_S = 3.0  # P search window about the committed P-peak time
PULSE_LEVEL = 0.03  # Pulse-end level, of the peak of the impulse response
PULSE_ALLOW_S = 0.5  # Allowance for the surface reflection near the source
D86_PERIODS = 2.0  # The pulse rule the rule replaces, in periods of the band's lowest frequency
LAST_LAG_S = 15  # Last lag, end of A2's first coda window
SPEED_KMS = 0.3  # Slowest surface-wave speed allowed
GATE_TOL = 0.20  # S5 gate tolerance
SELF_TOL = 0.03  # self-check: "a few percent"
IMPULSE_S = 150.0  # half-length of the impulse record: long against every filter response
GROUPS = ("exposed", "control")


def pulse_end(band: str, fs: float) -> float:
    """Seconds after an ideal impulse at which A2's smoothed envelope last stands at PULSE_LEVEL of its peak, plus the allowance."""
    n = int(round(IMPULSE_S * fs))
    x = np.zeros(2 * n + 1)
    x[n] = 1.0
    t, hil, _ = a2.envelopes(Trace(x, header=dict(sampling_rate=fs)), *a8.BANDS[band])
    return float(t[np.flatnonzero(hil >= PULSE_LEVEL * hil.max())[-1]] - t[n]) + PULSE_ALLOW_S


def shares(x: np.ndarray, dt: float, band: str, t_p: float, p_from: np.ndarray | None = None) -> tuple:
    """(time after the P peak, envelope as a share of the P peak, time of the P peak).

    The envelope is the simulations' (a8.band_envelope, no smoothing). The P peak is its largest value
    within P_SEARCH_S of `t_p`, in the trace itself, or in `p_from` when the incident P is a separate
    trace (the gate: the twin run)."""
    env = np.abs(a8.band_envelope(x, band, dt))
    ref = env if p_from is None else np.abs(a8.band_envelope(p_from, band, dt))
    t = np.arange(len(x)) * dt
    near = np.flatnonzero(np.abs(t - t_p) <= P_SEARCH_S)
    i = near[np.argmax(ref[near])]
    return t - t[i], env / ref[i], float(t[i])


def window_peak(lag: np.ndarray, share: np.ndarray, lo: float, hi: float) -> float:
    """Largest share in [lo, hi) seconds after the P peak."""
    sel = (lag >= lo) & (lag < hi)
    return float(share[sel].max()) if sel.any() else float("nan")


def statistic(lag: np.ndarray, share: np.ndarray, first: int, amp: float, b: float) -> dict:
    """The fixed statistic: the smallest over one-second bins of the bin peak over the reference share."""
    ks = np.arange(first, LAST_LAG_S)
    peaks = np.array([window_peak(lag, share, k, k + 1) for k in ks])
    ref = amp * (SPEED_KMS * ks) ** b  # range at the start of the bin: the nearest, so the largest share
    assert np.isfinite(peaks).all(), "a bin holds no sample"
    i = int(np.argmin(peaks / ref))
    return dict(statistic=peaks[i] / ref[i], bin_of_min_s=int(ks[i]), bin_peak=peaks[i], reference_share=ref[i], n_bins=len(ks))


def gate(setups: dict) -> pd.DataFrame:
    """The measurement on the saved simulated traces against the committed slow-window shares (forward)."""
    com = pd.read_csv(STAGE_A / "a8_production.csv", dtype={"band": str})
    com = com[(com.window == "slow") & (com.component == "vertical") & (com.azimuth == "forward")].set_index(["case", "band", "range_m"])
    assert next(iter(a8.AZIMUTHS)) == "forward"  # the forward receivers are the first columns of the saved traces
    rows = []
    for case, (_, _, bands) in a8.CASES.items():
        s = setups[case]
        twin, body = a8.load(case, False), a8.load(case, True)
        assert twin["dt"] == body["dt"] == s["dt"]
        for band in (b for b in BANDS if b in bands):
            for j, r in enumerate(s["ranges"]):
                c = com.loc[(case, band, float(r))]
                t_inc = s["t_surf"] + s["p"] * r  # the incident P at this receiver: the "committed P-peak time" of a simulated trace
                lag, share, t_pk = shares(body["vz"][:, j].astype(float), s["dt"], band, t_inc, p_from=twin["vz"][:, j].astype(float))
                lo, hi = c.t_lo_s - t_pk, c.t_hi_s - t_pk
                tail = window_peak(*shares(twin["vz"][:, j].astype(float), s["dt"], band, t_inc)[:2], lo, hi)
                got = window_peak(lag, share, lo, hi)
                counted = bool(tail < c.numerical)
                rows.append(dict(case=case, band=band, range_m=float(r), committed=c.numerical, measured=got, ratio=got / c.numerical, twin_tail=tail,
                                 counted=counted, passed=bool(abs(got / c.numerical - 1) <= GATE_TOL) if counted else "", p_peak_minus_predicted_s=t_pk - t_inc))
    return pd.DataFrame(rows)


def pulse_table(setups: dict, pulse: dict) -> pd.DataFrame:
    """Start of the slow window after the incident P at the receiver, against the pulse end (see the docstring for the origin)."""
    p = pd.read_csv(STAGE_A / "a8_production.csv", dtype={"band": str})
    p = p[(p.window == "slow") & (p.component == "vertical")]
    t_surf, ray = p.case.map(lambda c: setups[c]["t_surf"]), p.case.map(lambda c: setups[c]["p"])
    start = p.t_lo_s - t_surf - ray * p.range_m * p.azimuth.map(lambda a: a8.AZIMUTHS[a][0])
    out = pd.DataFrame(dict(case=p.case, azimuth=p.azimuth, band=p.band, range_m=p.range_m, slow_start_after_p_s=start.round(4), pulse_end_s=p.band.map(pulse)))
    out["inside_pulse"] = out.slow_start_after_p_s < out.pulse_end_s
    return out


def track_a_km(lon, lat) -> np.ndarray:
    """Distance (km) to the nearest Track A cell center, as s9_census computes km_to_track_a."""
    pr = np.load(prior_dir() / "p3_prior.npz")
    rows, cols = np.nonzero(pr["land"])
    cx, cy = Affine(*pr["transform"]) * (cols[pr["track_a"]] + 0.5, rows[pr["track_a"]] + 0.5)
    x, y = TO_AK.transform(np.asarray(lon), np.asarray(lat))
    return cKDTree(np.c_[cx, cy]).query(np.c_[x, y])[0] / 1000


def grouped(lv: pd.DataFrame, largest_km: float) -> pd.DataFrame:
    """S3's usable records with group (exposed, control, neither), distance to Track A and the match flags."""
    rec = pd.read_csv(HERE / "s9_receivers.csv")
    rec = rec[rec.cls == "passive"].reset_index(drop=True)
    rx, ry = TO_AK.transform(rec.lon.to_numpy(), rec.lat.to_numpy())
    codes = [{c.split(".")[-1] for c in s.split("+")} for s in rec.codes.fillna("")]
    pos = lv[["net", "sta", "lat", "lon"]].drop_duplicates().reset_index(drop=True)
    pos["grid_km"] = track_a_km(pos.lon, pos.lat)
    x, y = TO_AK.transform(pos.lon.to_numpy(), pos.lat.to_numpy())
    rows = []
    for i, s in pos.iterrows():
        same = np.flatnonzero([s.sta in c for c in codes])
        d = np.hypot(rx[same] - x[i], ry[same] - y[i]) / 1000
        matched = bool(len(same) and d.min() <= MATCH_KM)
        km, exposed = s.grid_km, False
        if matched:
            m = rec.loc[same[d.argmin()]]
            exposed = bool(m.on_track_a or m.km_to_track_a <= largest_km)
            if exposed:
                assert abs(m.km_to_track_a - s.grid_km) <= MATCH_KM + 0.01, (s.sta, m.km_to_track_a, s.grid_km)
                km = m.km_to_track_a
        rows.append(dict(code_in_census=bool(len(same)), matched=matched, km_to_track_a=km,
                         group="exposed" if exposed else "control" if s.grid_km > CONTROL_KM else "neither"))
    return lv.merge(pd.concat([pos, pd.DataFrame(rows)], axis=1), on=["net", "sta", "lat", "lon"])


def cached_records(want: pd.DataFrame):
    """(rows of `want` of one station and event, velocity trace, predicted P), from the cache S3 filled."""
    cat, wf_dir, sta_dir = data_dir("a5/cat") / "dprk_USGS.xml", data_dir("a5/wf"), data_dir("a5/sta")
    assert cat.exists(), f"{cat} is not cached: nothing is fetched here"
    ev = a5.explosions("dprk", ("USGS",))
    ev = ev[ev.time.str[:10].isin(s3.DATES)]
    for e in ev.itertuples():
        name = "dprk_" + e.time[:10]
        need = [wf_dir / f"{name}.mseed"] + [sta_dir / f"{name}_{box}.xml" for box in BOXES]
        assert all(p.exists() or p.with_suffix(".nodata").exists() for p in need), f"{name}: not cached, nothing is fetched here"
        sites, invs = s3.sites(name, T(e.time))
        st = a5.waveforms(need[0], [])
        for r in sites:
            w = want[(want.event == name) & (want.net == r["net"]) & (want.sta == r["sta"])]
            if w.empty:
                continue
            assert (w.cha == r["cha"]).all()
            tp, _ = a5.p_time(T(e.time), e.latitude, e.longitude, r["lat"], r["lon"])
            sel = st.select(network=r["net"], station=r["sta"], location=r["loc"], channel=r["cha"])
            yield w, a5.velocity(sel, invs[r["box"]], tp)[0], tp


def summary(out: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        for band in BANDS + ("both",):
            g = out[(out.group == group) & ((out.band == band) | (band == "both"))]
            low = g[g.statistic <= 1]
            q = np.percentile(g.statistic, [5, 25, 50]) if len(g) else [np.nan] * 3
            rows.append(dict(group=group, band=band, stations=(g.net + "." + g.sta).nunique(), records=len(g), records_statistic_le_1=len(low),
                             stations_with_such_record=(low.net + "." + low.sta).nunique(), p05=q[0], p25=q[1], p50=q[2], minimum=g.statistic.min()))
    return pd.DataFrame(rows)


def selfcheck() -> float:
    """A packet of share 0.2 at 8.4 s after a P pulse of the same shape: its bin returns 0.2."""
    dt, s, at = 0.02, 0.2, 8.4
    t = np.arange(15000) * dt

    def packet(t0: float) -> np.ndarray:
        return np.exp(-((t - t0) / 0.4) ** 2) * np.cos(2 * np.pi * 1.5 * (t - t0))
    lag, share, t_pk = shares(packet(100.0) + s * packet(100.0 + at), dt, "1-2", 101.0)  # a committed time one second off
    got = window_peak(lag, share, 8, 9)
    assert abs(t_pk - 100.0) <= dt and abs(got / s - 1) <= SELF_TOL, (t_pk, got)
    assert window_peak(lag, share, 13, 14) < 0.01 and np.isnan(window_peak(lag, share, 500, 501))  # the filter rings for two seconds: bins beside the packet are not empty
    stat = statistic(lag, share, 4, 1.0, 0.0)  # reference share one everywhere: the statistic is the quietest bin
    assert stat["n_bins"] == LAST_LAG_S - 4 and stat["statistic"] < 0.01 and stat["bin_of_min_s"] != 8
    return got / s


def main() -> None:
    lv = pd.read_csv(STAGE_A / "a5_dprk_levels.csv", dtype={"band": str})
    lv = lv[lv.usable & lv.band.isin(BANDS)]
    rates = sorted(lv.fs.unique())
    ends = {band: [pulse_end(band, fs) for fs in rates] for band in a8.BANDS}
    pulse = {band: max(v) for band, v in ends.items()}
    d86 = {band: D86_PERIODS / a8.BANDS[band][0] for band in a8.BANDS}
    assert all(pulse[b] > d86[b] for b in pulse), "the pulse must be longer than the fixed rule"
    first = {band: int(np.ceil(pulse[band])) for band in BANDS}
    recovered = selfcheck()

    _, info = sensitivity()  # the fit, as the census makes it: share = A r^b, r in km
    amp, b = info["fit_share_at_1km"], info["fit_exponent"]
    census = pd.read_csv(HERE / "s9_checks.csv").set_index("check").value
    assert float(census["fit_share_at_1km"]) == amp and float(census["fit_exponent"]) == b, "the fit differs from the committed census"
    largest_km = float(census["largest_range_generous_attenuation_km"])

    setups = {case: a8.setup(case) for case in a8.CASES}
    table = pulse_table(setups, pulse)
    table.to_csv(HERE / "s9_pulse_table.csv", index=False)
    inside = table.groupby(["case", "band", "azimuth"], sort=False).inside_pulse.agg(["sum", "size"])
    margin = table.pulse_end_s - table.slow_start_after_p_s
    tight = table.loc[margin.idxmin()]
    shortest = table.band.map({band: min(v) for band, v in ends.items()})  # had the shortest pulse over the sampling rates been used
    exposed_case = inside.loc[(CASE, BAND)]

    g = gate(setups)
    g.to_csv(HERE / "s9_station_gate.csv", index=False)
    cnt = g[g.counted]
    gate_pass = bool(len(cnt) and cnt.passed.astype(bool).all())
    worst = cnt.ratio[(cnt.ratio - 1).abs().idxmax()] if len(cnt) else np.nan
    checks = [dict(check=f"pulse_end_s_{band}", value=round(pulse[band], 3)) for band in pulse]
    checks += [dict(check=f"pulse_end_spread_over_sampling_rates_s_{band}", value=round(max(v) - min(v), 3)) for band, v in ends.items()]
    checks += [dict(check=f"d86_pulse_end_s_{band}", value=d86[band]) for band in d86]
    checks += [dict(check=f"first_bin_s_{band}", value=first[band]) for band in BANDS]
    checks += [dict(check=f"reference_share_first_bin_{band}", value=round(amp * (SPEED_KMS * first[band]) ** b, 4)) for band in BANDS]
    checks += [dict(check="reference_share_last_bin", value=round(amp * (SPEED_KMS * (LAST_LAG_S - 1)) ** b, 4)),
               dict(check="impulse_sampling_rates_hz", value=" ".join(f"{fs:g}" for fs in rates)),
               dict(check="fit_share_at_1km", value=amp), dict(check="fit_exponent", value=b), dict(check="fit_case_band", value=f"{CASE} {BAND}"),
               dict(check="largest_range_generous_attenuation_km", value=largest_km),
               dict(check="selfcheck_packet_recovered_over_true", value=round(recovered, 4)),
               dict(check="pulse_groups_case_band_direction", value=len(inside)),
               dict(check="pulse_groups_inside_at_every_range", value=int((inside["sum"] == inside["size"]).sum())),
               dict(check="most_exposed_case_inside_pulse_at_every_range", value=bool((exposed_case["sum"] == exposed_case["size"]).all()))]
    checks += [dict(check="pulse_table_least_margin_s", value=round(float(margin.min()), 4)),
               dict(check="pulse_table_least_margin_row", value=f"{tight.case} {tight.band} {tight.azimuth} {tight.range_m:g} m"),
               dict(check="pulse_table_rows_outside_with_shortest_pulse_over_sampling_rates", value=int((table.slow_start_after_p_s >= shortest).sum()))]
    checks += [dict(check=f"ranges_inside_pulse_{case}_{band}_{az}", value=f"{r['sum']} of {r['size']}") for (case, band, az), r in inside.iterrows()]
    # Added later: where the Rayleigh wave itself, at the host's Rayleigh speed,
    # arrives as the pulse ends. The slow window of a8_production starts earlier, with the body's shear speed.
    u_host = float(pd.read_csv(HERE.parent / "stage_a" / "a8_attenuation.csv").query("case == 'thawed_1200'").rayleigh_speed_ms.iloc[0])
    checks += [dict(check=f"rayleigh_arrival_at_pulse_end_km_{band}", value=round(pulse[band] * u_host / 1000, 2)) for band in pulse]
    checks += [dict(check="gate_ranges_passed", value=int(cnt.passed.astype(bool).sum())),
               dict(check="gate_counted_range_min_km", value=float(cnt.range_m.min()) / 1000), dict(check="gate_counted_range_max_km", value=float(cnt.range_m.max()) / 1000)]
    checks += [dict(check="gate_rows", value=len(g)), dict(check="gate_ranges_counted", value=len(cnt)),
               dict(check="gate_ranges_failed", value=int((~cnt.passed.astype(bool)).sum())), dict(check="gate_worst_ratio", value=round(float(worst), 4)),
               dict(check="gate_p_peak_minus_predicted_max_abs_s", value=round(float(g.p_peak_minus_predicted_s.abs().max()), 4)),
               dict(check="gate_pass", value=gate_pass)]
    if not gate_pass:
        for name in ("s9_station_records.csv", "s9_station_check.csv"):
            (HERE / name).unlink(missing_ok=True)
        pd.DataFrame(checks).to_csv(HERE / "s9_station_checks.csv", index=False)
        print(pd.DataFrame(checks).to_string(index=False))
        raise SystemExit("the gate failed: no record was read and nothing is reported")

    lv = grouped(lv, largest_km)
    sid = lv.net + "." + lv.sta
    want = lv[lv.group.isin(GROUPS) & (lv.p_peak_lag_s <= LAG_LIMIT_S)]
    checks += [dict(check="s3_stations_usable", value=sid.nunique()), dict(check="s3_records_usable", value=len(lv)),
               dict(check="stations_code_in_census", value=sid[lv.code_in_census].nunique()),
               dict(check="stations_dropped_by_position_match", value=sid[lv.code_in_census & ~lv.matched].nunique()),
               dict(check="stations_neither_exposed_nor_control", value=sid[lv.group == "neither"].nunique())]
    for group in GROUPS:
        sel = lv.group == group
        checks += [dict(check=f"stations_{group}_before_lag_rule", value=sid[sel].nunique()), dict(check=f"records_{group}_before_lag_rule", value=int(sel.sum())),
                   dict(check=f"records_{group}_dropped_by_lag_rule", value=int((sel & (lv.p_peak_lag_s > LAG_LIMIT_S)).sum())),
                   dict(check=f"stations_{group}_dropped_by_lag_rule", value=sid[sel].nunique() - sid[want.index][want.group == group].nunique())]

    rows, repro = [], 0.0
    for w, tr, tp in cached_records(want):
        lev = {x["band"]: x for x in a5.measure(tr, tp)[0]}
        for r in w.itertuples():
            assert lev[r.band]["p_peak_lag_s"] == r.p_peak_lag_s, "the cached record is not the one S3 measured"
            repro = max(repro, abs(lev[r.band]["p_peak_vel"] / r.p_peak_vel - 1))
            t_committed = (tp - tr.stats.starttime) + r.p_peak_lag_s
            lag, share, t_pk = shares(tr.data, tr.stats.delta, r.band, t_committed)
            rows.append(dict(net=r.net, sta=r.sta, group=r.group, km_to_track_a=round(r.km_to_track_a, 2), band=r.band, event=r.event,
                             p_peak_lag_s=r.p_peak_lag_s, p_peak_shift_s=round(t_pk - t_committed, 3), **statistic(lag, share, first[r.band], amp, b)))
    out = pd.DataFrame(rows).sort_values(["group", "net", "sta", "event", "band"], ascending=[False, True, True, True, True])
    assert len(out) == len(want) and repro < 1e-6, (len(out), len(want), repro)
    out.to_csv(HERE / "s9_station_records.csv", index=False)
    summ = summary(out)
    summ.to_csv(HERE / "s9_station_check.csv", index=False)

    both = summ[summ.band == "both"].set_index("group")
    checks += [dict(check="records_measured", value=len(out)), dict(check="s3_p_peak_reproduced_max_relative_difference", value=repro),
               dict(check="p_peak_shift_median_s", value=float(out.p_peak_shift_s.median())), dict(check="p_peak_shift_max_abs_s", value=float(out.p_peak_shift_s.abs().max()))]
    for group in GROUPS:
        checks += [dict(check=f"{k}_{group}", value=both.loc[group, k]) for k in ("stations", "records", "records_statistic_le_1", "stations_with_such_record")]
    checks.append(dict(check="prediction_of_d87_held", value=bool(both.loc["exposed", "records_statistic_le_1"] == 0)))
    pd.DataFrame(checks).to_csv(HERE / "s9_station_checks.csv", index=False)
    print(pd.DataFrame(checks).to_string(index=False))
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
