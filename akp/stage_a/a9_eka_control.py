"""A9: positive control of the array code on a known scatterer -- the Eskdalemuir array (EKA) and Moffat Water.

Question (step B): does the back-projection of
a7_coda_backprojection, as run on the Eielson array, find a scatterer that the literature has located
twice? Phase 1 (this run) fixes everything that needs no waveform. Phase 2 (--waveforms) reads the
records.

Inputs (all open, no account; nothing identifying is sent):
- Element positions and channel epochs: EarthScope station service, network IM, stations EK*, channel
  BHZ. StationXML cached under $AKP_DATA/a9.
- Catalogs: a6's cached list of M>=6 earthquakes ($AKP_DATA/a6), and the USGS catalog of a6's box
  around the Korean test site from 2006 (cached under $AKP_DATA/a9); the tests are the rows typed
  "nuclear explosion".
- Eielson: a7's constants (imported) and its committed event list a7_events.csv.
- Published values, with pages, as constants below: Key (1967, BSSA 57(1) 27-37); Gupta, Lynnes and
  Wagner (1990, GRL 17(2) 183-186).
- Phase 2 only: BHZ windows, predicted P -120..+300 s, cached as miniSEED under $AKP_DATA/a9.

Method. a7 and a6 are imported, not copied. Three pieces of a7 could not be reused and are rewritten
here: the source rule and the selection by signal-to-noise (inline in a7.main, tied to the Eielson
element table), a7.fetch (channel SHZ is written into it), and a7.psf (it asserts the imaging error and
averages four bearings at fixed ranges; wants one point and the error recorded). a7's blast
calibration is not run: there is no blast catalog here. Its sensitivity sweep and relief
step are not run either: the rules put the power test of step 9 in their place.
1. Elements: every IM EK* site with a BHZ channel. Local east, north about their centroid (a6).
2. Pre-check: a6.geometry and a6.response_rows on this geometry ("Band").
3. Sources by the Eielson rule (a7 step 1): deep teleseisms by a7's depth, magnitude,
   distance, isolation and pP rules, and the Korean tests. An event counts only if at least MIN_ELEM
   elements (a7's share of the array) have a BHZ epoch over its window. The N_CAND largest are the
   candidates. The selection by signal-to-noise needs waveforms and is phase 2.
4. Target: the mean of Key's and Gupta's positions, each set out by its range and bearing from
   the centroid.
5. Decisive lag floor: the largest floor, rounded down to FLOOR_STEP, that admits at the target
   point the scan speeds nearest Key's speed and the speed in Gupta's text, for every event: lead +
   range x slowness >= floor + WIN/2, as a7.valid has it. Phase 1 uses the candidates; phase 2
   recomputes it on the events used. The P-window control keeps a7's two offsets from the floor ("tied to it").
6. Blind zone: a7.valid keeps a window only if lead + range x slowness >= floor + WIN/2, and
   lead = -range x s x cos(bearing - back-azimuth). The nearest range imaged follows per event, bearing
   and scan speed. Given for Eielson (a7's floor, its used events) and for EKA under both floors.
7. Target cell: half-power half-extents (a7.extent) of the stacked energy of a noise-free
   synthetic at the target, at the scan speed nearest Key's, under the decisive floor. The imaging
   error is recorded. A point is in a cell when a7.same_cell says so for those half-extents.
8. Null cells: the target cell rotated about the centroid in steps of ROT_STEP. Left out:
   cells that share a grid cell with the target cell, and cells whose center the floor does not admit
   at both scan speeds of step 5 for every event. If fewer than MIN_NULLS remain the step is halved,
   once. Each cell carries its earliest admitted lag (smallest and median over events, at its center).
   The cells nearest the mirror bearings of the two arms (the target reflected in each arm's axis) and
   the antipode are marked.
9. Phase 2. Per event and band: a7.event_maps under the decisive floor and under
   a7's floor. Statistic: source groups whose strongest peak (a7 rule "strongest_peak") falls in the
   target cell. p = share of cells, target included, with a count at least the target's. Pass: p <=
   P_PASS and at most NOISE_GROUPS_MAX groups whose pre-P noise map peaks in the target cell. The lower
   band decides. Power, before the target cell is read: Key's arrival (the median of his amplitude
   ratios, his speed and range), built from each event's own P wavelet, is injected on the bearing of
   one kept null cell into the events of Key's sector only; power = share of kept cells at which that
   cell then passes the whole test against the kept cells that do not overlap it. Readings: pass |
   not_separable (p passes, the noise condition fails) | fail | uninformative (fail with power below
   POWER_MIN) | unable_to_decide (fewer than MIN_NULLS cells, declared before any map). Reported beside
   the reading, not deciding: the P-window control (a7 step 12), as its count in the target cell and
   that count's rank among the cells (on synthetics it cannot tell an arrival from its absence
   at this array, so only the rotation null guards against what the beam removal leaves of P); the run
   under a7's floor; the upper band; each event's back-azimuth, distance and sector; the back-azimuth
   spread of the events in the cell; the miss distance to both published positions.

Falsifier (--falsifier). Question: is the
pass repeated when the array is lit from other back-azimuths? New events only. The pipeline is frozen:
the lower band, the first run's target cell and rotation list (from its committed tables), the
statistic, the null and the noise condition.
F1. Events from a6's cached catalog from F_SINCE: a7's magnitude, distance and isolation rules, depth
    >= F_DEEP_KM, back-azimuth in F_SECTORS, at least MIN_ELEM epochs, none of the first run's
    candidates. a7's least pP delay is replaced by the window cut.
F2. Window cut: no window of an event ends later than F_MARGIN before its predicted pP, both
    times absolute (predicted P plus the iasp91 delay at the catalog depth). It is a per-event upper
    lag in place of a7's ceiling (see cut()); a7.valid then masks the coda map and every null of the
    event alike.
F3. Floor by the fixed rule on the new events. The fixed filter (admitted at both scan speeds for every
    event, now under each event's cut) is re-applied to the first run's rotation list. Fewer than
    MIN_NULLS cells: "unable to decide". Fewer than F_MIN_GROUPS source groups: "uninformative". Both
    are declared from the catalog, before any fetch, and again on the events used.
F4. Before any fetch: the fixed synthetic (P alone; P with Key's arrival) at one event per source group,
    with the cut. A group whose P alone puts the strongest peak in the target cell is flagged.
F5. Records: fetch, selection and groups as in phase 2; the cut from the picked P. An event whose own
    cut leaves no admitted window at the target is not usable and takes no part. Floor and
    cells again on the events used; the null-cell list written; a reading that the number of groups
    or cells fixes is written; then the target cell, as description only in that case. Readings:
    repeated (the decisive test passes with at least F_MIN_IN_CELL groups in the cell) | repeated from
    adjacent directions (every event in the cell lies within F_ADJACENT of a first-run back-azimuth or
    in Key's sector) | not repeated | uninformative | unable to decide. The same reading is given per
    cluster of back-azimuths (the sorted back-azimuths, split at gaps above F_ADJACENT).

Outputs (beside this file). Phase 1: a9_eka_elements.csv, a9_eka_precheck.csv, a9_eka_candidates.csv,
a9_eka_geometry.csv (`section`: target | floor | blind_zone | cell | null_cell), a9_eka_checks.csv.
Phase 2 (--waveforms): a9_eka_events.csv, a9_eka_cells.csv, a9_eka_power.csv, a9_eka_result.csv.
Falsifier (--falsifier), in the order written: a9_eka_falsifier_candidates.csv,
a9_eka_falsifier_synthetic.csv, a9_eka_falsifier_nulls.csv, a9_eka_falsifier_events.csv,
a9_eka_falsifier_cells.csv, a9_eka_falsifier_result.csv.
Run: PYTHONPATH=. conda run -n swe-akp python -m akp.stage_a.a9_eka_control
(phase 1; deterministic; offline once cached; one process, about 1 GB. Phase 2: add --waveforms. The
falsifier: --falsifier alone; it reads the first run's committed tables and rewrites none of them.)

Self-check: the helpers agree with a7 (the blind-zone formula against a7.valid on the grid). Then the
phase 2 path, from a raw 40 Hz stream to the reading, on synthetic events across the real element
positions: noise, a P wave with statics and site gains, and Key's arrival from the target. With the
arrival the count in the target cell is every group, no rotated cell matches it and the reading is
"pass". Without it the count is zero, the reading is not "pass", and the injection test has power.
Recorded in a9_eka_checks.csv (self_check_*), not asserted: the P-window control's count and rank and
the run under a7's floor. It proves the code path, not the power on real records: the synthetic is
coherent and has no dispersion. For the falsifier: the readings function returns each of its five
readings on made-up counts; and a plane arrival standing in for pP is the strongest peak without the
window cut and is kept out of every window with it.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import Stream, Trace, read_inventory
from obspy import UTCDateTime as T
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel
from scipy.signal import resample

from akp.paths import data_dir
from akp.stage_a import a6_array_precheck as a6
from akp.stage_a import a7_coda_backprojection as a7

HERE = Path(__file__).parent
NET, STA, CHA = "IM", "EK*", "BHZ"  # The open records of the array; the channel is forced by the array
ARMS = {"B": r"EKB\d+", "R": r"EKR\d+"}  # the two lines of the cross; EKB without a digit lies on neither

# Key (1967, BSSA 57(1) 27-37): the local wave from Moffat Water
KEY_U, KEY_BRG = 2.5, 315.0  # km/s, deg: velocity and bearing of the local wave (p. 32)
KEY_RANGE = 13.0  # km, "apparent distance of the source"; measured at "the center seismometer", not identified (p. 34)
KEY_LAG = 4.8  # s after P at that seismometer (p. 34)
KEY_P = (17.8, 8.0)  # km/s, deg: phase velocity and bearing of the P wave of that record (p. 32; printed "m/sec")
KEY_RATIOS = (.22, .31, .32, .23, .30, .27, .28, .20, .37, .41, .38, .29)  # local wave / P amplitude, Table 2 (p. 32): the events
KEY_RATIO = float(np.median(KEY_RATIOS))  # with a value (one more is listed as "<.1"); the injected amplitude is their median
KEY_SECTOR = (340.0, 30.0)  # deg, event azimuths with a detectable wave: strongest at 0-10, "up to 20 either side" (p. 35)
# Gupta, Lynnes & Wagner (1990, GRL 17(2) 183-186): residual f-k of an explosion in Nevada
GUPTA_U, GUPTA_BRG = 3.0, 310.0  # km/s, deg: "about 3 km/sec and back-azimuth of about 310" (p. 184)
GUPTA_RANGE = 15.0  # km "from the center element of the array" (p. 184)

ROT_STEP = 10.0  # deg, rotation step of the null; halved once if too few cells remain
MIN_NULLS = 19  # least number of null cells, so that p can reach P_PASS
FLOOR_STEP = 0.5  # s, the floor is rounded down to this
P_PASS, NOISE_GROUPS_MAX, POWER_MIN = 0.05, 1, 0.5  # Significance, allowed pre-P noise groups, power threshold
F_DEEP_KM = 70.0  # km, depth floor of the falsifier's events
F_SECTORS = ((40.0, 100.0), (290.0, 345.0))  # deg, back-azimuth sectors of the falsifier
F_SINCE = "2014-06-01"  # the catalog is used from here
F_MARGIN = 5.0  # s, no window ends later than this before the predicted pP
F_MIN_GROUPS, F_MIN_IN_CELL = 4, 2  # least source groups used; least groups in the target cell for "repeated"
F_ADJACENT = 20.0  # deg, what counts as an adjacent direction; here also the gap that separates clusters of back-azimuths
EIELSON_LAG = a7.LAG  # a7's floor and ceiling, kept before any run changes a7.LAG
PWIN_AFTER = (a7.PWIN_SHIFT - a7.LAG[0], a7.PWIN_LAG - a7.LAG[0])  # s: a7's P-window control, as offsets from the floor
NAME = {b: f"{b[0]:g}-{b[1]:g}" for b in a7.BANDS}
U_KEY = int(np.argmin(np.abs(1 / a7.P_SCAN - KEY_U)))  # scan index nearest Key's speed
U_GUPTA = int(np.argmin(np.abs(1 / a7.P_SCAN - GUPTA_U)))  # and the speed in Gupta's text
BOTH = [U_KEY, U_GUPTA]  # the two scan speeds the floor and the null cells must admit
FMT = dict(index=False, float_format="%.6g")


def polar(r, brg):
    """East, north (km) of the point at range r on bearing brg."""
    return r * np.array([np.sin(np.deg2rad(brg)), np.cos(np.deg2rad(brg))])


def bearing(p):
    return float(np.rad2deg(np.arctan2(p[0], p[1])) % 360)


KEY_XY, GUPTA_XY = polar(KEY_RANGE, KEY_BRG), polar(GUPTA_RANGE, GUPTA_BRG)
TARGET = (KEY_XY + GUPTA_XY) / 2  # km east, north of the centroid: the one target point


def in_sector(baz, sector=KEY_SECTOR):
    return bool((baz - sector[0]) % 360 <= (sector[1] - sector[0]) % 360)


def arc(a):
    """Smallest arc (deg) that holds every bearing in a."""
    a = np.sort(np.asarray(a, float) % 360)
    return float(360 - np.diff(np.r_[a, a[0] + 360]).max()) if len(a) else np.nan


def elements():
    """Element table from the station service: position, BHZ epoch, sensor; x, y about the centroid."""
    path = data_dir("a9") / "IM_EK.xml"
    if not path.exists():
        a7.Lazy().get_stations(network=NET, station=STA, channel=CHA, level="channel", filename=str(path))
    rows = []
    for sta in read_inventory(str(path))[0]:
        assert len(sta.channels) == 1, sta.code  # one BHZ epoch per site; a second one would need the epoch test below widened
        ch = sta.channels[0]
        rows.append(dict(station=sta.code, lat=sta.latitude, lon=sta.longitude, elev_m=sta.elevation, start=str(ch.start_date),
                         end="open" if ch.end_date is None else str(ch.end_date), channel=ch.code, sample_rate=ch.sample_rate,
                         sensor=ch.sensor.description if ch.sensor else ""))
    el = pd.DataFrame(rows).sort_values("station").reset_index(drop=True)
    x, y = a6.local_xy(el.lat, el.lon)
    return el.assign(x_km=x, y_km=y)


def arm_axes(el):
    """Per arm: axis bearing (deg, 0-180) of the least-squares line through its elements, and a point on it."""
    out = {}
    for name, pat in ARMS.items():
        p = el.loc[el.station.str.fullmatch(pat), ["x_km", "y_km"]].to_numpy()
        out[name] = (bearing(np.linalg.svd(p - p.mean(axis=0))[2][0]) % 180, p.mean(axis=0))
    return out


def catalog(el, min_elem, deep=a7.DEEP_KM, pp_min=a7.PP_MIN_S, sectors=None, since=None, skip=()):
    """Sources by the Eielson rule (a7 step 1, first half) for this array: rows for the tests and the N_CAND candidates.
    The falsifier changes the depth floor, drops the least pP delay, and adds back-azimuth sectors, a first
    date and the events to leave out; it takes no test."""
    la0, lo0 = el.lat.mean(), el.lon.mean()
    t0, t1 = [T(q) for q in el.start], [T("2100-01-01") if q == "open" else T(q) for q in el.end]
    taup = TauPyModel("iasp91")
    tele = a6.usgs(data_dir("a6"), "tele_m6")  # a6's cached global list; the distance is recomputed for this array
    tele["deg"] = [locations2degrees(a, o, la0, lo0) for a, o in zip(tele.latitude, tele.longitude)]
    cand = tele[(tele.deg >= a7.DEG[0]) & (tele.deg <= a7.DEG[1]) & (tele.depth >= deep)
                & (tele.mag >= a7.MAG[0]) & (tele.mag <= a7.MAG[1])]
    if sectors:
        baz = a6.GEOD.inv(np.full(len(cand), lo0), np.full(len(cand), la0), cand.longitude.to_numpy(), cand.latitude.to_numpy())[0] % 360
        cand = cand[np.array([any(in_sector(b, q) for q in sectors) for b in baz]) & (cand.t >= since).to_numpy()]
    dt = (tele.t.to_numpy()[None, :] - cand.t.to_numpy()[:, None]) / np.timedelta64(1, "m")
    alone = ((dt >= -a7.ISOLATED_MIN[0]) & (dt <= a7.ISOLATED_MIN[1])).sum(axis=1) == 1
    n_rule = dict(deep_mag_dist=len(cand), isolated=int(alone.sum()))
    cand = cand[alone].sort_values(["mag", "t"], ascending=[False, True], kind="stable")
    # a6's box, from the first test on. The type is filtered here: the service's eventtype filter returns only the 2016-2017 tests
    dprk = a6.usgs(data_dir("a9"), "korea_box", starttime="2006-01-01", endtime=a6.TELE_TO, minlatitude=40, maxlatitude=43,
                   minlongitude=127, maxlongitude=131)
    dprk = dprk[dprk.type == "nuclear explosion"]
    src = []
    for kind, cat in (("teleseism", cand),) if sectors else (("dprk", dprk), ("teleseism", cand)):
        for _, q in cat.iterrows():
            if kind == "teleseism" and sum(s["kind"] == "teleseism" for s in src) >= a7.N_CAND:
                break
            o, deg = T(q.time), locations2degrees(q.latitude, q.longitude, la0, lo0)
            depth = 0.0 if kind == "dprk" else float(q.depth)
            arr = {}
            for x in reversed(taup.get_travel_times(depth, deg, ["P", "pP", "PcP"])):
                arr[x.name] = x
            if "P" not in arr or (kind == "teleseism" and ("pP" not in arr or (pp_min and arr["pP"].time - arr["P"].time < pp_min))):
                continue
            if f"tele_{o.strftime('%Y%m%dT%H%M%S')}" in skip:
                continue
            tp = o + arr["P"].time
            n_epoch = sum(a <= tp - a6.PRE and tp + a6.POST <= b for a, b in zip(t0, t1))  # metadata only
            if kind == "teleseism" and n_epoch < min_elem:
                continue
            baz = a6.GEOD.inv(lo0, la0, q.longitude, q.latitude)[0] % 360
            src.append(dict(event=f"{kind}_{o.date}" if kind == "dprk" else f"tele_{o.strftime('%Y%m%dT%H%M%S')}", kind=kind,
                            time=str(o), lat=q.latitude, lon=q.longitude, depth_km=depth, mag=q.mag, dist_deg=deg, baz_deg=baz,
                            slow_skm=arr["P"].ray_param_sec_degree / 111.195, tp=tp,
                            pp_minus_p_s=arr["pP"].time - arr["P"].time if "pP" in arr else np.nan,
                            pcp_minus_p_s=arr["PcP"].time - arr["P"].time if "PcP" in arr else np.nan,
                            n_elem_epoch=n_epoch, candidate=n_epoch >= min_elem, in_key_sector=in_sector(baz)))
    return src, n_rule


def lags(point, src):
    """[event, scan speed]: lag after P (s) of the window at `point`, lead + range x slowness, as a7.valid has it."""
    lead = np.array([point @ a7.slow_vec(s["slow_skm"], s["baz_deg"]) for s in src])
    return lead[:, None] + np.hypot(*point) * a7.P_SCAN[None, :]


def decisive_floor(src):
    """The largest lag floor, rounded down to FLOOR_STEP, that admits both speeds at the target for every event."""
    return float(np.floor((lags(TARGET, src)[:, BOTH].min() - a7.WIN / 2) / FLOOR_STEP) * FLOOR_STEP)


def admitted(point, src, floor):
    """Filter and report: does the floor admit `point` at both speeds for every event; and the earliest admitted lag
    there over the whole scan, smallest and median over events."""
    t = lags(point, src)
    top = np.array([s.get("upper", EIELSON_LAG[1]) for s in src])[:, None]  # a7's ceiling, or the event's window cut
    ok = (t >= floor + a7.WIN / 2) & (t <= top - a7.WIN / 2)
    first = np.where(ok, t, np.inf).min(axis=1)
    return dict(admitted=bool(ok[:, BOTH].all()), earliest_lag_min_s=float(first.min()), earliest_lag_median_s=float(np.median(first)))


def nearest_range(floor, src):
    """[event, bearing (1 deg), scan speed]: nearest range (km) at which a7.valid keeps a window under `floor`."""
    s, baz = (np.array([q[k] for q in src])[:, None, None] for k in ("slow_skm", "baz_deg"))
    return (floor + a7.WIN / 2) / (a7.P_SCAN[None, None, :] - s * np.cos(np.deg2rad(np.arange(360.0)[None, :, None] - baz)))


def in_cell(center, pts, cell):
    """a7.same_cell for one cell of fixed half-extents: is each point inside the ellipse centered on `center`?"""
    return a7.same_cell(center, pts, dict(range_km=[1.0], radial=[cell["radial"]], cross=[cell["cross"]]))


def target_cell(xy, band, floor):
    """Cell: half-power half-extents (km) of a noise-free synthetic at the target, and its imaging error (recorded)."""
    nt, t_ref = int(100 * a7.FS), 20.0
    zero = np.zeros(len(a7.CELLS))
    syn = a7.scattered(a7.pulse(*band, nt, int(t_ref * a7.FS)), xy, TARGET, a7.P_SCAN[U_KEY], 0.0)
    mx, _ = a7.cell_max(a7.image(syn, t_ref, a7.times(xy, zero), power=True), a7.valid(zero, (floor, EIELSON_LAG[1])))
    gp = int(mx.argmax())
    radial, cross = a7.extent(mx, gp)
    return dict(radial=radial, cross=cross, err=float(np.hypot(*(a7.CELLS[gp] - TARGET))))


def null_cells(cell, floor, src, axes, step):
    """Null: the target cell at every rotation of `step`, with what leaves a cell out and which cells are marked."""
    r, b0, n = float(np.hypot(*TARGET)), bearing(TARGET), int(round(360 / step))
    in_t = in_cell(TARGET, a7.CELLS, cell)
    marks = {f"mirror_arm_{k}": (2 * ax - b0) % 360 for k, (ax, _) in axes.items()} | {"antipode": (b0 + 180) % 360}
    near = {int(round(((b - b0) % 360) / step)) % n: (k, b) for k, b in marks.items()}
    rows = []
    for k in range(1, n):
        c = polar(r, b0 + k * step)
        overlap = bool((in_t & in_cell(c, a7.CELLS, cell)).any())  # a peak is a grid cell, so shared grid cells are the overlap
        adm = admitted(c, src, floor)
        mark, mark_brg = near.get(k, ("", np.nan))
        rows.append(dict(step_deg=step, rot_deg=k * step, bearing_deg=bearing(c), x_km=c[0], y_km=c[1], overlaps_target=overlap,
                         **adm, kept=adm["admitted"] and not overlap, mark=mark, mark_bearing_deg=mark_brg))
    return pd.DataFrame(rows)


def geometry(src, xy, axes):
    """Floor, target cell and null cells per band for one source list. The null table holds both rotation
    steps; `used` marks the step the control takes: ROT_STEP, or half of it if ROT_STEP keeps fewer than MIN_NULLS."""
    floor = decisive_floor(src)
    assert admitted(TARGET, src, floor)["admitted"]  # The target passes the filter of the null cells
    cells, nulls = {}, {}
    for b in a7.BANDS:
        cells[b] = target_cell(xy, b, floor)
        tabs = [null_cells(cells[b], floor, src, axes, step) for step in (ROT_STEP, ROT_STEP / 2)]
        first = tabs[0].kept.sum() >= MIN_NULLS
        nulls[b] = pd.concat([tabs[0].assign(used=first), tabs[1].assign(used=not first)], ignore_index=True)
    return floor, cells, nulls


# --- phase 2: everything below reads waveforms (real, or the self-check's synthetic) -------------------------------------

def fetch(c, s, stations):
    """Raw BHZ window of one source, cached; None if the server has none (a7.fetch, for this channel)."""
    cache = data_dir("a9")
    miss = cache / f"{s['event']}.nodata"
    if miss.exists():
        return None
    st = a6.waveforms(c, cache, s["event"], NET, stations, CHA, s["tp"] - a6.PRE, s["tp"] + a6.POST)
    if st is None:
        miss.touch()
    return st


def prepare(s, st, stations, min_elem):
    """a7 step 1, second half, for one source: outcome, P reference time and the band-passed element matrices."""
    s["n_data"], s["outcome"] = 0, "no_data"
    if st is None:
        return
    for tr in st:  # forced by the array: BHZ brought to a7's rate, so a7's windows and filters apply unchanged
        k = tr.stats.sampling_rate / a7.FS
        assert k == round(k) and k >= 1, tr.id
        if k > 1:
            tr.decimate(int(round(k)))
    snr = {r.get("band_hz"): r for r in a6.snr_rows("EKA", s["event"], st, s["tp"], len(stations), {})}
    s["n_data"] = int(snr[next(iter(snr))]["n_data"])
    if s["n_data"] < min_elem:
        s["outcome"] = "few_elements"
        return
    lo = snr[NAME[a7.BANDS[0]]]
    s["p_snr"], s["t_ref"] = lo["p_snr"], a6.PRE + lo["t_pk_minus_pred_s"]
    s["outcome"] = "used" if s["kind"] == "dprk" or lo["p_snr"] >= a7.SNR_MIN else "low_snr"
    if s["outcome"] == "used":
        nt = int((a6.PRE + a6.POST) * a7.FS)
        s["X"] = {b: a7.matrix(a6.band(st, *b), stations, s["tp"] - a6.PRE, nt) for b in a7.BANDS}
        s["refine"] = {b: bool(snr[NAME[b]]["p_snr"] >= a7.REFINE_SNR) for b in a7.BANDS}


def select(src):
    """The N_TELE teleseisms of highest signal-to-noise plus the tests; source groups as in a7."""
    passed = sorted((s for s in src if s["kind"] == "teleseism" and s["outcome"] == "used"), key=lambda s: -s["p_snr"])
    for s in passed[a7.N_TELE:]:
        s["outcome"] = "passed_not_among_best"
    return groups([s for s in src if s["outcome"] == "used"])


def groups(used):
    """a7's source groups, in list order: events closer than GROUP_DEG share one."""
    for i, s in enumerate(used):
        s["group"] = next((u["group"] for u in used[:i] if locations2degrees(s["lat"], s["lon"], u["lat"], u["lon"]) < a7.GROUP_DEG), i)
    return used


def run(used, xy, band, floor):
    """a7.event_maps of every used event in one band under one lag floor. a7 reads its lag constants as globals."""
    keep = a7.LAG, a7.PWIN_SHIFT, a7.PWIN_LAG
    a7.PWIN_SHIFT, a7.PWIN_LAG = floor + PWIN_AFTER[0], floor + PWIN_AFTER[1]
    rng, out = np.random.default_rng(a7.SEED), []
    try:
        for s in used:
            a7.LAG = (floor, s.get("upper", EIELSON_LAG[1]))  # the ceiling masks the coda map and every null of the event alike
            out.append(a7.event_maps(s["X"][band][0], xy[s["X"][band][1]], a7.slow_vec(s["slow_skm"], s["baz_deg"]), s["t_ref"],
                                     s["refine"][band], rng))
        return out
    finally:
        a7.LAG, a7.PWIN_SHIFT, a7.PWIN_LAG = keep


def strongest(S, ok, thr):
    """a7 rule "strongest_peak": cell index of the largest peak above the threshold; -1 if the map has none."""
    pk = a7.peaks(S, ok, thr)
    return max(pk, key=lambda q: q[1])[0] if pk else -1


def groups_in(centers, g, grp, cell):
    """Per center: number of source groups with a peak (cell index g; -1 = none) inside the cell centered there."""
    has = g >= 0
    return np.array([len(set(grp[has][in_cell(c, a7.CELLS[g[has]], cell)])) for c in centers])


def pwin_peak(S, ok):
    """Cell index of the peak of a7's P-window control (a7 step 12: the largest value, no threshold)."""
    return int(a7.cell_max(S, ok)[0].argmax())


def tops(maps):
    """Per event: cell index of the strongest coda peak, of the strongest peak of each pre-P noise map, and of the P-window control."""
    return (np.array([strongest(m["S"], m["ok"], m["thr"]) for m in maps]),
            [np.array([strongest(m["nulls"][k], m["ok"], m["thr"]) for m in maps]) for k in range(len(a7.NOISE_REF))],
            np.array([pwin_peak(m["pwin"], m["ok_pwin"]) for m in maps]))


def event_row(s, m, cell, gi, noise_g, wi, keys=(), **head):
    """One event's row: where its strongest peak, its noise peaks and its P-window-control peak fall."""
    row = {k: s[k] for k in ("event", "kind", "group", "dist_deg", "baz_deg", "in_key_sector", "outcome", "p_snr", *keys)} | head | dict(
        n_elem_used=len(m["xy"]), p_window_residual_frac=m["resid_frac"], null_threshold=m["thr"],
        in_target_cell=bool(gi >= 0 and in_cell(TARGET, a7.CELLS[gi], cell)[0]),
        noise_peak_in_target_cell=any(q >= 0 and in_cell(TARGET, a7.CELLS[q], cell)[0] for q in noise_g),
        pwin_peak_in_target_cell=bool(in_cell(TARGET, a7.CELLS[wi], cell)[0]),
        pwin_peak_range_km=float(a7.RANGE[wi]), pwin_peak_bearing_deg=bearing(a7.CELLS[wi]))
    if gi >= 0:
        iu = int(a7.cell_max(m["S"], m["ok"])[1][gi])
        row |= dict(peak_x_km=a7.CELLS[gi, 0], peak_y_km=a7.CELLS[gi, 1], peak_range_km=float(a7.RANGE[gi]),
                    peak_bearing_deg=bearing(a7.CELLS[gi]), peak_u_kms=float(1 / a7.P_SCAN[iu]),
                    peak_lag_s=float(m["lag"][gi, iu]), peak_semblance=float(m["S"][gi, iu]),
                    **{f"miss_{k}_km": float(np.hypot(*(a7.CELLS[gi] - q))) for k, q in (("target", TARGET), ("key", KEY_XY), ("gupta", GUPTA_XY))})
    return row


def decide(centers, i, g, gn, grp, cell):
    """Test of the cell at centers[i] against all of `centers`: counts, rank p, noise groups, pass."""
    n = groups_in(centers, g, grp, cell)
    p = float((n >= n[i]).mean())
    noise = max(int(groups_in(centers[i:i + 1], q, grp, cell)[0]) for q in gn)  # as a7's noise_groups_same_cell
    return n, p, noise, bool(p <= P_PASS and noise <= NOISE_GROUPS_MAX)


def power(used, maps, kept, cell, g, gn, grp):
    """Power: Key's arrival, from each event's own P wavelet, injected on the bearing of one kept null cell into the
    events of Key's sector. Does that cell then pass the whole test? One row per cell. Its null is the kept cells that
    share no grid cell with it, as the target's null leaves out the cells that overlap the target. The target cell is
    not among them (power before the target cell is read)."""
    centers, rows = kept[["x_km", "y_km"]].to_numpy(), []
    member = np.array([in_cell(c, a7.CELLS, cell) for c in centers])
    for i, q in enumerate(kept.itertuples()):
        apart = ~(member & member[i]).any(axis=1)
        gi, pt = g.copy(), polar(KEY_RANGE, q.bearing_deg)
        for j, (s, ev) in enumerate(zip(used, maps)):
            if not s["in_key_sector"]:
                continue
            syn = a7.scattered(ev["wave"], ev["xy"], pt, 1 / KEY_U, float(pt @ a7.slow_vec(s["slow_skm"], s["baz_deg"])))
            # the residual is linear in the record for fixed delays and gains (a7 step 3), so the synthetic's residual is added
            R, tt = ev["R"] + KEY_RATIO * a7.residual(syn, ev["d"], ev["a"])[0], a7.times(ev["xy"], ev["lead"])
            gi[j] = strongest(a7.image(R, ev["t_ref"], tt), ev["ok"], ev["thr"])
        n, p, noise, ok = decide(np.vstack([centers[i], centers[apart]]), 0, gi, gn, grp, cell)
        rows.append(dict(rot_deg=q.rot_deg, n_cells=len(n), n_groups=int(n[0]), p=p, noise_groups=noise, passes=ok))
    return pd.DataFrame(rows)


def control(used, xy, floor, cells, nulls, bands=a7.BANDS, save=None):
    """On prepared sources: power, then the count in the target cell against the rotated cells, under both floors.
    With `save`, the power table is written there as soon as it exists, before the target cell is read."""
    grp, baz = np.array([s["group"] for s in used]), np.array([s["baz_deg"] for s in used])
    key = np.array([s["in_key_sector"] for s in used])
    ev_rows, cell_rows, res, pw = [], [], [], pd.DataFrame()

    def put(name, value, of=None, ok=None, note=""):
        res.append(dict(check=name, value=value, of=of, ok=ok, note=note))

    put("events_used", len(used), None, None,
        f"source groups: {len(set(grp))}; events in Key's sector: {int(key.sum())}, their groups: {len(set(grp[key]))}")
    for b in bands:
        nl = nulls[b][nulls[b].used]
        kept = nl[nl.kept]
        put(f"{NAME[b]}_null_cells", len(kept), MIN_NULLS, bool(len(kept) >= MIN_NULLS), f"rotation step {nl.step_deg.iloc[0]:g} deg")
        if len(kept) < MIN_NULLS:  # Declared before the target cell is read
            if b == a7.BANDS[0]:
                put("reading", "unable_to_decide", None, False, "too few null cells after halving the rotation step")
            continue
        centers = np.vstack([TARGET, kept[["x_km", "y_km"]].to_numpy()])
        early = [admitted(TARGET, used, floor), *kept.to_dict("records")]
        for run_name, fl in (("decisive", floor), ("eielson", EIELSON_LAG[0])):
            maps, tag = run(used, xy, b, fl), f"{NAME[b]}_{run_name}"
            decides = b == a7.BANDS[0] and run_name == "decisive"
            g, gn, gw = tops(maps)
            if decides:  # Power is measured before the target cell is read
                pw = power(used, maps, kept, cells[b], g, gn, grp)
                if save:
                    pw.to_csv(save, **FMT)
                put("power", float(pw.passes.mean()), POWER_MIN, bool(pw.passes.mean() >= POWER_MIN),
                    f"share of {len(pw)} rotated cells that pass the whole test when Key's arrival is injected there; fail on count or "
                    f"rank {int((pw.p > P_PASS).sum())}, on pre-P noise {int((pw.noise_groups > NOISE_GROUPS_MAX).sum())}; "
                    f"largest 1/cells {1 / pw.n_cells.min():.3f}")
            n, p, noise, ok = decide(centers, 0, g, gn, grp, cells[b])
            nw = groups_in(centers, gw, grp, cells[b])  # the P-window control, reported only
            hit = (g >= 0) & in_cell(TARGET, a7.CELLS[g], cells[b])
            miss = {k: np.hypot(*(a7.CELLS[g] - q).T) for k, q in (("key", KEY_XY), ("gupta", GUPTA_XY))}
            ev_rows += [event_row(s, m, cells[b], g[i], [q[i] for q in gn], gw[i], band_hz=NAME[b], run=run_name, floor_s=fl)
                        for i, (s, m) in enumerate(zip(used, maps))]
            cell_rows += [dict(band_hz=NAME[b], run=run_name, rot_deg=r, n_groups=int(k), pwin_groups=int(w), mark=mk)
                          | ({q: e[q] for q in ("earliest_lag_min_s", "earliest_lag_median_s")} if run_name == "decisive" else {})
                          for r, k, w, mk, e in zip([0.0, *kept.rot_deg], n, nw, ["target", *kept.mark], early)]
            put(f"{tag}_groups_in_target_cell", int(n[0]), len(set(grp)), None, "source groups whose strongest peak falls in the cell")
            put(f"{tag}_p", p, P_PASS, bool(p <= P_PASS), f"share of {len(n)} cells (target included) with a count >= the target's")
            put(f"{tag}_noise_groups_in_target_cell", noise, NOISE_GROUPS_MAX, bool(noise <= NOISE_GROUPS_MAX),
                "most groups, over the pre-P noise maps, whose strongest noise peak falls in the cell")
            put(f"{tag}_pwin_groups_in_target_cell", int(nw[0]), int(nw[1:].max()), None,
                "reported, not deciding: groups whose P-window-control peak falls in the cell; of = largest over the rotated cells")
            put(f"{tag}_pwin_rank", float((nw >= nw[0]).mean()), None, None, "share of cells (target included) with a control count >= the target's")
            put(f"{tag}_baz_spread_deg", arc(baz[hit]), None, None, "smallest arc holding the back-azimuths of the events in the cell: "
                + " ".join(f"{q:.0f}" for q in np.sort(baz[hit])))
            put(f"{tag}_in_cell_events_in_key_sector", int(key[hit].sum()), int(hit.sum()), None,
                "rests on one sector" if hit.any() and key[hit].all() else "")
            for k in ("key", "gupta"):
                put(f"{tag}_miss_{k}_km_median", float(np.median(miss[k][hit])) if hit.any() else np.nan, None, None,
                    "events in the cell: strongest peak to the published position")
            put(f"{tag}_peak_range_km_median", float(np.median(a7.RANGE[g[g >= 0]])) if (g >= 0).any() else np.nan,
                float(np.hypot(*TARGET)), None, "strongest peaks of all events; of = range of the target")
            if decides:
                reading = "pass" if ok else "not_separable" if p <= P_PASS else "fail" if pw.passes.mean() >= POWER_MIN else "uninformative"
                put("reading", reading, None, ok, "The lower band under the decisive floor decides, the rest is reported")
    return pd.DataFrame(ev_rows), pd.DataFrame(cell_rows), pw, pd.DataFrame(res)


def finish(src, xy, axes, bands=a7.BANDS, save=None):
    """Phase 2 once the records are prepared: selection, the floor and cells on the events used, the control.
    Returns the four phase 2 tables by name."""
    used = select(src)
    floor, cells, nulls = geometry(used, xy, axes)
    ev, cl, pw, res = control(used, xy, floor, cells, nulls, bands, save)
    rest = pd.DataFrame([{k: s.get(k) for k in ("event", "kind", "dist_deg", "baz_deg", "in_key_sector", "outcome", "n_data", "p_snr")}
                         for s in src if s["outcome"] != "used"])
    res = pd.concat([pd.DataFrame([dict(check="decisive_floor_s_events_used", value=floor, of=EIELSON_LAG[0],
                                        note=f"recomputed on the {len(used)} events used")]), res], ignore_index=True)
    return dict(events=pd.concat([ev, rest], ignore_index=True), cells=cl, power=pw, result=res)


def phase2(src, el, xy, axes, min_elem):
    """Fetch and prepare every candidate, run the control, write its four tables."""
    c, stations = a7.Lazy(), list(el.station)
    for s in src:
        prepare(s, fetch(c, s, stations), stations, min_elem)
    out = finish(src, xy, axes, save=HERE / "a9_eka_power.csv")
    for name, df in out.items():
        df.to_csv(HERE / f"a9_eka_{name}.csv", **FMT)
    print("\n== result")
    print(out["result"].to_string(index=False))


def synthetic(xy, stations, events, ratio, plane=None):
    """Prepared synthetic sources ('s synthetic): per event a plane P with statics and site gains, noise, Key's arrival
    from the target at `ratio` of P, and optionally a plane arrival (delay s, slowness, back-azimuth, amplitude)."""
    n, nt = len(xy), int((a6.PRE + a6.POST) * a7.FS)
    base = a7.pulse(a7.BANDS[0][0], a7.BANDS[-1][1], nt, int(a6.PRE * a7.FS))
    rng, out = np.random.default_rng(1), []
    for i, q in enumerate(events):
        s_vec, tp = a7.slow_vec(q["slow_skm"], q["baz_deg"]), T(2020, 1, 1) + 3600.0 * i
        X = a7.shift(np.tile(base, (n, 1)), xy @ s_vec + rng.normal(0, 0.03, n)) * rng.uniform(0.8, 1.25, n)[:, None] \
            + ratio * a7.scattered(base, xy, TARGET, 1 / KEY_U, float(TARGET @ s_vec)) + rng.normal(0, 0.02, (n, nt))
        if plane:
            X = X + plane[3] * a7.shift(np.tile(base, (n, 1)), plane[0] + xy @ a7.slow_vec(plane[1], plane[2]))
        st = Stream([Trace(x, dict(network=NET, station=k, channel=CHA, sampling_rate=2 * a7.FS, starttime=tp - a6.PRE))
                     for k, x in zip(stations, resample(X, 2 * nt, axis=1))])
        s = dict(q) | dict(tp=tp, in_key_sector=in_sector(q["baz_deg"]))
        prepare(s, st, stations, n)
        out.append(s)
    return out


def cut(s):
    """Window cut as a per-event upper lag. A window centered T after the picked P peak ends, in absolute time, at
    tp + (t_ref - PRE) + T + WIN/2; that may not pass tp + (pP - P) - F_MARGIN. a7.valid keeps T <= upper - WIN/2, so
    upper = (pP - P) - F_MARGIN - (t_ref - PRE), never above a7's ceiling. Before a record is read the pick is the predicted P."""
    s["upper"] = min(EIELSON_LAG[1], s["pp_minus_p_s"] - F_MARGIN - (s.get("t_ref", a6.PRE) - a6.PRE))


def clusters(src):
    """Cluster of back-azimuths per source: the sorted back-azimuths, split where a gap exceeds F_ADJACENT."""
    order = sorted(src, key=lambda s: s["baz_deg"])
    for i, s in enumerate(order):
        s["cluster"] = 0 if i == 0 else order[i - 1]["cluster"] + int(s["baz_deg"] - order[i - 1]["baz_deg"] > F_ADJACENT)


def reading_d110(groups_used, cells_kept, count, p, noise, adjacent):
    """The falsifier's reading from the numbers of the decisive test on the new events alone."""
    if cells_kept < MIN_NULLS:
        return "unable to decide"
    if groups_used < F_MIN_GROUPS:
        return "uninformative"
    if p <= P_PASS and noise <= NOISE_GROUPS_MAX and count >= F_MIN_IN_CELL:
        return "repeated from adjacent directions" if adjacent else "repeated"
    return "not repeated"


def falsifier():
    """The decisive test of the first run, frozen, on new events from other back-azimuths.
    Order: catalog, floor and null cells, the stops that fetch nothing, the synthetic flags, and only then the records."""
    el, xy, axes, min_elem, _ = setup()
    b, stations = a7.BANDS[0], list(el.station)
    first = pd.read_csv(HERE / "a9_eka_events.csv")
    baz1 = first[first.outcome == "used"].baz_deg.unique()  # back-azimuths used in the first run
    floor1 = float(pd.read_csv(HERE / "a9_eka_result.csv").set_index("check").value["decisive_floor_s_events_used"])
    rot1 = pd.read_csv(HERE / "a9_eka_cells.csv").query(f"band_hz == '{NAME[b]}' and run == 'decisive' and rot_deg > 0").rot_deg
    cell = target_cell(xy, b, floor1)  # the first run's target cell; its rotation list is rot1
    res = []

    def put(name, value, of=None, ok=None, note=""):
        res.append(dict(check=name, value=float(f"{value:.6g}") if isinstance(value, float) else value, of=of, ok=ok, note=note))

    def done(reading=None, note=""):
        if reading:
            put("reading", reading, None, None, note)
        pd.DataFrame(res).to_csv(HERE / "a9_eka_falsifier_result.csv", **FMT)
        pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400, "display.max_colwidth", 160)
        print("\n== falsifier result")
        print(pd.DataFrame(res).to_string(index=False))

    def near(baz):  # Within F_ADJACENT of a back-azimuth of the first run, or inside Key's sector
        return bool(in_sector(baz) or np.abs((baz - baz1 + 180) % 360 - 180).min() <= F_ADJACENT)

    def frame(evs, stage):  # floor by the fixed rule on these events; the fixed filter re-applied to the first run's rotation list
        floor = decisive_floor(evs)
        nl = null_cells(cell, floor, evs, axes, ROT_STEP)
        nl = nl[nl.rot_deg.isin(rot1)].reset_index(drop=True)
        n = int(nl.kept.sum())
        put(f"{stage}_floor_s", floor, floor1, None, "The fixed rule on these events; of = the first run's floor")
        put(f"{stage}_null_cells_kept", n, len(nl), bool(n >= MIN_NULLS), f"admitted at both speeds for every event, under each event's window cut; ok = at least {MIN_NULLS}")
        put(f"{stage}_smallest_p", 1 / (n + 1), P_PASS, bool(1 / (n + 1) <= P_PASS), "1 / (kept cells + target)")
        bad = [s["event"] for s in evs if not admitted(TARGET, [s], floor)["admitted"]]
        if bad:  # The target passes the same filter for every event. Where a window cut breaks that, no rule says what to do
            put("obstacle", len(bad), len(evs), False, f"{stage}: the window cut leaves no window that admits the target cell at both speeds for "
                + " ".join(bad) + "; no event is dropped for the cells and the target must be admitted for every event: stopped, target cell not read")
            pd.DataFrame(res).to_csv(HERE / "a9_eka_falsifier_result.csv", **FMT)
        return floor, nl, bad

    # --- 1-4. catalog only
    src, n_rule = catalog(el, min_elem, deep=F_DEEP_KM, pp_min=None, sectors=F_SECTORS, since=pd.Timestamp(F_SINCE, tz="UTC"),
                            skip=set(pd.read_csv(HERE / "a9_eka_candidates.csv").event))
    for s in src:
        cut(s)
    cand = groups(src)
    clusters(cand)
    for s in cand:
        s["adjacent"], s["last_window_lag_s"] = near(s["baz_deg"]), s["upper"] - a7.WIN / 2
    pd.DataFrame(cand).drop(columns=["tp", "candidate"]).to_csv(HERE / "a9_eka_falsifier_candidates.csv", **FMT)
    n_groups = len({s["group"] for s in cand})
    put("candidates", len(cand), n_rule["deep_mag_dist"], None,
        f"depth >= {F_DEEP_KM:g} km, sectors {F_SECTORS}, from {F_SINCE}, isolated, >= {min_elem} BHZ epochs, none of the first run's; "
        f"of = before the isolation rule ({n_rule['isolated']} after it)")
    put("candidate_source_groups", n_groups, F_MIN_GROUPS, bool(n_groups >= F_MIN_GROUPS), "a7 rule; ok = at least `of`")
    for k in sorted({s["cluster"] for s in cand}):
        q = [s for s in cand if s["cluster"] == k]
        put(f"candidate_cluster_{k}", len(q), None, None, f"back-azimuth {min(s['baz_deg'] for s in q):.0f} to {max(s['baz_deg'] for s in q):.0f} deg; "
            f"groups {len({s['group'] for s in q})}; adjacent to a first-run direction or in Key's sector: {sum(s['adjacent'] for s in q)}")
    floor, nl, bad = frame(cand, "candidates")
    if bad:
        raise RuntimeError(res[-1]["note"])
    if nl.kept.sum() < MIN_NULLS:
        return done("unable to decide", "too few null cells for the candidates; declared before any record is fetched")
    if n_groups < F_MIN_GROUPS:
        return done("uninformative", "the candidates cannot reach the least number of source groups; no record is fetched")

    # --- 5. The fixed synthetics at the candidates' back-azimuths and slownesses, one event per source group, with the window cut
    lead = [next(s for s in cand if s["group"] == k) for k in sorted({s["group"] for s in cand})]
    syn = {}
    for name, ratio in (("p_alone", 0.0), ("p_with_arrival", KEY_RATIO)):
        ev = synthetic(xy, stations, [{k: s[k] for k in ("event", "kind", "lat", "lon", "dist_deg", "baz_deg", "slow_skm", "pp_minus_p_s",
                                                         "group", "cluster")} for s in lead], ratio)
        assert all(s["outcome"] == "used" for s in ev), [s["outcome"] for s in ev]
        for s in ev:
            cut(s)
        maps = run(ev, xy, b, floor)
        g, _, gw = tops(maps)
        syn[name] = [event_row(s, m, cell, gi, [], wi, keys=("cluster", "upper")) for s, m, gi, wi in zip(ev, maps, g, gw)]
    flags = pd.DataFrame([dict(group=a["group"], cluster=a["cluster"], lead_event=a["event"], baz_deg=a["baz_deg"], upper_lag_s=a["upper"],
                               flag_leftover_p_in_target_cell=a["in_target_cell"], p_alone_peak_bearing_deg=a.get("peak_bearing_deg"),
                               p_alone_peak_range_km=a.get("peak_range_km"), p_alone_pwin_peak_in_target_cell=a["pwin_peak_in_target_cell"],
                               with_arrival_peak_in_target_cell=w["in_target_cell"], with_arrival_peak_bearing_deg=w.get("peak_bearing_deg"),
                               with_arrival_peak_range_km=w.get("peak_range_km")) for a, w in zip(syn["p_alone"], syn["p_with_arrival"])])
    flags.to_csv(HERE / "a9_eka_falsifier_synthetic.csv", **FMT)  # on disk before any fetch
    flagged = set(flags.group[flags.flag_leftover_p_in_target_cell])
    put("synthetic_groups_flagged", len(flagged), len(flags), None, "P alone: strongest coda peak in the target cell; lead events: "
        + " ".join(flags.lead_event[flags.flag_leftover_p_in_target_cell]))
    put("synthetic_pwin_peak_in_target_p_alone", int(flags.p_alone_pwin_peak_in_target_cell.sum()), len(flags), None, "P alone: P-window-control peak in the target cell")
    put("synthetic_arrival_found", int(flags.with_arrival_peak_in_target_cell.sum()), len(flags), None, "P with Key's arrival: strongest coda peak in the target cell")
    for s in cand:
        s["synthetic_flag"] = s["group"] in flagged

    # --- 6. records
    c = a7.Lazy()
    for s in cand:
        prepare(s, fetch(c, s, stations), stations, min_elem)
        s["pick_minus_pred_s"] = s.get("t_ref", np.nan) - a6.PRE
    for s in select(cand):
        cut(s)  # now from the picked P peak
        need = float(lags(TARGET, [s])[0, BOTH].max())
        if need > s["upper"] - a7.WIN / 2:  # No admitted window at the target under its own cut: not a usable event
            s["outcome"] = f"no_window_at_target (last window center {s['upper'] - a7.WIN / 2:.2f} s after the picked P; the target needs {need:.2f} s)"
    used = groups([s for s in cand if s["outcome"] == "used"])  # as the first run grouped the events it used
    grp, baz, clu = (np.array([s[k] for s in used]) for k in ("group", "baz_deg", "cluster"))
    extra = ("cluster", "depth_km", "slow_skm", "pp_minus_p_s", "pcp_minus_p_s", "pick_minus_pred_s", "upper", "adjacent", "synthetic_flag")
    rest = [{k: s.get(k) for k in ("event", "kind", "dist_deg", "baz_deg", "in_key_sector", "outcome", "n_data", "p_snr", *extra)}
            for s in cand if s["outcome"] != "used"]
    put("events_used", len(used), len(cand), None, "dropped: " + ", ".join(f"{k} {v}" for k, v in pd.Series(
        [s["outcome"].split(" ")[0] for s in cand if s["outcome"] != "used"], dtype=object).value_counts().items()))
    put("source_groups_used", len(set(grp)), F_MIN_GROUPS, bool(len(set(grp)) >= F_MIN_GROUPS), f"clusters of back-azimuths used: {sorted(set(clu.tolist()))}")
    if not used:
        pd.DataFrame(rest).to_csv(HERE / "a9_eka_falsifier_events.csv", **FMT)
        return done("uninformative", "no event used")
    floor, nl, bad = frame(used, "used")
    assert not bad, bad  # Took those events out
    nl.to_csv(HERE / "a9_eka_falsifier_nulls.csv", **FMT)  # the null-cell list, on disk before the target cell is read
    kept = nl[nl.kept]
    early = reading_d110(len(set(grp)), len(kept), 0, 1.0, 0, False)
    d = "" if early == "not repeated" else "descriptive_"
    if d:  # Fixed by the number of groups or cells alone, and on disk before the target cell is read
        done(early, "fixed before the target cell is read; the rows below are description and do not change it")
        if not len(kept):  # no rotated cell to count in
            return pd.DataFrame(rest).to_csv(HERE / "a9_eka_falsifier_events.csv", **FMT)
    maps = run(used, xy, b, floor)
    g, gn, gw = tops(maps)
    centers = np.vstack([TARGET, kept[["x_km", "y_km"]].to_numpy()])
    n, p, noise, _ = decide(centers, 0, g, gn, grp, cell)
    nw = groups_in(centers, gw, grp, cell)  # the P-window control, reported only
    hit = (g >= 0) & in_cell(TARGET, a7.CELLS[g], cell)
    rows = [event_row(s, m, cell, g[i], [q[i] for q in gn], gw[i], keys=extra, floor_s=floor) for i, (s, m) in enumerate(zip(used, maps))]
    pd.DataFrame(rows + rest).to_csv(HERE / "a9_eka_falsifier_events.csv", **FMT)
    pd.DataFrame([dict(rot_deg=r, bearing_deg=bg, n_groups=int(k), pwin_groups=int(w), mark=mk, earliest_lag_min_s=e["earliest_lag_min_s"],
                       earliest_lag_median_s=e["earliest_lag_median_s"])
                  for r, bg, k, w, mk, e in zip([0.0, *kept.rot_deg], [bearing(TARGET), *kept.bearing_deg], n, nw, ["target", *kept.mark],
                                                [admitted(TARGET, used, floor), *kept.to_dict("records")])]).to_csv(HERE / "a9_eka_falsifier_cells.csv", **FMT)
    put(f"{d}groups_in_target_cell", int(n[0]), F_MIN_IN_CELL, bool(n[0] >= F_MIN_IN_CELL), "source groups whose strongest peak falls in the cell; ok = at least `of`")
    put(f"{d}largest_rotated_count", int(n[1:].max()), None, None, "at rotations " + " ".join(f"{r:g}" for r in kept.rot_deg[n[1:] == n[1:].max()]))
    put(f"{d}marked_cells", int(kept.mark.ne("").sum()), int(nl.mark.ne("").sum()), None, "kept marked cells and their counts: " + "; ".join(
        f"{mk} at {bg:.0f} deg: {int(k)}" for mk, bg, k in zip(kept.mark, kept.bearing_deg, n[1:]) if mk) + "; of = marked cells in the rotation list")
    put(f"{d}p", p, P_PASS, bool(p <= P_PASS), f"share of {len(n)} cells (target included) with a count >= the target's")
    put(f"{d}noise_groups_in_target_cell", noise, NOISE_GROUPS_MAX, bool(noise <= NOISE_GROUPS_MAX), "most groups, over the pre-P noise maps, whose strongest noise peak falls in the cell")
    put(f"{d}pwin_groups_in_target_cell", int(nw[0]), int(nw[1:].max()), None, "reported, not deciding; of = largest over the rotated cells")
    put(f"{d}pwin_rank", float((nw >= nw[0]).mean()), None, None, "share of cells (target included) with a control count >= the target's")
    put(f"{d}in_cell_back_azimuths", int(hit.sum()), None, None, "; ".join(
        f"{s['event']} group {s['group']} baz {s['baz_deg']:.1f} ({'adjacent or in Key sector' if s['adjacent'] else 'new direction'}"
        f"{', flagged by the synthetic' if s['synthetic_flag'] else ''})" for s, h in zip(used, hit) if h))
    for k in sorted(set(clu.tolist())):  # the same counts on one cluster's events alone
        m = clu == k
        nk, pk, zk, _ = decide(centers, 0, np.where(m, g, -1), [np.where(m, q, -1) for q in gn], grp, cell)
        wk = groups_in(centers, np.where(m, gw, -1), grp, cell)
        put(f"{d}cluster_{k}_groups_in_target_cell", int(nk[0]), len(set(grp[m])), None,
            f"back-azimuth {baz[m].min():.0f} to {baz[m].max():.0f} deg; of = groups in the cluster; largest rotated count {int(nk[1:].max())}; p {pk:.3f}; "
            f"noise groups {zk}; P-window control {int(wk[0])} (largest rotated {int(wk[1:].max())})" + ("" if d else "; reading "
            + reading_d110(len(set(grp[m])), len(kept), int(nk[0]), pk, zk, all(s["adjacent"] for s, h in zip(used, hit & m) if h))))
    if d:
        return done()
    done(reading_d110(len(set(grp)), len(kept), int(n[0]), p, noise, all(s["adjacent"] for s, h in zip(used, hit) if h)),
         "The decisive test of the first run on the new events alone")


def setup():
    """Elements, their positions, the arm axes and the least number of elements at the Eielson share."""
    el = elements()
    n_ilar = len(pd.read_csv(HERE / "a6_array_precheck.csv").query("section == 'element' and array == 'ILAR'"))
    return el, np.c_[el.x_km, el.y_km], arm_axes(el), math.ceil(a7.MIN_ELEM / n_ilar * len(el)), n_ilar


def main(waveforms: bool, proved: dict) -> None:
    el, xy, axes, min_elem, n_ilar = setup()
    la0, lo0 = el.lat.mean(), el.lon.mean()
    checks = []

    def check(name, value, of=None, ok=None, note=""):  # the column also holds readings, so floats are rounded here, not by to_csv
        checks.append(dict(check=name, value=float(f"{value:.6g}") if isinstance(value, float) else value, of=of, ok=ok, note=note))

    for k, v in proved.items():
        check(f"self_check_{k}", v, None, None, "synthetic of self_check(); see its asserts for what is required")

    # --- 1. elements
    assert (el.sample_rate % a7.FS == 0).all()
    arm = el.station.str.fullmatch("|".join(ARMS.values()))
    cross = axes["B"][1] + np.linalg.solve(np.c_[polar(1, axes["B"][0]), -polar(1, axes["R"][0])], axes["R"][1] - axes["B"][1])[0] \
        * polar(1, axes["B"][0])
    check("elements", len(el), None, None, f"{NET} {STA} {CHA}; on the two arms: {int(arm.sum())}; off them: {' '.join(el.station[~arm])}")
    check("sample_rate_hz", float(el.sample_rate.iloc[0]), a7.FS, bool(el.sample_rate.nunique() == 1), "decimated to `of` in phase 2")
    check("sensor_models", int(el.sensor.str.split("=").str[0].nunique()), None, None, "; ".join(sorted(el.sensor.str.split("=").str[0].unique())))
    check("elements_with_closed_epoch", int((el.end != "open").sum()), None, None,
          " ".join(f"{s} to {e[:10]}" for s, e in zip(el.station, el.end) if e != "open"))
    check("min_elements", min_elem, len(el), None, f"ceil of a7's share {a7.MIN_ELEM}/{n_ilar}")
    check("centroid_lat", la0)
    check("centroid_lon", lo0)
    check("centroid_shift_without_off_arm_km", float(np.hypot(*xy[arm].mean(axis=0))), None, None, "centroid of the arm elements only, from the centroid used")
    check("arm_crossing_from_centroid_km", float(np.hypot(*cross)), None, None,
          f"limit: Key's center seismometer is not identified; the arms cross at bearing {bearing(cross):.0f} deg from the centroid")
    for k, (ax, _) in axes.items():
        check(f"arm_{k}_axis_deg", ax)

    # --- 2. pre-check
    geo = a6.geometry(xy[:, 0], xy[:, 1])
    summary, resp = a6.response_rows("EKA", xy[:, 0], xy[:, 1], geo["aperture_km"])
    pre = pd.DataFrame([dict(section="geometry", array="EKA", lat=la0, lon=lo0, **geo, **summary), *resp])
    for b in a7.BANDS:
        q = pre[pre.freq_hz.isin(b)]
        check(f"precheck_{NAME[b]}_rg_unaliased", int(q.rg_unaliased.sum()), len(q), bool(q.rg_unaliased.all()),
              f"a6 rows at the band edges, both Rg velocities; smallest alias radius {q.s_alias_skm.min():.3g} s/km; "
              f"largest sidelobe {q.sl_max.max():.2f}; against steep P: {int(q.rg_unaliased_vs_p.sum())} of {len(q)}")

    # --- 3. sources
    src, n_rule = catalog(el, min_elem)
    cand = groups([s for s in src if s["candidate"]])
    tele = [s for s in cand if s["kind"] == "teleseism"]
    check("teleseisms_passing_source_rule", n_rule["isolated"], n_rule["deep_mag_dist"],
          None, f"deep >= {a7.DEEP_KM:g} km, M {a7.MAG[0]}-{a7.MAG[1]}, {a7.DEG[0]:g}-{a7.DEG[1]:g} deg; of = before the isolation rule")
    check("teleseism_candidates", len(tele), a7.N_CAND, None, f"largest first, after the pP rule and >= {min_elem} BHZ epochs")
    check("korean_tests_with_bhz_epoch", sum(s["kind"] == "dprk" for s in cand), sum(s["kind"] == "dprk" for s in src), None,
          "; ".join(f"{s['event'][5:]}: {s['n_elem_epoch']}" for s in src if s["kind"] == "dprk") + " (elements with an epoch)")
    check("candidates_in_key_sector", sum(s["in_key_sector"] for s in cand), len(cand), None,
          f"back-azimuth {KEY_SECTOR[0]:g} to {KEY_SECTOR[1]:g} deg (Key p. 35); teleseisms only: {sum(s['in_key_sector'] for s in tele)}; "
          f"in Key's strongest sector 0-10 deg: {sum(in_sector(s['baz_deg'], (0.0, 10.0)) for s in cand)}; "
          f"within 5 deg outside the sector: {sum(in_sector(s['baz_deg'], (KEY_SECTOR[0] - 5, KEY_SECTOR[1] + 5)) for s in cand) - sum(s['in_key_sector'] for s in cand)}")
    check("candidate_source_groups", len({s["group"] for s in cand}), None, None,
          f"a7 rule, {a7.GROUP_DEG:g} deg; groups with an event in Key's sector: {len({s['group'] for s in cand if s['in_key_sector']})}; "
          "phase 2 regroups the events used")
    sect = pd.Series([int(s["baz_deg"] // 30) * 30 for s in cand]).value_counts().sort_index()
    check("candidate_baz_sectors", len(sect), 12, None, "30 deg sectors from north, start:count " + " ".join(f"{k}:{v}" for k, v in sect.items()))
    lead = [a7.CELLS @ a7.slow_vec(s["slow_skm"], s["baz_deg"]) for s in cand]
    late = max(float(a7.times(xy, q)[a7.valid(q, EIELSON_LAG).any(axis=1)].max()) for q in lead)
    check("a7_noise_window_margin_s", -10 - (a7.NOISE_REF[-1] + late + a7.WIN / 2), 0, None,
          "a7.event_maps asserts this is positive: the last pre-P noise window stays clear of P, worst candidate")

    # --- 4-8. target, floor, blind zones, cell, null cells
    floor, cells, nulls = geometry(cand, xy, axes)
    g_t = int(np.argmin(np.hypot(*(a7.CELLS - TARGET).T)))
    rows = []
    for name, p, u, note in (("key", KEY_XY, KEY_U, "Key 1967 pp. 32, 34"), ("gupta", GUPTA_XY, GUPTA_U, "Gupta et al. 1990 p. 184"),
                             ("target", TARGET, np.nan, "mean of the two")):
        lon, lat, _ = a6.GEOD.fwd(lo0, la0, bearing(p), float(np.hypot(*p)) * 1000)
        rows.append(dict(section="target", name=name, range_km=float(np.hypot(*p)), bearing_deg=bearing(p), x_km=p[0], y_km=p[1],
                         lat=lat, lon=lon, u_kms=u, from_target_km=float(np.hypot(*(p - TARGET))), note=note))
        if name == "target":
            for k in ("range_km", "bearing_deg", "x_km", "y_km", "lat", "lon"):
                check(f"target_{k}", rows[-1][k])
    check("target_from_key_km", rows[0]["from_target_km"])
    check("target_from_gupta_km", rows[1]["from_target_km"])
    check("key_lag_by_a7_formula_s", float(KEY_XY @ a7.slow_vec(1 / KEY_P[0], KEY_P[1]) + KEY_RANGE / KEY_U), KEY_LAG, None,
          "Key's range, bearing and speed with his P wave give his delay (of = p. 34): the sign conventions agree")
    t = lags(TARGET, cand)
    check("decisive_floor_s", floor, EIELSON_LAG[0], None, f"candidates; before rounding down {t[:, BOTH].min() - a7.WIN / 2:.3f} s; "
          f"binding event {cand[int(t[:, BOTH].min(axis=1).argmin())]['event']}; recomputed in phase 2 on the events used")
    for label, fl in (("eielson", EIELSON_LAG[0]), ("decisive", floor)):
        rows.append(dict(section="floor", name=label, floor_s=fl, earliest_window_center_s=fl + a7.WIN / 2, n_events=len(cand),
                         pwin_shift_s=fl + PWIN_AFTER[0], pwin_lag_s=fl + PWIN_AFTER[1],
                         **{f"target_lag_min_{k}_scan_speed_s": float(t[:, u].min()) for k, u in (("key", U_KEY), ("gupta", U_GUPTA))}))
        for k, u in (("key", U_KEY), ("gupta", U_GUPTA)):
            n = sum(bool(a7.valid(q, (fl, EIELSON_LAG[1]))[g_t, u]) for q in lead)
            check(f"{label}_floor_events_imaging_target_cell_at_{k}_scan_speed", n, len(cand), None,
                  f"a7.valid at the grid cell nearest the target, scan speed {1 / a7.P_SCAN[u]:.3f} km/s")
    ilar = pd.read_csv(HERE / "a7_events.csv").query("outcome == 'used'").drop_duplicates("event").to_dict("records")
    for array, label, fl, evs in (("ILAR", "eielson", EIELSON_LAG[0], ilar), ("EKA", "eielson", EIELSON_LAG[0], cand), ("EKA", "decisive", floor, cand)):
        near = nearest_range(fl, evs)
        rows += [dict(section="blind_zone", array=array, name=label, floor_s=fl, n_events=len(evs), u_kms=float(1 / a7.P_SCAN[u]),
                      nearest_min_km=float(near[:, :, u].min()), nearest_median_km=float(np.median(near[:, :, u])),
                      nearest_max_km=float(near[:, :, u].max())) for u in range(len(a7.P_SCAN))]
        check(f"blind_zone_{array}_{label}_nearest_km_fastest", float(near[:, :, 0].min()), float(near[:, :, 0].max()), None,
              f"scan speed {1 / a7.P_SCAN[0]:.3g} km/s, value = smallest, of = largest over events and bearings; median {np.median(near[:, :, 0]):.1f}")
        check(f"blind_zone_{array}_{label}_nearest_km_slowest", float(near[:, :, -1].min()), float(near[:, :, -1].max()), None,
              f"scan speed {1 / a7.P_SCAN[-1]:.3g} km/s; median {np.median(near[:, :, -1]):.1f}")
    for b in a7.BANDS:
        c, nl = cells[b], nulls[b]
        rows.append(dict(section="cell", band_hz=NAME[b], u_kms=float(1 / a7.P_SCAN[U_KEY]), floor_s=floor, radial_km=c["radial"],
                         cross_km=c["cross"], imaging_error_km=c["err"], **admitted(TARGET, cand, floor)))
        rows += nl.assign(section="null_cell", band_hz=NAME[b]).to_dict("records")
        check(f"cell_{NAME[b]}_radial_km", c["radial"])
        check(f"cell_{NAME[b]}_cross_km", c["cross"])
        check(f"cell_{NAME[b]}_imaging_error_km", c["err"], None, None, "peak of the noise-free image to the target point; recorded, not asserted")
        for step, q in nl.groupby("step_deg", sort=False):
            check(f"null_cells_kept_{NAME[b]}_step_{step:g}deg", int(q.kept.sum()), len(q), bool(q.kept.sum() >= MIN_NULLS),
                  f"left out: {int(q.overlaps_target.sum())} overlap the target cell, {int((~q.admitted).sum())} not admitted at both speeds "
                  f"for every candidate; ok = at least {MIN_NULLS}; {'used' if q.used.all() else 'not used'}")
        q = nl[nl.used]
        n_kept = int(q.kept.sum())
        check(f"smallest_p_{NAME[b]}", 1 / (n_kept + 1), P_PASS, bool(n_kept >= MIN_NULLS), f"1 / (kept cells + target), step {q.step_deg.iloc[0]:g} deg")
        check(f"marked_cells_{NAME[b]}", int(q[q.mark != ""].kept.sum()), int((q.mark != "").sum()), None,
              "; ".join(f"{m.mark} {m.mark_bearing_deg:.0f} deg -> cell at {m.bearing_deg:.0f} deg, {'kept' if m.kept else 'left out'}"
                        for m in q[q.mark != ""].itertuples()))
        check(f"earliest_lag_median_s_{NAME[b]}_kept_cells_range", float(q[q.kept].earliest_lag_median_s.min()) if n_kept else np.nan,
              float(q[q.kept].earliest_lag_median_s.max()) if n_kept else np.nan, None,
              f"value = smallest, of = largest over kept cells; target {admitted(TARGET, cand, floor)['earliest_lag_median_s']:.2f} s")

    el.to_csv(HERE / "a9_eka_elements.csv", **FMT)
    pre.to_csv(HERE / "a9_eka_precheck.csv", index=False, float_format="%.7g")
    pd.DataFrame(src).drop(columns="tp").to_csv(HERE / "a9_eka_candidates.csv", **FMT)
    lead_cols = ["section", "array", "name", "band_hz"]
    geom = pd.DataFrame(rows)
    geom[lead_cols + [k for k in geom.columns if k not in lead_cols]].to_csv(HERE / "a9_eka_geometry.csv", **FMT)
    pd.DataFrame(checks).to_csv(HERE / "a9_eka_checks.csv", **FMT)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400, "display.max_colwidth", 150)
    print("\n== checks")
    print(pd.DataFrame(checks).to_string(index=False))
    if waveforms:
        phase2(cand, el, xy, axes, min_elem)


def self_check() -> dict:
    assert np.allclose(polar(2.0, 90.0), [2.0, 0.0]) and abs(bearing(polar(3.0, 315.0)) - 315.0) < 1e-9
    assert in_sector(350.0) and in_sector(30.0) and not in_sector(31.0) and not in_sector(339.0)
    assert abs(arc([350.0, 10.0, 20.0]) - 30.0) < 1e-9 and arc([5.0]) == 0.0
    # the blind-zone formula against a7.valid on the grid: the nearest valid cell lies at or just beyond the formula's range
    ev = [dict(slow_skm=0.07, baz_deg=35.0)]
    lead = a7.CELLS @ a7.slow_vec(0.07, 35.0)
    for fl in (EIELSON_LAG[0], 1.5):
        d = a7.RANGE[:, None][a7.valid(lead, (fl, EIELSON_LAG[1])).any(axis=1)].min() - nearest_range(fl, ev).min()
        assert 0 <= d < 1.5 * a7.CELL, (fl, d)
    # phase 2 from a raw 40 Hz stream to the reading, on synthetic events across the real elements
    el = elements()
    xy, stations, axes = np.c_[el.x_km, el.y_km], list(el.station), arm_axes(el)
    four = [dict(event=f"syn_{i}", kind="teleseism", lat=10.0 * i, lon=0.0, dist_deg=60.0, baz_deg=baz, slow_skm=slow)
            for i, (baz, slow) in enumerate(((11.0, 0.056), (15.0, 0.0535), (22.0, 0.0515), (247.0, 0.0435)))]  # as the candidates lie: three in Key's sector
    out = {}
    lo = NAME[a7.BANDS[0]]
    for name, ratio in (("with_arrival", KEY_RATIO), ("without_arrival", 0.0)):
        src = synthetic(xy, stations, four, ratio)
        tab = finish(src, xy, axes, bands=a7.BANDS[:1])
        cl, pw, res, used = tab["cells"], tab["power"], tab["result"], [s for s in src if s["outcome"] == "used"]
        assert len(used) == 4 and len({s["group"] for s in used}) == 4, [s["outcome"] for s in src]
        v, q = res.set_index("check").value, cl[(cl.run == "decisive") & (cl.rot_deg > 0)]
        out |= {f"{name}_{k}": v[f"{lo}_{k}"] for k in (
            "decisive_groups_in_target_cell", "decisive_p", "decisive_noise_groups_in_target_cell", "decisive_pwin_groups_in_target_cell",
            "decisive_pwin_rank", "decisive_peak_range_km_median", "eielson_groups_in_target_cell", "eielson_p", "eielson_peak_range_km_median")}
        out |= {f"{name}_null_cells_max_groups": int(q.n_groups.max()), f"{name}_power": v["power"], f"{name}_reading": v["reading"]}
        if ratio:
            out |= dict(floor_s=v["decisive_floor_s_events_used"], null_cells=len(q))
            assert v[f"{lo}_decisive_groups_in_target_cell"] == 4 > q.n_groups.max(), cl  # the statistic finds it; no rotated cell does
            assert v[f"{lo}_decisive_p"] == 1 / (len(q) + 1) <= P_PASS and v["reading"] == "pass", res
        else:  # nothing there: no count, no pass; and an arrival injected at a rotated cell makes that cell pass
            assert v[f"{lo}_decisive_groups_in_target_cell"] == 0 and v[f"{lo}_decisive_p"] == 1.0 and v["reading"] != "pass", res
            assert v["power"] >= POWER_MIN, pw
    # falsifier: the five readings on made-up counts (groups used, cells kept, count, p, noise groups, all adjacent)
    assert reading_d110(5, MIN_NULLS - 1, 3, 0.03, 0, False) == "unable to decide" and reading_d110(3, 29, 3, 0.03, 0, False) == "uninformative"
    assert reading_d110(4, 29, 2, 0.03, 0, False) == "repeated" and reading_d110(4, 29, 2, 0.03, 0, True) == "repeated from adjacent directions"
    assert {reading_d110(4, 29, *q, True) for q in ((1, 0.03, 0), (3, 0.07, 0), (3, 0.03, 2))} == {"not repeated"}
    # falsifier: a plane arrival standing in for pP, 14 s after P and as large as the wave the test is after. Uncut, it is the
    # strongest peak; cut, it is not. (One as large as P itself still leaks through the filter's precursor at this noise level.)
    late, brg = 14.0, 200.0
    one = [dict(event="syn_pp", kind="teleseism", lat=0.0, lon=0.0, dist_deg=60.0, baz_deg=60.0, slow_skm=0.06, pp_minus_p_s=late)]
    for with_cut in (False, True):
        src = synthetic(xy, stations, one, 0.0, plane=(late, a7.P_SCAN[1], brg, KEY_RATIO))
        if with_cut:
            cut(src[0])
        m = run(src, xy, a7.BANDS[0], 1.5)[0]
        g = strongest(m["S"], m["ok"], m["thr"])
        on = g >= 0 and abs((bearing(a7.CELLS[g]) - brg + 180) % 360 - 180) < 10  # a peak on the arrival's bearing
        if with_cut:
            end = m["lag"][m["ok"]].max() + a7.WIN / 2 + src[0]["t_ref"] - a6.PRE  # last window end after the predicted P
            assert not on and end <= late - F_MARGIN + 1e-6, (g, end)
        else:
            iu = int(a7.cell_max(m["S"], m["ok"])[1][g])
            assert on and abs(m["lag"][g, iu] - late) <= a7.WIN / 2 and m["S"][g, iu] > 0.8, (g, m["lag"][g, iu], m["S"][g, iu])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--waveforms", action="store_true", help="phase 2: fetch BHZ windows and run the control")
    ap.add_argument("--falsifier", action="store_true", help="the falsifier on new events, instead of phases 1 and 2")
    args = ap.parse_args()
    proved = self_check()
    if args.falsifier:
        falsifier()
    else:
        main(args.waveforms, proved)
