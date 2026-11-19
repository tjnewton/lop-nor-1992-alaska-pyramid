"""A7: P4 item S4 -- the claimed method run as a positive control on the Eielson array (ILAR).

Question: teleseismic P-coda array analysis that locates
near-receiver P-to-Rg scatterers (the NORESS method) is the real method closest to the claim. Run on
the one open Alaskan array that passed the pre-check (a6), what does it resolve, and what is the
weakest scatterer it locates?

Inputs (all open, no account; nothing identifying is sent):
- Element positions: a6_array_precheck.csv (section element, array ILAR). Array geometry, catalog
  and waveform fetching and caching are imported from a6_array_precheck, not copied.
- Catalogs cached by a6 under $AKP_DATA/a6: the DPRK tests, M>=6 teleseisms, Alaskan explosions.
- Waveforms, IM IL01-IL19 SHZ: the three DPRK tests (a6 cache), teleseism candidates (predicted P
  -120..+300 s) and mine blasts (origin -60..+90 s), cached as miniSEED under $AKP_DATA/a7.
- Only if a scatterer is found: Copernicus GLO-90 DEM tiles (AWS open data), cached under $AKP_DATA/a7.

Method (design fixed before results; constants below are its inputs):
1. Sources. The three DPRK tests, plus teleseisms by rule: depth >= DEEP_KM, MAG range, 30-95 deg,
   no other M>=6 event from ISOLATED_MIN before to after, pP-P >= PP_MIN_S (the depth phase stays
   outside the coda window); the N_CAND largest are fetched; kept if >= MIN_ELEM elements have a
   full record and the 0.5-1 Hz P signal-to-noise (a6 definition) is >= SNR_MIN; the N_TELE with the
   highest signal-to-noise are used. Events within GROUP_DEG of each other form one source group.
2. Bands: 0.5-1 Hz primary, 1-2 Hz beside it (a6 zero-phase band-pass).
3. P beam removal. Element delays = plane wave at the iasp91 slowness and geodesic back-azimuth,
   refined by cross-correlation with the beam in the P window (when the band's P signal-to-noise is
   >= REFINE_SNR); gain per element = least-squares fit to the beam in the P window, clipped. The
   residual x_i(t) - a_i b(t - d_i) is linear in the data, so a synthetic can be passed through it.
4. Back-projection. Grid of CELL km cells out to GRID_KM from the array center. For cell g and Rg
   velocity U (13 values, uniform in slowness over 1.5-3.2 km/s) the Rg from a scatterer at g reaches
   element i at t_P + s.x_g + |x_g - r_i| / U: the P arrival time at the scatterer (iasp91 slowness
   vector s) plus the Rg travel time over the exact distance (curved front). t_P is the envelope peak
   of P (a6 t_pk). Semblance of the residual over a WIN s window centered on that time, over elements.
   The window's lag after P is T = s.x_g + |x_g| / U; only windows inside 5-60 s are used. So each
   (cell, velocity) is one sliding window, and a fixed scatterer is one cell whatever its window.
   Time-domain semblance of band-passed traces is the energy-weighted mean over the band of the
   per-frequency semblance.
5. Null. The same statistic (a) with the reference time moved into pre-P noise (NOISE_REF) and
   (b) on the coda with phases randomized independently per element (N_RAND draws). Threshold per
   event and band = the larger of the two 99th percentiles over all cells and windows.
6. Kill criterion. An event has a coherent peak above the null if its largest semblance exceeds the
   largest value in all its null maps. Kill if no event does. Because one in (1 + number of null
   maps) would do so by chance, the count is printed beside its binomial chance probability.
7. Stacking. Peaks = local maxima (within PEAK_SEP km) of the per-cell maximum over velocity, above
   the threshold. Peaks of different events are the same scatterer when inside one resolution cell
   (step 8) of the strongest. A scatterer needs peaks from >= 2 source groups. The same rule is run
   on each set of null maps: the number of "scatterers" it returns there is the chance level, and
   above_null says whether a scatterer has more groups than any null cluster. Added after the first
   full run, because that rule saturates (a 99th-percentile threshold leaves many peaks per event and
   the cell is wide): rule "strongest_peak" clusters only each event's single largest peak, with the
   same null. a7_scatterers.csv carries both, by column `rule`. noise_groups_same_cell and
   rand_groups_same_cell give, for the scatterer's own cell, the most source groups whose pre-P noise
   map (or phase-randomized map) has a peak there under the same rule: a cell that pre-P noise fills as
   often as the coda does is an ambient source or a favored direction, not a P-to-Rg scatterer.
8. Resolution cell. Noise-free synthetic scatterers at PSF_RANGES: half-extents, radial and
   cross-range, of the connected region where the per-cell stacked energy stays >= HALF of its peak
   (energy, because noise-free semblance stays near 1 on the filter's ringing).
9. Calibration. The largest cataloged explosion in each 45-degree by 10 km bin, 10-50 km from the
   array, back-projected as an ordinary source (catalog origin time plus Rg travel time, no beam
   removal): location error of the semblance peak against the catalog, the same for the peak of
   stacked energy, and the half-power extents of the real Rg image.
10. Sensitivity. A copy of the event's own P beam wavelet, scaled to Rg/P = ratio (peak amplitudes),
   injected as Rg from a cell at SENS_RANGES (azimuths and velocity drawn per trial from a fixed
   seed; cylindrical spreading across the array), into the band-passed record before beam removal.
   Recovered = the largest semblance of the whole map lies within one resolution cell of the cell
   and exceeds the event's threshold (recov_local: the largest value inside that cell exceeds the
   threshold; its ratio-0 row is the false-recovery rate). min_ratio_95 = smallest ratio from which
   recovery stays >= 95 percent of trials. The synthetic has no dispersion and is perfectly coherent
   across the array, so these ratios are a lower bound on what a real scatterer needs.
12. P-window control (added after the first full run; reported, not used by any rule). The earliest
   windows (lag <= PWIN_LAG) are re-imaged PWIN_SHIFT s earlier, so that they sit on the direct P
   residual instead of the coda. A coda peak in the same cell and band as this control's peak can be
   the array's response to what the beam removal leaves of P, or Rg from scatterers too near to image.
   The kill count is also given for late windows only (lag >= LATE_LAG), against the nulls' same windows.
11. Relief (only for scatterers found): GLO-90 elevation range within the equal-area circle of the
   resolution cell, and its percentile among all grid cells. An aid, not an identification.

Outputs (beside this file): a7_events.csv, a7_blast_calibration.csv, a7_scatterers.csv,
a7_sensitivity.csv, a7_checks.csv.
Run: PYTHONPATH=. conda run -n swe-akp python -m akp.stage_a.a7_coda_backprojection
(deterministic; re-runs offline from the cache; one process, about 1 GB of memory.)
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import Stream, Trace
from obspy import UTCDateTime as T
from obspy.clients.fdsn import Client
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel
from scipy.ndimage import label, maximum_filter, minimum_filter
from scipy.signal.windows import tukey
from scipy.stats import binom

from akp.paths import data_dir, data_glob
from akp.stage_a import a6_array_precheck as a6
from akp.zones import BOXES, in_box

HERE = Path(__file__).parent
FS = 20.0  # Hz, SHZ sample rate (a6 element table; asserted)
BANDS = ((0.5, 1.0), (1.0, 2.0))  # Hz; first is primary (design)
GRID_KM, CELL = 50.0, 1.0  # km (design)
P_SCAN = np.linspace(1 / 3.2, 1 / 1.5, 13)  # s/km, Rg slowness scan = 1.5-3.2 km/s (design)
WIN = 5.0  # s, sliding window (design)
LAG = (5.0, 60.0)  # s after P that a window must lie within (design)
OFFS = np.arange(0, int(WIN * FS) + 1, 2)  # window samples at 10 Hz
NOISE_REF = (-105.0, -80.0, -55.0)  # s from t_P: pseudo-P times for the pre-P noise maps
N_RAND = 3  # phase-randomized maps per event
RAND_SEG = (-5.0, 70.0)  # s from t_P: the segment whose phases are randomized
Q_NULL = 99.0  # percentile (design)
P_FIT = (-3.0, 5.0)  # s from t_P: window for delays, gains and the injected P wavelet
REFINE_SNR, MAX_REFINE_S, GAIN_CLIP = 3.0, 0.5, (0.5, 2.0)
DEEP_KM, MAG, DEG = 300.0, (6.0, 6.8), (30.0, 95.0)  # source rule: deep, large but short, teleseismic P
ISOLATED_MIN, PP_MIN_S = (60.0, 15.0), 65.0  # minutes before / after without another M>=6; s
N_CAND, N_TELE, MIN_ELEM, SNR_MIN = 60, 30, 17, 10.0
GROUP_DEG = 3.0  # events closer than this share a source group
PEAK_SEP = 3  # km, local-maximum neighborhood
PWIN_SHIFT, PWIN_LAG = 8.0, 10.5  # s: P-window control = windows up to this lag, imaged this much earlier
LATE_LAG = 15.0  # s: "late" windows start at least 12.5 s after the P peak, clear of the direct P pulse
HALF = 0.5  # resolution cell = region at or above this fraction of the peak stacked energy
PSF_RANGES = (15.0, 25.0, 35.0, 45.0)  # km
PSF_AZ = (45.0, 135.0, 225.0, 315.0)
PSF_U = 3  # index into P_SCAN (2.5 km/s) of the synthetic used for the resolution cell
SENS_RANGES = PSF_RANGES
RATIOS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)  # Rg/P peak amplitude
N_AZ_TELE, N_AZ_DPRK = 3, 8  # injection azimuths per range and event
BLAST_KM, BLAST_PRE, BLAST_POST, BLAST_NOISE_REF = (10.0, 50.0), 60.0, 90.0, 5.0
SEED = 7
DEM_URL = "https://copernicus-dem-90m.s3.amazonaws.com/{n}/{n}.tif"
DEM_STEP = 0.5  # km, sampling of the elevation grid
TINY = 1e-30

_ax = np.arange(-GRID_KM, GRID_KM + CELL, CELL)
_gx, _gy = np.meshgrid(_ax, _ax)
IJ = np.nonzero(np.hypot(_gx, _gy) <= GRID_KM)  # (row = north index, col = east index) of the cells in the disc
CELLS = np.c_[_gx[IJ], _gy[IJ]]  # km east, north of the array center
RANGE = np.hypot(*CELLS.T)


class Lazy:
    """FDSN client built on first use, so an offline re-run from the cache never opens a connection."""

    def __getattr__(self, name):
        if "_c" not in self.__dict__:
            self._c = Client(base_url=a6.ES, timeout=600)
        return getattr(self._c, name)


def fetch(c, cache, tag, stations, t1, t2):
    """a6.waveforms plus a marker for 'server has no data', so that outcome is cached too."""
    miss = cache / f"{tag}.nodata"
    if miss.exists():
        return None
    st = a6.waveforms(c, cache, tag, "IM", stations, "SHZ", t1, t2)
    if st is None:
        miss.touch()
    return st


def matrix(st, stations, t0, nt):
    """[n_ok, nt] float array from t0 (nearest sample) for stations with a full, live record; mask of those stations."""
    by = {tr.stats.station: tr for tr in st}
    rows, ok = [], np.zeros(len(stations), bool)
    for j, s in enumerate(stations):
        tr = by.get(s)
        if tr is None or tr.stats.npts < 0.9 * nt or abs(tr.stats.sampling_rate - FS) > 1e-6:
            continue
        k = int(round((tr.stats.starttime - t0) * FS))
        src = np.asarray(tr.data, float)[max(0, -k):][:nt - max(0, k)]
        x = np.zeros(nt)
        x[max(0, k):max(0, k) + len(src)] = src
        rows.append(x)
        ok[j] = True
    X = np.array(rows)
    if len(X):
        g = X.std(axis=1) / np.median(X.std(axis=1))
        live = (g > 0.2) & (g < 5.0)  # a dead or wild element would dominate the semblance
        X, ok[np.nonzero(ok)[0][~live]] = X[live], False
    return X, ok


def shift(X, tau):
    """x_i(t - tau_i) by a Fourier phase ramp."""
    f = np.fft.rfftfreq(X.shape[1], 1 / FS)
    return np.fft.irfft(np.fft.rfft(X, axis=1) * np.exp(-2j * np.pi * f * np.asarray(tau)[:, None]), X.shape[1], axis=1)


def slow_vec(s, baz):
    """Horizontal slowness vector (east, north; s/km) of a wave arriving from back-azimuth baz."""
    return -s * np.array([np.sin(np.deg2rad(baz)), np.cos(np.deg2rad(baz))])


def p_fit(X, xy, s_vec, t_ref, refine):
    """Element delays (s, relative to the array center) and gains of the P beam."""
    d, a = xy @ s_vec, np.ones(len(X))
    if not refine:
        return d, a
    w = slice(int((t_ref + P_FIT[0]) * FS), int((t_ref + P_FIT[1]) * FS))
    al = shift(X, -d)
    b = al[:, w].mean(axis=0)
    lags = np.arange(-int(MAX_REFINE_S * FS), int(MAX_REFINE_S * FS) + 1)
    cc = np.array([al[:, w.start + k:w.stop + k] @ b for k in lags]).T  # [element, lag]
    j = np.clip(cc.argmax(axis=1), 1, len(lags) - 2)
    i = np.arange(len(X))
    y0, y1, y2 = cc[i, j - 1], cc[i, j], cc[i, j + 1]
    d = d + (lags[j] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 - TINY)) / FS  # parabolic peak
    al = shift(X, -d)
    b = al[:, w].mean(axis=0)
    return d, np.clip(al[:, w] @ b / (b @ b), *GAIN_CLIP)


def residual(X, d, a):
    """(x_i(t) - a_i b(t - d_i), b): the record with the delay-and-sum P beam removed. Linear in X for fixed d, a."""
    b = shift(X, -d).mean(axis=0)
    return X - shift(a[:, None] * b[None, :], d), b


def times(xy, lead):
    """[cell, velocity, element] arrival time after the reference: lead (s, per cell) plus distance x slowness."""
    dist = np.hypot(CELLS[:, None, 0] - xy[None, :, 0], CELLS[:, None, 1] - xy[None, :, 1])
    return (lead[:, None, None] + dist[:, None, :] * P_SCAN[None, :, None]).astype(np.float32)


def valid(lead, lag):
    """[cell, velocity]: the window, centered at lag T = lead + range x slowness after the reference, lies inside `lag`."""
    t = lead[:, None] + RANGE[:, None] * P_SCAN[None, :]
    return (t >= lag[0] + WIN / 2) & (t <= lag[1] - WIN / 2)


def gather(R, t_ref, tt_u):
    """[cell, element, sample]: each element's WIN s window centered on its predicted arrival (nearest sample)."""
    n_el, nt = R.shape
    i0 = np.rint((t_ref + tt_u - WIN / 2) * FS).astype(np.int64)
    return R.ravel()[np.clip(i0[:, :, None] + OFFS, 0, nt - 1) + (np.arange(n_el) * nt)[None, :, None]]


def image(R, t_ref, tt, power=False):
    """Semblance [cell, velocity]: 1 for identical traces after the moveout, about 1/n_elements for incoherent ones.
    With power=True, the stacked (beam) energy of the window instead."""
    R = np.ascontiguousarray(R, np.float32)
    S = np.empty(tt.shape[:2], np.float32)
    for u in range(tt.shape[1]):
        G = gather(R, t_ref, tt[:, u])
        S[:, u] = (G.sum(axis=1) ** 2).sum(axis=-1)
        if not power:
            S[:, u] /= G.shape[1] * np.einsum("get,get->g", G, G) + TINY
    return S


def grid2d(v, fill=-1.0):
    out = np.full(_gx.shape, fill, float)
    out[IJ] = v
    return out


def cell_max(S, ok):
    """Per cell: largest semblance over its valid windows (-1 where it has none), and the velocity index."""
    m = np.where(ok, S, -1.0)
    return m.max(axis=1), m.argmax(axis=1)


def extent(mx, g):
    """Half-extents (radial, cross-range; km) of the connected region around cell g with mx >= HALF x mx[g]."""
    mx = np.where(mx < 0, 0.0, mx)
    lab, _ = label(grid2d(mx) >= HALF * mx[g])
    reg = CELLS[lab[IJ] == lab[IJ][g]]
    er = CELLS[g] / max(RANGE[g], TINY)
    r, t = reg @ er, reg @ np.array([er[1], -er[0]])
    return (np.ptp(r) + CELL) / 2, (np.ptp(t) + CELL) / 2


def same_cell(p, q, res):
    """Is each point of q inside the resolution cell (ellipse, radial x cross-range) centered on p?"""
    rng = max(float(np.hypot(*p)), TINY)
    ar, ac = np.interp(rng, res["range_km"], res["radial"]), np.interp(rng, res["range_km"], res["cross"])
    er = np.asarray(p) / rng
    dq = np.atleast_2d(q) - p
    return ((dq @ er) / ar) ** 2 + ((dq @ np.array([er[1], -er[0]])) / ac) ** 2 <= 1.0


def pulse(lo, hi, nt, i_c):
    """Band-limited pulse centered on sample i_c: the a6 band-pass of a spike."""
    x = np.zeros(nt)
    x[i_c] = 1.0
    y = a6.band(Stream([Trace(x, dict(delta=1 / FS))]), lo, hi)[0].data
    return y / np.abs(y).max()


def scattered(base, xy, cell, slow, lead):
    """Rg from `cell`: the record `base` (wavelet at the reference time) delayed at each element by lead +
    distance x slowness, with cylindrical spreading normalized at the array center."""
    dist = np.hypot(*(xy - cell).T)
    return shift(base[None, :] * np.sqrt(np.hypot(*cell) / dist)[:, None], lead + dist * slow)


def psf(xy, lo, hi):
    """Resolution cell by range: mean half-power half-extents of noise-free synthetic scatterers (slowness
    P_SCAN[PSF_U] where imaged). Stacked energy, not semblance: without noise the filter's ringing keeps the
    semblance near 1 along the whole bearing, which says nothing about a real record."""
    nt, t_ref = int(100 * FS), 20.0
    base = pulse(lo, hi, nt, int(t_ref * FS))
    zero = np.zeros(len(CELLS))
    tt, ok = times(xy, zero), valid(zero, LAG)
    rows = []
    for rng in PSF_RANGES:
        ext = []
        for az in PSF_AZ:
            g = int(np.argmin(np.hypot(*(CELLS - rng * np.array([np.sin(np.deg2rad(az)), np.cos(np.deg2rad(az))])).T)))
            u = max(PSF_U, int(np.argmax(ok[g])))  # nearer cells are only imaged at slower velocities
            mx, _ = cell_max(image(scattered(base, xy, CELLS[g], P_SCAN[u], 0.0), t_ref, tt, power=True), ok)
            gp = int(np.argmax(mx))
            assert np.hypot(*(CELLS[gp] - CELLS[g])) <= 1.5 * CELL, (rng, az, CELLS[gp], CELLS[g])  # images onto its cell
            ext.append(extent(mx, gp))
        rows.append((rng, *np.mean(ext, axis=0)))
    r = np.array(rows)
    return dict(range_km=r[:, 0], radial=r[:, 1], cross=r[:, 2])


def event_maps(X, xy, s_vec, t_ref, refine, rng):
    """Beam removal, the coda map and its null maps for one event and band."""
    d, a = p_fit(X, xy, s_vec, t_ref, refine)
    R, b = residual(X, d, a)
    lead = CELLS @ s_vec
    tt, ok = times(xy, lead), valid(lead, LAG)
    assert t_ref + NOISE_REF[-1] + tt[ok.any(axis=1)].max() + WIN / 2 < t_ref - 10  # noise windows stay clear of P
    i0, i1 = int((t_ref + RAND_SEG[0]) * FS), int((t_ref + RAND_SEG[1]) * FS)
    spec = np.fft.rfft(R[:, i0:i1] * tukey(i1 - i0, 0.1), axis=1)
    rand = []
    for _ in range(N_RAND):
        Q = R.copy()
        Q[:, i0:i1] = np.fft.irfft(spec * np.exp(2j * np.pi * rng.random(spec.shape)), i1 - i0, axis=1)
        rand.append(image(Q, t_ref, tt))
    noise = [image(R, t_ref + q, tt) for q in NOISE_REF]
    thr_n, thr_r = (float(np.percentile(np.concatenate([m[ok] for m in ms]), Q_NULL)) for ms in (noise, rand))
    w = slice(int((t_ref + P_FIT[0]) * FS), int((t_ref + P_FIT[1]) * FS))
    wave = np.zeros(X.shape[1])
    wave[w] = b[w] * tukey(w.stop - w.start, 0.25)
    p_amp = float(np.abs(b[w]).max())
    wave *= p_amp / np.abs(wave).max()  # injected wavelet: the P beam, peak amplitude = P amplitude
    lag = lead[:, None] + RANGE[:, None] * P_SCAN[None, :]
    return dict(xy=xy, d=d, a=a, R=R.astype(np.float32), lead=lead, ok=ok, t_ref=t_ref, wave=wave, p_amp=p_amp, lag=lag,
                pwin=image(R, t_ref - PWIN_SHIFT, tt), ok_pwin=ok & (lag <= PWIN_LAG), ok_late=ok & (lag >= LATE_LAG),
                S=image(R, t_ref, tt), nulls=noise + rand, thr_noise=thr_n, thr_rand=thr_r, thr=max(thr_n, thr_r),
                resid_frac=float(R[:, w].std() / X[:, w].std()))


def peaks(S, ok, thr):
    """Local maxima of the per-cell maximum above thr: rows (cell index, semblance, velocity km/s)."""
    mx, iu = cell_max(S, ok)
    top = maximum_filter(grid2d(mx), size=2 * PEAK_SEP + 1)[IJ]
    return [(int(g), float(mx[g]), float(1 / P_SCAN[iu[g]])) for g in np.nonzero((mx >= top) & (mx > thr))[0]]


def stack(pk, res):
    """Greedy clustering of peaks (DataFrame: event, group, g, sem, u, thr), strongest first, by resolution cell."""
    pk = pk.sort_values(["sem", "event"], ascending=[False, True], kind="stable").reset_index(drop=True)
    free, out = np.ones(len(pk), bool), []
    xy = CELLS[pk.g.to_numpy()] if len(pk) else np.zeros((0, 2))
    for i in range(len(pk)):
        if not free[i]:
            continue
        m = free & same_cell(xy[i], xy, res)
        free &= ~m
        mem = pk[m]
        out.append(dict(x_km=xy[i, 0], y_km=xy[i, 1], bearing_deg=float(np.rad2deg(np.arctan2(xy[i, 0], xy[i, 1])) % 360),
                        range_km=float(np.hypot(*xy[i])), n_events=mem.event.nunique(), n_groups=mem.group.nunique(),
                        peak_semblance=pk["sem"][i], null_threshold=float(mem.thr.median()), u_kms=pk["u"][i], lag_s=pk["lag"][i],
                        strongest_event=pk.event[i], events=";".join(sorted(mem.event.unique()))))
    return pd.DataFrame(out)


def inject_terms(ev, Gd, Bd, syn):
    """Cross terms of a unit-ratio synthetic with the data: S(A) = (a + 2Ab + A^2 c) / (n (e + 2Ax + A^2 f))."""
    Rs = np.ascontiguousarray(residual(syn, ev["d"], ev["a"])[0], np.float32)
    tt = times(ev["xy"], ev["lead"])
    b, c, x, f = (np.empty(tt.shape[:2], np.float32) for _ in range(4))
    for u in range(tt.shape[1]):
        Gs = gather(Rs, ev["t_ref"], tt[:, u])
        Bs = Gs.sum(axis=1)
        b[:, u], c[:, u] = (Bd[:, u] * Bs).sum(axis=-1), (Bs ** 2).sum(axis=-1)
        x[:, u], f[:, u] = np.einsum("get,get->g", Gd[:, u], Gs), np.einsum("get,get->g", Gs, Gs)
    return b, c, x, f


def data_terms(ev):
    tt = times(ev["xy"], ev["lead"])
    Gd = np.stack([gather(ev["R"], ev["t_ref"], tt[:, u]) for u in range(tt.shape[1])], axis=1)  # ~0.4 GB
    Bd = Gd.sum(axis=2)
    return Gd, Bd, (Bd ** 2).sum(axis=-1), np.einsum("guet,guet->gu", Gd, Gd)


def sensitivity(name, ev, res, rng, n_az):
    """Per-trial recovery of injected Rg for one event and band: rows (event, range, ratio, global, local)."""
    Gd, Bd, a, e = data_terms(ev)
    n, rows = Gd.shape[2], []
    for r in SENS_RANGES:
        for az in (rng.uniform(0, 360) + np.arange(n_az) * 360.0 / n_az):
            g = int(np.argmin(np.hypot(*(CELLS - r * np.array([np.sin(np.deg2rad(az)), np.cos(np.deg2rad(az))])).T)))
            p_min = max(P_SCAN[0], (LAG[0] + WIN / 2 - ev["lead"][g]) / RANGE[g])  # slowest lag bound: window after 5 s
            slow = rng.uniform(p_min, P_SCAN[-1])
            b, c, x, f = inject_terms(ev, Gd, Bd, scattered(ev["wave"], ev["xy"], CELLS[g], slow, ev["lead"][g]))
            near = same_cell(CELLS[g], CELLS, res)
            for A in RATIOS:
                S = np.where(ev["ok"], (a + 2 * A * b + A * A * c) / (n * (e + 2 * A * x + A * A * f) + TINY), -1.0)
                mx = S.max(axis=1)
                rows.append(dict(event=name, range_km=r, ratio=A, glob=bool(near[mx.argmax()] and mx.max() > ev["thr"]),
                                 loc=bool(mx[near].max() > ev["thr"])))
    return rows


def min_ratio(rate):
    """Smallest ratio from which the recovery rate (Series indexed by ratio, ascending) stays >= 0.95; NaN if never."""
    good = (rate >= 0.95).to_numpy()[::-1].cumprod()[::-1].astype(bool)
    return float(rate.index[good][0]) if good.any() else np.nan


def elevation(cache, la0, lo0):
    """GLO-90 elevation (m) on a DEM_STEP km grid over the back-projection square; tiles cached."""
    import rasterio

    ax = np.arange(-GRID_KM, GRID_KM + DEM_STEP, DEM_STEP)
    x, y = np.meshgrid(ax, ax)
    lon, lat, _ = a6.GEOD.fwd(np.full(x.shape, lo0), np.full(x.shape, la0), np.rad2deg(np.arctan2(x, y)), np.hypot(x, y) * 1000)
    assert not any(in_box(a, o, b) for b in BOXES.values() for a, o in zip(lat.ravel()[::97], lon.ravel()[::97])), \
        "the elevation grid must stay outside every claim box"
    z = np.full(x.shape, np.nan)
    for la in np.unique(np.floor(lat)).astype(int):
        for lo in np.unique(np.floor(lon)).astype(int):
            n = f"Copernicus_DSM_COG_30_N{la:02d}_00_W{-lo:03d}_00_DEM"
            path = cache / f"{n}.tif"
            if not path.exists():
                urllib.request.urlretrieve(DEM_URL.format(n=n), path)
            with rasterio.open(path) as ds:
                m = (np.floor(lat) == la) & (np.floor(lon) == lo)
                col, row = ~ds.transform * (lon[m], lat[m])
                z[m] = ds.read(1)[np.clip(row.astype(int), 0, ds.height - 1), np.clip(col.astype(int), 0, ds.width - 1)]
    return ax, z


def relief(ax, z, radius):
    """Elevation range (m) within `radius` km of every point of the elevation grid."""
    k = int(round(radius / DEM_STEP))
    yy, xx = np.ogrid[-k:k + 1, -k:k + 1]
    disc = xx ** 2 + yy ** 2 <= k ** 2
    return maximum_filter(z, footprint=disc, mode="nearest") - minimum_filter(z, footprint=disc, mode="nearest")


def main() -> None:
    cache6, cache7 = data_dir("a6"), data_dir("a7")
    c, taup = Lazy(), TauPyModel("iasp91")
    rng = np.random.default_rng(SEED)
    el = pd.read_csv(HERE / "a6_array_precheck.csv").query("section == 'element' and array == 'ILAR'").reset_index(drop=True)
    assert (el.sample_rate == FS).all()
    XY = np.c_[a6.local_xy(el.lat, el.lon)]
    la0, lo0 = el.lat.mean(), el.lon.mean()
    stations = list(el.station)
    band_name = {b: f"{b[0]:g}-{b[1]:g}" for b in BANDS}
    checks = []

    def check(name, value, ok=None, of=None, note=""):
        checks.append(dict(check=name, value=value, of=of, ok=ok, note=note))

    check("self_check_synthetic_scatterer_recovered_at_its_cell", 1, True, 1, "asserts in self_check()")
    res = {b: psf(XY, *b) for b in BANDS}

    # --- 1. sources
    tele = a6.usgs(cache6, "tele_m6")
    tele["deg"] = [locations2degrees(a, o, la0, lo0) for a, o in zip(tele.latitude, tele.longitude)]
    cand = tele[(tele.deg >= DEG[0]) & (tele.deg <= DEG[1]) & (tele.depth >= DEEP_KM) & (tele.mag >= MAG[0]) & (tele.mag <= MAG[1])]
    dt = (tele.t.to_numpy()[None, :] - cand.t.to_numpy()[:, None]) / np.timedelta64(1, "m")  # minutes, other minus candidate
    alone = ((dt >= -ISOLATED_MIN[0]) & (dt <= ISOLATED_MIN[1])).sum(axis=1) == 1  # only itself
    n_rule = dict(deep_mag_dist=len(cand), isolated=int(alone.sum()))
    cand = cand[alone].sort_values(["mag", "t"], ascending=[False, True], kind="stable")
    dprk = a6.usgs(cache6, "dprk", starttime="2016-01-01", endtime="2017-12-31", minlatitude=40, maxlatitude=43,
                   minlongitude=127, maxlongitude=131, eventtype="nuclear explosion")
    src = []
    for kind, cat in (("dprk", dprk), ("teleseism", cand)):
        for _, q in cat.iterrows():
            if kind == "teleseism" and sum(s["kind"] == "teleseism" for s in src) >= N_CAND:
                break
            o, deg = T(q.time), locations2degrees(q.latitude, q.longitude, la0, lo0)
            depth = 0.0 if kind == "dprk" else float(q.depth)
            arr = {}
            for x in reversed(taup.get_travel_times(depth, deg, ["P", "pP", "PcP"])):
                arr[x.name] = x
            if "P" not in arr or (kind == "teleseism" and ("pP" not in arr or arr["pP"].time - arr["P"].time < PP_MIN_S)):
                continue
            baz = a6.GEOD.inv(lo0, la0, q.longitude, q.latitude)[0] % 360
            src.append(dict(event=f"{kind}_{o.date}" if kind == "dprk" else f"tele_{o.strftime('%Y%m%dT%H%M%S')}", kind=kind,
                            time=str(o), lat=q.latitude, lon=q.longitude, depth_km=depth, mag=q.mag, dist_deg=deg, baz_deg=baz,
                            slow_skm=arr["P"].ray_param_sec_degree / 111.195, tp=o + arr["P"].time,
                            pp_minus_p_s=arr["pP"].time - arr["P"].time if "pP" in arr else np.nan,
                            pcp_minus_p_s=arr["PcP"].time - arr["P"].time if "PcP" in arr else np.nan))
    n_rule["fetched"] = sum(s["kind"] == "teleseism" for s in src)
    nt = int((a6.PRE + a6.POST) * FS)
    for s in src:
        tag, cache = (f"{s['event']}_ILAR", cache6) if s["kind"] == "dprk" else (f"{s['event']}", cache7)
        s["st"] = st = fetch(c, cache, tag, stations, s["tp"] - a6.PRE, s["tp"] + a6.POST)
        s["n_data"], s["outcome"] = 0, "no_data"
        if st is None:
            continue
        snr = {r.get("band_hz"): r for r in a6.snr_rows("ILAR", s["event"], st, s["tp"], len(el), {})}
        s["n_data"] = int(snr[next(iter(snr))]["n_data"])
        if s["n_data"] < MIN_ELEM:
            s["outcome"] = "few_elements"
            continue
        s["snr"] = snr
        s["t_ref"] = a6.PRE + snr["0.5-1"]["t_pk_minus_pred_s"]
        s["outcome"] = "used" if s["kind"] == "dprk" or snr["0.5-1"]["p_snr"] >= SNR_MIN else "low_snr"
    passed = sorted((s for s in src if s["kind"] == "teleseism" and s["outcome"] == "used"),
                    key=lambda s: -s["snr"]["0.5-1"]["p_snr"])
    for s in passed[N_TELE:]:
        s["outcome"] = "passed_not_among_best"
    used = [s for s in src if s["outcome"] == "used"]
    for i, s in enumerate(used):  # source groups, in list order
        s["group"] = next((u["group"] for u in used[:i] if locations2degrees(s["lat"], s["lon"], u["lat"], u["lon"]) < GROUP_DEG), i)
    check("teleseisms_passing_source_rule", n_rule["isolated"], None, n_rule["deep_mag_dist"],
          f"deep >= {DEEP_KM:g} km, M {MAG[0]}-{MAG[1]}, {DEG[0]:g}-{DEG[1]:g} deg; of = before the isolation rule")
    check("teleseisms_fetched", n_rule["fetched"], None, N_CAND, "largest first, after the pP rule")
    check("events_used", len(used), len(used) >= 4, None, "three DPRK tests plus teleseisms; source groups: "
          f"{len({s['group'] for s in used})}")

    # --- 3-6. beam removal, maps, nulls, kill criterion
    ev_rows, maps = [], {}
    for s in src:
        base = {k: s.get(k) for k in ("event", "kind", "time", "lat", "lon", "depth_km", "mag", "dist_deg", "baz_deg",
                                      "slow_skm", "pp_minus_p_s", "pcp_minus_p_s", "n_data", "outcome", "group")}
        if s["outcome"] != "used":
            ev_rows.append(base | ({"p_snr": s["snr"]["0.5-1"]["p_snr"], "band_hz": "0.5-1"} if "snr" in s else {}))
            continue
        for b in BANDS:
            X, ok_el = matrix(a6.band(s["st"], *b), stations, s["tp"] - a6.PRE, nt)
            q = s["snr"][band_name[b]]
            ev = event_maps(X, XY[ok_el], slow_vec(s["slow_skm"], s["baz_deg"]), s["t_ref"], q["p_snr"] >= REFINE_SNR, rng)
            ev["group"] = s["group"]
            maps[b, s["event"]] = ev
            mx, iu = cell_max(ev["S"], ev["ok"])
            g = int(mx.argmax())
            null_max = max(float(m[ev["ok"]].max()) for m in ev["nulls"])
            ev["pk"] = peaks(ev["S"], ev["ok"], ev["thr"])
            cm, cu = cell_max(ev["pwin"], ev["ok_pwin"])
            gc = int(cm.argmax())
            lm, lu = cell_max(ev["S"], ev["ok_late"])
            gl = int(lm.argmax())
            late_null = max(float(m[ev["ok_late"]].max()) for m in ev["nulls"])
            ev_rows.append(base | dict(
                band_hz=band_name[b], n_elem_used=len(X), p_snr=q["p_snr"], coda_snr_5_30=q["coda_snr_5_30"],
                beam_refined=bool(q["p_snr"] >= REFINE_SNR), p_window_residual_frac=ev["resid_frac"],
                peak_semblance=float(mx[g]), peak_range_km=float(RANGE[g]),
                peak_bearing_deg=float(np.rad2deg(np.arctan2(*CELLS[g])) % 360), peak_u_kms=float(1 / P_SCAN[iu[g]]),
                thr_noise_p99=ev["thr_noise"], thr_rand_p99=ev["thr_rand"], null_threshold=ev["thr"], null_max=null_max,
                exceed_frac=float((ev["S"][ev["ok"]] > ev["thr"]).mean()), peak_above_null_max=bool(mx[g] > null_max),
                peak_lag_s=float(ev["lag"][g, iu[g]]), pwin_peak_semblance=float(cm[gc]), pwin_peak_range_km=float(RANGE[gc]),
                pwin_peak_bearing_deg=float(np.rad2deg(np.arctan2(*CELLS[gc])) % 360), pwin_peak_u_kms=float(1 / P_SCAN[cu[gc]]),
                peak_in_pwin_cell=bool(same_cell(CELLS[gc], CELLS[g], res[b])[0]),
                late_peak_semblance=float(lm[gl]), late_peak_range_km=float(RANGE[gl]),
                late_peak_bearing_deg=float(np.rad2deg(np.arctan2(*CELLS[gl])) % 360), late_peak_u_kms=float(1 / P_SCAN[lu[gl]]),
                late_null_max=late_null, late_peak_above_null_max=bool(lm[gl] > late_null),
                n_peaks=len(ev["pk"]), lag_max_s=float((ev["lead"][:, None] + RANGE[:, None] * P_SCAN)[ev["ok"]].max())))
        s["st"] = None
    events = pd.DataFrame(ev_rows)
    n_null = len(NOISE_REF) + N_RAND
    killed = {}
    for b in BANDS:
        e = events[(events.band_hz == band_name[b]) & (events.outcome == "used")]
        k, n = int(e.peak_above_null_max.sum()), len(e)
        killed[b] = k == 0
        check(f"kill_{band_name[b]}_events_with_peak_above_null_max", k, k > 0, n,
              f"kill fires at 0; chance expects {n / (n_null + 1):.1f}; P(>= {k} by chance) = {binom.sf(k - 1, n, 1 / (n_null + 1)):.3g}")
        check(f"kill_{band_name[b]}_fired", int(killed[b]), not killed[b])
        check(f"exceed_frac_median_{band_name[b]}", float(e.exceed_frac.median()), None, 1 - Q_NULL / 100,
              "share of cell-windows above the threshold; of = the null's share by construction")
        kl = int(e.late_peak_above_null_max.sum())
        check(f"late_{band_name[b]}_events_with_peak_above_null_max", kl, None, n,
              f"windows at lag >= {LATE_LAG:g} s only; chance expects {n / (n_null + 1):.1f}; "
              f"P(>= {kl} by chance) = {binom.sf(kl - 1, n, 1 / (n_null + 1)):.3g}")
        check(f"peak_in_pwin_cell_{band_name[b]}", int(e.peak_in_pwin_cell.sum()), None, n,
              "events whose coda peak lies in the resolution cell of the P-window control's peak")
        check(f"lag_max_s_{band_name[b]}", float(e.lag_max_s.max()), None, LAG[1],
              "latest window that maps inside the grid; later windows up to `of` map beyond it")

    # --- 9. calibration on cataloged blasts
    bl = a6.usgs(cache6, "explosions")
    az, _, d = a6.GEOD.inv(np.full(len(bl), lo0), np.full(len(bl), la0), bl.longitude.to_numpy(), bl.latitude.to_numpy())
    bl = bl.assign(r=d / 1000.0, az=az % 360)
    bl = bl[(bl.r >= BLAST_KM[0]) & (bl.r < BLAST_KM[1])]
    bl = bl.assign(az_bin=(bl.az // 45).astype(int), r_bin=(bl.r // 10).astype(int))
    bl = bl.sort_values(["mag", "t"], ascending=[False, True], kind="stable").groupby(["az_bin", "r_bin"]).head(1).sort_values("t")
    nb = int((BLAST_PRE + BLAST_POST) * FS)
    zero = np.zeros(len(CELLS))
    ok_b = valid(zero, (-WIN / 2, np.inf))
    cal = []
    for _, q in bl.iterrows():
        o = T(q.time)
        name = f"blast_{o.strftime('%Y%m%dT%H%M%S')}"
        st = fetch(c, cache7, name, stations, o - BLAST_PRE, o + BLAST_POST)
        true = q.r * np.array([np.sin(np.deg2rad(q.az)), np.cos(np.deg2rad(q.az))])
        for b in BANDS:
            row = dict(event=name, band_hz=band_name[b], time=str(o), mag=q.mag, cat_range_km=q.r, cat_bearing_deg=q.az,
                       cat_x_km=true[0], cat_y_km=true[1], n_elem_used=0, coherent=False)
            cal.append(row)
            if st is None:
                continue
            X, ok_el = matrix(a6.band(st, *b), stations, o - BLAST_PRE, nb)
            if len(X) < MIN_ELEM:
                continue
            tt = times(XY[ok_el], zero)
            S, N = image(X, BLAST_PRE, tt), image(X, BLAST_NOISE_REF, tt)
            mx, iu = cell_max(S, ok_b)
            g = int(mx.argmax())
            w = slice(int((BLAST_PRE + q.r / 3.2) * FS), int((BLAST_PRE + q.r / 1.5) * FS))
            mp, _ = cell_max(image(X, BLAST_PRE, tt, power=True), ok_b)
            gp = int(mp.argmax())
            hr, hc = extent(mp, gp)
            dxy = CELLS[g] - true
            row |= dict(n_elem_used=len(X), rg_snr=float(np.median(X[:, w].std(axis=1) / X[:, int(5 * FS):int(50 * FS)].std(axis=1))),
                        peak_semblance=float(mx[g]), noise_p99=float(np.percentile(N[ok_b], Q_NULL)), noise_max=float(N[ok_b].max()),
                        coherent=bool(mx[g] > N[ok_b].max()), x_km=CELLS[g, 0], y_km=CELLS[g, 1], u_kms=float(1 / P_SCAN[iu[g]]),
                        err_km=float(np.hypot(*dxy)), range_err_km=float(RANGE[g] - q.r),
                        bearing_err_deg=float((np.rad2deg(np.arctan2(*CELLS[g])) - q.az + 180) % 360 - 180),
                        power_peak_err_km=float(np.hypot(*(CELLS[gp] - true))), cell_radial_km=hr, cell_cross_km=hc,
                        psf_radial_km=float(np.interp(q.r, res[b]["range_km"], res[b]["radial"])),
                        psf_cross_km=float(np.interp(q.r, res[b]["range_km"], res[b]["cross"])))
            row["within_one_cell"] = bool(same_cell(true, CELLS[g], res[b])[0])
    cal = pd.DataFrame(cal)
    for b in BANDS:
        q = cal[(cal.band_hz == band_name[b]) & cal.coherent]
        check(f"blasts_coherent_{band_name[b]}", len(q), None, int((cal.band_hz == band_name[b]).sum()),
              "peak semblance above the largest pre-origin noise value")
        check(f"blast_err_km_median_{band_name[b]}", float(q.err_km.median()) if len(q) else np.nan, None, None, "coherent blasts")
        check(f"blast_err_km_max_{band_name[b]}", float(q.err_km.max()) if len(q) else np.nan, None, None, "coherent blasts")
        check(f"blasts_within_one_cell_{band_name[b]}", int(q.within_one_cell.sum()) if len(q) else 0, None, len(q),
              "catalog location inside the synthetic resolution cell around the image peak")

    # --- 7. stacking, with the same rule on the null maps
    scat = []
    rules = {"design": lambda pk: pk, "strongest_peak": lambda pk: sorted(pk, key=lambda q: -q[1])[:1]}
    for b in BANDS:
        if killed[b]:
            check(f"scatterers_design_{band_name[b]}", 0, None, None, "kill criterion fired: not stacked")
            continue
        evs = {n: ev for (bb, n), ev in maps.items() if bb == b}

        def table(get):
            return pd.DataFrame([dict(event=n, group=ev["group"], g=g, sem=sem, u=u, thr=ev["thr"], lag=ev["lead"][g] + RANGE[g] / u)
                                 for n, ev in evs.items() for g, sem, u in get(ev)],
                                columns=["event", "group", "g", "sem", "u", "thr", "lag"])
        for rule, pick in rules.items():
            found = stack(table(lambda ev: pick(ev["pk"])), res[b])
            null_pk = [table(lambda ev, k=k: pick(peaks(ev["nulls"][k], ev["ok"], ev["thr"]))) for k in range(n_null)]
            null = [stack(q, res[b]) for q in null_pk]

            def groups_here(sets, xy0):  # most source groups, over null sets, with a null peak in the cell around xy0
                return max(int(q["group"][same_cell(xy0, CELLS[q.g.to_numpy()], res[b])].nunique()) if len(q) else 0 for q in sets)
            null_n = [int((q.n_groups >= 2).sum()) if len(q) else 0 for q in null]
            null_top = max([int(q.n_groups.max()) if len(q) else 0 for q in null])
            found = found[found.n_groups >= 2] if len(found) else found
            if len(found):
                at = found[["x_km", "y_km"]].to_numpy(float)
                found = found.assign(band_hz=band_name[b], rule=rule, n_groups_null_max=null_top, above_null=found.n_groups > null_top,
                                     noise_groups_same_cell=[groups_here(null_pk[:len(NOISE_REF)], q) for q in at],
                                     rand_groups_same_cell=[groups_here(null_pk[len(NOISE_REF):], q) for q in at],
                                     res_radial_km=np.interp(found.range_km, res[b]["range_km"], res[b]["radial"]),
                                     res_cross_km=np.interp(found.range_km, res[b]["range_km"], res[b]["cross"]))
                scat.append(found)
            check(f"scatterers_{rule}_{band_name[b]}", len(found), None, None, "clusters with peaks from >= 2 source groups")
            check(f"scatterers_{rule}_on_null_maps_{band_name[b]}", float(np.mean(null_n)), None, None,
                  f"same rule, mean over {n_null} null sets; per set: {null_n}")
            check(f"scatterers_{rule}_above_null_{band_name[b]}", int(found.above_null.sum()) if len(found) else 0, None, None,
                  f"more source groups than any null cluster ({null_top})")
    cols = ["band_hz", "rule", "x_km", "y_km", "bearing_deg", "range_km", "n_events", "n_groups", "peak_semblance", "null_threshold",
            "u_kms", "lag_s", "n_groups_null_max", "above_null", "noise_groups_same_cell", "rand_groups_same_cell", "res_radial_km", "res_cross_km", "relief_m", "relief_pctile",
            "strongest_event", "events"]
    scat = (pd.concat(scat, ignore_index=True) if scat else pd.DataFrame()).reindex(columns=cols)

    # --- 11. relief within the cell of each scatterer
    if len(scat):
        ax, z = elevation(cache7, la0, lo0)
        ig = np.rint((CELLS - ax[0]) / DEM_STEP).astype(int)
        rad = np.sqrt(scat.res_radial_km * scat.res_cross_km).round()
        for r in rad.unique():
            rel = relief(ax, z, r)[ig[:, 1], ig[:, 0]]  # at every grid cell
            for i in scat.index[rad == r]:
                v = rel[int(np.argmin(np.hypot(*(CELLS - scat.loc[i, ["x_km", "y_km"]].to_numpy(float)).T)))]
                scat.loc[i, "relief_m"], scat.loc[i, "relief_pctile"] = float(v), float((rel < v).mean() * 100)

    # --- 10. sensitivity
    sens = []
    for b in BANDS:
        tr = pd.DataFrame([r for (bb, n), ev in maps.items() if bb == b
                           for r in sensitivity(n, ev, res[b], rng, N_AZ_DPRK if n.startswith("dprk") else N_AZ_TELE)])
        sets = {"all": tr, "teleseisms": tr[tr.event.str.startswith("tele")], "dprk_2017-09-03": tr[tr.event == "dprk_2017-09-03"]}
        for label_, q in sets.items():
            for r, qr in q.groupby("range_km"):
                agg = qr.groupby("ratio").agg(n_trials=("glob", "size"), recovery_rate=("glob", "mean"), recovery_rate_local=("loc", "mean"))
                sens.append(agg.reset_index().assign(
                    band_hz=band_name[b], range_km=r, events=label_, min_ratio_95=min_ratio(agg.recovery_rate),
                    min_ratio_95_local=min_ratio(agg.recovery_rate_local),
                    res_radial_km=float(np.interp(r, res[b]["range_km"], res[b]["radial"])),
                    res_cross_km=float(np.interp(r, res[b]["range_km"], res[b]["cross"]))))
    sens = pd.concat(sens, ignore_index=True)
    sens = sens[["band_hz", "range_km", "events", "ratio", "n_trials", "recovery_rate", "recovery_rate_local", "min_ratio_95",
                 "min_ratio_95_local", "res_radial_km", "res_cross_km"]]

    size = sum(p.stat().st_size for p in data_glob(cache7, "*") if p.is_file()) + \
        sum(p.stat().st_size for p in data_glob(cache6, "dprk_*_ILAR.mseed"))
    check("fetch_bytes", size, size < 1e9, 1e9, "a7 cache plus the three DPRK windows reused from a6")

    fmt = dict(index=False, float_format="%.6g")
    events.to_csv(HERE / "a7_events.csv", **fmt)
    cal.to_csv(HERE / "a7_blast_calibration.csv", **fmt)
    scat.to_csv(HERE / "a7_scatterers.csv", **fmt)
    sens.to_csv(HERE / "a7_sensitivity.csv", **fmt)
    pd.DataFrame(checks).to_csv(HERE / "a7_checks.csv", **fmt)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400)
    for name, df in (("events", events), ("blast calibration", cal), ("scatterers", scat),
                     ("sensitivity (ratio 0 rows carry min_ratio_95)", sens[sens.ratio == 0]), ("checks", pd.DataFrame(checks))):
        print(f"\n== {name}")
        print(df.drop(columns=["time", "lat", "lon", "events"], errors="ignore").to_string(index=False))


def self_check() -> None:
    # a wave from the west (back-azimuth 270) travels east: an element to the east is hit later
    assert np.allclose(slow_vec(0.07, 270.0), [0.07, 0.0], atol=1e-12)
    el = pd.read_csv(HERE / "a6_array_precheck.csv").query("section == 'element' and array == 'ILAR'")
    xy = np.c_[a6.local_xy(el.lat, el.lon)]
    rng = np.random.default_rng(1)
    nt, t_ref, lo, hi = int(420 * FS), 120.0, *BANDS[0]
    s_vec = slow_vec(0.07, 285.0)
    base = pulse(lo, hi, nt, int(t_ref * FS))
    g = int(np.argmin(np.hypot(CELLS[:, 0] + 18.0, CELLS[:, 1] - 12.0)))  # scatterer 18 km west, 12 km north
    lead = CELLS @ s_vec
    rg = scattered(base, xy, CELLS[g], P_SCAN[5], lead[g])
    X = shift(np.tile(base, (len(xy), 1)), xy @ s_vec + rng.normal(0, 0.03, len(xy))) * rng.uniform(0.8, 1.25, len(xy))[:, None] \
        + 0.3 * rg + rng.normal(0, 0.01, (len(xy), nt))  # P with statics and site gains, Rg at 0.3 of P, noise
    ev = event_maps(X, xy, s_vec, t_ref, True, rng)
    assert ev["resid_frac"] < 0.05, ev["resid_frac"]  # the P beam is removed
    mx, iu = cell_max(ev["S"], ev["ok"])
    assert int(mx.argmax()) == g and iu[g] == 5 and mx[g] > 0.9, (CELLS[int(mx.argmax())], iu[g], mx[g])  # recovered at its cell
    assert mx[g] > ev["thr"] and max(m[ev["ok"]].max() for m in ev["nulls"]) < mx[g]
    assert any(p[0] == g for p in peaks(ev["S"], ev["ok"], ev["thr"]))
    # the injection algebra equals a direct map of data plus synthetic
    Gd, Bd, a, e = data_terms(ev)
    g2 = int(np.argmin(np.hypot(CELLS[:, 0] - 20.0, CELLS[:, 1] + 25.0)))
    syn = scattered(ev["wave"], xy, CELLS[g2], 0.45, lead[g2])
    b, c, x, f = inject_terms(ev, Gd, Bd, syn)
    A = 0.4
    direct = image(residual(X + A * syn, ev["d"], ev["a"])[0], t_ref, times(xy, lead))
    assert np.abs(direct - (a + 2 * A * b + A * A * c) / (len(xy) * (e + 2 * A * x + A * A * f) + TINY))[ev["ok"]].max() < 1e-3
    assert cell_max(direct, ev["ok"])[0][g2] > 0.8 > cell_max(ev["S"], ev["ok"])[0][g2]  # and the injection shows at its cell
    # an ordinary source with a known origin time (blast mode) is located at its cell
    zero = np.zeros(len(CELLS))
    S = image(scattered(base, xy, CELLS[g], P_SCAN[8], 0.0), t_ref, times(xy, zero))
    assert int(cell_max(S, valid(zero, (-WIN / 2, np.inf)))[0].argmax()) == g
    assert min_ratio(pd.Series([0.0, 0.96, 0.5, 1.0, 1.0], index=[0, 1, 2, 3, 4])) == 3.0


if __name__ == "__main__":
    self_check()
    main()
