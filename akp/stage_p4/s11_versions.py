"""P4 S11: the stations nearest each version of the claim.

Question: for each place a version of the claim states, which station that recorded the 1992
explosion stood nearest, which operating station with no reading stood nearest, did any station
stand inside the stated area, how much of the area lies within reach of a station, and how large
could the scattered wave be at the nearest station outside it?

Rules implemented here, none changed:
- A committed table (a3c_spacing.csv) gave a distance from the belt around Denali to a
  station that was the gap to a sampling grid. This module recomputes the distances. The old
  module and its table are not edited.
- The versions, the stations, the distances, the columns, the two ranges, the bound and
  the self-check, as written there.
- A bound may be set against the measured coda level only where the bounded
  wave would arrive inside the measured windows, and the comparison comes from committed code.

Superseded columns: the bound columns of s11_versions.csv (*_bound_spreading, *_bound_attenuated, the delays) and
the whole of s11_bound_against_coda.csv are no longer used by the paper. They are still written, unchanged in rule, because
akp/stage_a/a5_late_windows.py takes from them which pairs of version and station carry a bound. The paper's bounds are
those of a5_late_readings.csv.

Inputs:
- akp/zones.py: the anchors (Denali, Nome, Valdez) and the two boxes.
- akp/stage_a/a1_stations.csv: one row per station code. Recorded: evidence is recorded_test.
  Operating: evidence is operating_at_test (readings of other events within three days before
  and after the explosion, and no reading of it). Names: reg_name. Open waveform: waveforms_open.
- akp/stage_a/a8_smoothing.csv (the first four simulated cases) and a8_generous_shares.csv (the
  four later cases): the amplitude ratio by case, azimuth, range and band.
- akp/stage_a/a5_dprk_site_spread.csv, through a8_smoothing.clutter: the median coda level by band.
  The same table (box mckinley_nome, event all) gives the windows after P (column lag: 5-15,
  15-30, 30-60 s) with the median and the fifth percentile (p05) of each band and window.
- a8_attenuation.factor, CENTRE_HZ and Q_VALUES; the Rayleigh speeds of a8_attenuation.csv and
  a8_generous_attenuation.csv.
- a3c_spacing.csv, a1_line_coverage.csv: the old values, for the checks only.

Method:
- Places. PLACES holds one entry per stated place with a committed construction. NOT_PLACED
  holds the stated places with no committed coordinates. No coordinate is invented here.
- Distances are geodesic on WGS84 (pyproj.Geod). A place is a point, a line or an area, each
  given by exact boundary curves. The distance from a station to a curve is found by a bounded
  minimization along the curve, started from the best of N_CURVE points. It is zero if the
  station stands inside the area.
- Inside. A disc, a belt or a part of a belt: from the station's own distance and bearing from
  Denali. A box: from the station's latitude and longitude (zones.in_box). Never from samples of
  the area. An edge counts as inside, to TOL_KM and TOL_DEG. The north-west part runs from
  bearing 270 to bearing 360, both included.
- Amplitude ratio. The larger of the table's two measures (share, and share_smoothed, which is
  measured as the coda is). Edge-wave flag: the committed column for the later cases. The first
  four cases have no committed flag per row. It is computed here by the rule of a8_generous
  (edge_delay less than edge_needed), by import.
- The two ranges. The largest range at which the ratio is at or above the median coda level of
  its band, and the last simulated receiver, both over every case, band and azimuth, flagged
  rows included.
- Shares. A regular grid in EPSG:3338 (equal area) with a cell of a fifth of the smaller range.
  Only cells within the range of a station are built. A cell counts if its center is within the
  range of a station and inside the area. The area itself is the geodesic area of its boundary.
  For a point the share is 1 or 0. For a line it is 0 when no station is within the range, and
  left empty otherwise.
- The bound. The largest ratio at the last receiver with no edge-wave flag, over cases, bands
  and azimuths, times the square root of that range over the station's distance. Beside it: the
  same times the attenuation factor at the largest Q of the attenuation table, over the whole
  distance, at the center frequency of the band and the Rayleigh speed of the ground of the case
  that sets it; and the delay, distance over that speed. One more column is not: the
  largest attenuated value over every case and band, since a lower band is attenuated less. It
  leaves out the frozen cases, which have no Q. No bound is given for a station nearer than the
  last receiver: the wave is carried outward only.

- Bound against the coda (s11_bound_against_coda.csv). One row per version, kind of station and
  simulated band where a bound exists, and per window. A window is closed at both ends, as
  a2_wavefield cuts it, so a delay on a shared edge is in both windows. Delays are distance over
  a Rayleigh speed and are counted from P; the windows count from the P peak. Three delays
  decide the windows: the existing one (the speed of the case that sets the fixed bound); the
  earliest (the fastest tabled speed among the cases that hold the band, flagged or not); and
  each case's own (its own ground; a slower ground arrives later). window_by says which of them
  fall in the row's window; a row that only slower grounds reach is marked. A delay outside the
  windows gets a row with no comparison. Inside a window the row holds: the fixed bound by
  spreading alone (the largest over all bands); the largest by spreading in that band; the
  largest with attenuation in that band; the window's median and p05 in that band; and whether
  each bound is below each level. The bounds of a band are the largest over the cases that have
  the band, a last unflagged receiver and a tabled speed, made as the any-band column is made:
  ratio, times the square root of its range over the distance, times the factor at the band's
  center frequency and the case's own speed. The largest over bands is the any-band column
  (asserted). The frozen cases have no tabled speed; none would set a band's spreading bound
  (asserted). In a row that only slower grounds reach, the bounds still include the faster
  cases, so "not below" there is on the safe side.

Outputs, beside this file: s11_bound_against_coda.csv (above); s11_versions.csv (one row per version), s11_stations_inside.csv (one
row per station inside an area), s11_ranges.csv (the two ranges, the start of the bound, and the
last unflagged receiver of each case and band), s11_checks.csv (counts, agreement with the old
table, each difference from it with its cause).

Run: PYTHONPATH=. conda run -n swe-akp python -m akp.stage_p4.s11_versions

Self-check, asserted on every run. It proves: a station on an edge of a belt, of its north-west
part or of a box counts as inside, and one a meter outside does not; a station inside a made-up
belt, between the samples the old script would have taken, is at distance zero; the distance to
a belt equals the difference of radii; the area of a disc is right, by its boundary and by the
grid; the edge-wave rule, called as here, gives the committed flags of the later cases; the old
table's values for the most specific point and for the peak come out to their printed precision;
a delay on a window's edge is in every window whose label contains it, and one outside is named
before or after; the Nome end of the line falls in the first window.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod, Transformer
from scipy.optimize import minimize_scalar

from akp.stage_a import a8_generous as gen
from akp.stage_a import a8_production as prod
from akp.stage_a.a8_attenuation import CENTER_HZ, Q_VALUES, factor
from akp.stage_a.a8_smoothing import clutter
from akp.zones import ANCHORS, BOXES, in_box

HERE = Path(__file__).parent
A = HERE.parent / "stage_a"
G = Geod(ellps="WGS84")
TR = Transformer.from_crs("EPSG:4326", "EPSG:3338", always_xy=True)
MI = 1.609344  # km in a statute mile
TOL_KM, TOL_DEG = 1e-6, 1e-9  # an edge counts as inside: one millimeter, and the same in bearing
N_CURVE = 721  # points that start the search along a boundary curve (and 5 times as many trace it for the area)
CELL_PER_RANGE = 5  # "fine against those ranges": cells across the smaller range
Q_GENEROUS = max(Q_VALUES)  # the attenuation table's generous value
KINDS = {"recorded_test": "recorded", "operating_at_test": "operating"}
DEN = ANCHORS["denali"]


def inv(lat1, lon1, lat2, lon2):
    """(km, bearing in [0, 360)) from point 1 to point 2, as arrays."""
    a = np.broadcast_arrays(*(np.atleast_1d(np.asarray(v, float)) for v in (lon1, lat1, lon2, lat2)))
    az, _, m = G.inv(*(np.ascontiguousarray(v) for v in a))
    return m / 1e3, az % 360.0


def fwd(bearing, km):
    """(lon, lat) at a bearing and a distance from the Denali anchor."""
    b, d = (np.ascontiguousarray(v) for v in np.broadcast_arrays(np.atleast_1d(np.asarray(bearing, float)), np.atleast_1d(np.asarray(km, float))))
    lon, lat, _ = G.fwd(np.full(b.shape, DEN[1]), np.full(b.shape, DEN[0]), b, d * 1e3)
    return lon, lat


AZ_NOME = float(inv(*DEN, *ANCHORS["nome"])[1][0])
LINE_KM = float(inv(*DEN, *ANCHORS["nome"])[0][0])
# Three places the register names without coordinates. Positions (lat, lon) of the named features, from the
# Geographic Names Information System as mirrored on the web, read 2026-11-16; akp/zones.py is frozen, so they stand here.
NAMED = dict(
    unalakleet=(63.8789, -160.7897),  # the city of Unalakleet (GNIS 1411517)
    carey_lake=(63.4067, -152.6030),  # Carey Lake, 34 miles south-south-west of Lake Minchumina (GNIS 1399952)
    haarp=(62.3917, -145.1500),  # the antenna array of the research facility near Gakona
    # Places of versions found by the census of 2026-11-19 (Supplement One, section S1.12); GNIS points unless stated
    coords_c12=(63 + 17 / 60 + 51 / 3600, -(152 + 31 / 60 + 24 / 3600)),  # the coordinates C12 states, 63 17 51 N 152 31 24 W
    farewell_lake=(62.554720, -153.639280),  # Farewell Lake (GNIS 1402002), for C13's "Farewell base"
    telida=(63.383890, -153.282247),  # Telida (GNIS 1410727), C14
    kaltag=(64.327222, -158.721969),  # Kaltag (GNIS 1404379), C14
    charley_mouth=(65.317225, -142.780582),  # the mouth of the Charley River on the Yukon (GNIS 1400160), for C15's "Yukon-Charley"
    fort_richardson=(61.254446, -149.688356),  # Fort Richardson (GNIS 1402275), C16
    mcgrath=(62.956389, -155.595857),  # McGrath (GNIS 1406131), C17
)
SUSITNA_VALLEY = [(61.359677, -150.579976), (61.599142, -150.455753), (61.707382, -150.285463), (61.856722, -150.186584), (62.127572, -150.120664),
                  (62.382022, -150.156374), (62.550892, -150.095954), (62.688572, -149.862494), (62.789212, -149.634524), (62.809302, -149.255494),
                  (62.796752, -148.678714), (62.831882, -148.269474), (62.776652, -147.887694), (62.701360, -147.623282)]  # Susitna Valley (GNIS 2791579), C18
AZ_UNK = float(inv(*DEN, *NAMED["unalakleet"])[1][0])
UNK_KM = float(inv(*DEN, *NAMED["unalakleet"])[0][0])


# ---------------------------------------------------------------- places: exact boundary curves, t in [0, 1] -> (lon, lat)
def arc(r, b0, b1):
    return lambda t: fwd(b0 + (b1 - b0) * t, r)


def ray(b, r0, r1):
    return lambda t: fwd(b, r0 + (r1 - r0) * t)


def edge(p, q):
    return lambda t: (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)


def point(lat, lon):
    return dict(kind="point", rings=[[edge((lon, lat), (lon, lat))]])


def line(r0, r1):
    """The part of the geodesic from Denali to Nome between two distances from Denali (km)."""
    return dict(kind="line", rings=[[ray(AZ_NOME, r0, r1)]])


def belt(r0, r1, b0=0.0, b1=360.0):
    """The points r0 to r1 km from Denali, between bearings b0 and b1 (all bearings by default). Edges are inside."""
    full = b1 - b0 >= 360.0

    def inside(lat, lon):
        d, b = inv(*DEN, lat, lon)
        turn = (b - b0) % 360.0
        return (d >= r0 - TOL_KM) & (d <= r1 + TOL_KM) & (full | (turn <= b1 - b0 + TOL_DEG) | (turn >= 360.0 - TOL_DEG))
    if full:
        rings = [[arc(r1, 0.0, 360.0)]] + ([[arc(r0, 0.0, 360.0)]] if r0 > 0 else [])
    else:
        rings = [[arc(r1, b0, b1), ray(b1, r1, r0), arc(r0, b1, b0), ray(b0, r0, r1)]]
    return dict(kind="area", inside=inside, rings=rings)


def box(b):
    w, e, s, n = b["minlongitude"], b["maxlongitude"], b["minlatitude"], b["maxlatitude"]
    c = [(w, s), (e, s), (e, n), (w, n), (w, s)]
    return dict(kind="area", inside=np.vectorize(lambda la, lo: in_box(la, lo, b), otypes=[bool]),
                rings=[[edge(p, q) for p, q in zip(c, c[1:])]])


def to_curve(lat, lon, curve):
    """km from a point to the nearest point of a curve."""
    t = np.linspace(0.0, 1.0, N_CURVE)
    lo, la = curve(t)
    d = inv(lat, lon, la, lo)[0]
    i = int(np.argmin(d))

    def f(x):
        lo, la = curve(np.array([x]))
        return float(inv(lat, lon, la, lo)[0][0])
    best = minimize_scalar(f, bounds=(t[max(i - 1, 0)], t[min(i + 1, N_CURVE - 1)]), method="bounded", options={"xatol": 1e-12})
    return min(float(best.fun), float(d[i]))


def km_to(p, lat, lon):
    """km from a station to the nearest point of a place; zero if it stands inside."""
    if "inside" in p and p["inside"](lat, lon).all():
        return 0.0
    return min(to_curve(lat, lon, c) for ring in p["rings"] for c in ring)


def area_km2(p):
    """Geodesic area inside the first ring, less the area inside the others."""
    t = np.linspace(0.0, 1.0, 5 * N_CURVE)
    a = [abs(G.polygon_area_perimeter(*np.concatenate([c(t) for c in ring], axis=1))[0]) for ring in p["rings"]]
    return (a[0] - sum(a[1:])) / 1e6


def cells(lat, lon, r_km, cell):
    """Cells of the regular EPSG:3338 grid (origin 0, 0) with centers within r_km of a point: (key, lat, lon)."""
    x, y = TR.transform(lon, lat)
    n = int(np.ceil(1.1 * r_km * 1e3 / cell)) + 2  # the projection's scale is within a few percent of one
    di, dj = (v.ravel() for v in np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1)))
    i, j = di + int(x // cell), dj + int(y // cell)
    lo, la = TR.transform((i + 0.5) * cell, (j + 0.5) * cell, direction="INVERSE")
    keep = inv(lat, lon, la, lo)[0] <= r_km
    assert j.min() >= 0 and j.max() < 10**7 and not keep[(abs(di) == n) | (abs(dj) == n)].any(), "window too small"
    return i[keep] * 10**7 + j[keep], la[keep], lo[keep]


R_C3 = (45.0 * MI, 55.0 * MI)
STATED = dict(  # the register's words for each place
    between="C1, C2; C3 (03:42): 'between Mt McKinley (Denali) and Nome'",
    c3="C3 (29:12-29:52): 'within 50 miles of Denali', a 45-55 mile belt, 'a little bit northwest of Denali'",
    c4="C4 (07:50): 'about 60 miles west of Mount McKinley ... on a straight-line bearing to Nome'",
    c4map="C4 (06:01), on a map: 'about 60 to 70 miles from Mount McKinley'",
    c7="C7 (00:56): 'About 60 miles west of Mount McKinley'",
    c9="C9: 'near Valdez' (a lead: recollection of TV coverage)",
    c7b="C7 (19:29, 20:23): 'east of Unalakleet and west of Mount Denali'",
    c9b="C9: Gakona variant (the HAARP site near Gakona); a lead",
    c10="C10: 'Carey Lake, northwest Denali National Park'; Reddit, 2024-11-16 (read from an archive capture)",
    c12="C12 (Before It's News, 2013-03-12): coordinates 63 17 51 N 152 31 24 W, 'about 50 miles' from Mount McKinley",
    c13="C13 (Coast to Coast AM, 2012-11-29): 'near McKinley', the 'Farewell base'",
    c14="C14 (Earthfiles podcast, 2024-02-28, show notes): Kaltag, Telida",
    c15="C15 (whatdoesitmean.com, 2012-12-10): 'Yukon-Charley'",
    c16="C16 (UFO Casebook, 2024-02-03): the Fort Richardson area",
    c17="C17 (Substack, 2024-07-03): McGrath",
    c18="C18 (Wondergressive, 2025-08-06): the Susitna valley")
PLACES = {
    "line_denali_nome": dict(ids="C1 C2 C3", stated=STATED["between"], anchor="denali",
                             construction="geodesic from the Denali anchor to the Nome anchor", **line(0.0, LINE_KM)),
    "box_denali_nome": dict(ids="C1 C2 C3 C4 C7", stated=STATED["between"] + "; the register's zone for C1-C4 and C7", anchor="denali",
                            construction="akp.zones.BOXES['mckinley_nome']", **box(BOXES["mckinley_nome"])),
    "disc_denali": dict(ids="C3", stated=STATED["c3"], anchor="denali",
                        construction="within 50 statute miles of the Denali anchor", **belt(0.0, 50.0 * MI)),
    "belt_denali": dict(ids="C3", stated=STATED["c3"], anchor="denali",
                        construction="45 to 55 statute miles from the Denali anchor", **belt(*R_C3)),
    "belt_denali_northwest": dict(ids="C3", stated=STATED["c3"], anchor="denali",
                                  construction="the belt between bearings 270 and 360 from the Denali anchor, both included (the quadrant of a3c_spacing)",
                                  **belt(*R_C3, 270.0, 360.0)),
    "point_c4": dict(ids="C4", stated=STATED["c4"], anchor="denali",
                     construction="60 statute miles from the Denali anchor along the geodesic to the Nome anchor",
                     **point(*[float(v[0]) for v in fwd(AZ_NOME, 60.0 * MI)][::-1])),
    "stretch_c4_map": dict(ids="C4", stated=STATED["c4map"], anchor="denali",
                           construction="the geodesic from Denali to Nome between 60 and 70 statute miles from the Denali anchor",
                           **line(60.0 * MI, 70.0 * MI)),
    "point_c7_west": dict(ids="C7", stated=STATED["c7"], anchor="denali",
                          construction="60 statute miles from the Denali anchor at bearing 270 (due west)",
                          **point(*[float(v[0]) for v in fwd(270.0, 60.0 * MI)][::-1])),
    "valdez_town": dict(ids="C9", stated=STATED["c9"], anchor="valdez", construction="the Valdez anchor", **point(*ANCHORS["valdez"])),
    "box_valdez": dict(ids="C9", stated=STATED["c9"], anchor="valdez", construction="akp.zones.BOXES['valdez']", **box(BOXES["valdez"])),
    "line_c7_unalakleet": dict(ids="C7", stated=STATED["c7b"], anchor="denali",
                               construction="geodesic from the Denali anchor to the city of Unalakleet: our construction for a stated region",
                               kind="line", rings=[[ray(AZ_UNK, 0.0, UNK_KM)]]),
    "point_c9_haarp": dict(ids="C9", stated=STATED["c9b"], anchor="valdez",
                           construction="the antenna array of the facility near Gakona", **point(*NAMED["haarp"])),
    "point_c10_carey_lake": dict(ids="C10", stated=STATED["c10"], anchor="denali",
                                 construction="the named lake", **point(*NAMED["carey_lake"])),
    # The places of versions found by the census; a named feature stands for a place stated by name
    "point_c12_coordinates": dict(ids="C12", stated=STATED["c12"], anchor="denali", construction="the coordinates as stated", **point(*NAMED["coords_c12"])),
    "point_c13_farewell": dict(ids="C13", stated=STATED["c13"], anchor="denali", construction="Farewell Lake (GNIS 1402002)", **point(*NAMED["farewell_lake"])),
    "point_c14_telida": dict(ids="C14", stated=STATED["c14"], anchor="denali", construction="Telida (GNIS 1410727)", **point(*NAMED["telida"])),
    "point_c14_kaltag": dict(ids="C14", stated=STATED["c14"], anchor="denali", construction="Kaltag (GNIS 1404379)", **point(*NAMED["kaltag"])),
    "point_c15_yukon_charley": dict(ids="C15", stated=STATED["c15"], anchor="denali",
                                    construction="the mouth of the Charley River on the Yukon (GNIS 1400160)", **point(*NAMED["charley_mouth"])),
    "point_c16_fort_richardson": dict(ids="C16", stated=STATED["c16"], anchor="denali", construction="Fort Richardson (GNIS 1402275)",
                                      **point(*NAMED["fort_richardson"])),
    "point_c17_mcgrath": dict(ids="C17", stated=STATED["c17"], anchor="denali", construction="McGrath (GNIS 1406131)", **point(*NAMED["mcgrath"])),
    "line_c18_susitna_valley": dict(ids="C18", stated=STATED["c18"], anchor="denali", construction="the line of the Susitna Valley (GNIS 2791579)",
                                    kind="line", rings=[[edge((lo0, la0), (lo1, la1)) for (la0, lo0), (la1, lo1) in zip(SUSITNA_VALLEY, SUSITNA_VALLEY[1:])]]),
}
NOT_PLACED = [  # Listed with the source, no coordinates (placed the three that name a feature)
    *[dict(version=f"{c.lower()}_television", ids=c, status="no place extracted", stated=f"{c}: location 'not extracted' (source unavailable)",
           construction="the register states no place") for c in ("C5", "C6", "C8")],
]


# ---------------------------------------------------------------- simulations: the two ranges and the bound
def simulations():
    """Every simulated row with its ratio, its band's median coda level and its edge-wave flag."""
    a = pd.read_csv(A / "a8_smoothing.csv")
    s = {c: prod.setup(c) for c in a.case.unique()}
    a["edge_flag"] = [gen.edge_delay(s[c]) < gen.edge_needed(s[c], b, r) for c, b, r in zip(a.case, a.band, a.range_m)]
    d = pd.concat([a, pd.read_csv(A / "a8_generous_shares.csv")], ignore_index=True)
    d = d[["case", "azimuth", "range_m", "band", "share", "share_smoothed", "edge_flag"]].astype({"edge_flag": bool})
    d = d.assign(ratio=d[["share", "share_smoothed"]].max(axis=1))
    # The further case. Its ratio is the upper end of its bracket for the source-edge artifact, the larger
    # of the two measures. Every row carries the edge flag, so it sets no start of the bound of this module. It stops at 1.5 km.
    g = pd.read_csv(A / "a8_gradient_shares.csv")
    g = g.assign(ratio=g[["share_high", "share_smoothed_high"]].max(axis=1))[list(d.columns)].astype({"edge_flag": bool})
    d = pd.concat([d, g], ignore_index=True)
    return d.assign(median_coda=d.band.map(clutter()["median"]))


def speeds():
    """The Rayleigh speed of each case's ground, as the two attenuation tables give it (m/s)."""
    return pd.concat([pd.read_csv(A / f) for f in ("a8_attenuation.csv", "a8_generous_attenuation.csv")]).groupby("case").rayleigh_speed_ms.first()


def ranges(d):
    """(rows of s11_ranges.csv, the row that starts the bound, the candidates with a Rayleigh speed)."""
    d = d.assign(rayleigh_speed_ms=d.case.map(speeds()), center_hz=d.band.map(CENTER_HZ), q=Q_GENEROUS)
    over = d[d.ratio >= d.median_coda]
    at = over[over.range_m == over.range_m.max()]
    far = d[d.range_m == d.range_m.max()]
    assert set(d.case) - set(far.case) <= {"R5"}, "the last receiver is the same in every case but the further one, which stops sooner"
    free = d[~d.edge_flag]
    last = free[free.range_m == free.groupby(["case", "band"]).range_m.transform("max")]
    last = last.loc[last.groupby(["case", "band"]).ratio.idxmax()]  # the largest of the three azimuths
    start = last.loc[last.ratio.idxmax()]
    rows = [
        dict(quantity="range_median_coda", **at.loc[(at.ratio / at.median_coda).idxmax()],
             note=f"largest range with ratio at or above the median coda level; {len(at)} rows reach it there: "
                  + ", ".join(sorted(set(at.case + " " + at.band)))),
        dict(quantity="range_last_receiver", **far.loc[far.ratio.idxmax()], note="the last receiver of every case but R5, which stops at 1.5 km; the row shown has the largest ratio there"),
        dict(quantity="bound_start", **start, note="The largest ratio at the last receiver with no edge-wave flag, over cases, bands and azimuths"),
        *[dict(quantity="last_unflagged_receiver", **r, note="largest of the three azimuths" + ("" if r["rayleigh_speed_ms"] > 0 else "; frozen ground: no Q, no speed tabled"))
          for r in last.to_dict("records")],
    ]
    return pd.DataFrame(rows), start, last[last.rayleigh_speed_ms.notna()]


def bound(km, start, cand):
    """The fixed bound at a station km from a place, and the largest attenuated value over cases and bands."""
    m = km * 1e3
    if not m >= start.range_m:
        return dict(bound_note="inside" if km == 0 else "nearer than the last receiver: not carried outward; see the simulation tables")
    spread = start.ratio * np.sqrt(start.range_m / m)
    every = cand.ratio * np.sqrt(cand.range_m / m) * [factor(f, m, Q_GENEROUS, u) for f, u in zip(cand.center_hz, cand.rayleigh_speed_ms)]
    top = cand.loc[every.idxmax()]
    return dict(bound_spreading=spread, bound_attenuated=spread * factor(start.center_hz, m, Q_GENEROUS, start.rayleigh_speed_ms),
                delay_after_p_s=m / start.rayleigh_speed_ms, bound_attenuated_any_band=every.max(),
                any_band_set_by=f"{top.case} {top.band} Hz", any_band_delay_after_p_s=m / top.rayleigh_speed_ms, bound_note="")


# ---------------------------------------------------------------- the bound against the coda level
def coda_levels():
    """The coda table by (band, window): median and p05, in-box sites, the three later explosions pooled."""
    return pd.read_csv(A / "a5_dprk_site_spread.csv").query("box == 'mckinley_nome' and event == 'all'").set_index(["band", "lag"])


def windows(delay, lags):
    """The windows (labels 'a-b', seconds after P, both ends included) that hold a delay; else where it lies."""
    span = [tuple(map(float, w.split("-"))) for w in lags]
    return [w for w, (lo, hi) in zip(lags, span) if lo <= delay <= hi] or ["before the first window" if delay < span[0][0] else "after the last window"]


def against_coda(rows, rng, cand, sim):
    """Rows of s11_bound_against_coda.csv: each bound that exists, by band, in the window its delay falls in."""
    level = coda_levels()
    lags = list(dict.fromkeys(level.index.get_level_values("lag")))
    fastest = sim.assign(u=sim.case.map(speeds())).groupby("band").u.max()
    every = rng[rng.quantity == "last_unflagged_receiver"]  # the frozen cases too
    out = []
    for row in rows:
        for kind in (k for k in KINDS.values() if row[f"{k}_bound_note"] == ""):
            m = row[f"{kind}_km"] * 1e3
            top = []
            for band, c in cand.groupby("band"):
                carried = c.ratio * np.sqrt(c.range_m / m)
                att = carried * [factor(CENTER_HZ[band], m, Q_GENEROUS, u) for u in c.rayleigh_speed_ms]
                e = every[every.band == band]
                assert (e.ratio * np.sqrt(e.range_m / m)).max() == carried.max(), "a frozen case would set the band's spreading bound"
                own = m / c.rayleigh_speed_ms
                where = {}  # window -> the delays that fall in it
                for label, t in [("existing", row[f"{kind}_delay_after_p_s"]), ("earliest", m / fastest[band]), *zip(c.case, own)]:
                    for w in windows(t, lags):
                        where.setdefault(w, []).append(label)
                top.append(att.max())
                for w, labels in where.items():
                    first = bool({"existing", "earliest"} & set(labels))
                    if not first and w not in lags:
                        continue  # a slower ground outside the windows: nothing to compare
                    r = dict(version=row["version"], kind=kind, code=row[f"{kind}_code"], name=row[f"{kind}_name"], km=m / 1e3, band=band,
                             delay_after_p_s=row[f"{kind}_delay_after_p_s"], earliest_delay_s=m / fastest[band], latest_delay_s=own.max(),
                             window=w, window_by="; ".join(labels), slower_ground_only=not first)
                    if w in lags:
                        b = dict(spreading=row[f"{kind}_bound_spreading"], spreading_band=carried.max(), attenuated_band=att.max())
                        lv = level.loc[(band, w)]
                        r |= {f"bound_{k}": v for k, v in b.items()}
                        r |= dict(spreading_band_set_by=c.case[carried.idxmax()], attenuated_band_set_by=c.case[att.idxmax()], median=lv["median"], p05=lv["p05"])
                        r |= {f"{k}_below_{q}": bool(v < lv[q]) for k, v in b.items() for q in ("median", "p05")}
                    out.append(r)
            assert np.isclose(max(top), row[f"{kind}_bound_attenuated_any_band"], rtol=1e-12), "the largest over bands is the any-band column"
    return pd.DataFrame(out), lags


# ---------------------------------------------------------------- self-check
def selfcheck():
    b, nw, bx = PLACES["belt_denali"], PLACES["belt_denali_northwest"], BOXES["valdez"]
    for bearing in (0.0, 123.0, 270.0, 360.0):  # on the outer and the inner edge
        for r in R_C3:
            assert b["inside"](*fwd(bearing, r)[::-1]).all() and km_to(b, *[v[0] for v in fwd(bearing, r)][::-1]) == 0.0
    lo, la = (v[0] for v in fwd(123.0, R_C3[1] + 1e-3))
    assert not b["inside"](la, lo).all() and abs(km_to(b, la, lo) - 1e-3) < 1e-6, "a meter outside the belt"
    for bearing, ok in ((270.0, True), (360.0, True), (0.0, True), (315.0, True), (269.99, False), (0.01, False), (135.0, False)):
        assert nw["inside"](*fwd(bearing, 50.0 * MI)[::-1]).all() == ok, f"north-west part at bearing {bearing}"
    lo, la = (v[0] for v in fwd(40.0, 55.0))  # outside the part's bearings: the nearest point is the corner at bearing 360
    assert abs(km_to(nw, la, lo) - inv(la, lo, *fwd(360.0, R_C3[0])[::-1])[0][0]) < 1e-6
    box_v = PLACES["box_valdez"]
    assert box_v["inside"](bx["maxlatitude"], bx["minlongitude"]).all() and not box_v["inside"](bx["maxlatitude"] + 1e-5, bx["minlongitude"]).all()
    assert abs(km_to(box_v, bx["maxlatitude"] + 0.01, -146.5) - inv(bx["maxlatitude"], -146.5, bx["maxlatitude"] + 0.01, -146.5)[0][0]) < 1e-4
    made = belt(10.0, 20.0)  # a made-up belt; the station lies between the points a 1 km by 1 degree sampling would take
    lo, la = (v[0] for v in fwd(77.5, 15.4))
    assert made["inside"](la, lo).all() and km_to(made, la, lo) == 0.0, "a station inside a made-up belt"
    for r, want in ((25.0, 5.0), (4.0, 6.0)):  # outside it and inside its inner circle: the difference of radii
        assert abs(km_to(made, *[v[0] for v in fwd(77.5, r)][::-1]) - want) < 1e-6
    disc = belt(0.0, 10.0)
    flat = np.pi * 100.0
    assert abs(area_km2(disc) / flat - 1) < 1e-4 and abs(area_km2(made) / (3 * flat) - 1) < 1e-4
    key, la, lo = cells(*DEN, 10.5, 50.0)
    assert len(key) == len(set(key)) and abs(disc["inside"](la, lo).sum() * 50.0**2 / 1e6 / flat - 1) < 2e-3, "area of a disc on the grid"
    lags = list(dict.fromkeys(coda_levels().index.get_level_values("lag")))
    lo, hi = (tuple(map(float, w.split("-"))) for w in lags[:2])
    assert lo[1] == hi[0] and windows(lo[1], lags) == lags[:2], "a delay on a shared edge is in both windows, by their labels"
    assert windows(lo[0], lags) == lags[:1] and windows(float(lags[-1].split("-")[1]), lags) == lags[-1:], "the first and the last edge"
    assert windows(lo[0] - 0.01, lags) == ["before the first window"] and windows(1e3, lags) == ["after the last window"]
    g = pd.read_csv(A / "a8_generous_shares.csv")
    for case in g.case.unique():  # the edge-wave rule, called as simulations() calls it, gives the committed flags
        ch, _ = gen.settings(case, quiet=True)
        with gen.swapped(gen.numerics(ch)):
            s = prod.setup(case, gen.spec(case, ch))
        d = g[g.case == case]
        assert ([gen.edge_delay(s) < gen.edge_needed(s, bd, r) for bd, r in zip(d.band, d.range_m)] == d.edge_flag).all(), case


# ---------------------------------------------------------------- the table
def main() -> None:
    selfcheck()
    st = pd.read_csv(A / "a1_stations.csv")
    every = st.copy()
    st = st[st.evidence.isin(KINDS)].reset_index(drop=True)
    st["kind"] = st.evidence.map(KINDS)
    assert st[["lat", "lon", "reg_name"]].notna().all().all()
    rec = (st.kind == "recorded").to_numpy()
    sim = simulations()
    rng, start, cand = ranges(sim)
    r_km = {"median_range": float(rng.range_m[0]) / 1e3, "last_range": float(rng.range_m[1]) / 1e3}
    cell = min(r_km.values()) * 1e3 / CELL_PER_RANGE
    grid = {}
    for k, r in r_km.items():  # the cells within the range of each station, once for every area
        parts = [cells(la, lo, r, cell) for la, lo in zip(st.lat, st.lon)]
        grid[k] = (*(np.concatenate([p[i] for p in parts]) for i in range(3)), np.repeat(rec, [len(p[0]) for p in parts]))

    rows, inside_rows, dist = [], [], {}
    for name, p in PLACES.items():
        anchor = ANCHORS[p["anchor"]]
        d = dist[name] = np.array([km_to(p, la, lo) for la, lo in zip(st.lat, st.lon)])
        to_anchor = inv(*anchor, st.lat, st.lon)[0]
        order = np.lexsort((to_anchor, d))  # nearest first; stations inside an area by their distance from its anchor
        row = dict(version=name, ids=p["ids"], status="placed", kind=p["kind"], stated=p["stated"], construction=p["construction"],
                   anchor=p["anchor"], area_km2=area_km2(p) if p["kind"] == "area" else np.nan)
        for kind in KINDS.values():
            i = next(i for i in order if st.kind[i] == kind)
            row |= {f"{kind}_code": st.code[i], f"{kind}_name": st.reg_name[i], f"{kind}_km": d[i], f"{kind}_mi": d[i] / MI}
        if p["kind"] == "area":
            ins = [i for i in order if d[i] == 0.0]
            row |= dict(n_inside=len(ins), **{f"inside_{kind}": "; ".join(f"{st.code[i]} {st.reg_name[i]}" for i in ins if st.kind[i] == kind) for kind in KINDS.values()})
            inside_rows += [dict(version=name, code=st.code[i], name=st.reg_name[i], kind=st.kind[i], anchor=p["anchor"],
                                 km_anchor=to_anchor[i], mi_anchor=to_anchor[i] / MI) for i in ins]
        for k, r in r_km.items():
            key, la, lo, is_rec = grid[k]
            for who, sel, dd in (("any", np.ones(len(key), bool), d), ("recorded", is_rec, d[rec]), ("operating", ~is_rec, d[~rec])):
                if p["kind"] == "area":
                    share = len(np.unique(key[sel & p["inside"](la, lo)])) * cell**2 / 1e6 / row["area_km2"]
                else:  # a point is within the range or not; a line with no station within the range has no part within it
                    share = float(dd.min() <= r) if p["kind"] == "point" or dd.min() > r else np.nan
                row[f"share_{who}_{k}"] = share
        i = next(i for i in order if st.waveforms_open[i])
        row |= dict(open_code=st.code[i], open_name=st.reg_name[i], open_km=d[i], open_mi=d[i] / MI)
        for kind in KINDS.values():
            row |= {f"{kind}_{k}": v for k, v in bound(row[f"{kind}_km"], start, cand).items()}
        rows.append(row)
    out = pd.concat([pd.DataFrame(rows), pd.DataFrame(NOT_PLACED)], ignore_index=True)

    # checks: counts, the old table (a3c_spacing.csv), the station table's own columns
    old = pd.read_csv(A / "a3c_spacing.csv").set_index("check")
    v = out.set_index("version")
    checks = [dict(check="stations_recorded", value=int(rec.sum()), note="codes; evidence recorded_test"),
              dict(check="stations_operating_no_reading", value=int((~rec).sum()), note="codes; evidence operating_at_test"),
              dict(check="stations_open_waveform", value=int(st.waveforms_open.sum()), note=" ".join(st.code[st.waveforms_open])),
              dict(check="versions_placed", value=len(PLACES), note=" ".join(PLACES)),
              dict(check="versions_not_placed", value=sum(n["status"] == "not placed" for n in NOT_PLACED),
                   note=" ".join(n["version"] for n in NOT_PLACED if n["status"] == "not placed")),
              dict(check="versions_no_place_extracted", value=sum(n["status"] != "not placed" for n in NOT_PLACED), note="C5 C6 C8"),
              dict(check="simulated_cases", value=sim.case.nunique(), note=" ".join(sim.case.unique()) + "; R5 by the upper end of its bracket"),
              dict(check="rows_with_edge_flag", value=int(sim.edge_flag.sum()), note=f"of {len(sim)}; first four cases by the rule of a8_generous, computed here"),
              dict(check="cell_m", value=cell, note=f"EPSG:3338; a fifth of the smaller range ({min(r_km.values())} km)"),
              *[dict(check=f"{k}_km", value=r, note="s11_ranges.csv") for k, r in r_km.items()]]

    def nearest(lat, lon, sel):
        d = inv(lat, lon, st.lat[sel], st.lon[sel])[0]
        return round(float(d.min()), 1), st.code[sel].iloc[int(np.argmin(d))]
    c4 = [float(x[0]) for x in fwd(AZ_NOME, 60.0 * MI)][::-1]
    for label, pt in (("c4_point", c4), ("denali", DEN)):  # where the old table is right: reproduced to its printed precision
        for which, sel in (("recorders", rec), ("recorders_and_operating", np.ones(len(st), bool))):
            o, got = old.loc[f"km_{label}_to_nearest_{which}"], nearest(*pt, sel)
            assert got == (o.value, o.note), (label, which, got)
            checks.append(dict(check=f"old_km_{label}_to_nearest_{which}", value=got[0], note=f"{got[1]}; agrees with a3c_spacing.csv ({o.value} {o.note})"))
    assert v.recorded_km["point_c4"].round(1) == old.value["km_c4_point_to_nearest_recorders"]
    assert v.operating_km["point_c4"].round(1) == old.value["km_c4_point_to_nearest_recorders_and_operating"]
    disc = sorted(st.code[dist["disc_denali"] == 0.0])
    assert "/".join(disc) == old.note["c3_disc_operating_sites"]
    checks.append(dict(check="old_c3_disc_operating_sites", value=len(disc), note="/".join(disc) + "; agrees with a3c_spacing.csv"))
    # where it is wrong: its sampling, redone here (radii every 1 km, bearings 0 to 359 by 1 degree)
    rr, aa = (x.ravel() for x in np.meshgrid(np.arange(R_C3[0], R_C3[1] + 1e-9, 1.0), np.arange(0.0, 360.0, 1.0)))
    lo, la = fwd(aa, rr)
    near = np.min([inv(a, o, la, lo)[0] for a, o in zip(st.lat, st.lon)], axis=0)
    was = dict(belt=round(float(near.min()), 1), nw=round(float(near[aa >= 270.0].min()), 1), nw_closed=round(float(near[(aa >= 270.0) | (aa == 0.0)].min()), 1))
    assert was["belt"] == old.value["c3_belt_min_km_to_operating"] and was["nw"] == old.value["c3_belt_nw_min_km_to_operating"], was
    cut = st.index[dist["belt_denali"] == 0.0]
    shared = ~rec & st.site_id.isin(st.site_id[rec]).to_numpy()
    b, nw = v.loc["belt_denali"], v.loc["belt_denali_northwest"]
    checks += [
        dict(check="differs_c3_belt_min_km_to_operating", value=b.operating_km,
             note=f"a3c_spacing.csv has {was['belt']}. Cause: that is the gap between {b.operating_code} and the nearest of the script's sample points "
                  f"(reproduced here: {was['belt']}). {b.operating_code} ({b.operating_name}) stands {inv(*DEN, st.lat[cut], st.lon[cut])[0][0]:.1f} km "
                  f"from Denali, inside the belt ({R_C3[0]:.2f} to {R_C3[1]:.2f} km)"),
        dict(check="differs_c3_belt_nw_min_km_to_operating", value=nw.operating_km,
             note=f"a3c_spacing.csv has {was['nw']} (reproduced here: {was['nw']}). Cause: its bearings stop at 359, so the bounding bearing 360 is left out; "
                  f"the nearest point to {nw.operating_code} is the corner at bearing 360 on the inner edge. The old sampling with bearing 360 added gives {was['nw_closed']}"),
        dict(check="old_rows_not_recomputed", value=len(old) - 7,
             note="a3c_spacing.csv: station spacing (12 rows) and belt shares within the analytic radii of a3b4_apply (4 rows); other quantities"),
        dict(check="old_positions", value=int(st.site_id.nunique()), note=f"a3c_spacing.py takes one position per site (first code); here every code has its own ({len(st)} codes)"),
        dict(check="operating_codes_at_a_site_that_recorded", value=int(shared.sum()),
             note="; ".join(f"{c} (site {s})" for c, s in zip(st.code[shared], st.site_id[shared])) + ": no reading under this code, a reading under another code of the site"),
    ]
    # the station table's own columns and the line coverage
    lc = pd.read_csv(A / "a1_line_coverage.csv")
    ln = v.loc["line_denali_nome"]
    col = every.set_index("code").loc[st.code]
    checks += [
        dict(check="line_km_max_abs_diff_from_a1_column", value=float(np.abs(dist["line_denali_nome"] - col.km_denali_nome_line.to_numpy()).max()),
             note="a1_stations.csv km_denali_nome_line is printed to 0.1 km from a line sampled every 1 km"),
        dict(check="valdez_km_max_abs_diff_from_a1_column", value=float(np.abs(dist["valdez_town"] - col.km_valdez.to_numpy()).max()), note="a1_stations.csv km_valdez, printed to 0.1 km"),
        dict(check="line_nearest_recorder_km", value=ln.recorded_km, note=f"{ln.recorded_code}; a1_line_coverage.csv minimum over its points: {lc.km_nearest_recorder.min()}"),
        dict(check="line_nearest_operating_km", value=ln.operating_km,
             note=f"{ln.operating_code}; operating with no reading. a1_line_coverage.csv counts recorders as operating: minimum {lc.km_nearest_operating.min()}"),
        *[dict(check=f"inside_{z}_agrees_with_a1_column", value=int((dist[name] == 0.0).sum()),
               note="same stations as a1_stations.csv in_" + z) for z, name in (("mckinley_nome", "box_denali_nome"), ("valdez", "box_valdez"))],
    ]
    for z, name in (("mckinley_nome", "box_denali_nome"), ("valdez", "box_valdez")):
        assert ((dist[name] == 0.0) == col[f"in_{z}"].to_numpy(bool)).all(), z
    key, la, lo = cells(*DEN, R_C3[1] + 1.0, cell)
    on_grid = PLACES["belt_denali"]["inside"](la, lo).sum() * cell**2 / 1e6
    assert abs(on_grid / b.area_km2 - 1) < 1e-3
    checks.append(dict(check="belt_area_on_grid_over_geodesic_area", value=on_grid / b.area_km2, note=f"{on_grid:.1f} km2 by cells of {cell:g} m; {b.area_km2:.1f} km2 by the boundary"))

    # the bound against the coda level, where the wave would arrive inside the measured windows
    coda, lags = against_coda(rows, rng, cand, sim)
    nome = coda[(coda.version == "line_denali_nome") & (coda.kind == "recorded") & ~coda.slower_ground_only]
    assert set(nome.window) == {lags[0]} and windows(nome.delay_after_p_s.iloc[0], lags) == lags[:1], "the Nome end falls in the first window"
    seen = coda[coda.window.isin(lags)]
    pairs = {True: [], False: []}  # by: is the existing or the earliest delay inside a window
    for (ver, kind), g in coda[~coda.slower_ground_only].groupby(["version", "kind"], sort=False):
        pairs[bool(g.window.isin(lags).any())].append(f"{ver} {kind} {g.code.iloc[0]} {g.delay_after_p_s.iloc[0]:.1f} s ({', '.join(dict.fromkeys(g.window))})")

    def yes(r, k):
        return f"below the median: {'yes' if r[f'{k}_below_median'] else 'no'}; below the fifth percentile: {'yes' if r[f'{k}_below_p05'] else 'no'}"
    checks += [
        dict(check="coda_pairs_inside_windows", value=len(pairs[True]), note="version, kind, station, delay (window): " + "; ".join(pairs[True])),
        dict(check="coda_pairs_outside_windows", value=len(pairs[False]), note="no comparison: " + "; ".join(pairs[False])),
        *[dict(check=f"coda_{'slower_ground_' * r.slower_ground_only}{r.version}_{r.kind}_{r.band}_window_{r.window}", value=r.bound_spreading,
               note=f"{r.code}; in the window by: {r.window_by}. Median {r['median']:.3g}, fifth percentile {r.p05:.3g}. "
                    f"Spreading alone, largest of all bands ({r.bound_spreading:.3g}): {yes(r, 'spreading')}. "
                    f"Spreading alone, this band ({r.bound_spreading_band:.3g}, {r.spreading_band_set_by}): {yes(r, 'spreading_band')}. "
                    f"With attenuation, this band ({r.bound_attenuated_band:.3g}, {r.attenuated_band_set_by}): {yes(r, 'attenuated_band')}")
          for _, r in seen.iterrows()],
    ]
    coda.to_csv(HERE / "s11_bound_against_coda.csv", index=False, float_format="%.6g")
    for name, df in (("versions", out), ("stations_inside", pd.DataFrame(inside_rows)), ("ranges", rng), ("checks", pd.DataFrame(checks))):
        df.to_csv(HERE / f"s11_{name}.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60)
    print(out.drop(columns=["stated", "construction"]).T.to_string())
    print(rng.to_string(index=False))
    print(coda.to_string(index=False))


if __name__ == "__main__":
    main()
