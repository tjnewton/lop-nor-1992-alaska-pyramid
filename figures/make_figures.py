"""Manuscript figures from committed tables only (no raster is read).

fig2_timeline: the registered versions of the claim and the documented record (the test, the press,
  the critique), each at its date as the register supports it (fig2_claims.csv, transcribed from the sources of the paper;
  a dot is a day, an open dot a month, a capped line a year or a range), the broadcast as the versions date it
  as a band, the archive window as a box, and an overview of the whole span; the kind of source and the stated
  depths and height in a table. Undated versions are listed, not placed.
fig3_distance: distance from each claim-box grid point to the nearest 1992 recorder and to the
  nearest station operating at the test (akp/stage_a/a1_box_coverage.csv), against the largest
  single-station detection radius with cited attenuation (a3b4_apply.csv).
fig4_scatter: the Rayleigh wave the body scatters, as a share of the incident P at the same station,
  against range and band, for every simulated case (a8_production.csv: the body of doubled stiffness
  in thawed ground at two speeds and under a frozen layer; a8_generous_shares.csv: R1 to R4; R1 is
  drawn where production has no run), over the clutter measured at in-box sites
  (a5_dprk_site_spread.csv: median, cleanest twentieth and interquartile band). Rows with the
  edge-wave flag are dashed: the committed flag of R1 to R4, and for the first four cases the flag
  that s11_versions.simulations computes by the same rule.
fig5_pd: detection probability against relief, per base: measured on the claim zone
  (p2e_claim_pd.csv, all plants) and predicted from the controls at the frozen cut
  (p2e_pd.summary on p2e_pd_trials.csv, third seed), each with its Clopper-Pearson interval.
fig1_map: the stations that recorded the test and those operating at it (a1_stations.csv) against the stated
  places, in EPSG:3338: a, the region of the claimed locations and the boxes (akp/zones.py) with the station nearest each
  version (s11_versions.csv), the stations elsewhere counted in a note; b and c, around Denali and around Valdez, the
  claimed locations by the boundary curves of s11_versions.PLACES, the stations with their codes, and the last simulated
  range (s11_ranges.csv) around each station of s11_stations_inside.csv. No screening anomaly is shown.
fig8_screen: Supplement Two, the screened area (p2a_searched_area.geojson) and the control tiles (p2a_controls.csv)
  of the terrain screen over Alaska, with the two boxes.
  The land mask is kept in $AKP_DATA/p2a/fig1_land.npz, so the grid (about 6 GB) is built once.
fig6_exposure: the share of the prior within a distance of any receiver (s9_exposure.csv), with the
  ranges of the census rows: the body of doubled stiffness (s9_checks.csv) and stone (R2 in
  s10_sensitivity.csv); a range beyond the last simulated receiver is dashed.
fig7_lead: left, per band: the range of the simulated ratio over cases and lines to the last receiver
  (a5_late_windows.simulated), the range at which its largest value meets the median (a5_late_crossing.csv), and
  beyond it the band's own bound by geometrical spreading alone, with the nearest stations of each version
  (a5_late_by_band_summary.csv). Right, same vertical axis: the P coda before later phases and the 1992 record against time
  after the P peak (a5_late_levels.csv), the region with no level of the P coda before later phases shaded, and the largest
  bound that could still arrive at each time. Asserts that the curve gives the table's bounds.
Palette: categorical slots 1-6 of a validated palette. Slots 1-2 validate alone; the seven lines of
  fig4 cannot pass on every pair, so each case also has its own marker.
Run: PYTHONPATH=. python figures/make_figures.py [fig4 fig7 ...]
  (no name: every figure; the names are the functions below)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path(__file__).parent
REPO = OUT.parent
sys.path.insert(0, str(REPO))
A, P = REPO / "akp/stage_a", REPO / "akp/stage_p2"
S1, S2, S1_LIGHT = "#2a78d6", "#eb6834", "#86b6ef"  # series 1, series 2, series-1 step 250
S3, S4, S5, S6 = "#1baf7a", "#eda100", "#e87ba4", "#008300"  # series 3 to 6
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "axes.axisbelow": True})
CASE = {"thawed_1200": "high-contrast body, 1200 m/s ground", "thawed_850": "high-contrast body, 850 m/s ground",  # plain names
        "frozen_1200": "high-contrast body, frozen layer over 1200 m/s ground", "frozen_850": "high-contrast body, frozen layer over 850 m/s ground",
        "R1": "high-contrast body, 850 m/s ground, finer grid", "R2": "crystalline body, 1200 m/s ground",
        "R3": "high-contrast body, 400 m base, 1200 m/s ground", "R4": "soft-ground case", "R5": "velocity-gradient case (to 1.5 km)"}
PLACE = {"line_denali_nome": "Line from Denali to Nome", "box_denali_nome": "Denali\u2013Nome box", "disc_denali": "Disc around Denali",  # as Table 2 names them, in its order
         "belt_denali": "Belt around Denali", "belt_denali_northwest": "Northwest part of the belt", "point_c4": "Point on the line to Nome",
         "stretch_c4_map": "Stretch marked on a map", "point_c7_west": "Due west of Denali", "valdez_town": "Valdez, the town",
         "box_valdez": "Valdez box", "line_c7_unalakleet": "Line from Denali to Unalakleet", "point_c9_haarp": "Facility near Gakona",
         "point_c10_carey_lake": "Carey Lake"}
RECORDED, OPERATING = "station that recorded the explosion", "station operating at the time, no reading reported"


def value(path: Path, check: str) -> float:
    d = pd.read_csv(path)
    return float(d.loc[d.check == check, "value"].iloc[0])


def save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)  # the journal asks for 300 dots per inch or more
    plt.close(fig)


def _land():
    """The land mask of the strata grid and its extent (x0, y1, x1, y0) in EPSG:3338 meters, from a cache on the data drive."""
    from akp.paths import data_dir

    cache = data_dir("p2a") / "fig1_land.npz"  # simplification: no check that the grid is unchanged; delete the file after a change to p2a_strata
    if not cache.exists():  # building the grid peaks near 6 GB: once, and only its land mask is kept
        from akp.stage_p2.p2a_strata import strata_grid

        transform, grid = strata_grid()
        np.savez_compressed(cache, land=grid >= 0, extent=[*(transform * (0, 0)), *(transform * (grid.shape[1], grid.shape[0]))])
        del grid
    z = np.load(cache)
    return z["land"], z["extent"]


def _tidy_land(land, max_hole: int):
    """Fill the holes in the land mask smaller than max_hole cells (wet ground read as water at display scale); lakes stay."""
    from scipy import ndimage

    holes = ndimage.binary_fill_holes(land) & ~land
    lab, n = ndimage.label(holes)
    sizes = ndimage.sum(holes, lab, range(1, n + 1))
    return land | np.isin(lab, np.nonzero(sizes < max_hole)[0] + 1)


def _halo(width=2.4):
    import matplotlib.patheffects as pe

    return [pe.withStroke(linewidth=width, foreground="white")]


def fig1() -> None:
    """Figure 2 of the paper: the stations of 1992 against the places the versions state.

    a: the region of the claimed locations and of the station nearest each, cropped from Alaska (the stations elsewhere are
    counted in a note), with a light graticule; b and c: around Denali and around Valdez, with the last simulated range
    about each station inside a claimed area. Places of the claim are blue and their names blue italic; towns black;
    station codes gray; every name has a white halo. The searches of Supplement Two are mapped in fig8_screen."""
    import textwrap

    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from pyproj import Geod
    from shapely.geometry import box, shape

    from akp.stage_p2.p2a_strata import TO3338, to_3338
    from akp.stage_p4.s11_versions import NAMED, PLACES, SUSITNA_VALLEY
    from akp.zones import ANCHORS, BOXES

    full, (x0, y1, x1, y0) = _land()
    st = pd.read_csv(A / "a1_stations.csv")
    st = st[st.evidence.isin(("operating_at_test", "recorded_test"))].copy()
    st["x"], st["y"] = TO3338.transform(st.lon, st.lat)
    ver = pd.read_csv(REPO / "akp/stage_p4/s11_versions.csv")
    inside = pd.read_csv(REPO / "akp/stage_p4/s11_stations_inside.csv")
    nearest = set(ver.recorded_code.dropna()) | set(ver.operating_code.dropna())
    geod = Geod(ellps="WGS84")

    def pt(lo, la):
        return to_3338(shape({"type": "Point", "coordinates": [lo, la]})).coords[0]

    # the crop: every claimed location and box, the station nearest each version, the inside stations and COL, with a margin
    keep = [pt(lo, la) for la, lo in (*ANCHORS.values(), *NAMED.values())]
    for b in BOXES.values():
        keep += [pt(b[f"{e}longitude"], b[f"{n}latitude"]) for e in ("min", "max") for n in ("min", "max")]
    sel = st[st.code.isin(nearest | set(inside.code) | {"COL"})]
    keep += list(zip(sel.x, sel.y))
    M = 60e3
    X0, X1 = min(p[0] for p in keep) - M, max(p[0] for p in keep) + M
    Y0, Y1 = min(p[1] for p in keep) - M, max(p[1] for p in keep) + M
    shown = st.x.between(X0, X1) & st.y.between(Y0, Y1)
    assert st[~shown].code.isin(nearest).sum() == 0, "the nearest station of every version is on the map"

    win = {"denali": (-170, 135, 109, 50, -160), "valdez": (-75, 75, 53.5, 20, 45)}  # km from the anchor: west, east, half-height; scale bar and its start
    W, GAPX, GAPY, PAD, KEYH = 6.9, 0.10, 0.10, 0.02, 0.70  # with the 0.1 in that a saved figure adds about it, 180 mm
    wa = W - 2 * PAD
    ha = wa * (Y1 - Y0) / (X1 - X0)
    wb = (wa - GAPX) / 2
    hb = wb * (2 * win["denali"][2]) / (win["denali"][1] - win["denali"][0])
    H = PAD + ha + KEYH + hb + PAD
    assert H <= 200 / 25.4, f"the figure is {H:.2f} in tall; the journal allows 200 mm"
    fig = plt.figure(figsize=(W, H))

    def axes(x, top, w, h):
        return fig.add_axes((x / W, (H - top - h) / H, w / W, h / H))

    ax = axes(PAD, PAD, wa, ha)
    kax = axes(PAD, PAD + ha + 0.04, wa, KEYH - 0.10)
    insets = {"denali": axes(PAD, PAD + ha + KEYH, wb, hb), "valdez": axes(PAD + wb + GAPX, PAD + ha + KEYH, wb, hb)}
    gray = matplotlib.colors.ListedColormap([GRID])
    k = 2  # display cells of 2 km in panel a, of the native grid in the insets
    ax.imshow(np.where(_tidy_land(full[::k, ::k], 40), 1.0, np.nan), extent=(x0, x1, y0, y1), cmap=gray, interpolation="nearest", zorder=0)
    for a in insets.values():
        a.imshow(np.where(_tidy_land(full, 160), 1.0, np.nan), extent=(x0, x1, y0, y1), cmap=gray, interpolation="nearest", zorder=0)

    # graticule, labeled where it leaves the frame
    LINE = "#cfcec8"
    for la in range(58, 70, 2):
        lo = np.linspace(-175, -135, 400)
        gx, gy = TO3338.transform(lo, np.full_like(lo, la))
        ax.plot(gx, gy, color=LINE, lw=0.5, zorder=1)
        i = np.argmin(np.abs(gx - X0))
        if Y0 + 0.08 * (Y1 - Y0) < gy[i] < Y1 - 0.10 * (Y1 - Y0) and abs(gx[i] - X0) < 20e3:
            ax.text(X0 + 8e3, gy[i], f"{la}°N", fontsize=7, color=INK2, va="bottom", ha="left", path_effects=_halo(), zorder=6)
    for lo in range(-170, -135, 5):
        la = np.linspace(55, 72, 400)
        gx, gy = TO3338.transform(np.full_like(la, lo), la)
        ax.plot(gx, gy, color=LINE, lw=0.5, zorder=1)
        i = np.argmin(np.abs(gy - Y0))
        if X0 + 0.04 * (X1 - X0) < gx[i] < X1 - 0.04 * (X1 - X0) and abs(gy[i] - Y0) < 20e3:
            ax.text(gx[i], Y0 + 10e3, f"{-lo}°W", fontsize=7, color=INK2, va="bottom", ha="center", path_effects=_halo(), zorder=6)

    # boxes, lines and places of the claim
    for b in BOXES.values():
        g = to_3338(box(b["minlongitude"], b["minlatitude"], b["maxlongitude"], b["maxlatitude"]).segmentize(0.05))
        for a in (ax, *insets.values()):
            a.plot(*g.exterior.xy, color=INK2, lw=0.8, ls="--", zorder=2)
    (dla, dlo), (nla, nlo) = ANCHORS["denali"], ANCHORS["nome"]
    ula, ulo = NAMED["unalakleet"]
    for (la2, lo2), ls in (((nla, nlo), ":"), ((ula, ulo), "-.")):
        way = [(dlo, dla), *geod.npts(dlo, dla, lo2, la2, 80), (lo2, la2)]
        ax.plot(*TO3338.transform([p[0] for p in way], [p[1] for p in way]), color=S1, lw=1.0, ls=ls, zorder=2)
    az, _, _ = geod.inv(dlo, dla, nlo, nla)
    c4lo, c4la, _ = geod.fwd(dlo, dla, az, 60.0 * 1609.344)  # the point on the line to Nome: 60 statute miles west of Denali, as stated
    ax.scatter(*pt(c4lo, c4la), s=46, marker="x", color="white", lw=3.6, zorder=5)
    ax.scatter(*pt(c4lo, c4la), s=46, marker="x", color=S1, lw=1.6, zorder=5)
    for key in ("carey_lake", "haarp"):
        la, lo = NAMED[key]
        ax.scatter(*pt(lo, la), s=120, marker="D", facecolor="none", edgecolor=S1, lw=1.2, zorder=4)
    # The places of the versions found by the search of Supplement One, marked by their ids (Table 2)
    sx, sy = zip(*(pt(lo, la) for la, lo in SUSITNA_VALLEY))
    ax.plot(sx, sy, color=S1, lw=1.4, zorder=4)
    later = {"coords_c12": None, "farewell_lake": ("C13", (6, -5), "left"), "telida": ("C14", (-7, 7), "right"),  # C12 is named with Carey Lake
             "kaltag": ("C14", (6, 4), "left"), "charley_mouth": ("C15", (-6, 0), "right"), "fort_richardson": ("C16", (-6, -6), "right"),
             "mcgrath": ("C17", (-5, 8), "right")}
    for key in later:
        la, lo = NAMED[key]
        ax.scatter(*pt(lo, la), s=40, marker="D", facecolor="white", edgecolor=S1, lw=1.1, zorder=8)  # above the labels' halos

    # stations, then the Eielson array
    kinds = (("operating_at_test", "^", S2, 30, 40), ("recorded_test", "o", INK, 24, 34))  # evidence, marker, color, size in a, in b and c
    for ev, marker, c, sa, sb in kinds:
        d = st[st.evidence == ev]
        ax.scatter(d.x, d.y, s=sa, marker=marker, color=c, edgecolor="white", lw=0.5, zorder=3)
        for a in insets.values():
            a.scatter(d.x, d.y, s=sb, marker=marker, color=c, edgecolor="white", lw=0.5, zorder=3)
    il = pd.read_csv(A / "a6_array_precheck.csv").query("section == 'element' and array == 'ILAR'")
    ax.scatter(*pt(il.lon.mean(), il.lat.mean()), s=34, marker="s", facecolor="none", edgecolor=INK, lw=1.1, zorder=4)

    # names: towns black, places of the claim blue italic, station codes gray; offsets in points
    def name(a, xy, text, off, kind, ha="left", size=7.5):
        style = {"town": dict(color=INK), "place": dict(color=S1, fontstyle="italic"), "code": dict(color=INK2)}[kind]
        a.annotate(text, xy, xytext=off, textcoords="offset points", ha=ha, va="center", fontsize=size if kind != "code" else 7,
                   path_effects=_halo(), zorder=7, **style)

    for town, (la, lo), off, ha in (("Denali", ANCHORS["denali"], (-6, -9), "right"), ("Nome", ANCHORS["nome"], (5, 6), "left"),
                                    ("Valdez", ANCHORS["valdez"], (0, -12), "center"), ("Unalakleet", NAMED["unalakleet"], (-5, -7), "right")):
        x, y = pt(lo, la)
        ax.plot(x, y, marker="+", ms=6, mew=1.0, color=INK, zorder=5)
        name(ax, (x, y), town, off, "town", ha)
    name(ax, pt(c4lo, c4la), textwrap.fill(PLACE["point_c4"], 12), (-7, -15), "place", "right")
    for key, off, ha in (("carey_lake", (0, 14), "center"), ("haarp", (-17, -1), "right")):
        la, lo = NAMED[key]
        name(ax, pt(lo, la), "Carey Lake; C12" if key == "carey_lake" else textwrap.fill(PLACE["point_c9_haarp"], 13), off, "place", ha)
    for key, v in later.items():
        if v:
            la, lo = NAMED[key]
            name(ax, pt(lo, la), v[0], v[1], "place", v[2], size=7)
    name(ax, (sx[2], sy[2]), "C18", (-6, 0), "place", "right", size=7)
    for label, (la, lo), off, ha in ((PLACE["box_denali_nome"], (65.85, -158.5), (0, 0), "center"),
                                     ("line to Nome", (64.25, -161.6), (0, 6), "center"), ("line to Unalakleet", (63.45, -158.8), (0, -7), "center")):
        name(ax, pt(lo, la), label, off, "place", ha, size=7)
    named = {"COL": (-4, 6, "right"), "TTA": (-5, -6, "right"), "TRF": (5, 5, "left"), "HUR": (5, -5, "left"), "KLU": (6, -7, "left"),
             "RND": (5, 4, "left"), "ANM": (-5, -6, "right"), "SDG": (-6, 8, "right"), "TOA": (6, -8, "left"), "PMR": (-6, 5, "right"),
             "SKT": (-6, -6, "right"), "PMS": (6, -6, "left"), "MLY": (6, 4, "left"), "PRP": (6, 4, "left")}  # The nearest stations of the later places
    for code, (dx, dy, ha) in named.items():
        r = st[st.code == code].iloc[0]
        name(ax, (r.x, r.y), code, (dx, dy), "code", ha)

    # where b and c lie
    for key, letter in (("denali", "b"), ("valdez", "c")):
        la, lo = ANCHORS[key]
        cx, cy = pt(lo, la)
        west, east, h = (1e3 * v for v in win[key][:3])
        ax.add_patch(Rectangle((cx + west, cy - h), east - west, 2 * h, facecolor="none", edgecolor=INK, lw=0.8, zorder=6))
        ax.text(cx + west + 6e3, cy + h - 6e3, letter, fontsize=8.5, fontweight="bold", color=INK, va="top", ha="left", path_effects=_halo(), zorder=7)

    ax.set(xlim=(X0, X1), ylim=(Y0, Y1), aspect="equal", xticks=[], yticks=[])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set(visible=True, color=INK2, lw=0.6)
    ax.text(X0 + 0.012 * (X1 - X0), Y1 - 0.02 * (Y1 - Y0), "a", fontsize=9, fontweight="bold", color=INK, va="top", path_effects=_halo(), zorder=7)
    sb0, sby = X0 + 0.03 * (X1 - X0), Y0 + 0.11 * (Y1 - Y0)
    ax.plot([sb0, sb0 + 200e3], [sby] * 2, color=INK, lw=1.5, zorder=6)
    ax.text(sb0 + 100e3, sby + 10e3, "200 km", ha="center", va="bottom", fontsize=7.5, color=INK, path_effects=_halo(), zorder=7)
    off_rec, off_op = int((~shown & (st.evidence == "recorded_test")).sum()), int((~shown & (st.evidence == "operating_at_test")).sum())
    handles = [Line2D([], [], ls="", marker="o", ms=5, color=INK, mec="white"), Line2D([], [], ls="", marker="^", ms=5.5, color=S2, mec="white"),
               Line2D([], [], ls="", marker="s", ms=5, mfc="none", mec=INK, mew=1.1), Line2D([], [], ls="", marker="+", ms=6, mew=1.0, color=INK),
               Line2D([], [], ls="", marker="x", ms=6, color=S1, mew=1.6), Line2D([], [], ls="", marker="D", ms=6, mfc="none", mec=S1, mew=1.2),
               Line2D([], [], color=S1, lw=1.0, ls=":"), Line2D([], [], color=S1, lw=1.0, ls="-."),
               Line2D([], [], color=INK2, lw=0.8, ls="--"), Line2D([], [], color=INK, lw=0.7)]
    reach = pd.read_csv(REPO / "akp/stage_p4/s11_ranges.csv").set_index("quantity").range_m["range_last_receiver"]  # m
    labels = ["station that recorded the explosion", "station operating, no reading reported", "Eielson array", "town",
              PLACE["point_c4"], "place named by a later version", "line from Denali to Nome", "line from Denali to Unalakleet",
              "box around claimed locations", f"in b and c: last simulated range, {reach / 1e3:g} km"]
    kax.axis("off")
    kax.legend(handles, labels, loc="upper left", ncol=3, fontsize=7, frameon=False, handlelength=1.6, handletextpad=0.5,
               columnspacing=1.1, labelspacing=0.3, borderaxespad=0)
    off_rec, off_op = int((~shown & (st.evidence == "recorded_test")).sum()), int((~shown & (st.evidence == "operating_at_test")).sum())
    kax.text(1.0, 0.0, f"Not on map a: {off_rec} stations that recorded the explosion and {off_op} that operated, none the nearest to a claimed location.",
             transform=kax.transAxes, ha="right", va="bottom", fontsize=7, color=INK2)

    # the insets: the claimed locations by their boundary curves, the stations by code, the last simulated range about each station inside an area
    def curves(key):
        t = np.linspace(0.0, 1.0, 361)
        return [np.array(TO3338.transform(*np.concatenate([c(t) for c in ring], axis=1))) for ring in PLACES[key]["rings"]]

    den = insets["denali"]
    outer, inner = curves("belt_denali")
    den.fill(np.r_[outer[0], inner[0][::-1]], np.r_[outer[1], inner[1][::-1]], color=S1, alpha=0.18, lw=0, zorder=1)
    den.fill(*curves("belt_denali_northwest")[0], color=S1, alpha=0.25, lw=0, zorder=1)
    den.plot(*curves("disc_denali")[0], color=S1, lw=1.0, zorder=2)
    den.plot(*curves("line_denali_nome")[0], color=S1, lw=0.9, ls=":", zorder=2)
    den.plot(*curves("stretch_c4_map")[0], color=S1, lw=2.6, solid_capstyle="butt", zorder=2)
    den.scatter(*curves("point_c4")[0][:, 0], s=46, marker="x", color=S1, lw=1.6, zorder=4)
    den.scatter(*curves("point_c7_west")[0][:, 0], s=46, marker="+", color=S1, lw=1.6, zorder=4)
    labels_in = {"denali": ((PLACE["disc_denali"], -5, -65, "center"), (PLACE["belt_denali"], -5, -98, "center"),  # place; km east and north of the anchor
                            (textwrap.fill(PLACE["belt_denali_northwest"], 16), -105, 80, "center"), ("line to Nome", -165, 63, "left"),
                            (textwrap.fill(PLACE["stretch_c4_map"], 8), -112, 12, "right"), (textwrap.fill(PLACE["point_c7_west"], 9), -91, -20, "right"),
                            (PLACE["box_denali_nome"], 88, 100, "right"), (textwrap.fill(PLACE["point_c4"], 12), -95, 46, "center")),
                 "valdez": ((PLACE["box_valdez"], -48, 41, "left"),)}
    off = {"HUR": (-5, 4), "VZW": (-6, 5), "VLZ": (9, 0)}  # code labels that would otherwise sit on a line or a mark (points)
    vlz_km = float(ver.set_index("version").operating_km["valdez_town"])
    for key, a in insets.items():
        la, lo = ANCHORS[key]
        cx, cy = pt(lo, la)
        west, east, h, bar, bx = (1e3 * v for v in win[key])
        a.plot(cx, cy, marker="+", ms=7, mew=1.0, color=INK, zorder=5)
        a.annotate(key.capitalize(), (cx, cy), xytext=(6, -8) if key == "denali" else (0, 11), textcoords="offset points", fontsize=7.5, color=INK,
                   ha="left" if key == "denali" else "center", path_effects=_halo(), zorder=7)
        for label, e, n, ha in labels_in[key]:
            a.text(cx + e * 1e3, cy + n * 1e3, label, ha=ha, va="center", fontsize=7, color=S1, fontstyle="italic", path_effects=_halo(), zorder=7)
        ring_done = False
        for r in st[st.x.between(cx + west, cx + east) & st.y.between(cy - h, cy + h)].itertuples():
            dx, dy = off.get(r.code, (6, 4))
            a.annotate(r.code, (r.x, r.y), xytext=(dx, dy), textcoords="offset points", ha="left" if dx > 0 else "right", fontsize=7, color=INK2,
                       path_effects=_halo(), zorder=7)
            if r.code in set(inside.code):
                ring = geod.fwd(np.full(181, r.lon), np.full(181, r.lat), np.linspace(0, 360, 181), np.full(181, reach))
                a.plot(*TO3338.transform(ring[0], ring[1]), color=INK, lw=0.8, zorder=2)
                if key == "denali" and r.code == "TRF" and not ring_done:
                    a.annotate(f"{reach / 1e3:g} km", (r.x + reach * 0.7, r.y - reach * 0.7), xytext=(14, -12), textcoords="offset points", fontsize=7,
                               color=INK, path_effects=_halo(), zorder=7, arrowprops=dict(arrowstyle="-", color=INK, lw=0.5, shrinkA=0, shrinkB=0))
                    ring_done = True
        if key == "valdez":
            r = st[st.code == "VLZ"].iloc[0]
            a.annotate(f"VLZ stands {vlz_km:.1f} km\nfrom the town", (r.x, r.y), xytext=(16, -30), textcoords="offset points", fontsize=7, color=INK, ha="left",
                       path_effects=_halo(), zorder=7, arrowprops=dict(arrowstyle="-", color=INK, lw=0.5, shrinkA=2, shrinkB=3))
        a.plot([cx + bx, cx + bx + bar], [cy - 0.9 * h] * 2, color=INK, lw=1.5, zorder=6)
        a.text(cx + bx + bar / 2, cy - 0.86 * h, f"{bar / 1e3:g} km", ha="center", fontsize=7, color=INK, path_effects=_halo(), zorder=7)
        a.text(cx + west + 0.02 * (east - west), cy + 0.95 * h, {"denali": "b", "valdez": "c"}[key], fontsize=9, fontweight="bold", color=INK, va="top",
               path_effects=_halo(), zorder=7)
        a.set(xlim=(cx + west, cx + east), ylim=(cy - h, cy + h), aspect="equal", xticks=[], yticks=[])
        a.grid(False)
        for sp in a.spines.values():
            sp.set(visible=True, color=INK2, lw=0.6)
    fig.canvas.draw()
    over = fig.get_tightbbox(fig.canvas.get_renderer())
    assert over.x1 <= W + 0.01 and over.y1 <= H + 0.01, f"text runs past the figure ({over.x1:.2f} x {over.y1:.2f} in)"
    save(fig, "fig1_map")


def fig8() -> None:
    """Supplement Two: the area screened by the terrain and airborne searches, and the control tiles of the terrain screen,
    over Alaska, with the two boxes. Near Valdez the screened area is the Valdez box."""
    import json

    from shapely.geometry import box, shape
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from akp.stage_p2.p2a_strata import TO3338, to_3338
    from akp.zones import ANCHORS, BOXES

    full, (x0, y1, x1, y0) = _land()
    X0, X1, Y0, Y1 = -1.25e6, 1.55e6, 0.38e6, 2.4e6
    W = 6.85  # with the 0.1 in that a saved figure adds about it, 180 mm
    H = W * (Y1 - Y0) / (X1 - X0)
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    gray = matplotlib.colors.ListedColormap([GRID])
    ax.imshow(np.where(_tidy_land(full[::4, ::4], 12), 1.0, np.nan), extent=(x0, x1, y0, y1), cmap=gray, interpolation="nearest", zorder=0)
    for t in pd.read_csv(P / "p2a_controls.csv").itertuples():
        g = to_3338(box(t.min_lon, t.min_lat, t.max_lon, t.max_lat).segmentize(0.05))
        ax.fill(*g.exterior.xy, color=INK2, alpha=0.4, lw=0, zorder=1)
    for f in json.loads((P / "p2a_searched_area.geojson").read_text())["features"]:
        g = to_3338(shape(f["geometry"]))
        ax.fill(*g.exterior.xy, color=S1, alpha=0.22, lw=0, zorder=2)
        ax.plot(*g.exterior.xy, color=S1, lw=1.1, zorder=3)
    for b in BOXES.values():
        g = to_3338(box(b["minlongitude"], b["minlatitude"], b["maxlongitude"], b["maxlatitude"]).segmentize(0.05))
        ax.plot(*g.exterior.xy, color=INK2, lw=0.8, ls="--", zorder=4)
    for name, (la, lo) in ANCHORS.items():
        x, y = TO3338.transform(lo, la)
        ax.plot(x, y, marker="+", ms=6, mew=1.0, color=INK, zorder=5)
        ax.annotate(name.capitalize(), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=7.5, color=INK, path_effects=_halo(), zorder=6)
    ax.plot([1.0e6, 1.2e6], [0.47e6, 0.47e6], color=INK, lw=1.5)
    ax.text(1.1e6, 0.49e6, "200 km", ha="center", fontsize=7.5, color=INK)
    handles = [Patch(facecolor=S1, alpha=0.3, edgecolor=S1), Patch(facecolor=INK2, alpha=0.4, edgecolor="none"),
               Line2D([], [], color=INK2, lw=0.8, ls="--"), Line2D([], [], ls="", marker="+", ms=6, mew=1.0, color=INK)]
    ax.legend(handles, ["screened area", "control tile of the terrain screen", "box around claimed locations (our construction)", "town"],
              loc="upper right", fontsize=7.5, frameon=False, bbox_to_anchor=(0.99, 0.98))
    ax.set(xlim=(X0, X1), ylim=(Y0, Y1), aspect="equal", xticks=[], yticks=[])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, "fig8_screen")


ROWS_B = {"claimant": "The claimant", "c2c": "Coast to Coast AM, radio", "earthfiles": "Earthfiles, L. M. Howe", "dreamland": "Dreamland, radio",
          "gaia": "Gaia", "history": "History: Ancient Aliens", "travel": "Travel Channel, discovery+", "whyfiles": "The Why Files",
          "reddit": "Reddit, r/aliens", "critique": "Critiques", "other": "Other web pages, podcasts, videos, print"}


def fig2_timeline() -> None:
    """Figure 1 of the paper: when the story's versions appear, against the record of 1992-93.

    One mark per meaning: a filled dot is a date known to the day, an open dot a date known to the month, a capped
    line a date known only to a year or a range (a span of possible dates, not a duration), a gray box an archive
    window searched. The broadcast that the versions describe is a band across panel a. Labels stand in one column at
    the left, wrapped to fit it; the kind of source and the stated depths stand in a table at the right of panel b.
    The layout is set in inches, so the figure prints at 180 mm wide without scaling its text."""
    import textwrap

    import matplotlib.dates as mdates
    from matplotlib.font_manager import FontProperties
    from matplotlib.legend_handler import HandlerLine2D
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.textpath import TextPath

    d = pd.read_csv(OUT / "fig2_claims.csv", dtype=str).fillna("")
    assert (d.label.str.split().str.len() <= 8).all(), "labels: at most 8 words, own words (no transcript text)"
    d = d[d.date != ""].copy()
    span = d.date.str.split("/")  # "2012" is the whole year, "1993-02" the month, "a/b" from a to b
    d["t0"] = [pd.Period(s[0]).start_time for s in span]
    d["t1"] = [pd.Period(s[-1]).end_time.normalize() for s in span]
    d["mark"] = ["day" if p == "day" else "month" if len(s) == 1 and len(s[0]) == 7 else "span"
                 for p, s in zip(d.date_precision, span)]
    stated = d[d.kind == "claim_stated"].iloc[0]  # the broadcast as the versions date it: a band, not a row
    searched = d[d.date_precision == "archive window searched"]
    assert (searched.date == searched.date.iloc[0]).all(), "one archive window"
    d = d[(d.kind != "claim_stated") & (d.date_precision != "archive window searched")].sort_values(["t0", "t1"], kind="stable")
    early = d[d.t0.dt.year < 2000]
    v = pd.read_csv(OUT / "fig2_versions.csv", dtype=str).fillna("")
    undated, v = v[v.date == ""], v[v.date != ""].copy()
    vs = v.date.str.split("/")
    v["t0"] = [pd.Period(s[0]).start_time for s in vs]
    v["t1"] = [pd.Period(s[-1]).end_time.normalize() for s in vs]
    v["mark"] = ["day" if p == "day" else "month" if p == "month" else "span" for p in v.precision]
    v["kind"] = np.where(v.type == "critique", "context", "claim")
    assert set(v.row) <= set(ROWS_B), "every version falls in a row of panel b"
    late = v
    color = {"claim": S1, "context": S2}
    SHADE, BOX = "#f3f2ee", "#d9d8d2"  # alternate rows; an archive window
    LAB, DATE, SMALL = 7.5, 7.0, 7.0  # pt

    def when(r) -> str:
        return r.date.replace("/", " to ") + ("" if r.date_precision == "day" else f" ({r.date_precision})")

    rows_a = [dict(id=r.id, label=r.label, when=when(r), kind=r.kind, mark=r.mark, t0=r.t0, t1=r.t1) for r in early.itertuples()]
    s0 = searched.iloc[0]
    rows_a.append(dict(id=", ".join(searched.id), label="Anchorage Daily News, AP and New York Times archives searched: no such story",
                       when=s0.date.replace("/", " to "), kind="context", mark="searched", t0=s0.t0, t1=s0.t1))
    rows_b = []
    for key, label in ROWS_B.items():
        g = v[v.row == key]
        ids = ", ".join(sorted((i for i in g.id if i), key=lambda i: (i[0], int(i[1:]))))
        rows_b.append(dict(id="", label=label + (f" ({ids})" if ids else ""), kind="claim", mark="many", t0=g.t0.min(), t1=g.t1.max(),
                           when=f"{len(g)} item{'s' * (len(g) > 1)}", marks=list(zip(g.t0, g.t1, g.mark, g.kind))))

    # geometry, inches
    W, L0, L1, P0, P1B, T0, R = 7.0, 0.06, 2.92, 3.02, 4.92, 5.08, 6.94
    ROW, ROW_B, LINE = 0.30, 0.20, 0.115  # a one-line row (panel a: label and date; panel b: one line); each further line of a wrapped label
    font = FontProperties(family=plt.rcParams["font.family"])

    def wrap(s: str, width_in: float) -> list[str]:
        words, out = s.split(), [""]
        for w in words:
            t = (out[-1] + " " + w).strip()
            if TextPath((0, 0), t, size=LAB, prop=font).get_extents().width / 72 <= width_in or not out[-1]:
                out[-1] = t
            else:
                out.append(w)
        return out

    for r in rows_a + rows_b:
        r["lines"] = wrap((r["id"] + "\u2002" if r["id"] else "") + r["label"] + (f", {r['when']}" if r["mark"] == "many" else ""), L1 - L0 - 0.08)
        r["h"] = (ROW_B if r["mark"] == "many" else ROW) + LINE * (len(r["lines"]) - 1)
    KEY, OV, TITLE, HEAD, TICKS, GAP, FOOT = 0.40, 0.34, 0.26, 0.16, 0.40, 0.14, 0.30
    h_a, h_b = sum(r["h"] for r in rows_a), sum(r["h"] for r in rows_b)
    H = KEY + 0.10 + OV + 0.26 + GAP + TITLE + h_a + TICKS + GAP + TITLE + HEAD + h_b + 0.44 + FOOT
    assert H <= 200 / 25.4, f"the figure is {H:.2f} in tall; the journal allows 200 mm"
    fig = plt.figure(figsize=(W, H))

    def axes(x0, x1, top, height):  # inches from the top-left corner
        return fig.add_axes((x0 / W, (H - top - height) / H, (x1 - x0) / W, height / H))

    # the key
    y = 0.04
    key = axes(L0, R, y, KEY)
    key.axis("off")
    span_h = Line2D([], [], color=INK, lw=1.4, marker="|", ms=7, mew=1.4)
    handles = [Line2D([], [], color=S1, marker="o", ls="", ms=5.5), Line2D([], [], color=S2, marker="o", ls="", ms=5.5),
               Patch(facecolor=S1_LIGHT, alpha=0.45, edgecolor="none"), Patch(facecolor=BOX, edgecolor=INK2, lw=0.6),
               Line2D([], [], color=INK, marker="o", ls="", ms=5.5), Line2D([], [], color=INK, marker="o", ls="", ms=5.5, mfc="white", mew=1.2),
               span_h]
    names = ["telling of the claim", "documented record or critique", "broadcast, as the versions date it", "archive searched",
             "exact date", "month only", "year or range only"]
    key.legend(handles, names, loc="upper left", ncol=4, fontsize=LAB, handlelength=1.8, columnspacing=1.2, borderaxespad=0,
               handler_map={span_h: HandlerLine2D(numpoints=2)})

    # the whole record: the two panels boxed, the gap between them named
    y += KEY + 0.10
    ov = axes(P0, R, y, OV)
    fig.text(L0 / W, 1 - (y + OV / 2) / H, "The whole record, 1990–2026", fontsize=9, fontweight="bold", color=INK, va="center")
    ov.set(xlim=(pd.Timestamp("1990-01-01"), pd.Timestamp("2027-12-31")), ylim=(-1, 1), yticks=[])
    ov.set_facecolor("none")
    ov.grid(False)
    ov.spines["left"].set_visible(False)
    ov.xaxis.set_major_locator(mdates.YearLocator(5))
    ov.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ov.tick_params(axis="x", labelsize=LAB)
    boxes = {}
    for (a, b), letter in (((min(early.t0.min(), s0.t0), s0.t1), "a"), ((v.t0.min(), v.t1.max()), "b")):
        a, b = a - pd.Timedelta(days=120), b + pd.Timedelta(days=120)
        boxes[letter] = (a, b)
        ov.add_patch(Rectangle((mdates.date2num(a), -0.8), mdates.date2num(b) - mdates.date2num(a), 1.6, facecolor="none", edgecolor=INK2, lw=0.8))
        ov.text(a, 0.9, letter, ha="left", va="bottom", fontsize=8.5, fontweight="bold", color=INK)
    marks_b = [dict(t0=t0, t1=t1, mark=mk, kind=k) for r in rows_b for t0, t1, mk, k in r["marks"]]
    for r in sorted(rows_a + marks_b, key=lambda r: r["mark"] != "searched"):  # the archive window first, the marks over it
        ov.plot([r["t0"], max(r["t1"], r["t0"] + pd.Timedelta(days=50))], [0, 0], color=BOX if r["mark"] == "searched" else color[r["kind"]],
                lw=3, solid_capstyle="butt")
    g0, g1 = boxes["a"][1] + pd.Timedelta(days=90), boxes["b"][0] - pd.Timedelta(days=90)
    ov.annotate("", xy=(g1, -0.1), xytext=(g0, -0.1), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.8))
    ov.text(g0 + (g1 - g0) / 2, 0.1, f"nothing registered, {s0.t1.year + 1}\u2013{v.t0.min().year - 1}",
            ha="center", va="bottom", fontsize=SMALL, color=INK)

    def panel(top, rows, x1, lo, hi, major, minor, fmt, title, table):
        fig.text(L0 / W, 1 - (top + TITLE / 2) / H, title, fontsize=9, fontweight="bold", color=INK, va="center")
        top += TITLE + (HEAD if table else 0)
        hgt = sum(r["h"] for r in rows)
        lab, ax = axes(L0, L1, top, hgt), axes(P0, x1, top, hgt)
        tab = axes(T0, R, top, hgt) if table else None
        edges = np.cumsum([0.0] + [r["h"] for r in rows])  # inches from the top of the rows
        mids = (edges[:-1] + edges[1:]) / 2
        for a in (lab, ax, tab):
            if a is None:
                continue
            a.set_ylim(hgt, 0)
            for i in range(1, len(rows), 2):
                a.axhspan(edges[i], edges[i + 1], color=SHADE, lw=0, zorder=0)
        lab.set_xlim(0, 1)
        lab.axis("off")
        for r, m, e in zip(rows, mids, edges):
            n = len(r["lines"])
            lab.text(0.02, e + 0.045, "\n".join(r["lines"]), fontsize=LAB, color=INK, va="top", linespacing=1.15)
            if r["mark"] != "many":
                lab.text(0.02, e + 0.045 + LINE * n + 0.005, r["when"], fontsize=DATE, color=INK2, va="top")
        ax.set(xlim=(lo, hi), yticks=[])
        ax.set_facecolor("none")
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.xaxis.set_major_locator(major)
        ax.xaxis.set_minor_locator(minor)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
        ax.tick_params(axis="x", labelsize=LAB)
        flat = [(dict(t0=t0, t1=t1, mark=mk, kind=k, ms=4.0, alpha=0.8), m) for r, m in zip(rows, mids) if r["mark"] == "many" for t0, t1, mk, k in r["marks"]]
        for r, m in [(r, m) for r, m in zip(rows, mids) if r["mark"] != "many"] + flat:
            c = color[r["kind"]]
            ms, al = r.get("ms", 5.5), r.get("alpha", 1.0)  # panel b: smaller, so that marks close in time stay apart
            if r["mark"] == "day":
                ax.plot(r["t0"], m, "o", ms=ms, color=c, alpha=al, mec="white", mew=0.4, zorder=3)
            elif r["mark"] == "month":
                ax.plot(r["t0"] + (r["t1"] - r["t0"]) / 2, m, "o", ms=ms, mfc="white", mec=c, mew=1.3, zorder=3)
            elif r["mark"] == "searched":
                x0, x1_ = mdates.date2num(r["t0"]), mdates.date2num(r["t1"])
                ax.add_patch(Rectangle((x0, m - 0.07), x1_ - x0, 0.14, facecolor=BOX, edgecolor=INK2, lw=0.6, zorder=2))
            else:
                ax.plot([r["t0"], r["t1"]], [m, m], color=c, lw=1.6, zorder=3)
                ax.plot([r["t0"], r["t1"]], [m, m], ls="", marker="|", ms=7, mew=1.6, color=c, zorder=3)
        if tab is not None:
            tab.set_xlim(0, 1)
            tab.axis("off")
            cols = (("src", "Source", 0.0, "left"), ("top", "Top", 0.58, "right"), ("base", "Base", 0.78, "right"), ("height", "Height", 1.0, "right"))
            tab.text(1.0, -HEAD - TITLE / 2, "in meters", ha="right", va="center", fontsize=SMALL, color=INK2)
            for _, head, x, ha in cols:
                tab.text(x, -0.04, head, ha=ha, va="bottom", fontsize=LAB, color=INK2)
            tab.plot([0.0, 1.0], [-0.02, -0.02], color=INK2, lw=0.5, clip_on=False)
            for r, m in zip(rows, mids):
                if r["kind"] == "claim":
                    for k, _, x, ha in cols:
                        tab.text(x, m, r[k] or "–", ha=ha, va="center", fontsize=LAB, color=INK)
        return ax

    y += OV + 0.26 + GAP
    ax_a = panel(y, rows_a, R, min(early.t0.min(), s0.t0) - pd.Timedelta(days=15), s0.t1 + pd.Timedelta(days=20),
                 mdates.MonthLocator(bymonth=(1, 4, 7, 10)), mdates.MonthLocator(), "%b\n%Y", "a  The explosion and the record, 1992–93", False)
    b0, b1 = mdates.date2num(stated.t0), mdates.date2num(stated.t1)
    ax_a.axvspan(b0, b1, color=S1_LIGHT, alpha=0.45, lw=0, zorder=1)
    ax_a.text(b1 + 4, 0.04, "broadcast as the versions date it:\nlate November or early December 1992", ha="left", va="top",
              fontsize=SMALL, color=INK, linespacing=1.1, zorder=4)
    y += TITLE + h_a + TICKS + GAP
    panel(y, rows_b, R, pd.Timestamp("2011-06-01"), pd.Timestamp("2026-12-31"), mdates.YearLocator(2), mdates.YearLocator(1), "%Y",
          "b  Tellings of the claim and critiques, by outlet, 2012–26", False)
    fig.canvas.draw()
    over = fig.get_tightbbox(fig.canvas.get_renderer())
    assert over.x1 <= W + 0.01 and over.y1 <= H + 0.01, f"text runs past the figure ({over.x1:.2f} x {over.y1:.2f} in)"
    fig.text(L0 / W, 0.12 / H, textwrap.fill("Not placed, no source located: " + "; ".join(f"{r.id} {r.outlet}" for r in undated.itertuples()) +
                                             ". Every item, with its date and source, is listed in Supplement One.", 128),
             fontsize=SMALL, color=INK2, va="bottom")
    save(fig, "fig2_timeline")


def fig3() -> None:
    d = pd.read_csv(A / "a1_box_coverage.csv")
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for col, color, ls, label in (("km_nearest_recorder", S1, "-", "nearest station that recorded the explosion"),
                                  ("km_nearest_operating", S2, "--", "nearest station, recorded or operating")):  # the table counts a station that recorded as operating
        x = np.sort(d[col].to_numpy())
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=color, ls=ls, lw=1.5, label=label)
    ax.set(xlabel="Distance from a point of the Denali\u2013Nome box (km)", ylabel="Cumulative fraction of the box",
           xlim=(0, None), ylim=(0, 1.02))
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "fig3_distance")


def fig4() -> None:
    from matplotlib.lines import Line2D

    from akp.stage_p4.s11_versions import simulations

    d = pd.read_csv(A / "a8_production.csv")
    d = d[(d.window == "slow") & (d.component == "vertical")].rename(columns={"numerical": "share"})
    d = pd.concat([d, pd.read_csv(A / "a8_generous_shares.csv"), pd.read_csv(A / "a8_gradient_shares.csv")], ignore_index=True).query("azimuth == 'forward'")  # the velocity-gradient case as measured
    key = ["case", "band", "range_m"]
    flag = simulations().query("azimuth == 'forward'").set_index(key).edge_flag  # the first four cases carry no committed flag: the rule, as the table of versions applies it
    d = d.drop(columns="edge_flag").join(flag.rename("flag"), on=key)
    assert d.flag.notna().all()
    d["case"] = d.case.replace("R1", "thawed_850")  # R1 is the slower thawed ground again, on a finer grid
    d = d.drop_duplicates(key)  # production comes first: R1 is left where production has no run
    clutter = pd.read_csv(A / "a5_dprk_site_spread.csv").query("box == 'mckinley_nome' and event == 'all' and lag == '5-15'").set_index("band")
    bands = ["0.5-1", "1-2", "2-4"]
    cases = (("thawed_1200", S1, "o", S1, ""),  # color, marker and its face: no case rests on color alone
             ("thawed_850", S1_LIGHT, "o", "white", " (finer grid at 2–4 Hz)"),
             ("frozen_1200", S2, "s", S2, ""), ("frozen_850", S4, "s", "white", ""),
             ("R2", S6, "D", S6, ""), ("R3", S5, "^", S5, ""), ("R4", S3, "v", S3, ""), ("R5", INK, "X", "white", ""))
    fig, axes = plt.subplots(1, len(bands), figsize=(7.0, 3.3), sharey=True)
    fig.subplots_adjust(wspace=0.6)  # room for the labels of the coda amplitude
    for ax, band in zip(axes, bands):
        c = clutter.loc[band]
        ax.axhspan(c.q25, c.q75, color=GRID, lw=0)
        ax.axhline(c["median"], color=INK2, lw=0.8)
        ax.axhline(c.p05, color=INK2, lw=0.8, ls=":")
        for y, va, label in ((c.q75, "bottom", "middle half\nof records"), (c["median"], "center", "median"), (c.p05, "center", "5th percentile\nof records")):
            ax.annotate(label, (1, y), xycoords=ax.get_yaxis_transform(), xytext=(3, 0), textcoords="offset points", va=va,
                        fontsize=7, color=INK2, annotation_clip=False)
        for k, (case, color, marker, face, _) in enumerate(cases):
            g = d[(d.case == case) & (d.band == band)].sort_values("range_m")
            for rows, ls in ((~g.flag, "-"), (g.flag | g.flag.shift(-1, fill_value=False), "--")):  # the dashes start at the last clean row
                ax.plot(g.range_m / 1e3, g.share.where(rows), color=color, ls=ls, lw=1.5, marker=marker, ms=4, mfc=face,
                        markevery=(k, len(cases)))  # staggered, so the markers of two cases do not share a range
        ax.set_yscale("log")
        ax.set(title=f"{band.replace('-', '–')} Hz", xlabel="Range from the body (km)", xlim=(0, 3.1), ylim=(3e-3, 1), xticks=range(4))
        ax.title.set_fontsize(9)
    axes[0].set_ylabel("Amplitude ratio (scattered wave over incident P)")
    handles = [Line2D([], [], color=color, lw=1.5, marker=marker, ms=4, mfc=face, label=CASE[case] + more) for case, color, marker, face, more in cases]
    handles.append(Line2D([], [], color=INK2, lw=1.5, ls="--", label="dashed: the source-edge effect can arrive in the window"))
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.96), fontsize=7, columnspacing=1.0, handlelength=2.6)
    save(fig, "fig4_scatter")


def fig5() -> None:
    from akp.stage_p2.p2e_pd import summary
    from akp.stage_p2.p2e_vintage import frozen_cut

    run = pd.read_csv(P / "p2e_cut.csv").iloc[0]
    v = pd.read_csv(P / "p2e_vetoes.csv").set_index("stratum").loc["_total"]
    blind = float(1 - v.searched_unvetoed_km2 / v.searched_km2)
    ctl = summary(pd.read_csv(P / "p2e_pd_trials.csv", dtype={"tile_id": str}), float(run.t), float(frozen_cut().c), blind)
    clm = pd.read_csv(P / "p2e_claim_pd.csv").query("scope == 'all'")
    bases = sorted(clm.base.unique())
    fig, axes = plt.subplots(1, len(bases), figsize=(7.0, 2.4), sharey=True)
    for ax, b in zip(axes, bases):
        for d, color, ls, label in ((clm, S1, "-", "screened area, measured"), (ctl, S2, "--", "control prediction")):
            g = d[d.base == b].sort_values("height")
            ax.fill_between(g.height, g.ci_low, g.ci_high, color=color, alpha=0.15, lw=0)
            ax.plot(g.height, g.p_detect, color=color, ls=ls, lw=1.5, marker="o", ms=3.5, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.5, 2, 8, 16], ["0.5", "2", "8", "16"])
        ax.set_title(f"base {b:.0f} m", fontsize=9, color=INK)
        ax.axhline(0.9, color=INK2, lw=0.8, ls=":")
        ax.set_xlabel("Relief (m)")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Detection probability")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    save(fig, "fig5_pd")


def fig6() -> None:
    """Prior mass within a distance of any seismic receiver ever archived (akp/stage_p4/s9_census.py), and the
    ranges of the census rows for the body of doubled stiffness (s9) and for stone (s10_census_generous.py, case R2)."""
    from matplotlib.lines import Line2D

    s9 = REPO / "akp/stage_p4"
    e = pd.read_csv(s9 / "s9_exposure.csv").query("cls == 'all' and uncertainty == 'no'")
    c = pd.read_csv(s9 / "s9_checks.csv").set_index("check").value
    h = pd.read_csv(s9 / "s9_high_prior.csv").set_index("quantity").value
    t = pd.read_csv(s9 / "s10_sensitivity.csv").query("case == 'R2' and cls == 'all' and uncertainty == 'added'")
    last = float(t.last_receiver_km.max())  # beyond the last simulated receiver a range comes from the fitted law alone

    def doubled(check: str) -> tuple[float, bool]:
        return float(c[check]), float(c[check]) > last

    def stone(variant: str, attenuation: str, level: str | None = None) -> tuple[float, bool]:
        g = t[(t.body_variant == variant) & (t.attenuation == attenuation)]
        r = (g[g.clutter_level == level] if level else g.nlargest(1, "range_km")).iloc[0]  # one clutter level, or every choice
        return float(r.range_km), bool(r.at_least)

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.plot(e.r_km, e.prior, color=S1, lw=2, marker="o", ms=4, label="claim-weighted prior")
    ax.plot(e.r_km, e.cover, color=S2, lw=2, marker="o", ms=4, label="cover-weighted prior")
    two, every, sim = "high-contrast body:", "all favorable assumptions,", "simulated effect of the base"  # plain terms
    marks = ((doubled("typical_site_range_km"), INK2, two.replace(" ", "\n", 1) + " median\ncoda amplitude", 1, "left"),  # row of the label above the panel, and its side
             (doubled("d83_row_range_km"), INK2, two + " 5th\npercentile of records,\nno attenuation", 1, "right"),
             (stone("as_simulated", "none", "cleanest_twentieth"), S6, "crystalline body: 5th percentile\nof records, no attenuation", 2, "right"),
             (stone("measured_base", "generous"), S6, f"crystalline body: {every}\nQ of 100, {sim}", 3, "right"),
             (doubled("largest_range_generous_attenuation_km"), INK2, f"{two} {every}\nQ of 100, assumed scaling with base", 3, "left"),
             (stone("measured_base", "none"), S6, f"crystalline body: all favorable\nassumptions, no attenuation,\n{sim}", 2, "left"),
             (doubled("largest_range_no_attenuation_km"), INK2, f"{two} all favorable\nassumptions, no attenuation,\nassumed scaling with base", 1, "left"))
    row = 29 / (72 * fig.get_figheight() * ax.get_position().height)  # 29 points (three lines of text), as a share of the panel's height
    for (x, far), color, label, level, ha in marks:
        ax.plot([x, x], [0, 1 + level * row], transform=ax.get_xaxis_transform(), color=color, lw=0.8 if color == INK2 else 1.2,
                ls="--" if far else ":", clip_on=False)
        ax.annotate(label, (x, 1 + level * row), xycoords=ax.get_xaxis_transform(), xytext=(3 if ha == "left" else -3, 0),
                    textcoords="offset points", ha=ha, va="top", fontsize=7, color=INK2, annotation_clip=False)
    x = float(h["nearest_receiver_km_min"])
    ax.axvline(x, color=INK, lw=0.8)
    ax.text(x * 1.07, 1.6e-4, "nearest\nreceiver to\nthe smallest\narea holding\nhalf of the\ncover-weighted\nprior", fontsize=7.5, color=INK, va="bottom")
    ax.set(xscale="log", yscale="log", xlim=(0.15, 60), ylim=(1e-4, 1.5), xlabel="Distance from the nearest seismic receiver (km)",
           ylabel="Fraction of the location prior within that distance")
    handles = ax.get_legend_handles_labels()[0] + [
        Line2D([], [], color=INK2, lw=0.8, label="gray: exceedance range of the high-contrast body"),
        Line2D([], [], color=S6, lw=1.2, label="green, heavier: exceedance range of the crystalline body"),
        Line2D([], [], color=INK, lw=0.8, ls=":", label="dotted: range inside the simulated receivers"),
        Line2D([], [], color=INK, lw=0.8, ls="--", label="dashed: range beyond the last simulated receiver, from the fitted law")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=7.5)
    save(fig, "fig6_exposure")


def fig7() -> None:
    """Figure 3 of the paper: how large the scattered wave could be, against distance (left) and against the coda amplitude it
    would arrive in (right), by band.

    Left: the range of the simulated amplitude ratio over the cases and lines to the last receiver, the range at which its
    largest value meets the median coda amplitude (a5_late_crossing.csv), and beyond the last receiver each band's own bound by
    geometrical spreading alone, with the nearest stations of the versions on it. Right, on the same vertical axis: the P coda before later phases
    over the reference records, the 1992 record at IU.COL, and the largest bound that could still arrive at each time: it
    steps down at each station's latest arrival, so it is the upper envelope of the stations' bounds. Past the end of the
    clean level the region is shaded: there the rule gives no outcome. Asserts that the curve drawn gives the table's bounds."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    from akp.stage_a import a5_late_windows as late

    sim = late.simulated()
    far = sim[sim.range_m >= late.START_M].assign(k=lambda d: d.ratio * np.sqrt(d.range_m))
    k = far.groupby("band").k.max()  # each band's own start: the largest ratio times root range at or beyond START_M
    last = float(sim.range_m.max()) / 1e3
    marks = pd.read_csv(A / "a5_late_by_band_summary.csv").query("kind != 'open waveform'")
    assert np.allclose(marks.bound_band, [k[b] / np.sqrt(km * 1e3) for b, km in zip(marks.band, marks.km)], rtol=0, atol=2e-6), "the curve drawn is the bound of the table"
    lev = pd.read_csv(A / "a5_late_levels.csv").query("box == 'mckinley_nome' and way == 'valid' and level == 'clean'")
    lev = lev.assign(t0=lev.lag.str.split("-").str[0].astype(float), t1=lev.lag.str.split("-").str[1].astype(float)).sort_values("t0")
    cross = pd.read_csv(A / "a5_late_crossing.csv").query("level == 'pooled median, way valid'").set_index("band").range_m / 1e3  # km
    clean_end = value(A / "a5_late_checks.csv", "clean_level_ends_s_max")  # s after P
    style = {"recorded": dict(marker="o", mfc=INK), "operating": dict(marker="^", mfc="white")}
    bands = list(k.index)
    xlim, tlim, ylim, dash = (0.09, 420.0), (5.0, 700.0), (5e-4, 6.0), chr(0x2013)
    NOLEVEL = "#efeee9"

    W, H = 6.9, 6.55  # with the 0.1 in that a saved figure adds about it, 180 mm wide
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(len(bands), 2, width_ratios=[1.45, 1.0], hspace=0.10, wspace=0.05, left=0.095, right=0.99, top=0.855, bottom=0.10)
    for i, band in enumerate(bands):
        ax, tx = fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])
        g = sim[sim.band == band]
        env = g.groupby("range_m").ratio.agg(["min", "max"])
        ax.fill_between(env.index / 1e3, env["min"], env["max"], color=S1, alpha=0.22, lw=0)
        ax.plot(env.index / 1e3, env["max"], color=S1, lw=1.5)
        x = np.geomspace(late.START_M / 1e3, xlim[1], 100)
        ax.plot(x, k[band] / np.sqrt(x * 1e3), color=S2, lw=1.5, ls="--")
        ax.axvline(last, color=INK, lw=0.6)
        first = lev[(lev.band == band) & (lev.t0 == lev.t0.min())].iloc[0]  # the first window, the one a wave from so near arrives in
        near = [xlim[0], last]
        ax.plot(near, [first["median"]] * 2, color=INK2, lw=1.1)
        ax.plot(near, [first.p05] * 2, color=INK2, lw=1.1, ls=":")
        ax.plot(near, [first.col1992] * 2, color=S3, lw=1.3)
        xc = float(cross[band])
        if xc > 0:  # where the largest simulated ratio meets the median coda amplitude
            ax.plot([xc, xc], [first["median"] / 1.8, first["median"] * 1.8], color=INK, lw=1.0, zorder=5)
            ax.annotate(f"reaches the median\nwithin {xc:g} km", (xc, first["median"] * 1.8), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, color=INK, zorder=5)
        else:
            ax.text(0.55, first["median"] * 1.5, "below the median\nat every simulated range", fontsize=7, color=INK, va="bottom", ha="center", zorder=5)
        mb = marks[marks.band == band].sort_values("km")
        for m in mb.itertuples():
            ax.plot(m.km, m.bound_band, ls="none", color=INK, ms=4, mew=0.8, marker=style[m.kind]["marker"], mfc=style[m.kind]["mfc"], zorder=4)
        if i == 0:
            for m, ha, dy in ((mb.iloc[0], "center", 9), (mb.iloc[-1], "right", 9)):  # the farthest label ends at its mark, inside the panel
                ax.annotate(f"{m.station}, {m.km:.1f} km", (m.km, m.bound_band), xytext=(0, dy), textcoords="offset points", ha=ha, va="bottom",
                            fontsize=7, color=INK)
            ax.text(last * 1.1, ylim[0] * 1.3, "beyond the last receiver:\nbound by geometrical spreading alone", va="bottom", fontsize=7, color=S2)

        # right: the levels, the region with no level of the P coda before later phases, and the largest bound that could still arrive
        tx.axvspan(clean_end, tlim[1], color=NOLEVEL, lw=0, zorder=0)
        c = lev[lev.band == band]
        ok = c[c.n > 0]
        edges = np.r_[ok.t0.to_numpy(), clean_end]  # the last level ends where the latest reference record meets its first later phase
        tx.stairs(ok["median"], edges, baseline=None, color=INK2, lw=1.1, zorder=2)
        tx.stairs(ok.p05, edges, baseline=None, color=INK2, lw=1.1, ls=":", zorder=2)
        tx.stairs(c.col1992, np.r_[c.t0.to_numpy(), c.t1.iloc[-1]], baseline=None, color=S3, lw=1.3, zorder=2)
        by_end = mb.sort_values("delay_max_s")
        ends = by_end.delay_max_s.to_numpy()
        top = [float(by_end[by_end.delay_max_s >= e].bound_band.max()) for e in ends]  # on each interval, the largest bound still to arrive
        tx.stairs(top, np.r_[tlim[0], ends], baseline=None, color=S2, lw=1.6, zorder=3)
        for m in by_end.itertuples():
            tx.plot(m.delay_max_s, m.bound_band, ls="none", color=INK, ms=3.6, mew=0.7, marker=style[m.kind]["marker"], mfc=style[m.kind]["mfc"], zorder=4)
        tx.axvline(clean_end, color=INK, lw=0.6)
        if i == 0:
            tx.text(clean_end * 1.06, 4.5, "no reference\nP-coda level:\nno outcome", va="top", fontsize=7, color=INK)
        ax.set(xscale="log", yscale="log", xlim=xlim, ylim=ylim)
        tx.set(xscale="log", yscale="log", xlim=tlim, ylim=ylim)
        tx.tick_params(labelleft=False)
        for a_ in (ax, tx):
            a_.grid(axis="x", visible=False)
            a_.tick_params(labelbottom=i == len(bands) - 1)
        ax.text(0.985, 0.95, f"{band.replace('-', dash)} Hz", transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK)
        ax.text(xlim[0] * 1.05, ylim[0] * 1.3, f"simulated, {g.case.nunique()} cases", va="bottom", fontsize=7, color=S1)
        for a_, letter in ((ax, "abc"[i]), (tx, "def"[i])):
            a_.text(0.012, 0.95, letter, transform=a_.transAxes, ha="left", va="top", fontsize=9, fontweight="bold", color=INK)
        if i == 0:
            ax.set_title("Near the body, and at the stations' distances", fontsize=8.5, color=INK, loc="left", pad=4)
            tx.set_title("Against the coda, by time after the P peak", fontsize=8.5, color=INK, loc="left", pad=4)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    tx.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    tx.set_xticks([5, 15, 30, 60, 120, 240, 480])
    tx.minorticks_off()
    ax.set_xlabel("Range from the body; for a station, distance from the claimed location (km)", fontsize=8)
    tx.set_xlabel("Time after the P peak (s)", fontsize=8)
    fig.supylabel("Amplitude, as a fraction of the P wave", x=0.012, y=0.49, fontsize=8.5)
    handles = [Line2D([], [], lw=6, color=S1, alpha=0.3), Line2D([], [], lw=1.5, ls="--", color=S2), Line2D([], [], lw=1.6, color=S2),
               Line2D([], [], lw=1.1, color=INK2), Line2D([], [], lw=1.1, ls=":", color=INK2), Line2D([], [], lw=1.3, color=S3),
               Line2D([], [], ms=4.5, mew=0.8, mec=INK, color=INK, ls="none", marker="o", mfc=INK),
               Line2D([], [], ms=4.5, mew=0.8, mec=INK, color=INK, ls="none", marker="^", mfc="white")]
    labels = ["simulated: range over cases and lines", "bound by geometrical spreading alone", "largest bound that could still arrive",
              "P coda of the reference records: median", "the same: 5th percentile", "the 1992 record at IU.COL",
              "nearest station that recorded the explosion", "nearest station operating, no reading reported"]
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.54, 1.0), fontsize=7, columnspacing=1.6, handlelength=2.2, frameon=False)
    fig.text(0.095, 0.012, "Stations inside a claimed area have no distance to mark and are not shown (Table 2). Every bound is listed in Table S6.",
             fontsize=7, color=INK2, va="bottom")
    fig.canvas.draw()
    over = fig.get_tightbbox(fig.canvas.get_renderer())
    assert over.x1 <= W + 0.01 and over.y1 <= H + 0.01 and over.x0 >= -0.01, f"text runs past the figure ({over.x0:.2f}, {over.x1:.2f} x {over.y1:.2f} in)"
    save(fig, "fig7_lead")


if __name__ == "__main__":
    for name in sys.argv[1:] or ("fig1", "fig2_timeline", "fig3", "fig4", "fig5", "fig6", "fig7", "fig8"):
        globals()[name]()
    print("wrote", sorted(p.name for p in OUT.glob("fig*.png")))
