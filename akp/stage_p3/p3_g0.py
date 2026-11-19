"""P3 G0: the share of the Track A prior mass within detection range of a flight line.

Surveys: p3_survey_inventory.csv (W1 rebuild, metadata only; each value has a source in
p3_inventory_evidence.csv). A survey counts for a class only if all of these hold:
- it is a line survey with line data, intersecting Track A;
- its line spacing is known;
- for magnetic classes, it flew magnetics and its magnetometer height is known;
- for T3, it is frequency-domain helicopter EM (DIGHEM or RESOLVE) with its bird height known.
Time-domain EM (SkyTEM, VTEM) is not modeled yet, so it counts as unsearched here.

Footprints:
- the inventory v5.0 outline where the survey has one;
- else the survey boundary from its DGGS vector package (BOUNDARY_SHP, fetched by p3_vector_fetch);
- else the FGDC bounding box of its metadata ($AKP_DATA/p3/inventory/meta/dggs_meta). A bounding
  box overstates the footprint. "v5" in the output means inventory outlines only; "all" adds the rest.

Inside a footprint, the share of land within the half-swath w of a line is min(1, 2w / spacing)
for parallel lines. w is the largest line offset with single-line P_d >= G1_FLOOR, modeled at the
survey's own reported height (no rounding to a clearance grid: corrects the rounding-up):
- magnetic classes: p3_mag_forward.snr_offsets (along-line median clutter, the smaller over two
  turns), with the threshold z for the survey's spacing;
- T3: p3_em_forward.snr_offsets_em (along-line median clutter, the 150 m footprint).
Priors: W and W_EQUAL (cover-weighted and not), and each claim component alone (comp_*), so the
dependence on the weights is visible.
Overlapping surveys are not added: a cell takes its largest share. G0 is sum(mass x share) over
land, for the cover-weighted prior (p3_cover) and the uncovered one (p3_prior), each for W and
W_EQUAL. Writes p3_g0.csv and p3_g0_surveys.csv. Run: python -m akp.stage_p3.p3_g0
"""
from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from rasterio.features import geometry_mask
from shapely.geometry import box, mapping

from akp.paths import data_dir
from akp.stage_p3.p3_g1a_decide import HOST, MAG_CLASSES
from akp.stage_p3.p3_g1a_mag import snr_at
from akp.stage_p3.p3_em_forward import snr_offsets_em
from akp.stage_p3.p3_mag_forward import DK_REF, G1_FLOOR, OFFSETS_M, snr_offsets, z_threshold
from akp.stage_p3.p3_prior import out_dir

HERE = Path(__file__).parent
FDEM = ("DIGHEM", "RESOLVE")
GEOMS = [(b, t) for b in (200.0, 300.0, 400.0) for t in (30.48, 45.72)]
NOISE = "cols_ns_median"
TURNS = (0.0, 45.0)
# Survey boundaries from the DGGS vector packages (p3_vector_fetch; boundary members only).
BOUNDARY_SHP = {
    "DGGS_GPR2022-1-a": "tanana_big_delta_tanana_flown_surveyboundary",
    "DGGS_GPR2022-1-b": "tanana_big_delta_chena_flown_surveyboundary",
    "DGGS_GPR2022-1-c": "tanana_big_delta_healy_flown_surveyboundary",
    "DGGS_GPR2022-1-d": "tanana_big_delta_eisen_flown_surveyboundary",
    "DGGS_GPR2022-1-e": "tanana_big_delta_north_pole_flown_surveyboundary",
    "DGGS_GPR2022-1-f": "tanana_big_delta_eisen_infill_flown_surveyboundary",
    "DGGS_GPR2023-1": "kuskokwim_northernmtns_flown_surveyboundary",
    "DGGS_GPR2024-1": "kuskokwim_sischu_boundary",
    "DGGS_GPR2024-2": "seward_pen_em_kigluaik_survey_boundary",
    "DGGS_GPR2025-2": "kuskokwim_em_kaiyuh_survey_boundary",
    "DGGS_GPR2025-3": "kaiyuh_boundary_pg",
    "DGGS_GPR2026-1": "ganes_creek_boundary_pg",
    "DGGS_GPR2026-2": "mcgrath_boundary_pg",
    "DGGS_GPR2026-3": "candle_hills_boundary",
}


def footprints(inv: pd.DataFrame) -> dict:
    g = gpd.read_file(f"zip://{data_dir('probes/airborne_inventory') / 'footprint.zip'}"
                      "!AirborneGeophysicalSurvey_footprint.shp")
    g = g[g.state == "AK"].to_crs(3338)
    v5 = {s: geom for s, geom in zip(g.Survey, g.geometry)}
    meta = data_dir("p3/inventory/meta/dggs_meta")
    vec = data_dir("p3/inventory/vector")
    out = {}
    for r in inv.itertuples():
        if r.survey_id in v5 and str(r.footprint_source).startswith("inventory v5.0"):
            out[r.survey_id] = (v5[r.survey_id], "v5")
            continue
        if r.survey_id in BOUNDARY_SHP:
            shp = next(vec.glob(f"*/**/{BOUNDARY_SHP[r.survey_id]}.shp"), None)
            if shp is not None:
                out[r.survey_id] = (gpd.read_file(shp).to_crs(3338).union_all(), "boundary")
                continue
        m = re.search(r"GPR\s?(\d{4}-\d+)", str(r.publication))
        f = meta / f"dggs.alaska.gov_metadata_GPR{m.group(1)}.faq.html" if m else None
        if f is None or not f.exists():
            continue
        t = re.sub(r"<[^>]+>", " ", f.read_text(errors="ignore"))
        v = [re.search(k + r"_Bounding_Coordinate:?\s*(-?[\d.]+)", t) for k in ("West", "South", "East", "North")]
        if all(v):
            out[r.survey_id] = (gpd.GeoSeries([box(*(float(x.group(1)) for x in v))], crs=4326).to_crs(3338).iloc[0], "bbox")
    return out


def half_swath(snr_ref: np.ndarray, scale: float, z: float, spacing: float) -> float:
    d = np.linspace(0.0, spacing / 2, 401)
    ok = snr_at(np.array(OFFSETS_M), snr_ref, d) * scale >= z
    return float(d[ok].max()) if ok.any() else 0.0


def main() -> None:
    inv = pd.read_csv(HERE / "p3_survey_inventory.csv")
    inv = inv[(inv.intersects_track_a.astype(str) == "yes") & (inv.line_data_available == "yes")
              & (inv.product_type == "line survey") & inv.line_spacing_m.notna()]
    pr, cv = np.load(out_dir() / "p3_prior.npz"), np.load(out_dir() / "p3_cover.npz")
    land = pr["land"]
    transform = Affine(*pr["transform"])
    track_a = float(pd.read_csv(HERE / "p3_prior.csv").query("check == 'track_a_land_km2'").value.iloc[0])
    masses = {("W", "cover"): cv["mass_cover"], ("W_EQUAL", "cover"): cv["mass_equal_cover"],
              ("W", "none"): pr["mass"], ("W_EQUAL", "none"): pr["mass_equal"]}
    for k in [f for f in pr.files if f.startswith("comp_")]:
        mc = pr[k].astype(np.float64) * cv["cover"]
        masses[(k, "cover")] = mc / mc.sum()
        masses[(k, "none")] = pr[k]
    fps = footprints(inv)
    cells = {sid: geometry_mask([mapping(g)], land.shape, transform, invert=True)[land] for sid, (g, _) in fps.items()}
    ef = pd.read_csv(HERE / "p3_em_forward.csv").query("noise == @NOISE and r_fp_m == 150")
    classes = {f"{c}_{lab}": ("mag", k - HOST) for c, ks in MAG_CLASSES.items() for lab, k in zip(("low", "high"), ks)}
    classes.update({f"T3_host{h:.0f}": ("em", h) for h in ef[ef.cls == "T3"].rho_host.unique()})
    rows, srows = [], []
    for cname, (kind, par) in classes.items():
        for base, top in GEOMS:
            share = {"all": np.zeros(land.sum()), "v5": np.zeros(land.sum())}
            for r in inv.itertuples():
                if r.survey_id not in fps:
                    continue
                s = float(r.line_spacing_m)
                if kind == "mag":
                    if str(r.mag) != "1" or pd.isna(r.mag_sensor_height_m):
                        continue
                    c = float(r.mag_sensor_height_m)
                    w = min(half_swath(snr_offsets(base, top, turn, c, NOISE), par / DK_REF, z_threshold(track_a, s), s)
                            for turn in TURNS)
                else:
                    if not any(k in str(r.em_system) for k in FDEM) or pd.isna(r.em_bird_height_m):
                        continue
                    c = float(r.em_bird_height_m)
                    w = half_swath(snr_offsets_em(par, base, top, c, 150.0, NOISE), 1.0, z_threshold(track_a, s), s)
                frac = min(1.0, 2 * w / s)
                src = fps[r.survey_id][1]
                for key in ("all",) + (("v5",) if src == "v5" else ()):
                    share[key] = np.where(cells[r.survey_id], np.maximum(share[key], frac), share[key])
                if base == 300.0 and top == 45.72:
                    srows.append(dict(cls=cname, survey_id=r.survey_id, name=r.name, footprint=src, spacing_m=s,
                                      clearance_used_m=c, half_swath_m=w, share_in_footprint=round(frac, 4),
                                      cover_mass_in_footprint=round(float(masses[("W", "cover")][cells[r.survey_id]].sum()), 4)))
            for (prior, cover), m in masses.items():
                for key, sh in share.items():
                    rows.append(dict(cls=cname, base_m=base, top_m=round(top, 2), prior=prior, cover=cover,
                                     footprints=key, g0=round(float((m * sh).sum()), 4)))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "p3_g0.csv", index=False)
    pd.DataFrame(srows).to_csv(HERE / "p3_g0_surveys.csv", index=False)
    print(out.query("prior == 'W' and cover == 'cover'").pivot_table(index=["cls", "footprints"], columns=["base_m", "top_m"], values="g0").to_string())


if __name__ == "__main__":
    main()
