"""P3 M6: Bayesian search map and hide map from per-cell detection probabilities.

For class T and a prior p (p3_prior or p3_cover masses, unit sum over land), each land cell gets a
detection probability d(cell). With no detection (no candidate):
    posterior(cell) = p(cell) (1 - d(cell)) / (1 - sum p d).
The total searched mass is sum p d, the probability that a body of class T present under the prior
would have been found. The hide map is the posterior itself; its top cells are where a body is most
likely to remain unseen.
d(cell) comes from a detection table, one of:
- "g1a": the forward bound. Within a survey footprint, d = the offset-averaged single-line P_d of
  p3_g1a_mag / p3_g1a_em for the survey's spacing and clearance (half-swath geometry as in p3_g0).
  This is optimistic, so it is labeled a bound;
- "g1b": injected P_d per survey and stratum at the frozen cut (W2). The table format is the same.
d is capped at D_CAP, as in the AF447 search. Overlapping surveys combine as
1 - prod(1 - d): independent looks.
Outputs are labeled prior-conditional. Writes p3_search_map.csv and
$AKP_DATA/p3/search_map/<cls>_<source>.npz. Run: python -m akp.stage_p3.p3_search_map g1a
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from affine import Affine
from rasterio.features import geometry_mask
from scipy.stats import norm
from shapely.geometry import mapping

from akp.paths import data_dir
from akp.stage_p3.p3_em_forward import snr_offsets_em
from akp.stage_p3.p3_g0 import FDEM, TURNS, footprints
from akp.stage_p3.p3_g1a_decide import HOST, MAG_CLASSES
from akp.stage_p3.p3_g1a_mag import snr_at
from akp.stage_p3.p3_mag_forward import DK_REF, OFFSETS_M, snr_offsets, z_threshold
from akp.stage_p3.p3_prior import out_dir

HERE = Path(__file__).parent
D_CAP = 0.9
BASE_M, TOP_M = 300.0, 45.72  # the headline geometry
NOISE = "cols_ns_median"


def pd_mean(snr_ref: np.ndarray, scale: float, z: float, spacing: float) -> float:
    d = np.linspace(0.0, spacing / 2, 401)
    return float(norm.cdf(snr_at(np.array(OFFSETS_M), snr_ref, d) * scale - z).mean())


def g1a_cells(inv, fps, cells, land_n, track_a_km2):
    ef = pd.read_csv(HERE / "p3_em_forward.csv").query("noise == @NOISE and r_fp_m == 150 and cls == 'T3'")
    classes = {f"{c}_{lab}": ("mag", k - HOST) for c, ks in MAG_CLASSES.items() for lab, k in zip(("low", "high"), ks)}
    classes.update({f"T3_host{h:.0f}": ("em", h) for h in ef.rho_host.unique()})
    out = {}
    for cname, (kind, par) in classes.items():
        miss = np.ones(land_n)
        for r in inv.itertuples():
            if r.survey_id not in fps:
                continue
            s = float(r.line_spacing_m)
            z = z_threshold(track_a_km2, s)
            if kind == "mag":
                if str(r.mag) != "1" or pd.isna(r.mag_sensor_height_m):
                    continue
                c = float(r.mag_sensor_height_m)
                d = min(pd_mean(snr_offsets(BASE_M, TOP_M, turn, c, NOISE), par / DK_REF, z, s) for turn in TURNS)
            else:
                if not any(k in str(r.em_system) for k in FDEM) or pd.isna(r.em_bird_height_m):
                    continue
                d = pd_mean(snr_offsets_em(par, BASE_M, TOP_M, float(r.em_bird_height_m), 150.0, NOISE), 1.0, z, s)
            miss = np.where(cells[r.survey_id], miss * (1 - min(d, D_CAP)), miss)
        out[cname] = 1 - miss
    return out


def main(source: str) -> None:
    if source != "g1a":
        raise SystemExit("only the g1a bound exists before G1b (W2)")
    inv = pd.read_csv(HERE / "p3_survey_inventory.csv")
    inv = inv[(inv.line_data_available == "yes") & (inv.product_type == "line survey") & inv.line_spacing_m.notna()
              & (inv.intersects_track_a.astype(str) == "yes")]
    pr, cv = np.load(out_dir() / "p3_prior.npz"), np.load(out_dir() / "p3_cover.npz")
    land, transform = pr["land"], Affine(*pr["transform"])
    fps = footprints(inv)
    cells = {sid: geometry_mask([mapping(g)], land.shape, transform, invert=True)[land] for sid, (g, _) in fps.items()}
    track_a = float(pd.read_csv(HERE / "p3_prior.csv").query("check == 'track_a_land_km2'").value.iloc[0])
    dmaps = g1a_cells(inv, fps, cells, int(land.sum()), track_a)
    rows, od = [], data_dir("p3/search_map")
    for cname, d in dmaps.items():
        for prior, m in (("W_cover", cv["mass_cover"].astype(float)), ("W_none", pr["mass"].astype(float))):
            found = float((m * d).sum())
            post = m * (1 - d) / (1 - found)
            rows.append(dict(cls=cname, source=source, prior=prior, searched_mass=round(found, 4),
                             cells_with_d=int((d > 0).sum()), top1pct_post_share=round(float(np.sort(post)[::-1][: max(1, post.size // 100)].sum()), 4)))
            np.savez_compressed(od / f"{cname}_{source}_{prior}.npz", d=d.astype(np.float32), posterior=post.astype(np.float32))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "p3_search_map.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "g1a")
