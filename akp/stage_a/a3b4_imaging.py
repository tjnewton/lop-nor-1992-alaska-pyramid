"""A3c imaging criterion at the A3b-4 Rg detection radii.

The A3c necessary condition (at least 3 recorder sites within r* spanning at least 90 deg of
azimuth) evaluated at the largest Rg r* at the 1200 m/s floor: unattenuated (strong-stage
corrected) and with each A3-0 Rg Q. Reads a3b4_apply.csv. Writes a3b4_imaging.csv.
Run: python -m akp.stage_a.a3b4_imaging
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from akp.stage_a.a3c_imaging import imaging_fraction, site_matrices

HERE = Path(__file__).parent
RADII = {"q_inf": "max_r_corr_km_vp_1200", "q_100": "rq_max_r_km_generous_100_vp_1200",
         "q_50": "rq_max_r_km_sediment_50_vp_1200", "q_20": "rq_max_r_km_sediment_20_vp_1200"}


def main() -> None:
    ap = pd.read_csv(HERE / "a3b4_apply.csv").set_index("check").value
    rows = []
    for sname, (D, A, _) in site_matrices().items():
        for tag, key in RADII.items():
            frac, valdez = imaging_fraction(D, A, float(ap[key]))
            rows.append(dict(check=f"rg_imaging_box_frac_{sname}_{tag}", value=round(frac, 4)))
            rows.append(dict(check=f"rg_imaging_valdez_{sname}_{tag}", value=valdez))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "a3b4_imaging.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
