"""Critical amplitude scales of the A3 r* table (input to the A3b-2 fourth-run falsifier).

The amplitude factor s on every row of a contrast that would bring box coverage to the 1%
materiality threshold: s_crit = r_1% / max r*, for each rule (generous, geometry-correct), station
set and host velocity. Uses a3b_cases_corrected.csv (A3b-1 amplitudes) and a1_box_coverage.csv.
Writes a3b2_critical.csv. Run: python -m akp.stage_a.a3b2_critical
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from akp.stage_a.a3b4_apply import MATERIALITY, r_at_fraction

HERE = Path(__file__).parent


def main() -> None:
    cases = pd.read_csv(HERE / "a3b_cases_corrected.csv")
    box = pd.read_csv(HERE / "a1_box_coverage.csv")
    sets = {"recorders": box.km_nearest_recorder.to_numpy(), "operating": box.km_nearest_operating.to_numpy()}
    rows = []
    for vp in (850.0, 1200.0, 1500.0):
        sub = cases[cases.host_vp == vp]
        for rule, col in (("generous", "r_star_qs_km"), ("geo", "r_star_geo_qs_km")):
            for name, d in sets.items():
                s = r_at_fraction(d, MATERIALITY) / float(sub[col].max())
                rows.append(dict(check=f"s_crit_{rule}_{name}_vp_{int(vp)}", value=round(s, 3)))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "a3b2_critical.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
