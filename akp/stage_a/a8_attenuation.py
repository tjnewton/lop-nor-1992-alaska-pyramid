"""A8 attenuation table: the simulated scattered surface wave with near-surface attenuation applied by formula.

The production runs (a8_production.csv) are elastic. A Rayleigh wave of frequency f that travels a
range r at speed U in ground of quality factor Q loses a factor exp(-pi f r / (Q U)) in amplitude.
For each thawed case, band and range, the slow-window vertical share of a8_production.csv (forward
azimuth) is multiplied by that factor, with:
- f the band's center frequency (as a8_production's T);
- r the range from the point above the body;
- U the host's Rayleigh speed (a3b4_rg.rayleigh_speed for the host's vp and vp / VP_VS);
- Q each of Q_VALUES, the values the stage fixed in advance from opened sources
  (a3_0_PROPERTIES.md: the generous value from a crystalline shield, and the two sediment values).
Loss of the incident P wave is not applied: it lowers the signal and the clutter alike.
No value is applied under the frozen layer: no measured Q exists for frozen ground.
This is a first-order correction (no dispersion, the phase speed for the group speed); has it
checked against one attenuated pair of simulations.
Writes a8_attenuation.csv: case, band, range_m, q, share_elastic, factor, share_attenuated.
Self-check: the factor is 1 at zero range and falls with range, frequency and 1 / Q.
Run: python -m akp.stage_a.a8_attenuation
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from akp.stage_a.a3b4_rg import rayleigh_speed

HERE = Path(__file__).parent
Q_VALUES = (100.0, 50.0, 20.0)  # a3_0_PROPERTIES.md, "Q values A3b-4 uses"
VP_VS = 1.8  # the stage's host ratio (a8_production)
HOST_VP = {"thawed_1200": 1200.0, "thawed_850": 850.0}
CENTER_HZ = {"0.5-1": 0.75, "1-2": 1.5, "2-4": 3.0}


def factor(f: float, r: float, q: float, u: float) -> float:
    return float(np.exp(-np.pi * f * r / (q * u)))


def main() -> None:
    assert factor(3.0, 0.0, 50.0, 600.0) == 1.0
    assert factor(3.0, 2000.0, 50.0, 600.0) < factor(3.0, 1000.0, 50.0, 600.0) < factor(1.5, 1000.0, 50.0, 600.0) < factor(1.5, 1000.0, 100.0, 600.0)
    d = pd.read_csv(HERE / "a8_production.csv")
    d = d[(d.window == "slow") & (d.component == "vertical") & (d.azimuth == "forward") & d.case.isin(HOST_VP)]
    rows = []
    for r in d.itertuples():
        u = rayleigh_speed(HOST_VP[r.case], HOST_VP[r.case] / VP_VS)
        for q in Q_VALUES:
            fac = factor(CENTER_HZ[r.band], r.range_m, q, u)
            rows.append(dict(case=r.case, band=r.band, range_m=r.range_m, q=q, rayleigh_speed_ms=round(u, 1),
                             share_elastic=round(r.numerical, 5), factor=round(fac, 5), share_attenuated=round(r.numerical * fac, 5)))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "a8_attenuation.csv", index=False)
    print(out[out.range_m.isin([1000.0, 3000.0])].pivot_table(index=["case", "band", "range_m"], columns="q", values="share_attenuated").round(4).to_string())


if __name__ == "__main__":
    main()
