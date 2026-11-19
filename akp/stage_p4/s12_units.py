"""P4 S12: the sources' miles and feet in SI units, for the paper.

The versions of the claim, one paper and one news report give distances in statute miles and
heights or depths in feet. The paper states every value in kilometers or meters and lists the
originals once, in Supplement One. This module is the one place that converts them.

Each row: the quantity, its source, the value and unit as stated, and the value in SI units,
exact and as the paper shows it. A value is shown to the precision of its source: an "about 60
miles" is about 97 km, not 96.56 km. Exact factors: 1.609344 km per statute mile, 0.3048 m per
foot (international definitions).

Where the analysis already holds a stated value as a constant (the belt of C3 in s11_versions
and p3_prior, the center and spread of the C4 component of the prior), the self-check asserts
that the constant equals the value here, so this table and the analysis cannot drift apart.

Output: akp/stage_p4/s12_units.csv. Run: PYTHONPATH=. python akp/stage_p4/s12_units.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

KM_PER_MILE, M_PER_FT = 1.609344, 0.3048
OUT = Path(__file__).with_name("s12_units.csv")

# quantity, source, stated value, unit, rounding step of the SI value as shown
ROWS = [
    ("C4 point, west of Denali on the line to Nome", "Earthfiles Ep 170, 07:50", 60, "mile", 1),
    ("C3 disc around Denali, radius", "Earthfiles Ep 123, 29:12", 50, "mile", 1),
    ("C3 belt around Denali, inner radius", "Earthfiles Ep 123, 29:12-29:52", 45, "mile", 1),
    ("C3 belt around Denali, outer radius", "Earthfiles Ep 123, 29:12-29:52", 55, "mile", 1),
    ("C4 stretch on the map, near end", "Earthfiles Ep 170, 06:01", 60, "mile", 1),
    ("C4 stretch on the map, far end", "Earthfiles Ep 170, 06:01", 70, "mile", 1),
    ("C7 place due west of Denali", "The Why Files 2024, 00:56", 60, "mile", 1),
    ("prior, C4 component, center from Denali", "our construction (p3_prior.C4_MILES)", 65, "mile", 1),
    ("prior, C4 component, standard deviation", "our construction (p3_prior.C4_SD_MILES)", 5, "mile", 1),
    ("body, depth of the base", "Earthfiles Ep 170, 08:06", 700, "ft", 1),
    ("body, depth of the top", "Earthfiles Ep 170, 08:15", 150, "ft", 1),
    ("body, height", "Earthfiles Ep 170, 08:31", 550, "ft", 1),
    ("body, depth of the top in the retelling", "The Why Files 2024", 100, "ft", 1),
    ("Eskdalemuir, depth of the located valley", "Key 1967, p. 34", 600, "ft", 10),
    ("body, sides of the base in a web repost", "Before It's News, 2012-08-21 (C19)", 1510, "ft", 10),
    ("deep structure, depth in the news report", "Browne 1993", 2000, "mile", 100),
]


def table() -> pd.DataFrame:
    out = []
    for q, src, v, unit, step in ROWS:
        si, si_unit = (v * KM_PER_MILE, "km") if unit == "mile" else (v * M_PER_FT, "m")
        out.append(dict(quantity=q, source=src, value_stated=v, unit_stated=unit, unit_si=si_unit,
                        value_si=round(si, 3), value_si_shown=int(round(si / step) * step)))
    return pd.DataFrame(out)


def self_check(t: pd.DataFrame) -> None:
    from akp.stage_p3 import p3_prior
    from akp.stage_p4 import s11_versions
    shown = dict(zip(t.quantity, t.value_si_shown))
    stated = dict(zip(t.quantity, t.value_stated))
    assert shown["C4 point, west of Denali on the line to Nome"] == 97 and shown["body, height"] == 168
    assert shown["body, depth of the top"] == 46 and shown["Eskdalemuir, depth of the located valley"] == 180
    assert shown["deep structure, depth in the news report"] == 3200 and shown["body, sides of the base in a web repost"] == 460
    belt = (stated["C3 belt around Denali, inner radius"], stated["C3 belt around Denali, outer radius"])
    assert tuple(p3_prior.C3_MILES) == belt
    assert s11_versions.R_C3 == tuple(b * KM_PER_MILE for b in belt) and s11_versions.MI == KM_PER_MILE
    assert p3_prior.C4_MILES == stated["prior, C4 component, center from Denali"]
    assert p3_prior.C4_SD_MILES == stated["prior, C4 component, standard deviation"]


def main() -> None:
    t = table()
    self_check(t)
    t.to_csv(OUT, index=False)
    print(f"wrote {OUT.name}: {len(t)} rows")


if __name__ == "__main__":
    main()
