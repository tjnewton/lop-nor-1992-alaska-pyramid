"""P3: the sensor height each member was actually flown at, and its system class.

The inventory's nominal height is a line of metadata, and for several open surveys it disagrees with
the altimeter by a factor of three or more (a higher survey described with the series' standard
text; a radar altimeter that reads to the forest canopy). The detector and the plants already use
the altimeter sample by sample. The system class now follows the same channel:
- height(row): the median over all cached traverse samples of the member's altitude channel, in
  meters after the row's units, less sensor_below_alt_m where the map gives one (a towed bird under
  a helicopter altimeter), with dropouts cleaned as the detector does (p3_mag_inject.clean_heights);
  the EM height uses em_alt where the map names one, else the same channel;
- member_class(table, package, member, sensor): the sensor and the height band (p3_null_rates.band)
  of that median. Where a member has no row in the table, the inventory height is used
  (p3_null_rates.system_class), as before.
Track A surveys are allocated to classes before they are opened by their inventory heights
(p3_null_rates.track_a_area); after they are opened, the report classes each member by its measured
height and prints both.
Writes p3_member_heights.csv for the open members (package, member, survey_id, mag_height_m,
em_height_m, inventory heights, classes by both rules). Run: python -m akp.stage_p3.p3_member_heights
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from akp.stage_p3.p3_mag_inject import FT_M, clean_heights
from akp.stage_p3.p3_null_rates import band, system_class

HERE = Path(__file__).parent


def height(row, channel: str, tow: bool = True) -> float:
    """Median height (m) of one altitude channel over the member's cached chunks; NaN if the channel is absent.

    tow: take sensor_below_alt_m off (the channel is the altimeter the sensor hangs under)."""
    from akp.stage_p3.p3_g1b_run import ensure_cache

    if not (isinstance(channel, str) and channel.strip()):
        return float("nan")
    unit = FT_M if str(row.alt_units).lower().startswith("f") else 1.0
    below = getattr(row, "sensor_below_alt_m", float("nan"))
    below = float(below) if pd.notna(below) and tow else 0.0
    vals = []
    for p in ensure_cache(row):
        z = np.load(p)
        if channel.strip().upper() not in z.files:
            return float("nan")
        v = z[channel.strip().upper()][~z["is_tie"]]
        v = v[np.isfinite(v)]
        if v.size:
            vals.append(clean_heights(v * unit - below))
    return float(np.median(np.concatenate(vals))) if vals else float("nan")


def heights(row) -> dict:
    row = SimpleNamespace(**row) if isinstance(row, dict) else row
    mag = height(row, row.alt)
    own = isinstance(row.em_alt, str) and row.em_alt.strip() and row.em_alt.strip().upper() != str(row.alt).strip().upper()
    em = height(row, row.em_alt, tow=False) if own else mag  # a separate em_alt is the bird's own altimeter
    has_em = isinstance(row.em_channels, str) and row.em_channels.strip()
    return dict(mag_height_m=round(mag, 1) if isinstance(row.mag, str) and row.mag.strip() else np.nan,
                em_height_m=round(em, 1) if has_em else np.nan)


def member_class(table: pd.DataFrame | None, inv: pd.DataFrame, package: str, member: str, survey_id: str, sensor: str) -> str | None:
    """System class by the measured height in table (indexed by package and member), else by the inventory."""
    col = "mag_height_m" if sensor == "mag" else "em_height_m"
    if table is not None and (package, member) in table.index:
        h = table.at[(package, member), col]
        return f"{sensor}_{band(float(h))}" if pd.notna(h) and band(float(h)) else None
    return system_class(inv, survey_id, sensor)


def load() -> pd.DataFrame | None:
    f = HERE / "p3_member_heights.csv"
    return pd.read_csv(f).set_index(["package", "member"]) if f.exists() else None


def main() -> None:
    ch = pd.read_csv(HERE / "p3_channels.csv").query("reader_ok == 'yes'")
    inv = pd.read_csv(HERE / "p3_survey_inventory.csv").set_index("survey_id")
    rows = []
    for r in ch.itertuples():
        h = heights(r)
        cls = {s: (f"{s}_{band(h[f'{k}_height_m'])}" if pd.notna(h[f"{k}_height_m"]) and band(h[f"{k}_height_m"]) else None)
               for s, k in (("mag", "mag"), ("em", "em"))}
        rows.append(dict(package=r.package, member=r.member, survey_id=r.survey_id, **h,
                         mag_class=cls["mag"], em_class=cls["em"],
                         mag_class_inventory=system_class(inv, r.survey_id, "mag"), em_class_inventory=system_class(inv, r.survey_id, "em")))
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    assert out.mag_height_m.dropna().between(5, 2500).all()
    out.to_csv(HERE / "p3_member_heights.csv", index=False)
    d = out[(out.mag_class != out.mag_class_inventory) | ((out.em_class != out.em_class_inventory) & out.em_class.notna())]
    print("\nclass differs from the inventory's:\n", d[["package", "mag_height_m", "mag_class", "mag_class_inventory", "em_height_m", "em_class", "em_class_inventory"]].to_string(index=False))


if __name__ == "__main__":
    main()
