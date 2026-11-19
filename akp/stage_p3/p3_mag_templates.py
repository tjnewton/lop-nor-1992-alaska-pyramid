"""P3 M4 magnetics: template bank spanning magnetization direction and line heading.

Basalt is often remanence-dominated, so induced-only templates can miss it. Profiles run along +x
(east) with the line heading of HEADINGS_DEG (azimuth from north; a heading and its reverse share
a bin). The field, and each total-magnetization direction, are therefore given relative to the line:
the declination is rotated by -(heading - 90). Directions:
- the induced field direction;
- MAG_INC_DEG x MAG_DEC_DEG (the total magnetization of a remanence-dominated body).
Geometries: bases BASES_M, the middle top depth TOP_M, turn 0, offsets OFFSETS_M, at every
clearance of p3_mag_forward. Amplitude does not matter: templates are unit-normalized in
p3_mag_detect. Writes $AKP_DATA/p3/mag_forward/templates_vector.npz, keyed
"hd{heading}_inc{inc}_dec{dec}_b{base}_o{offset}_h{clearance}" (dec "ind" for induced), each
sampled like the profiles (p3_mag_forward.NPTS at DX). With --depths, every top depth of TOPS_M is modeled and written to templates_vector_depths.npz
(keys carry _t<top>), for the depth-classified statistic.
Run: python -m akp.stage_p3.p3_mag_templates [--depths]
"""
from __future__ import annotations

import itertools

import numpy as np

from akp.paths import data_dir
from akp.stage_p3.p3_mag_forward import CLEARANCES_M, DX, FIELD, NPTS
from akp.stage_p3.p3_mag_inject import response

HEADINGS_DEG = (0.0, 45.0, 90.0, 135.0)
MAG_INC_DEG = (-60.0, -20.0, 20.0, 60.0)
MAG_DEC_DEG = (0.0, 90.0, 180.0, 270.0)
BASES_M = (200.0, 400.0)
TOP_M = 38.1  # 125 ft: the middle of the stated 100-150 ft
TOPS_M = (5.0, TOP_M, 120.0, 300.0)  # Depth classes; a cluster is a target only if its best template has TOP_M
OFFSETS_M = (-150.0, 0.0, 150.0)  # both sides of the line


def main(depths: bool = False) -> None:
    x = (np.arange(NPTS) - NPTS // 2) * DX
    out = {}
    for hd, clr, base, off, top in itertools.product(HEADINGS_DEG, CLEARANCES_M, BASES_M, OFFSETS_M,
                                                     TOPS_M if depths else (TOP_M,)):
        rot = -(hd - 90.0)
        dirs = [("ind", FIELD["inclination"], FIELD["declination"])]
        dirs += [(f"{dec:.0f}", inc, dec) for inc, dec in itertools.product(MAG_INC_DEG, MAG_DEC_DEG)]
        field = dict(FIELD, declination=FIELD["declination"] + rot)  # the inducing field, in the line's frame
        for name, inc, dec in dirs:
            if name == "ind":
                p = response(x, np.full(NPTS, off), np.full(NPTS, clr), base, top, 0.0, 0.01, field=field)
            else:
                # a pure remanent direction: Q is large, so the induced part is negligible (amplitude is normalized later)
                p = response(x, np.full(NPTS, off), np.full(NPTS, clr), base, top, 0.0, 1e-4, q=100.0,
                             rem=(inc, dec + rot), field=field)
            key = f"hd{hd:.0f}_inc{'ind' if name == 'ind' else f'{inc:.0f}'}_dec{name}_b{base:.0f}_o{off:.0f}"
            key += f"_t{top:.0f}_h{clr:.0f}" if depths else f"_h{clr:.0f}"
            out[key] = p
    name = "templates_vector_depths.npz" if depths else "templates_vector.npz"
    np.savez_compressed(data_dir("p3/mag_forward") / name, **out)
    print(f"{len(out)} templates written")


if __name__ == "__main__":
    import sys

    main(depths="--depths" in sys.argv)
