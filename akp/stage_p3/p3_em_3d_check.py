"""P3 M1: 3D spot check of the footprint-weighted 1D EM bound for T3 ("a 3D check follows").

One case: a conductive pyramid (BODY ohm m, base BASE_M, top TOP_M) in a resistive half-space
(HOST ohm m). A DIGHEM-like coplanar pair (vertical magnetic dipoles, separation SEP_M, along +x)
flies at CLEARANCE_M, with the transmitter at the positions XS_M (two, to keep the direct solve within memory) along the line through the apex
(offset 0) and along a line OFFSET_M to the side.
- 3D: SimPEG Simulation3DMagneticFluxDensity on an octree mesh (coarse, to fit the direct solve in
  memory: cells MIN_CELL_M near the
  transmitter-receiver pairs, finer over the body, coarse padding), solved with MUMPS. It is solved
  twice, with and without the body, so the difference is the body's own response and mesh error
  mostly cancels. ppm = 1e6 x the change in vertical flux density / the free-space primary at the
  receiver.
- 1D: p3_em_model.delta_ppm at the same transmitter-receiver midpoints, for footprint radii 100 and
  150 m.
Writes p3_em_3d_check.csv (freq_hz, x_m, y_m, component, ppm_3d, ppm_1d_r100, ppm_1d_r150).
Run: python -m akp.stage_p3.p3_em_3d_check
"""
from __future__ import annotations

from pathlib import Path

import discretize
import numpy as np
import pandas as pd
from pymatsolver import Mumps
from simpeg import maps
from simpeg.electromagnetics import frequency_domain as fdem

from akp.stage_p3.p3_em_model import delta_ppm
from akp.stage_p3.p3_mag_forward import HEIGHT_M, pyramid_cells

HERE = Path(__file__).parent
BODY, HOST = 10.0, 1000.0
BASE_M, TOP_M = 300.0, 38.1
SEP_M, CLEARANCE_M = 8.0, 30.0
FREQS = (900.0,)
XS_M = (0.0, 150.0)
OFFSET_M = 120.0
MIN_CELL_M = 8.0
MU0 = 4e-7 * np.pi


def mesh():
    h = MIN_CELL_M
    n = 2 ** 9  # 4096 m wide at 8 m (skin depth at 900 Hz in the host is about 530 m)
    m = discretize.TreeMesh([[(h, n)], [(h, n)], [(h, n)]], origin=[-n * h / 2, -n * h / 2, -n * h / 2], diagonal_balance=True)
    pts = np.array([[x + d, y, CLEARANCE_M] for x in XS_M for y in (0.0, OFFSET_M) for d in (0.0, SEP_M)])
    half = 0.5 * BASE_M + 20.0
    m.refine_box(np.array([[-half, -half, -(TOP_M + HEIGHT_M + 20.0)]]), np.array([[half, half, 0.0]]), levels=[m.max_level - 1], finalize=False)
    m.refine_box(np.array([[-1000.0, -1000.0, -1000.0]]), np.array([[1000.0, 1000.0, 400.0]]), levels=[m.max_level - 4], finalize=False)
    m.refine_points(pts, level=m.max_level, padding_cells_by_level=[2, 2, 2], finalize=False)
    m.finalize()
    return m


def main() -> None:
    m = mesh()
    cc = m.cell_centers
    air = cc[:, 2] > 0
    sig_host = np.where(air, 1e-8, 1.0 / HOST)
    body = pyramid_cells(m, BASE_M, TOP_M, 0.0) & ~air
    sig_body = sig_host.copy()
    sig_body[body] = 1.0 / BODY
    print(f"mesh {m.n_cells} cells, body {int(body.sum())} cells", flush=True)
    rows = []
    for f in FREQS:
        srcs = []
        for x in XS_M:
            for y in (0.0, OFFSET_M):
                rx = [fdem.receivers.PointMagneticFluxDensity(np.array([[x + SEP_M, y, CLEARANCE_M]]), orientation="z", component=c)
                      for c in ("real", "imag")]
                srcs.append(fdem.sources.MagDipole(rx, frequency=f, location=np.array([x, y, CLEARANCE_M]), orientation="z"))
        sim = fdem.Simulation3DMagneticFluxDensity(m, survey=fdem.survey.Survey(srcs), sigmaMap=maps.IdentityMap(nP=m.n_cells),
                                                   solver=Mumps)
        d_body = sim.dpred(sig_body)
        d_host = sim.dpred(sig_host)
        primary = -MU0 / (4 * np.pi * SEP_M ** 3)  # Bz of a unit vertical dipole in its equatorial plane (negative)
        dppm = (d_body - d_host).reshape(-1, 2) / primary * 1e6
        k = 0
        for x in XS_M:
            for y in (0.0, OFFSET_M):
                mid = np.array([x + SEP_M / 2])
                one = {r: delta_ppm(mid, np.array([y]), np.array([CLEARANCE_M]), BASE_M, TOP_M, BODY, HOST, (f,), SEP_M, r_fp=r)[0, :, 0]
                       for r in (100.0, 150.0)}
                for ci, comp in enumerate(("inphase", "quadrature")):
                    rows.append(dict(freq_hz=f, x_m=x, y_m=y, component=comp, ppm_3d=round(float(dppm[k, ci]), 3),
                                     ppm_1d_r100=round(float(one[100.0][ci]), 3), ppm_1d_r150=round(float(one[150.0][ci]), 3)))
                k += 1
        print(f"{f:.0f} Hz done", flush=True)
    out = pd.DataFrame(rows)
    out["ratio_3d_over_1d_r150"] = (out.ppm_3d / out.ppm_1d_r150).round(3)
    out.to_csv(HERE / "p3_em_3d_check.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
