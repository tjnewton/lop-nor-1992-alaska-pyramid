"""P3 M3 magnetics: plant a modeled pyramid into real flight-line samples.

For a plant at (x0, y0), every sample within REACH_M gets the TMI of the pyramid at its own position
and its own sensor height above ground (the survey's altitude channel). Since P4 item M0 the
plant uses the analytic model p3_mag_fast.response; response() here is the SimPEG model on
p3_mag_forward.pyramid_cells that the analytic one was validated against (p3_mag_fast_check.csv). Flat ground is assumed: the body is buried below the
sensor's local ground. Magnetization is a vector in effective-susceptibility units (SimPEG
model_type "vector"): dk along the inducing field, plus Q * dk along a remanent direction when Q > 0.
The response is added to the IGRF-removed channel. plant_lines returns the modified frame and the
indices it touched.
Self-check: python -m akp.stage_p3.p3_mag_inject (one plant into an open control package: the
added field peaks near the plant and decays with distance).
"""
from __future__ import annotations

import discretize
import numpy as np
import pandas as pd
from simpeg import maps
from simpeg.potential_fields import magnetics as mag

from akp.stage_p3.p3_mag_fast import response as fast_response
from akp.stage_p3.p3_mag_forward import FIELD, HEIGHT_M, pyramid_cells

REACH_M = 1500.0
CELL_M = 10.0
FT_M = 0.3048
HEIGHT_OK = (0.5, 2.0)


def clean_heights(h: np.ndarray) -> np.ndarray:
    """Altimeter dropouts and spikes (outside HEIGHT_OK times the local median) take the median.

    Bethel's radar altimeter reads below 1 m on some samples of a 90 m survey; a plant there would be
    modeled with the sensor at ground level."""
    h = np.asarray(h, float).copy()
    med = np.nanmedian(h)
    bad = ~np.isfinite(h) | (h < HEIGHT_OK[0] * med) | (h > HEIGHT_OK[1] * med)
    h[bad] = med
    return h


def unit(inc: float, dec: float) -> np.ndarray:
    i, d = np.radians(inc), np.radians(dec)
    return np.array([np.cos(i) * np.sin(d), np.cos(i) * np.cos(d), -np.sin(i)])  # x east, y north, z up


def response(xr, yr, hr, base, top, turn, dk, q=0.0, rem=(0.0, 0.0), field: dict | None = None) -> np.ndarray:
    """TMI (nT) at receivers (relative coordinates, heights above ground) for one pyramid, in field (default FIELD)."""
    field = FIELD if field is None else field
    half_w = 0.5 * base * np.sqrt(2) + 2 * CELL_M
    n_xy = int(np.ceil(2 * half_w / CELL_M))
    dz = 0.6 * CELL_M
    n_z = int(np.ceil((top + HEIGHT_M + dz) / dz))
    mesh = discretize.TensorMesh([[(CELL_M, n_xy)], [(CELL_M, n_xy)], [(dz, n_z)]],
                                 origin=[-n_xy * CELL_M / 2, -n_xy * CELL_M / 2, -n_z * dz])
    act = pyramid_cells(mesh, base, top, turn)
    n = int(act.sum())
    m = dk * unit(field["inclination"], field["declination"]) + q * dk * unit(*rem)
    rx = mag.receivers.Point(np.c_[xr, yr, hr], components="tmi")
    src = mag.sources.UniformBackgroundField(receiver_list=[rx], **field)
    sim = mag.simulation.Simulation3DIntegral(mesh=mesh, survey=mag.survey.Survey(src), model_type="vector",
                                              chiMap=maps.IdentityMap(nP=3 * n), active_cells=act,
                                              store_sensitivities="forward_only")
    return sim.dpred(np.tile(m, (n, 1)).T.ravel())


def plant_lines(df: pd.DataFrame, mag_ch: str, alt: str, alt_in_feet: bool, x0: float, y0: float,
                base: float, top: float, turn: float, dk: float, q: float = 0.0, rem=(0.0, 0.0)):
    near = np.flatnonzero((np.hypot(df.X.to_numpy() - x0, df.Y.to_numpy() - y0) <= REACH_M)
                          & np.isfinite(df[mag_ch].to_numpy()) & np.isfinite(df[alt].to_numpy()))
    out = df.copy()
    if near.size:
        h = clean_heights(df[alt].to_numpy()[near] * (FT_M if alt_in_feet else 1.0))
        add = fast_response(df.X.to_numpy()[near] - x0, df.Y.to_numpy()[near] - y0, h, base, top, turn, dk, q, rem)
        col = out.columns.get_loc(mag_ch)
        out.iloc[near, col] = out.iloc[near, col].to_numpy() + add
    return out, near


if __name__ == "__main__":
    import time

    from akp.paths import data_dir, data_glob
    from akp.stage_p3.p3_lines import read_package

    pkg = [p for p in data_glob(data_dir("p3/lines/open"), "*.zip") if "iron-creek" in p.name][0]
    _, df = next(read_package(pkg))
    lines = df[df.kind == "line"]
    i = len(lines) // 2
    x0, y0 = float(lines.X.iloc[i]), float(lines.Y.iloc[i])
    t0 = time.time()
    out, near = plant_lines(df, "MAGIGRF", "ALTR", True, x0, y0, 300.0, 45.72, 0.0, 0.03)
    add = (out.MAGIGRF - df.MAGIGRF).to_numpy()[near]
    r = np.hypot(df.X.to_numpy()[near] - x0, df.Y.to_numpy()[near] - y0)
    print(f"{near.size} samples within {REACH_M:.0f} m; {time.time() - t0:.1f} s; peak {np.nanmax(np.abs(add)):.1f} nT "
          f"at {r[np.nanargmax(np.abs(add))]:.0f} m; median |add| beyond 1 km {np.nanmedian(np.abs(add[r > 1000])):.2f} nT")
    assert r[np.nanargmax(np.abs(add))] < 300 and np.nanmax(np.abs(add)) > 10 * np.nanmedian(np.abs(add[r > 1000]))
    print("p3_mag_inject self-check OK")
