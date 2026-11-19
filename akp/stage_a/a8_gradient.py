"""A8 gradient column: one more simulation, case R5, and its validations.

Question: beyond one kilometer, does the wave a stone body scatters stay below the median coda level of its
band when the ground is a sediment column whose shear speed rises with depth, up to the highest band?

Rules implemented here, none changed. Where they differ the later one wins.
- The column from one source; three departures; the body of R4; the two higher bands; a wrapper, no
  production code rewritten; the production measurement; validations before the body run is read.
- The column fixed completely; box depth; the sheet at the production depth; the edge wave measured
  against a one-dimensional reference; each validation with a number and a tolerance; all validations on one
  set of settings and one repeat; the readings; what the wrapper replaces.
- Receivers to 1.5 km; no convergence pair; the grid step half of R4's, read from R4's settings; the
  sponge R4's thickness in meters; a repeat changes the time-step factor or the sponge, never the grid step;
  the edge level L, the row term L_r and the bracket; the frequencies of the check against the reference;
  the uniform-column check bit for bit; the readings completed; the two sentences that compare grounds left out.
- The surface-wave check's window; what "the full modal sum" is; the check of the twin against the
  reference ends before the edge wave; a row breaks by either measure and the bracket uses one L for both; a
  change of settings is logged when it is used; the order of work; the checksums of the two modules in the log.
- Tolerances are those of a8_gates. The smoothed measure is that of a8_smoothing.

Inputs: a8_elastic3d (solver), a8_production (set-up, measurement), a8_generous (R4's case, the memory model,
the `swapped` helper), a8_smoothing, a8_gates (windows and tolerances), a8_reference1d (the exact layered
reference). Committed tables: a8_generous_checks.csv (R4's grid step, sponge and speed as run),
a8_production_checks.csv (memory model), a8_smoothing.csv (production shares), a5_dprk_site_spread.csv and
a5_dprk_snr_sensitivity_checks.csv (the committed and the lowest clutter medians). Traces: R4's archived twin.

Method.
- Column (`column`): the source's relations with the three departures. Shear speed: its relation for the top
  tens of meters, its conversion from P speed below, never falling with depth (the value above the source's
  junction is held until the deeper relation exceeds it), and not below the floor. P speed: the source's
  relations for saturated deposits, not below that of water; its own steps are kept. Density: Gardner's rule.
- Case: R4's stone body, base, top and height. Grid step half of R4's. Sponge R4's thickness in meters.
  Receivers every 100 m to 1.5 km on the three azimuths. Bands 1-2 and 2-4 Hz.
- Set-up (`setup`, swapped in for a8_production.setup): the production set-up for the surface values of the
  column, then the arrival times of the incident wave at the body's top and at the surface by the integral of
  vertical slowness through the column, the window speeds as the largest and smallest in the model, and the
  step count from them. With a uniform column it returns the production set-up bit for bit.
- Run (`run_case`): a8_production.run_case with the column as the profile, the sheet set for the P speed at its
  own depth, the sponge tuned to the surface P speed, and one more receiver under the origin at the depth of
  the body's top. The sheet lies on the node plane nearest the production depth, where the solver puts it
  anyway, and the timing uses that depth.
- Validations (`--validate`), each row with its number, tolerance and verdict:
  reference: a8_reference1d's self-checks with this column, and the reference against R4's archived twin at
    the three receivers nearest the body (amplitude at the rule's frequencies that R4 measured; arrival);
  uniform: on a miniature box with R4's uniform ground, the old path (a8_production's run, measurement and
    a8_smoothing) against the new path (this module's run and the swapped set-up): traces bit for bit, shares
    within a8_smoothing's tolerance;
  surface: an explosion two grid steps deep in the column, no body; at the center of each band the phase
    speed between receivers at one and two kilometers against the fundamental mode's, and the vertical
    amplitude against the sum of the modes trapped above the box bottom, both in a window about the
    fundamental mode's group arrival, SURFACE_PERIODS periods of the frequency either side. The table
    gives the reference's own reading of the speed beside the run's. Only the fundamental's term of the modal
    sum has an exact check;
  twin: the run without the body against the reference: amplitude at four frequencies and arrival at the
    three nearest receivers; the wrapper's travel times against the reference's own integral; the apparent
    slowness along the incidence plane; and the edge level L, reported. The window of this check opens the
    pulse's half-length before the predicted arrival and ends where the edge wave's pulse begins at that
    receiver: its arrival by ray times in the column, less the pulse's half-length. A receiver is used
    only if the pulse's first half-length fits; the others are named in the table.
  A failed validation may be repeated once with --set (time-step factor, sponge thickness or strength) and
  --note; then every validation is run on those settings. The body run needs all four passed on one set.
- Measure: a8_production.measure and a8_smoothing.measure unchanged, through the swapped set-up. Added: the
  lag of the slow window's peak after P and its particle motion; L per band and L_r per row; the
  bracket and its outcome per row and measure; the statements; the quality factor only if a row breaks.
- Statements: each sentence of a8_generous's table except the two that compare grounds, for this case's rows,
  by its own rule, with the bracket: a row breaks a sentence if the low end of its bracket crosses the level,
  supports it if the high end does not, and is untested otherwise.

Outputs, in this folder (with AKP_A8_SMOKE=1 in $AKP_DATA/a8/gradient_smoke): a8_gradient_size.csv,
a8_gradient_column.csv, a8_gradient_validation.csv (every attempt, and every change of settings when it is
used), a8_gradient_shares.csv (the columns of a8_generous_shares.csv, then the bracket),
a8_gradient_statements.csv, a8_gradient_checks.csv, and a8_gradient_steps.txt: one line at the start of every
step with the time, the step, its settings, the sha256 of this module and of a8_reference1d.py, and the commit.
Traces: $AKP_DATA/a8/gradient.

Run, in this order, one command at a time. GUARD stands for
`OMP_NUM_THREADS=6 PYTHONPATH=. python scripts/mem_guard.py --every 0.5`; the size table gives each run's memory and time.
  PYTHONPATH=. python -m akp.stage_a.a8_gradient                                    (--size: column, sizes, edge wave; nothing heavy)
  GUARD --gb 3 -- python -m akp.stage_a.a8_gradient --validate reference           (no simulation)
  GUARD --gb 3 -- ... --validate uniform                                            (four miniature simulations)
  GUARD --gb 8 -- ... --validate surface                                            (one simulation)
  GUARD --frac 0.40 -- ... --run twin                                               (the run without the body)
  GUARD --gb 3 -- ... --validate twin                                               (saved traces)
  GUARD --frac 0.40 -- ... --run body                                               (refused unless all four passed on one set of settings)
  GUARD --gb 3 -- ... --measure                                                     (saved traces; all tables)
  ... --validate NAME --set dt_safety=0.6 --note "why"                              (the one repeat; then all, and --run twin, with the same --set)
  GUARD --frac 0.40 -- ... --next                                                   ('s order from where the log stands: prints the plan, does the
                                                                                     next step only, exits with an error at a failed validation)
  AKP_A8_SMOKE=1 ...                                                                (miniature boxes, coarse grid: the pipeline only)

Self-check (--selfcheck, no simulation): the column never falls with depth, keeps its floors and equals the
running maximum of the source's relations; the grid step and sponge follow R4; a uniform column gives the
production set-up exactly; the first-arrival time in a uniform column is the straight ray's; the explosion's
closed form equals gate three's reference; the validation log allows one repeat and binds the body run to one
set of settings; --next follows the fixed order and stops at a failure; the twin check's window ends where the edge
wave begins and leaves out a receiver where the pulse's first half does not fit; the bracket gives its three
outcomes; with no bracket the statements equal a8_generous's.
"""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from akp.stage_a import a8_elastic3d as fd  # first: sets the Devito environment
from akp.stage_a import a8_attenuated as attenuated
from akp.stage_a import a8_gates as gates
from akp.stage_a import a8_generous as gen
from akp.stage_a import a8_production as prod
from akp.stage_a import a8_reference1d as ref
from akp.stage_a import a8_smoothing as smoothing

HERE = Path(__file__).parent
SMOKE = prod.SMOKE
CASE = "R5"
# Brocher (2005), U.S. Geological Survey Open-File Report 2005-1317. Depth z in km, speeds in km/s, density in g/cm3.
EQ41 = (0.215, 10.932, -138.1)  # eqn 41, p. 31: Vs of Plio-Quaternary deposits, depths 0 to 0.037 km
EQ42 = (0.7, 42.968, -575.8, 2931.6, -3977.6)  # eqn 42, p. 31: Vp, surface value fixed at 0.7 km/s, depths 0 to 0.04 km
EQ43 = (1.5, 3.735, -3.543)  # eqn 43, p. 32: Vp, depths 0.04 to 0.5 km
EQ35 = (2.24, 0.6)  # eqn 35, p. 32: Vp, depths 0.5 to 2 km
MUDLINE = (1.36, 1.16)  # the "mudline", pp. 31 and 32: Vs = (Vp - 1.36) / 1.16
Z41, Z42, Z43 = 0.037, 0.04, 0.5  # km, pp. 31-32: eqn 41 above 0.037 and the mudline below; eqn 42 to 0.04; eqn 43 to 0.5
GARDNER = (1.74, 0.25)  # eqn 7, p. 12: density = 1.74 Vp^0.25
QS_SOFT, QS_BELOW = 10.0, 0.5  # p. 15: Qs = 10 where Vs is below 0.5 km/s
# Inputs fixed before any code or run.
VS_FLOOR = 300.0  # m/s. Floor on the shear speed (the soft ground)
VP_FLOOR = 1500.0  # m/s. P speed of water
RMAX, RANGE_MIN = 1500.0, 1500.0  # m. Receiver range; the run is not made below this range
DEPTH = 700.0  # m. Box depth
BANDS = ("1-2", "2-4")  # The two higher bands
FREQS = (1.5, 2.0, 3.0, 4.0)  # Hz. Frequencies of the check against the reference
TIME_TOL = gates.SPEED_TOL  # Timing tolerance, as the speed tolerance
SURFACE_RANGES, SURFACE_STEPS = (1000.0, 2000.0), 2  # Receivers of the surface-wave check (m); the explosion's depth in grid steps
SURFACE_PERIODS = 2.0  # The surface-wave check's window is this many periods of the frequency measured either side of the group arrival
ORDER = (("validate", "reference"), ("validate", "uniform"), ("validate", "surface"), ("run", "twin"), ("validate", "twin"),
         ("run", "body"), ("measure", ""))  # The order of work
FAR_M = smoothing.FAR_M  # the statement concerns ranges of 1 km or more
MINI = (400.0, 200.0, 400.0, 300.0)  # m: receiver range, margin, depth and sheet depth of the uniform-column check (a8_production's miniature box)
RETRY = ("dt_safety", "sponge_m", "sponge_a")  # What the one repeat may change
VALIDATIONS = ("reference", "uniform", "surface", "twin")
VAL_COLS = ["validation", "attempt", "settings", "quantity", "at", "value", "reference", "ratio", "tolerance", "passed", "note"]
COARSEN = 4.0 if SMOKE else 1.0  # smoke only: a grid this much coarser; its numbers mean nothing
if SMOKE:
    RMAX, DEPTH, SURFACE_RANGES, FAR_M = prod.RMAX, prod.DEPTH, (300.0, 600.0), 200.0
    MINI = (prod.RMAX, prod.MARGIN, prod.DEPTH, prod.Z_SHEET)
PROD_SETUP, MARGIN0, SHEET0 = prod.setup, prod.MARGIN, prod.Z_SHEET  # the production values, kept before any swap


# ---------------------------------------------------------------- the column
def _poly(c, zk):
    return np.polyval(c[::-1], zk)


def source_vp(zk):
    """The source's P speed (km/s) at depth zk (km), as written."""
    return np.where(zk <= Z42, _poly(EQ42, zk), np.where(zk <= Z43, _poly(EQ43, zk), _poly(EQ35, zk)))


def source_vs(zk):
    """The source's S speed (km/s), as written: it drops at 0.037 km."""
    return np.where(zk <= Z41, _poly(EQ41, zk), (source_vp(zk) - MUDLINE[0]) / MUDLINE[1])


VS_HELD = float(_poly(EQ41, Z41))  # km/s: the shear speed just above the source's junction, held below it


def column(z):
    """The R5 column: (vp, vs, rho) in m/s and kg/m3 at depths z (m), vectorized."""
    zk = np.asarray(z, float) / 1e3
    vs = np.maximum(1e3 * np.maximum(source_vs(zk), np.where(zk > Z41, VS_HELD, 0.0)), VS_FLOOR)
    vp = np.maximum(1e3 * source_vp(zk), VP_FLOOR)
    return vp, vs, 1e3 * GARDNER[0] * (vp / 1e3) ** GARDNER[1]


def at(profile, z):
    """A profile's three values at one depth, as floats."""
    return tuple(float(np.ravel(q)[0]) for q in profile(np.array([float(z)])))


def junctions():
    """Depths (m) at which a relation or a departure begins or ends, computed from the constants."""
    from scipy.optimize import brentq
    floor_end = brentq(lambda zk: _poly(EQ41, zk) - VS_FLOOR / 1e3, 0.0, Z41) * 1e3
    water_end = brentq(lambda zk: _poly(EQ42, zk) - VP_FLOOR / 1e3, 0.0, Z42) * 1e3
    hold_end = brentq(lambda zk: float(source_vs(zk)) - VS_HELD, Z42 + 1e-9, Z43) * 1e3
    q_end = brentq(lambda zk: float(source_vs(zk)) - QS_BELOW, hold_end / 1e3, 2.0) * 1e3
    eps = 1e-9
    return [dict(depth_m=floor_end, note=f"the floor of {VS_FLOOR:g} m/s on the shear speed acts from the surface to here"),
            dict(depth_m=water_end, note=f"the P speed is held at {VP_FLOOR:g} m/s from the surface to here"),
            dict(depth_m=Z41 * 1e3, note=f"the source's junction: its shear speed falls from {VS_HELD * 1e3:.1f} to {float(source_vs(Z41 + eps)) * 1e3:.1f} m/s; "
                 f"the column holds {VS_HELD * 1e3:.1f} m/s from here"),
            dict(depth_m=Z42 * 1e3, note=f"the source's P speed steps from {float(source_vp(Z42)) * 1e3:.1f} to {float(source_vp(Z42 + eps)) * 1e3:.1f} m/s (kept)"),
            dict(depth_m=hold_end, note="the deeper relation regains the held shear speed: the hold ends here"),
            dict(depth_m=q_end, note=f"the shear speed reaches {QS_BELOW * 1e3:g} m/s: the source's Qs is {QS_SOFT:g} above here"),
            dict(depth_m=Z43 * 1e3, note=f"the source's P speed steps from {float(source_vp(Z43)) * 1e3:.1f} to {float(source_vp(Z43 + eps)) * 1e3:.1f} m/s (kept)")]


def column_table():
    body = gen.CASES["R4"]["body"]
    depths = sorted({0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 600.0, DEPTH,
                     body["apex_depth"], body["apex_depth"] + body["height"]})
    j = junctions()
    rows = []
    for z in depths:
        vp, vs, rho = at(column, z)
        svp, svs = float(source_vp(z / 1e3)) * 1e3, float(source_vs(z / 1e3)) * 1e3
        acts = [name for name, on in (("floor on the shear speed", z < j[0]["depth_m"]), ("P speed of water", z < j[1]["depth_m"]),
                                      ("shear speed held", j[2]["depth_m"] < z < j[4]["depth_m"])) if on]
        rows.append(dict(kind="depth", depth_m=z, vs_ms=round(vs, 1), vp_ms=round(vp, 1), rho_kgm3=round(rho, 1), vp_over_vs=round(vp / vs, 3),
                         source_vs_ms=round(svs, 1), source_vp_ms=round(svp, 1), departure="; ".join(acts), note=""))
    for q in j:
        vp, vs, rho = at(column, q["depth_m"])
        rows.append(dict(kind="junction", depth_m=round(q["depth_m"], 2), vs_ms=round(vs, 1), vp_ms=round(vp, 1), rho_kgm3=round(rho, 1),
                         vp_over_vs=round(vp / vs, 3), source_vs_ms="", source_vp_ms="", departure="", note=q["note"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- R4's settings, the case, the constants a run takes
@lru_cache(maxsize=1)
def r4():
    """R4 as its gates passed and as it ran: numerical changes, grid step (m), sponge cells, point-steps per second."""
    changes, _ = gen.settings("R4", quiet=True)
    c = pd.read_csv(HERE / "a8_generous_checks.csv").set_index("check")
    npts = np.prod([int(v) for v in c.value["R4_grid_shape"].split("x")])
    steps = int(c.detail["R4_time_step_s"].split()[0])
    words = c.detail["R4_grid_shape"].split()  # "h 9.28... m, sponge 30 cells, ..."
    return dict(changes=changes, h=changes.get("h", gen.CASES["R4"]["h"]), nbl=int(changes.get("nbl", fd.NBL)),
                speed=min(npts * steps / float(c.value[f"R4_{run}_run_seconds"]) for run in ("twin", "body")),
                h_as_run=float(words[1]), nbl_as_run=int(words[4]))


H = 0.5 * r4()["h"] * COARSEN  # Half of R4's grid step
H_MINI = r4()["h"] * (COARSEN / 2 if SMOKE else 1.0)  # the uniform-column check runs on R4's step
SPONGE_M = r4()["nbl"] * r4()["h"]  # R4's sponge thickness in meters


def case_spec(name):
    """The case as the set-up takes it. R5: the column and R4's body. The uniform-column check: R4's ground, uniform."""
    c = gen.CASES["R4"]
    base = dict(bands=BANDS, stone=c["stone"], body=c["body"])
    if name == CASE:
        return dict(base, h=H, profile=column)
    return dict(base, h=H_MINI, host=c["host"], **({} if name == "uniform_old" else dict(profile=lambda z: c["host"])))


def numerics(changes, h):
    """Solver settings for grid step h: the sponge's thickness in meters as R4's, and what the repeat changed."""
    out = [(fd, "NBL", int(round(changes.get("sponge_m", SPONGE_M) / h)))]
    out += [(fd, "DT_SAFETY", changes["dt_safety"])] if "dt_safety" in changes else []
    return out + ([(fd, "SPONGE_A", changes["sponge_a"])] if "sponge_a" in changes else [])


def constants(name, changes):
    """The module constants one run and its measurement take, for a8_generous.swapped: the box, the sheet on its
    node plane, the solver settings."""
    mini = name.startswith("uniform")
    h = H_MINI if mini else H
    rmax, margin, depth, sheet = MINI if mini else (RMAX, MARGIN0, DEPTH, SHEET0)
    return numerics(changes, h) + [(prod, "RMAX", rmax), (prod, "MARGIN", margin), (prod, "DEPTH", depth), (prod, "Z_SHEET", round(sheet / h) * h)]


def said(changes):
    return "; ".join(f"{k}={v:g}" for k, v in sorted(changes.items()))


def unsaid(text):
    return {k: float(v) for k, v in (kv.split("=") for kv in str(text).split("; ") if kv)}


def folder():
    from akp.paths import data_dir
    d = data_dir("a8") / ("gradient_smoke" if SMOKE else "gradient")
    d.mkdir(parents=True, exist_ok=True)
    return d


def out_dir():
    return folder() if SMOKE else HERE


def archive(name, changes):
    return folder() / (name + "".join(f"_{k}={v:g}" for k, v in sorted(changes.items())) + ".npz")


# ---------------------------------------------------------------- set-up and run in a column
def q_time(profile, p, z0, z1):
    """Travel time (s) of the plane P wave between depths z0 < z1: the integral of vertical slowness, written as
    the uniform formula at z0 plus the integral of the difference, which is exactly zero in a uniform column."""
    z = np.linspace(z0, z1, int(np.ceil((z1 - z0) / 0.05)) + 2)
    q = np.sqrt(1 / np.broadcast_to(np.asarray(profile(z)[0], float), z.shape) ** 2 - p**2)
    return (z1 - z0) * q[0] + float(trapezoid(q - q[0], z))


def setup(case, spec=None):
    """a8_production.setup for a case in a column (spec["profile"]); any other case goes to the production set-up.

    Replaced: t0 and t_surf (integral of vertical slowness from the sheet), the window speeds (the largest and
    smallest in the model, the sponge's depths included), the time step and the step count from them. `host` is
    the column at the surface; c_r is the Rayleigh speed of that floor, the wave the grid step is sized for."""
    if spec is None or "profile" not in spec:
        return PROD_SETUP(case, spec)
    profile = spec["profile"]
    s = PROD_SETUP(case, dict({k: v for k, v in spec.items() if k != "profile"}, host=at(profile, 0.0)))
    g, stone = s["g"], s["stone"]
    vp, vs, _ = (np.broadcast_to(np.asarray(q, float), (200,)) for q in profile(np.linspace(0.0, (g["shape"][2] - 1 - fd.KF) * g["h"], 200)))
    vp_fast, vs_fast, vs_slow = max(stone[0], float(vp.max())), max(stone[1], float(vs.max())), min(stone[1], float(vs.min()))
    dt = fd.time_step(g, vp_fast)
    t0 = s["t_c"] + q_time(profile, s["p"], s["body"]["apex_depth"], prod.Z_SHEET)
    t_surf = s["t_c"] + q_time(profile, s["p"], 0.0, prod.Z_SHEET)
    t_low = 1 / np.mean(prod.BANDS["0.5-1"])
    nt = int((t0 + prod.RMAX / (0.85 * vs_slow) + 1.5 * t_low + 1.5) / dt) + 2  # a8_production.setup's formula
    return dict(s, vp_fast=vp_fast, vs_fast=vs_fast, vs_slow=vs_slow, dt=dt, t0=t0, t_surf=t_surf, nt=nt, profile=profile)


def top_point(s):
    """The vertical-velocity sample nearest the point above the body's center at the depth of its top."""
    k = round(s["body"]["apex_depth"] / s["h"] - 0.5)
    return fd.snap(s["g"], [(0.0, 0.0, 0.0)])[0] + np.array([0.0, 0.0, (k + 0.5) * s["h"]])


def run_case(case, with_body, spec):
    """One simulation in a column: as a8_production.run_case, with the profile, the sheet set for the P speed at its
    own depth, the sponge tuned to the surface P speed, and the receiver of `top_point` (saved as vz_top)."""
    s = setup(case, spec)
    profile = spec["profile"]
    body = fd.square_body(x=0.0, y=0.0, **s["body"]) if with_body else None
    par = fd.effective_params(s["g"], profile, body, s["stone"] if with_body else None)
    vp_sheet, half = at(profile, prod.Z_SHEET)[0], s["half"]
    rec = fd.native(s["nodes"], s["h"])
    rec["vz"] = np.vstack([rec["vz"], top_point(s)])
    out = fd.simulate(s["g"], par, dt=s["dt"], nt=s["nt"], v_ref=at(profile, 0.0)[0], rec=rec,
                      plane=dict(z=prod.Z_SHEET, angle_deg=float(np.degrees(np.arcsin(s["p"] * vp_sheet))), vp=vp_sheet,
                                 f0=prod.F0, t0=s["t_c"], flat=(half, half)))
    out["vz_top"], out["vz"] = out["vz"][:, -1], out["vz"][:, :-1]
    return {k: (v.astype(np.float32) if isinstance(v, np.ndarray) else v) for k, v in out.items()}


def run_member(name, with_body, changes):
    """One simulation in its own process: R5, a member of the uniform-column check, or the surface-wave check."""
    if name == "surface":
        with gen.swapped(numerics(changes, H)):
            return run_surface()
    sp = case_spec(name)
    with gen.swapped(constants(name, changes)):
        return prod.run_case(name, with_body, sp) if name == "uniform_old" else run_case(name, with_body, sp)


def run_or_load(path, *args):
    if not path.exists():
        out = attenuated.in_fresh_process(run_member, *args)
        np.savez_compressed(path, **out)
        print(f"{path.stem}: {out['npts'] / 1e6:.1f}M points, {out['nt']} steps, {out['seconds']:.0f} s, "
              f"{out['mpts_per_s']:.0f} Mpts/s, peak rss {out['rss_gb']:.2f} GB", flush=True)
    return attenuated.load(path)


# ---------------------------------------------------------------- the reference of the twin, the edge wave
def ref_stack(profile, h, depth):
    """Layers no thicker than half the grid step, from the surface to the box bottom, over a half-space."""
    return ref.stack(profile, np.arange(int(round(2 * depth / h)) + 1) * h / 2)


def reference(s, profile, depths, x):
    """Vertical velocity (steps, points) of the one-dimensional reference under the solver's sheet and wavelet, at
    points of depth `depths` and horizontal position `x` along the incidence plane. Uses prod.Z_SHEET and prod.DEPTH."""
    dt, nt, h = s["dt"], s["nt"], s["h"]
    depths, x = np.asarray(depths, float), np.asarray(x, float)
    uniq, inv = np.unique(depths, return_inverse=True)
    fr = np.fft.rfftfreq(2 * nt, dt)
    use = (fr > 0) & (fr <= 5 * prod.F0)  # the wavelet has nothing above
    wav = np.fft.rfft(fd.ricker(np.arange(nt) * dt, prod.F0, s["t_c"]), 2 * nt)[use]
    vz = ref.plane_response(ref_stack(profile, h, prod.DEPTH), s["p"], prod.Z_SHEET, uniq, fr[use])[1][:, inv]
    spec = np.zeros((len(fr), len(depths)), complex)
    # numpy's FFT is the conjugate of the exp(-i w t) convention (a8_gates)
    spec[use] = np.conj(vz * h * np.exp(2j * np.pi * fr[use][:, None] * s["p"] * x[None, :])) * wav[:, None]
    return np.fft.irfft(spec, 2 * nt, axis=0)[:nt]


def twin_reference(s):
    """The reference at the surface receivers (their own vz samples, half a step deep) and at `top_point`."""
    top = top_point(s)
    return reference(s, s["profile"], [s["h"] / 2] * len(s["nodes"]) + [top[2]], list(s["nodes"][:, 0]) + [top[0]])


@lru_cache(maxsize=8)
def _rays(profile, z_src, z_rec, bottom):
    """Ray branches from depth z_src up to depth z_rec in a column: for scanned ray parameters p, the intercept time
    tau(p) and the ray's horizontal distance, -d tau / d p, of the direct ray and of the ray that dives under the
    source and turns above `bottom` (with the parameters for which it does turn there)."""
    z = np.linspace(z_rec, bottom, int(round((bottom - z_rec) / 0.5)) + 1)
    v = np.broadcast_to(np.asarray(profile(z)[0], float), z.shape)
    up = z <= z_src
    ps = (1 - np.linspace(1, 0, 4002)[1:-1] ** 2) / v[up].max()  # dense near the ray that leaves the source level

    def tau(sel):
        return trapezoid(np.sqrt(np.maximum(1 / v[sel][None, :] ** 2 - ps[:, None] ** 2, 0.0)), z[sel], axis=1)
    direct = tau(up)
    return tuple((ps, tt, -np.gradient(tt, ps), ok) for tt, ok in ((direct, np.ones(len(ps), bool)), (direct + 2 * tau(~up), ps > 1 / v[~up].max())))


def first_arrival(profile, z_src, z_rec, x, bottom):
    """First arrival (s) of a P wave from depth z_src at a point at depth z_rec above it, a horizontal distance x
    away, in a column: the direct ray, or one that dives under the source and turns above `bottom`. The time is
    tau(p) + p x where the ray's own distance equals x."""
    best = np.inf
    for ps, tt, dist, ok in _rays(profile, float(z_src), float(z_rec), float(bottom)):
        for i in np.flatnonzero(ok[:-1] & ok[1:] & ((dist[:-1] - x) * (dist[1:] - x) <= 0)):
            w = (x - dist[i]) / (dist[i + 1] - dist[i])
            best = min(best, tt[i] + w * (tt[i + 1] - tt[i]) + (ps[i] + w * (ps[i + 1] - ps[i])) * x)
    return float(best)


def edge_delay(s, z_rec=None, x_rec=0.0, y_rec=0.0):
    """Seconds by which the sheet's edge wave follows the plane wave at a point (default: the body's top), by ray
    times in the column: the earliest of the sheet's edges, each firing as the tilted wave leaves it. The edges are
    the two ends across the incidence plane and the side; the mirror plane has none. In a uniform column the end
    the wave leaves first gives a8_generous.edge_delay."""
    z_rec = s["body"]["apex_depth"] if z_rec is None else z_rec
    half, p = s["half"], s["p"]

    def first(d):
        return first_arrival(s["profile"], prod.Z_SHEET, z_rec, d, prod.DEPTH)
    side = min(xe * p + first(float(np.hypot(xe - x_rec, half - y_rec))) for xe in np.linspace(-half, half, 37))
    return min(-half * p + first(half + x_rec), half * p + first(half - x_rec), side) - q_time(s["profile"], p, z_rec, prod.Z_SHEET) - p * x_rec


def edge_onset(s):
    """At each surface receiver, the seconds after the incident pulse's center at which the edge wave's pulse
    begins: its arrival by ray times in the column, less the pulse's half-length."""
    return np.array([edge_delay(s, 0.0, float(x), float(y)) for x, y in s["nodes"][:, :2]]) - 1 / prod.F0


def window_ends(onsets, need, what):
    """The window of the check against the reference ends where the edge wave's pulse begins, and at the
    incident pulse's own end at the latest. A receiver is used when at least the pulse's first half-length fits,
    that is when the edge wave does not begin before the pulse's center; the window is not shortened below that.
    Returns (which receivers are used, the window's end in seconds after the pulse's center)."""
    onsets = np.asarray(onsets, float)
    ok = onsets >= 0
    if ok.sum() < need:
        msg = (f"{what}: the edge wave begins before the incident pulse's center at all but {int(ok.sum())} of {len(ok)} receivers, and {need} are needed. "
               "The window is not shortened below the pulse's first half-length")
        if not SMOKE:
            raise SystemExit(msg)
        print(f"smoke only; at full size this stops here: {msg}")
        ok = np.ones(len(ok), bool)
    return ok, np.clip(onsets, 0.0, 1 / prod.F0)


def edge_level(s, twin, rf):
    """. L of a band: the largest band envelope of the twin less the reference, vertical, at `top_point` and at
    the three receivers at the first range, up to the end of the slow window of the last receiver, over the
    reference's own peak at the same point and band. The interval opens where a8_production's incident window
    opens (from the arrival of the plane wave). L_r of a receiver and band: the twin's incident peak, as
    a8_production takes it, against the reference's in the same window, as a fraction.
    Returns (rows of the four points, {band: L}, table of L_r)."""
    t, n = np.arange(s["nt"]) * s["dt"], len(s["ranges"])
    tw = np.column_stack([twin["vz"].astype(float), twin["vz_top"].astype(float)])
    az = list(prod.AZIMUTHS.items())
    near = [(f"{name} {s['ranges'][0]:g} m", a * n, s["t_surf"] + s["p"] * s["ranges"][0] * ux) for a, (name, (ux, _)) in enumerate(az)]
    near.append(("under the origin at the depth of the body's top", 3 * n, s["t0"]))
    points, level, rows = [], {}, []
    for band in s["bands"]:
        T = 1 / np.mean(prod.BANDS[band])
        e_tw, e_rf, e_d = (np.abs(prod.band_envelope(q, band, s["dt"])) for q in (tw, rf, tw - rf))
        end = s["t0"] + s["ranges"][-1] / (0.85 * s["vs_slow"]) + 1.5 * T
        for name, j, t_arr in near:
            lo = t_arr - 1 / prod.F0 - T
            sel = (t >= lo) & (t <= end)
            points.append(dict(band=band, point=name, level=float(e_d[sel, j].max() / prod.peak(e_rf[:, j], t, lo, t_arr + 1 / prod.F0 + T)),
                               largest_at_s_after_arrival=round(float(t[sel][np.argmax(e_d[sel, j])] - t_arr), 3)))
        level[band] = max(q["level"] for q in points if q["band"] == band)
        for a, (name, (ux, _)) in enumerate(az):
            for i, r in enumerate(s["ranges"]):
                t_inc = s["t_surf"] + s["p"] * r * ux
                win = (t_inc - 1 / prod.F0 - T, t_inc + 1 / prod.F0 + T)
                rows.append(dict(azimuth=name, range_m=float(r), band=band,
                                 L_r=abs(prod.peak(e_tw[:, a * n + i], t, *win) / prod.peak(e_rf[:, a * n + i], t, *win) - 1)))
    return points, level, pd.DataFrame(rows)


# ---------------------------------------------------------------- modes of the column
@lru_cache(maxsize=4)
def mode_speeds(h, depth):
    """Fundamental mode of the reference stack by the determinant: {frequency: (phase, group speed)} at the
    frequencies of the checks, and {band: (group speed at the center, fastest group speed across the band)}."""
    st = ref_stack(column, h, depth)
    at_f = {f: ref.fundamental(f, st) for f in sorted({*ref.MODE_FREQS, *FREQS, *(float(np.mean(prod.BANDS[b])) for b in BANDS)})}
    band = {}
    for b in BANDS:
        lo, hi = prod.BANDS[b]
        us = [at_f[f][1] if f in at_f else ref.fundamental(f, st)[1] for f in np.linspace(lo, hi, 5)]
        band[b] = (at_f[float(np.mean(prod.BANDS[b]))][1], max(us))
    return at_f, band


def reference_column():
    """(profile, layer thickness, depth) of the reference stack, for a8_reference1d's self-check."""
    return column, H / 2, DEPTH


# ---------------------------------------------------------------- validations
def rel(quantity, where, value, reference, tol="", note=""):
    return dict(quantity=quantity, at=where, value=float(value), reference=float(reference), ratio=float(value / reference),
                tolerance=tol, passed=bool(abs(value / reference - 1) <= tol) if tol != "" else "", note=note)


def within(quantity, where, value, limit, note=""):
    return dict(quantity=quantity, at=where, value=float(value), reference="", ratio="", tolerance=float(limit),
                passed=bool(abs(value) <= limit), note=note)


def lag_s(a, b, dt):
    """Seconds by which trace a follows trace b: cross-correlation, parabolic peak (as a8_gates.measure_surface)."""
    from scipy.signal import correlate
    xc = correlate(a, b, mode="full")
    i = int(np.argmax(xc))
    return (i - (len(b) - 1) + 0.5 * (xc[i - 1] - xc[i + 1]) / (xc[i - 1] - 2 * xc[i] + xc[i + 1])) * dt


def pulse_rows(s, vz, rf, labels, centers, onsets, travel, freqs):
    """The incident pulse of a run without a body against the reference's, trace by trace: amplitude at `freqs`
    (windowed DFT, the same window on both) within the fixed amplitude tolerance, and the lag within the fixed share of
    the sheet-to-surface time `travel`. The window opens the pulse's half-length before the predicted arrival and
    ends by `window_ends`; onsets: where the edge wave's pulse begins, seconds after the pulse's center."""
    t, rows = np.arange(s["nt"]) * s["dt"], []
    ok, ends = window_ends(onsets, 1, "the check against the reference")
    for j, (label, c) in enumerate(zip(labels, centers)):
        if not ok[j]:
            rows.append(dict(quantity="receiver_not_used", at=label, value=float(onsets[j]), reference="", ratio="", tolerance="", passed="",
                             note="seconds after the incident pulse's center at which the edge wave begins: the pulse's first half-length does not fit"))
            continue
        lo, hi = c - 1 / prod.F0, c + ends[j]
        note = f"window ends {ends[j]:.3f} s after the pulse's center, where the edge wave begins or the pulse ends"
        for f in freqs:
            a, b = (abs(fd.win_dft(q[:, j], f, lo, hi, s["dt"], gates.TAPER)) for q in (vz, rf))
            rows.append(rel("incident_amplitude_run_over_reference", f"{label}, {f:g} Hz", a, b, gates.TOL, note))
        w = fd.window(t, lo, hi, gates.TAPER)
        rows.append(within("incident_arrival_run_after_reference_s", label, lag_s(vz[:, j] * w, rf[:, j] * w, s["dt"]), TIME_TOL * travel,
                           f"limit: {TIME_TOL:g} of the {travel:.4f} s from sheet to surface; {note}"))
    return rows


def r4_twin():
    """R4's archived twin. In smoke a miniature one is run by a8_generous's own function and kept in this module's folder."""
    if not SMOKE:
        return attenuated.load(gen.archive("R4", False))
    path = folder() / "R4_twin.npz"
    if not path.exists():
        np.savez_compressed(path, **attenuated.in_fresh_process(gen.run_member, "R4", False, {}))
    return attenuated.load(path)


def validate_reference(changes):
    """The reference checks itself first."""
    rows = [dict(quantity=c["check"], at="", value=c["value"], reference=c["reference"], ratio=c["value"] / c["reference"],
                 tolerance=c["tolerance"], passed=c["passed"], note="a8_reference1d.checks") for c in ref.checks(reference_column())]
    q = r4()
    twin = r4_twin()
    with gen.swapped(gen.numerics(q["changes"])):
        s = PROD_SETUP("R4", gen.spec("R4", q["changes"]))
    assert twin["dt"] == s["dt"] and twin["nt"] == s["nt"], "R4's archived twin does not match R4's set-up"
    host, n, sheet = s["host"], len(s["ranges"]), round(prod.Z_SHEET / s["h"]) * s["h"]  # the node plane the solver put R4's sheet on
    idx = [a * n for a in range(len(prod.AZIMUTHS))]
    x = s["nodes"][idx, 0]
    with gen.swapped([(prod, "Z_SHEET", sheet)]):
        rf = reference(s, lambda z: host, [s["h"] / 2] * len(idx), x)
        travel = ref.vertical_time(ref_stack(lambda z: host, s["h"], prod.DEPTH), s["p"], 0.0, sheet)
    labels = [f"R4 twin, {az} {s['ranges'][0]:g} m" for az in prod.AZIMUTHS]
    onsets = np.full(len(idx), gen.edge_delay(s) - 1 / prod.F0)  # R4's ground is uniform: a8_generous's own edge delay
    return rows + pulse_rows(s, twin["vz"][:, idx].astype(float), rf, labels, s["t_surf"] + s["p"] * x, onsets, travel, [f for f in FREQS if f <= gen.F_SOFT])


def validate_uniform(changes):
    """With a uniform column, through the swapped set-up and this module's run, the old path's numbers."""
    old, new = ([run_or_load(archive(f"{name}_{'body' if b else 'twin'}", changes), name, b, changes) for b in (False, True)]
                for name in ("uniform_old", "uniform_new"))
    rows = [dict(quantity="traces_identical", at=f"{run} {c}", value=float(np.array_equal(a[c], b[c])), reference=1.0, ratio="", tolerance="bit for bit",
                 passed=bool(np.array_equal(a[c], b[c])), note="new path against a8_production.run_case")
            for run, a, b in (("twin", old[0], new[0]), ("body", old[1], new[1])) for c in ("vx", "vy", "vz")]
    key = ["azimuth", "range_m", "band"]
    with gen.swapped(constants("uniform_old", changes)):
        sp_old, sp_new = case_spec("uniform_old"), case_spec("uniform_new")
        s_old = PROD_SETUP("uniform", sp_old)
        m_old, sm_old = pd.DataFrame(prod.measure("uniform", sp_old, old)[0]), pd.DataFrame(smoothing.measure("uniform", sp_old, old))
        with gen.swapped([(prod, "setup", setup)]):
            s_new = setup("uniform", sp_new)
            m_new, sm_new = pd.DataFrame(prod.measure("uniform", sp_new, new)[0]), pd.DataFrame(smoothing.measure("uniform", sp_new, new))
    same = all(s_old[k] == s_new[k] for k in ("t0", "t_surf", "nt", "dt", "vp_fast", "vs_fast", "vs_slow", "t_c"))
    rows.append(dict(quantity="set_up_identical", at="t0, t_surf, steps, time step, window speeds", value=float(same), reference=1.0, ratio="",
                     tolerance="exact", passed=bool(same), note="the swapped set-up against a8_production.setup"))
    for name, a, b, cols in (("a8_production.measure", m_old, m_new, ["numerical"]), ("a8_smoothing.measure", sm_old, sm_new, ["share", "share_smoothed"])):
        on = key + (["window", "component"] if "window" in a else [])
        m = a.merge(b, on=on, validate="one_to_one", suffixes=("_old", "_new"))
        assert len(m) == len(a) == len(b)
        for c in cols:
            ok = np.isfinite(m[f"{c}_old"])
            worst = float((m[f"{c}_new"][ok] / m[f"{c}_old"][ok] - 1).abs().max())
            rows.append(within(f"{c}_max_relative_difference", f"{name}, {int(ok.sum())} rows", worst, smoothing.REPRO_TOL, "a8_smoothing.REPRO_TOL"))
    return rows


def surface_setup():
    """The surface-wave check: an explosion two grid steps deep in the column, vertical receivers at one and
    two kilometers on the mirror plane. Margins as gate three's (a8_gates.surface_setup, rule 4b of a8_elastic3d):
    a wave reflected at a sponge's inner edge starts after the last window has closed; with the fundamental mode's
    group speeds at the band centers in place of the half-space's Rayleigh speed, and this check's window."""
    u = [q[0] for q in mode_speeds(H, DEPTH)[1].values()]
    halves = [SURFACE_PERIODS / float(np.mean(prod.BANDS[b])) for b in BANDS]  # the windows' half-lengths, s
    extra, last = max(u) * (max(halves) + 1 / prod.F0 + 2 * gates.TAPER), SURFACE_RANGES[-1]
    wy = float(np.ceil(np.sqrt((last + extra) ** 2 - last**2) / 2 / H) * H)
    xm = float(np.ceil(extra / 2 / H) * H)
    g = fd.make_grid(lo=(-xm, 0.0, 0.0), hi=(last + xm, wy, DEPTH), h=H, free_surface=True, sym_y=True, nbl=fd.NBL)
    dt = fd.time_step(g, gen.CASES["R4"]["stone"][0])  # the run's time step: the stone's P speed
    t0 = 1 / prod.F0 + 0.1
    nt = int((t0 + max(last / q + w for q, w in zip(u, halves)) + 3 * gates.TAPER) / dt) + 2
    return dict(g=g, dt=dt, nt=nt, t0=t0, h=H, nodes=fd.snap(g, [(r, 0.0, 0.0) for r in SURFACE_RANGES]), z_src=SURFACE_STEPS * H, wy=wy, xm=xm)


def run_surface():
    s = surface_setup()
    par = fd.effective_params(s["g"], column)
    nat = fd.native(s["nodes"], H)
    return fd.simulate(s["g"], par, dt=s["dt"], nt=s["nt"], v_ref=at(column, 0.0)[0], rec=dict(vx=nat["vx"], vz=nat["vz"]),
                       point=dict(xyz=(0.0, 0.0, s["z_src"]), f0=prod.F0, t0=s["t0"]))


def modal_traces(s, profile, depth, ranges, c_max_share=0.98):
    """Vertical velocity (steps, ranges) at half a step's depth from the modal sum of a8_reference1d, for the
    explosion of the surface-wave check, and the same from the fundamental mode alone. Modes slower than
    `c_max_share` of the half-space's S speed: those trapped above the box bottom."""
    dt, nt, h = s["dt"], s["nt"], s["h"]
    st = ref_stack(profile, h, depth)
    fr = np.fft.rfftfreq(2 * nt, dt)
    use = (fr >= 0.1 * prod.F0) & (fr <= 3.5 * prod.F0)
    wav = np.fft.rfft(fd.ricker(np.arange(nt) * dt, prod.F0, s["t0"]), 2 * nt)[use] * h**3  # moment rate: w h^3 (a8_elastic3d)
    total, modes = ref.modal_response(st, fr[use], s["z_src"], h / 2, ranges, c_max_share * st["vs"][-1])
    first = np.array([ref.excitation(m[0], s["z_src"], h / 2) * np.sqrt(2 / (np.pi * m[0]["k"] * ranges)) * np.exp(1j * (m[0]["k"] * ranges + np.pi / 4))
                      if m else 0 * ranges for m in modes])
    out = []
    for resp in (total, first):
        spec = np.zeros((len(fr), len(ranges)), complex)
        spec[use] = np.conj(resp) * wav[:, None]
        out.append(np.fft.irfft(spec, 2 * nt, axis=0)[:nt])
    return out[0], out[1], [len(m) for m in modes]


def surface_rows(s, vz, profile, depth, speeds):
    """Rows of the surface-wave check from the run's vertical traces at the two receivers. speeds: {f: (c, U)}."""
    r = s["nodes"][:, 0]
    full, first, count = modal_traces(s, profile, depth, r)
    rows = []
    for f, (c0, u0) in speeds.items():
        half = SURFACE_PERIODS / f
        win = [(s["t0"] + q / u0 - half, s["t0"] + q / u0 + half) for q in r]
        amp = {name: np.array([fd.win_dft(tr[:, j], f, *win[j], s["dt"], gates.TAPER) for j in range(len(r))])
               for name, tr in (("run", vz), ("modes", full), ("fundamental", first))}

        def speed(a):  # phase between the receivers; the whole cycles are those of the mode's own speed
            dphi = np.angle(a[-1] / a[0])
            cycles = round((2 * np.pi * f / c0 * (r[-1] - r[0]) - dphi) / (2 * np.pi))
            return 2 * np.pi * f * (r[-1] - r[0]) / (dphi + 2 * np.pi * cycles)
        where = f"{f:g} Hz, {r[0]:.0f} to {r[-1]:.0f} m"
        rows.append(rel("phase_speed_run_over_fundamental_mode", where, speed(amp["run"]), c0, gates.SPEED_TOL, f"group speed {u0:.1f} m/s sets the window"))
        rows.append(rel("phase_speed_modal_sum_over_fundamental_mode", where, speed(amp["modes"]), c0, "",
                        "context: what the other modes in the window do to the same measurement"))
        for j, q in enumerate(r):
            rows.append(rel("vertical_amplitude_run_over_modal_sum", f"{f:g} Hz, {q:.0f} m", abs(amp["run"][j]), abs(amp["modes"][j]), gates.TOL,
                            f"modes trapped above the box bottom, far field; {min(count)} to {max(count)} modes across the pulse's spectrum; "
                            "only the fundamental's term has an exact check"))
            rows.append(rel("vertical_amplitude_modal_sum_over_fundamental_alone", f"{f:g} Hz, {q:.0f} m", abs(amp["modes"][j]), abs(amp["fundamental"][j]), "",
                            "context: the share of the other modes inside the window"))
    return rows


def validate_surface(changes):
    tr = run_or_load(archive("surface", changes), "surface", False, changes)
    with gen.swapped(numerics(changes, H)):
        s = surface_setup()
    assert tr["dt"] == s["dt"] and tr["nt"] == s["nt"]
    at_f = mode_speeds(H, DEPTH)[0]
    return surface_rows(s, tr["vz"].astype(float), column, DEPTH, {f: at_f[f] for f in (float(np.mean(prod.BANDS[b])) for b in BANDS)})


def validate_twin(changes):
    """The run without the body against the reference."""
    path = archive("twin", changes)
    if not path.exists():
        raise SystemExit(f"no twin on these settings: run --run twin{''.join(f' --set {k}={v:g}' for k, v in changes.items())} first")
    twin = attenuated.load(path)
    with gen.swapped(constants(CASE, changes)):
        s = setup(CASE, case_spec(CASE))
        assert twin["dt"] == s["dt"] and twin["nt"] == s["nt"]
        rf = twin_reference(s)
        st = ref_stack(column, s["h"], prod.DEPTH)
        travel, to_top = ref.vertical_time(st, s["p"], 0.0, prod.Z_SHEET), ref.vertical_time(st, s["p"], s["body"]["apex_depth"], prod.Z_SHEET)
        points, level, l_r = edge_level(s, twin, rf)
        onsets = edge_onset(s)
    n, vz, t = len(s["ranges"]), twin["vz"].astype(float), np.arange(s["nt"]) * s["dt"]
    idx = [a * n for a in range(len(prod.AZIMUTHS))]
    x = s["nodes"][:, 0]
    labels = [f"{az} {s['ranges'][0]:g} m" for az in prod.AZIMUTHS]
    rows = pulse_rows(s, vz[:, idx], rf[:, idx], labels, s["t_surf"] + s["p"] * x[idx], onsets[idx], travel, FREQS)
    rows += [within("wrapper_travel_time_less_reference_integral_s", "sheet to surface", s["t_surf"] - s["t_c"] - travel, s["dt"], "limit: one sample"),
             within("wrapper_travel_time_less_reference_integral_s", "sheet to the body's top", s["t0"] - s["t_c"] - to_top, s["dt"], "limit: one sample")]
    line = np.flatnonzero(np.array([ux for ux, _ in prod.AZIMUTHS.values()]).repeat(n) != 0)  # the receivers in the incidence plane
    ok, ends = window_ends(onsets[line], 2, "the apparent slowness")
    use = line[ok]

    def pulse(tr, j, end):  # the incident pulse at receiver j, cut by the window of `pulse_rows`
        c = s["t_surf"] + s["p"] * x[j]
        return tr[:, j] * fd.window(t, c - 1 / prod.F0, c + end, gates.TAPER)
    lags = np.array([lag_s(pulse(vz, j, e), pulse(rf, j, e), s["dt"]) for j, e in zip(use, ends[ok])])  # the twin behind the reference, receiver by receiver
    rows.append(rel("apparent_slowness_over_ray_parameter", f"{len(use)} of {len(line)} receivers in the incidence plane, x {x[use].min():.0f} to {x[use].max():.0f} m, least squares",
                    s["p"] + np.polyfit(x[use], lags, 1)[0], s["p"], TIME_TOL,
                    "the reference moves at the ray parameter; the fit is of the twin's lag behind it against position. Receivers where the pulse's "
                    "first half-length does not fit before the edge wave are left out"))
    for q in points:
        rows.append(dict(quantity="edge_level", at=f"{q['band']} Hz, {q['point']}", value=q["level"], reference="", ratio="", tolerance="", passed="",
                         note=f"reported, enters the bracket; largest {q['largest_at_s_after_arrival']} s after the arrival"))
    for band in s["bands"]:
        rows.append(dict(quantity="edge_level_L", at=f"{band} Hz", value=level[band], reference="", ratio="", tolerance="", passed="", note="the largest of the four points"))
        rows.append(dict(quantity="incident_peak_term_L_r_largest", at=f"{band} Hz", value=float(l_r[l_r.band == band].L_r.max()), reference="", ratio="", tolerance="",
                         passed="", note="over the receivers; reported"))
    return rows


VALIDATE = dict(reference=validate_reference, uniform=validate_uniform, surface=validate_surface, twin=validate_twin)


# ---------------------------------------------------------------- the log of validations and the stop rule
def val_table():
    path = out_dir() / "a8_gradient_validation.csv"
    return pd.read_csv(path, keep_default_na=False, dtype=str) if path.exists() else pd.DataFrame(columns=VAL_COLS)


def state(table):
    """{validation: {settings: passed}} from the log; every row with a tolerance decides."""
    out = {}
    for (name, key), g in table.groupby(["validation", "settings"]):
        out.setdefault(name, {})[key] = bool((g.passed[g.passed != ""] == "True").all())
    return out


def repeat(table):
    """The settings of the one repeat ('' when none was made)."""
    used = sorted(set(table.settings) - {""})
    assert len(used) <= 1, f"the log holds more than one changed set of settings: {used}"
    return used[0] if used else ""


def check_settings(changes, table):
    """Fixed settings first; one change of numerical settings after a failure; then everything on it."""
    key, used, st = said(changes), repeat(table), state(table)
    if key == "" and used:
        raise SystemExit(f"the settings were changed to '{used}' after a failure. Every validation and the run use them: add --set {' --set '.join(used.split('; '))}")
    if key and used and key != used:
        raise SystemExit(f"one repeat only: the settings were already changed to '{used}'")
    if key and not used and not any(v.get("") is False for v in st.values()):
        raise SystemExit("--set is for the one repeat after a validation has failed on the fixed settings")


def next_attempt(name, changes, note, table):
    check_settings(changes, table)
    key, mine = said(changes), state(table).get(name, {})
    if key in mine:
        raise SystemExit(f"{name}: already {'passed' if mine[key] else 'FAILED'} on {'the fixed settings' if not key else key}; it is not run again"
                         + ("" if mine[key] or key else f". The one repeat: --set {' | '.join(RETRY)}=VALUE and --note")
                         + (". The run with the body is not read; the paper says the question is untested for such a column" if key and not mine[key] else ""))
    if key and not note:
        raise SystemExit("--note: say why the repeat changes what it changes")
    return 2 if key else 1


def run_settings(table):
    """The settings on which every validation passed, for the body run and the measurement."""
    key, st = repeat(table), state(table)
    bad = [f"{v} {'FAILED' if st.get(v, {}).get(key) is False else 'not run'}" for v in VALIDATIONS if st.get(v, {}).get(key) is not True]
    if bad:
        raise SystemExit(f"{CASE} with the body is not run or read: on {'the fixed settings' if not key else key}: {', '.join(bad)}")
    return unsaid(key)


def settings():
    """In smoke a coarse validation's verdict means nothing: a missing or failed one is reported and passed over."""
    table = val_table()
    try:
        return run_settings(table)
    except SystemExit as e:
        if not SMOKE:
            raise
        print(f"smoke only; at full size this stops here: {e}")
        return unsaid(repeat(table))


def log_settings(changes, note, where):
    """A change of settings is logged when it is used, by a run as by a validation."""
    key, table = said(changes), val_table()
    if key and not ((table.validation == "settings") & (table.settings == key) & (table["at"] == where)).any():
        row = dict(dict.fromkeys(VAL_COLS, ""), validation="settings", attempt="2", settings=key, quantity="settings_changed", at=where, note=note)
        pd.concat([table, pd.DataFrame([row])], ignore_index=True).to_csv(out_dir() / "a8_gradient_validation.csv", index=False)


def code_sha():
    """sha256 of this module and of the reference module, as they stand on disk."""
    import hashlib
    return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in ("a8_gradient.py", "a8_reference1d.py")}


def step_log(step, changes):
    """At the start of every step, one line in a8_gradient_steps.txt: the time, the step, its settings, the
    checksums of the two modules, the commit, and whether the two modules are committed as they stand."""
    import subprocess
    from datetime import datetime, timezone

    def git(*args):
        return subprocess.run(["git", "-C", str(HERE), *args], capture_output=True, text=True).stdout.strip()
    sha = code_sha()
    line = (f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} {'smoke ' if SMOKE else ''}{step}; settings: {said(changes) or 'fixed'}; "
            + "; ".join(f"{k} sha256 {v}" for k, v in sha.items()) + f"; HEAD {git('rev-parse', 'HEAD')}; the two modules are "
            + ("committed as they stand" if not git("status", "--porcelain", "--", *sha) else "NOT committed as they stand"))
    with open(out_dir() / "a8_gradient_steps.txt", "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def validate(name, changes, note):
    """One validation: its rows into the log. Returns whether every judged row passed."""
    attempt = next_attempt(name, changes, note, val_table())
    log_settings(changes, note, f"--validate {name}")
    step_log(f"--validate {name}", changes)
    rows = pd.DataFrame([dict(r, validation=name, attempt=attempt, settings=said(changes)) for r in VALIDATE[name](changes)])
    rows["note"] = [("; ".join(q for q in (n, note) if q)) for n in rows.note]
    out = pd.concat([val_table(), rows.reindex(columns=VAL_COLS).astype(str)], ignore_index=True)  # the log is read after the step
    out.to_csv(out_dir() / "a8_gradient_validation.csv", index=False)
    judged = rows[rows.passed.astype(str) != ""]
    failed = judged[judged.passed.astype(str) != "True"]
    print(rows[["quantity", "at", "value", "reference", "ratio", "tolerance", "passed"]].to_string(index=False))
    print(f"{name} attempt {attempt}: {'passed' if failed.empty else 'FAILED'} ({len(judged)} rows judged, {len(failed)} outside tolerance)"
          + ("; smoke: coarse grid, the verdict means nothing" if SMOKE else ""))
    return failed.empty


# ---------------------------------------------------------------- the two runs
def run(which, changes, note):
    rules()
    table = val_table()
    if which == "twin":
        check_settings(changes, table)
        if changes and not note:
            raise SystemExit("--note: say why the repeat changes what it changes")
        log_settings(changes, note, "--run twin")
    else:
        if changes:
            raise SystemExit("the body run takes the settings every validation passed on; no --set")
        changes = settings()
        if not archive("twin", changes).exists():
            raise SystemExit("the twin on these settings is missing")
    with gen.swapped(constants(CASE, changes)):
        g = setup(CASE, case_spec(CASE))["g"]
    if not gen.fits(g):
        raise SystemExit(f"{CASE}: predicted {gen.predicted_gb(g):.1f} GB, over {gen.SAFETY:g} of the budget of {gen.budget_gb():.1f} GB. The run is not made")
    step_log(f"--run {which}", changes)
    run_or_load(archive(which, changes), CASE, which == "body", changes)


# ---------------------------------------------------------------- the fixed order of work
def plan(table, done):
    """The fixed order from where the log stands: (settings in force, [(kind, name, status)], index of the next step or
    None, why it stops or ''). done(kind, name, changes): whether a run's traces or the measurement's tables exist.
    A validation that failed stops the order: on the fixed settings until the one repeat is made by hand (its
    change is a choice), on the changed settings for good."""
    key, st = repeat(table), state(table)
    changes, steps = unsaid(key), []
    for kind, name in ORDER:
        if kind == "validate":
            v = st.get(name, {}).get(key)
            steps.append((kind, name, "not run" if v is None else "passed" if v else "FAILED"))
        else:
            steps.append((kind, name, "done" if done(kind, name, changes) else "not run"))
    failed, stop = [n for _, n, status in steps if status == "FAILED"], ""
    if failed and not SMOKE:  # in smoke a coarse validation's verdict means nothing and the order goes on
        stop = (f"{failed[0]} FAILED on the fixed settings. The one repeat: --validate {failed[0]} --set {' | '.join(RETRY)}=VALUE --note WHY; "
                "then --next goes on with those settings" if not key else
                f"{failed[0]} FAILED on {key}, after the one repeat. The run with the body is not read; the paper says the question is untested for such a column")
    nxt = next((i for i, q in enumerate(steps) if q[2] == "not run"), None)
    return changes, steps, None if stop else nxt, stop


def next_step():
    """--next: print the plan, do the next step only. Exits with an error when a validation has failed."""
    table = val_table()

    def done(kind, name, changes):
        if kind == "run":
            return archive(name, changes).exists()
        shares, body = out_dir() / "a8_gradient_shares.csv", archive("body", changes)
        return shares.exists() and body.exists() and shares.stat().st_mtime >= body.stat().st_mtime
    changes, steps, nxt, stop = plan(table, done)
    note = "; ".join(sorted(set(table.note[table.validation == "settings"]) - {""}))
    print(f"settings in force: {said(changes) or 'the fixed settings'}")
    for i, (kind, name, status) in enumerate(steps):
        print(f"  {'next ->' if i == nxt else '       '} --{kind} {name}".rstrip() + f": {status}")
    if stop:
        raise SystemExit(stop)
    if nxt is None:
        print("nothing is left: every step of the fixed order is done")
        return
    kind, name, _ = steps[nxt]
    if kind == "validate":
        if not validate(name, changes, note) and not SMOKE:
            raise SystemExit(f"{name} FAILED: --next stops here")
    elif kind == "run":
        run(name, changes if name == "twin" else {}, note)
    else:
        measure()


# ---------------------------------------------------------------- measurement
def peak_rows(s, runs, speed):
    """Per receiver and band: the time of the slow window's peak after the incident P; the part of the slow
    window that held it, before or from the fundamental mode's earliest arrival in the band (less half a period, as
    every production window opens), with the ratio of horizontal to vertical motion and the phase lag there; and
    a8_generous's edge-wave flag with the column's ray times and group speed. speed: {band: (U at the center, fastest U)}."""
    twin, body = runs
    t, n, delay, out = np.arange(s["nt"]) * s["dt"], len(s["ranges"]), edge_delay(s), []
    for band in s["bands"]:
        T = 1 / np.mean(prod.BANDS[band])
        env = {c: prod.band_envelope(body[c].astype(float) - twin[c].astype(float), band, s["dt"]) for c in ("vx", "vy", "vz")}
        for a, (az, (ux, _)) in enumerate(prod.AZIMUTHS.items()):
            for i, r in enumerate(s["ranges"]):
                j = a * n + i
                ver, rad = env["vz"][:, j], (ux * env["vx"][:, j]) if ux else env["vy"][:, j]
                lo, hi = s["t0"] + r / s["vs_fast"] - T / 2, s["t0"] + r / (0.85 * s["vs_slow"]) + 1.5 * T  # the slow window
                t_ray = s["t0"] + r / speed[band][1] - T / 2
                sel = (t >= lo) & (t <= hi)
                t_pk = t[sel][np.argmax(np.abs(ver[sel]))]
                name, win = ("rayleigh", (t_ray, hi)) if t_pk >= t_ray else ("before_rayleigh", (lo, t_ray))
                hv, lag = prod.rayleigh_evidence(rad, ver, t, *win)
                out.append(dict(case=s["case"], azimuth=az, range_m=float(r), band=band, lag_after_p_s=round(float(t_pk - s["t_surf"] - s["p"] * r * ux), 3),
                                clutter_lag_s=smoothing.CLUTTER_LAG, peak_window=name, peak_h_over_v=hv, peak_phase_lag_deg=lag,
                                edge_flag=bool(delay < gen.edge_needed(dict(s, c_r=speed[band][0]), band, r))))
    return pd.DataFrame(out)


def medians():
    """Per band: the committed median clutter and cleanest twentieth (a8_smoothing.clutter), and the lowest median."""
    clut = smoothing.clutter()
    low = pd.read_csv(HERE / "a5_dprk_snr_sensitivity_checks.csv").drop_duplicates("check").set_index("check").value
    return clut["median"], clut["p05"], {b: float(low[f"pooled_median_min_{b}"]) for b in BANDS}


def bracket(shares, level, l_r):
    """The bracket of every row and its outcome against the committed median, for both measures."""
    med, _, lowest = medians()
    d = shares.merge(l_r, on=["azimuth", "range_m", "band"], validate="one_to_one")
    d["L"] = d.band.map(level)
    d["median"], d["lowest_median"] = d.band.map(med), d.band.map(lowest)
    for m in ("share", "share_smoothed"):
        d[f"{m}_low"] = d[m] * (1 - d.L) / (1 + d.L_r)
        d[f"{m}_high"] = np.where(d.L_r < 1, d[m] * (1 + d.L) / (1 - d.L_r.clip(upper=1 - 1e-12)), np.inf)
        d[f"{m}_over_median"], d[f"{m}_over_lowest_median"] = d[m] / d["median"], d[m] / d.lowest_median
        d[f"outcome_{m}"] = np.where(d[f"{m}_low"] >= d["median"], "breaks", np.where(d[f"{m}_high"] < d["median"], "supports", "untested"))
    return d


def reading(n, breaks, untested):
    """The fixed readings from the counts of rows."""
    return ("no rows" if n == 0 else "broken" if breaks else "holds" if not untested else
            "could not decide: every row untested" if untested == n else "holds in the rows tested; some untested")


def statements(d):
    """Every sentence of a8_generous.statements but the two that compare grounds, for this case's rows, by its own
    rule, with the bracket. d: the table of `bracket`."""
    med, p05, _ = medians()
    o = pd.read_csv(HERE / "a8_smoothing.csv")
    o = o[o.azimuth == "forward"]  # the production cases, forward, as a8_generous compares
    fwd, rows = d[d.azimuth == "forward"], []
    for m in ("share", "share_smoothed"):
        far_o = o[o.range_m >= FAR_M]

        def add(name, sentence, rule, dom, above, below=None, was="", m=m):
            lo, hi = dom[f"{m}_low"], dom[f"{m}_high"]
            brk, sup = above(lo, dom), ~above(hi, dom)
            if below is not None:
                brk, sup = brk | below(hi, dom), sup & ~below(lo, dom)
            unt = ~brk & ~sup
            rows.append(dict(statement=name, measure=m, outcome=reading(len(dom), int(brk.sum()), int(unt.sum())), rows=len(dom),
                             rows_supporting=int(sup.sum()), rows_untested=int(unt.sum()), rows_breaking=int(brk.sum()),
                             broken_where=gen.where(dom[brk], m) if brk.any() else "", untested_where=gen.where(dom[unt], m) if unt.any() else "",
                             largest=round(float(dom[m].max()), 4) if len(dom) else "", production=was,
                             L="; ".join(f"{b}: {v:.3f}" for b, v in dom.groupby("band").L.first().items()), sentence=sentence, rule=rule))
        far, at1, far_all = fwd[fwd.range_m >= FAR_M], fwd[fwd.range_m == FAR_M], d[d.range_m >= FAR_M]
        add("below_median_beyond_1km", "beyond 1 km the scattered wave stays below the median clutter in every simulated case",
            "share below its band's median clutter at every range of 1 km or more, on all three azimuths", far_all,
            lambda v, q: v / q.band.map(med) >= 1, was=round(float((far_o[m] / far_o.band.map(med)).max()), 3))
        pct = np.round(100 * o[m][o.range_m == FAR_M])
        add("abstract_range_at_1km", "sends a surface wave of ... to ... percent of the incident P to 1 km",
            "the share at 1 km, in whole percent, stays inside the range of the production cases", at1,
            lambda v, q: np.round(100 * v) > pct.max(), lambda v, q: np.round(100 * v) < pct.min(), was=f"{pct.min():g} to {pct.max():g}")
        top = far_o.groupby("range_m")[m].max()
        add("few_percent_to_few_km", "the body sends a few percent at most to a few kilometers in these bands",
            "at each range of 1 km or more, no new case exceeds the largest production share (the sentence gives no number)", far,
            lambda v, q: v > q.range_m.map(top), was=round(float(far_o[m].max()), 3))
        second = (far_o[m] / far_o.band.map(med)).groupby([far_o.case, far_o.band]).max().nlargest(2).iloc[-1]
        add("the_two_closest", "in the two closest cases it is about half the clutter",
            "no new case and band comes nearer the median clutter, at 1 km or more, than the second of the production cases", far,
            lambda v, q: v / q.band.map(med) > second, was=round(float(second), 3))
        for name, sentence, lev in (("cleanest_twentieth_range", "against the cleanest twentieth of sites, those two cases reach the clutter out to ... km", p05),
                                    ("few_hundred_meters", "less than ordinary hills and valleys do except within a few hundred meters of it", med)):
            hit = o.range_m[o[m] >= o.band.map(lev)]
            reach = float(hit.max()) if len(hit) else 0.0
            add(name, sentence, "no new case reaches the level (cleanest twentieth; median) farther out than the farthest production case", fwd,
                lambda v, q, lev=lev, reach=reach: (v >= q.band.map(lev)) & (q.range_m > reach), was=f"{reach:g} m")
    for name, sentence in (("contrast_is_the_extreme", "these are upper values in contrast ... the contrast is the extreme"),
                           ("stiffest_contrast_considered", "at the stiffest contrast considered")):
        rows.append(dict(statement=name, measure="", outcome="left out", rows=0, sentence=sentence,
                         rule="compares grounds: this case is left out, and no nominal ground is invented for a column"))
    return pd.DataFrame(rows, dtype=object).fillna("")


def quality_factor(far, speed):
    """Only if a row breaks. The quality factor at which the largest ratio to the median among the breaking
    rows would fall to the median by a8_attenuation's formula, at that row's band center and range, with the
    fundamental mode's group speed there (speed[band][0]); the source's value beside it. A check row, or none."""
    worst = max((far[far[f"outcome_{m}"] == "breaks"].assign(over=lambda q, m=m: q[f"{m}_over_median"], measure=m) for m in ("share", "share_smoothed")),
                key=lambda q: q.over.max() if len(q) else -np.inf)
    if not len(worst):
        return []
    w = worst.loc[worst.over.idxmax()]
    f, u = float(np.mean(prod.BANDS[w.band])), speed[w.band][0]
    return [dict(check=f"{CASE}_quality_factor_at_which_the_largest_ratio_falls_to_the_median", value=float(np.pi * f * w.range_m / (u * np.log(w.over))),
                 detail=f"{w.band} Hz {w.azimuth} {w.range_m:g} m, {w.measure} {w.over:.3f} of the median; {f:g} Hz, group speed {u:.1f} m/s; "
                        f"the source's Qs is {QS_SOFT:g} where the shear speed is below {QS_BELOW * 1e3:g} m/s (p. 15)")]


def measure():
    rules()
    changes = settings()
    step_log("--measure", changes)
    with gen.swapped(constants(CASE, changes) + [(prod, "setup", setup)]):
        sp = case_spec(CASE)
        s = setup(CASE, sp)
        runs = tuple(attenuated.load(archive(w, changes)) for w in ("twin", "body"))
        rows, checks = prod.measure(CASE, sp, runs)
        shares = pd.DataFrame(smoothing.measure(CASE, sp, runs))
        at_f, speed = mode_speeds(s["h"], prod.DEPTH)
        peaks = peak_rows(s, runs, speed)
        points, level, l_r = edge_level(s, runs[0], twin_reference(s))
        delay = edge_delay(s)
    key = ["case", "azimuth", "range_m", "band"]
    slow = pd.DataFrame(rows).query("window == 'slow' and component == 'vertical'")[key + ["numerical"]]
    shares = shares.merge(slow, on=key, validate="one_to_one")
    repro = float((shares.share / shares.numerical - 1).abs().max())
    assert repro <= smoothing.REPRO_TOL, f"the share recomputed by a8_smoothing differs from a8_production.measure ({repro:.1e}); nothing is reported"
    d = bracket(shares.drop(columns="numerical").merge(peaks, on=key, validate="one_to_one"), level, l_r)
    st = statements(d)
    out = out_dir()
    d.round(6).to_csv(out / "a8_gradient_shares.csv", index=False)
    st.to_csv(out / "a8_gradient_statements.csv", index=False)
    for c in checks:  # two details of a8_production.measure speak of production's limits
        if c["check"].endswith("_peak_rss_gb"):
            c["detail"] = f"budget {gen.budget_gb():.1f} GB"
        if c["check"].endswith("_bands_measured"):
            c["detail"] = "the two higher bands; the lowest is not read: its surface wave reaches below the box"
        if c["check"].endswith("_incidence_deg"):
            c["detail"] = "at the surface of the column"
    far = d[d.range_m >= FAR_M]
    both = np.where((far.outcome_share == "breaks") | (far.outcome_share_smoothed == "breaks"), "breaks",
                    np.where((far.outcome_share == "supports") & (far.outcome_share_smoothed == "supports"), "supports", "untested"))
    checks += [dict(check=f"{CASE}_settings", value=said(changes) or "fixed settings", detail="every validation passed on them" if not SMOKE else "smoke"),
               dict(check=f"{CASE}_share_recomputed_max_relative_difference", value=repro, detail="a8_smoothing's unsmoothed share against a8_production.measure"),
               dict(check=f"{CASE}_reading_beyond_1km_either_measure", value=reading(len(far), int((both == "breaks").sum()), int((both == "untested").sum())),
                    detail=f"{len(far)} rows from {FAR_M:g} m to {s['ranges'][-1]:g} m, two bands, three azimuths: {int((both == 'supports').sum())} support, "
                           f"{int((both == 'untested').sum())} untested, {int((both == 'breaks').sum())} break (a row breaks if either measure breaks)"),
               dict(check=f"{CASE}_largest_share_over_median_beyond_1km", value=float(far.share_over_median.max()) if len(far) else "", detail="unsmoothed, committed medians"),
               dict(check=f"{CASE}_largest_share_over_lowest_median_beyond_1km", value=float(far.share_over_lowest_median.max()) if len(far) else "",
                    detail="reported beside it: the lowest medians"),
               dict(check=f"{CASE}_edge_wave_delay_at_body_top_s", value=round(delay, 3), detail="by ray times in the column, after the plane wave"),
               dict(check=f"{CASE}_rows_with_edge_flag", value=int(d.edge_flag.sum()), detail=f"of {len(d)}; a8_generous's rule with the column's times; the bracket decides, not the flag"),
               dict(check=f"{CASE}_bracket_assumption", value="", detail="the bracket assumes that the edge wave scatters as efficiently as the plane wave")]
    checks += [dict(check=f"{CASE}_edge_level_L_{b}", value=v, detail="; ".join(f"{q['point']}: {q['level']:.3f}" for q in points if q["band"] == b)) for b, v in level.items()]
    checks += [dict(check=f"{CASE}_incident_peak_term_L_r_largest_{b}", value=float(l_r[l_r.band == b].L_r.max()), detail="over the receivers") for b in s["bands"]]
    checks += [dict(check=f"{CASE}_fundamental_mode_{f:g}hz_phase_and_group_speed_ms", value=f"{c:.1f} {u:.1f}", detail="a8_reference1d, by the determinant") for f, (c, u) in at_f.items()]
    checks += quality_factor(far, speed)
    pd.DataFrame(checks).reindex(columns=["check", "value", "detail"]).to_csv(out / "a8_gradient_checks.csv", index=False)
    print(st[["statement", "measure", "outcome", "rows", "rows_supporting", "rows_untested", "rows_breaking", "largest", "production"]].to_string(index=False))
    print(f"L: {level}; wrote the tables of {CASE} to {out}" + ("; smoke: the numbers mean nothing" if SMOKE else ""))


# ---------------------------------------------------------------- rules, size
def rules():
    """Assertions on the column and the grid, made before anything is sized or run."""
    q, c4 = r4(), gen.CASES["R4"]
    z = np.linspace(0.0, 2000.0, 400001)
    vp, vs, rho = column(z)
    run_max = np.maximum(np.maximum.accumulate(source_vs(z / 1e3)), VS_FLOOR / 1e3) * 1e3
    assert np.allclose(vs, run_max, rtol=1e-12) and (np.diff(vs) >= 0).all(), "The running maximum of the source's shear speed, with the floor"
    assert vs.min() == VS_FLOOR and vp.min() == VP_FLOOR and np.allclose(rho, 1e3 * GARDNER[0] * (vp / 1e3) ** GARDNER[1])
    assert (VP_FLOOR, VS_FLOOR) == c4["host"][:2], "The floor is the soft ground, so its ratio of P to S speed is the one its gates validated"
    if not SMOKE:
        assert q["h"] == q["h_as_run"] and q["nbl"] == q["nbl_as_run"], "R4's settings differ from R4 as run (a8_generous_checks.csv)"
        assert H == q["h"] / 2 and np.isclose(gen.ppw(c4["host"], BANDS, H), gen.PPW_PROD, rtol=1e-12), "Half of R4's step; R4's points per wavelength at the top band"
        assert abs(int(round(SPONGE_M / H)) * H - SPONGE_M) < 1e-9, "The sponge has R4's thickness in meters"
        assert RMAX >= RANGE_MIN and prod.DR == 100.0, "Receivers every 100 m to at least 1.5 km"
    assert case_spec(CASE)["body"] == c4["body"] and case_spec(CASE)["stone"] == c4["stone"], "The body of R4"


def size():
    rules()
    out, q = out_dir(), r4()
    col = column_table()
    col.to_csv(out / "a8_gradient_column.csv", index=False)
    print(col.to_string(index=False))
    with gen.swapped(constants(CASE, {})):
        s = setup(CASE, case_spec(CASE))
        st = ref_stack(column, s["h"], prod.DEPTH)
        times = dict(sheet_depth_m=prod.Z_SHEET, sheet_to_surface_s=s["t_surf"] - s["t_c"], body_top_to_sheet_s=s["t0"] - s["t_c"],
                     reference_sheet_to_surface_s=ref.vertical_time(st, s["p"], 0.0, prod.Z_SHEET),
                     uniform_formula_sheet_to_surface_s=prod.Z_SHEET * np.sqrt(1 / s["host"][0] ** 2 - s["p"] ** 2),
                     edge_wave_after_plane_wave_at_body_top_s=edge_delay(s),
                     edge_wave_after_plane_wave_first_receiver_forward_s=edge_delay(s, 0.0, s["ranges"][0]),
                     edge_wave_after_plane_wave_first_receiver_backward_s=edge_delay(s, 0.0, -s["ranges"][0]))
        surf = surface_setup()
        onsets = edge_onset(s)
    names = [f"{az} {r:g} m" for az in prod.AZIMUTHS for r in s["ranges"]]
    usable = "; ".join(f"{name} (window ends {min(o, 1 / prod.F0):.3f} s after the pulse's center)" for name, o in zip(names, onsets) if o >= 0) or "none"
    with gen.swapped(constants("uniform_new", {})):
        mini = setup("uniform", case_spec("uniform_new"))

    def row(name, kind, g, dt, nt, runs, note):
        n = int(np.prod(g["shape"]))
        return dict(name=name, kind=kind, h_m=round(g["h"], 4), sponge_cells=g["nbl"], sponge_m=round(g["nbl"] * g["h"], 1), grid="x".join(str(v) for v in g["shape"]),
                    points_millions=round(n / 1e6, 1), dt_ms=round(dt * 1e3, 4), steps=nt, predicted_gb=round(gen.predicted_gb(g), 1), fits=gen.fits(g),
                    predicted_hours_each_at_r4_speed=round(n * nt / q["speed"] / 3600, 2), simulations=runs, note=note)
    rows = [row(CASE, "run", s["g"], s["dt"], s["nt"], 2, f"receivers to {s['ranges'][-1]:g} m; half-width {s['half']:g} m; box depth {DEPTH:g} m; "
                + "; ".join(f"{k} {v:.4f}" for k, v in times.items())),
            row("surface", "validation", surf["g"], surf["dt"], surf["nt"], 1, f"explosion {surf['z_src']:.2f} m deep; receivers at {SURFACE_RANGES[0]:g} and {SURFACE_RANGES[1]:g} m; "
                f"side half-width {surf['wy']:.0f} m, end margins {surf['xm']:.0f} m"),
            row("uniform", "validation", mini["g"], mini["dt"], mini["nt"], 4, "miniature box, R4's uniform ground on R4's grid step: old path and new path, twin and body"),
            dict(name="reference", kind="validation", simulations=0, note="no simulation: a8_reference1d's self-checks and R4's archived twin"),
            dict(name="twin", kind="validation", simulations=0, note=f"no simulation: the saved run without the body of {CASE}. Receivers where the pulse's "
                 f"first half-length fits before the edge wave begins: {usable}"),
            *[dict(name=k, kind="sha256", note=v) for k, v in code_sha().items()]]
    df = pd.DataFrame(rows, dtype=object).fillna("")
    df.to_csv(out / "a8_gradient_size.csv", index=False)
    at_f = mode_speeds(s["h"], DEPTH)[0]
    thin = {f: ref.tlm_modes(st, f)[0] for f in ref.MODE_FREQS}
    print(f"\nmemory model: {gen.production_log()[0]:.0f} bytes per point; budget {gen.budget_gb():.1f} GB, fits at {gen.SAFETY:g} of it "
          f"({gen.SAFETY * gen.budget_gb():.1f} GB); R4's speed as run {q['speed'] / 1e6:.0f} million point-steps per second; R4's step {q['h']:.4f} m, sponge {SPONGE_M:.1f} m")
    print(df.drop(columns="note").to_string(index=False))
    print(df[["name", "note"]].to_string(index=False))
    print("\nfundamental mode of the column (Hz: phase, group speed m/s by the determinant | by thin layers):")
    for f in ref.MODE_FREQS:
        print(f"  {f:g}: {at_f[f][0]:.1f} {at_f[f][1]:.1f} | {thin[f]['c']:.1f} {thin[f]['U']:.1f}")
    if SMOKE:
        print("smoke: a coarse miniature; these numbers mean nothing")


# ---------------------------------------------------------------- self-check
def _selfcheck() -> None:
    """No simulation (module docstring)."""
    rules()
    j = junctions()
    assert 0 < j[0]["depth_m"] < j[1]["depth_m"] < j[2]["depth_m"] < j[4]["depth_m"] < j[5]["depth_m"] < 500.0
    assert at(column, j[4]["depth_m"] - 1.0)[1] == VS_HELD * 1e3 and at(column, 0.0)[:2] == (VP_FLOOR, VS_FLOOR)
    # a uniform column gives the production set-up exactly, and the column's times differ from the uniform formula
    for name in ("uniform_new", CASE):
        with gen.swapped(constants(name, {})):
            sp = case_spec(name)
            new = setup(name, sp)
            old = PROD_SETUP(name, dict({k: v for k, v in sp.items() if k != "profile"}, host=at(sp["profile"], 0.0)))
            assert all(np.array_equal(new["g"][k], old["g"][k]) for k in old["g"])
            if name == CASE:
                assert new["t_surf"] < old["t_surf"] and new["t0"] < old["t0"] and (new["vp_fast"], new["vs_fast"], new["vs_slow"]) == (5500.0, 3200.0, VS_FLOOR)
                st = ref_stack(column, new["h"], prod.DEPTH)
                assert abs(new["t_surf"] - new["t_c"] - ref.vertical_time(st, new["p"], 0.0, prod.Z_SHEET)) < new["dt"], "Within one sample"
                assert abs(top_point(new)[2] - new["body"]["apex_depth"]) <= new["h"] / 2
            else:
                assert all(new[k] == old[k] for k in ("t0", "t_surf", "nt", "dt", "t_c", "vs_slow")), "uniform column: the production set-up, bit for bit"
                d = prod.Z_SHEET - new["body"]["apex_depth"]
                straight = np.hypot(new["half"], d) / new["host"][0]
                assert abs(first_arrival(new["profile"], prod.Z_SHEET, new["body"]["apex_depth"], new["half"], prod.DEPTH) / straight - 1) < 1e-3
                assert abs(edge_delay(new) / gen.edge_delay(new) - 1) < 5e-3, "uniform column: a8_generous's edge delay"
    # the explosion's closed form against gate three's reference (Lamb's residue), in gate three's own ground
    for f in gates.RG_FREQS:
        want = gates.rayleigh_reference(f, gates.RANGES[0], gates.H / 2)[0]
        k = 2 * np.pi * f / ref.rayleigh_exact(*gates.HOST[:2])
        got = abs(ref.half_space_excitation(*gates.HOST, f, gates.SRC_DEPTH, gates.H / 2)) * np.sqrt(2 / (np.pi * k * gates.RANGES[0])) * gates.H**3
        assert abs(got / want - 1) < 1e-3, (f, got, want)
    # the surface-wave measurement on synthetic traces of a uniform half-space: the mode's speed and the modal sum come back
    host = gen.CASES["R4"]["host"]
    s = dict(dt=2e-3, nt=3000, h=10.0, t0=0.6, z_src=20.0, nodes=np.array([[500.0, 0, 0], [1000.0, 0, 0]]))
    c = ref.rayleigh_exact(*host[:2])
    full = modal_traces(s, lambda z: host, 300.0, s["nodes"][:, 0])[0]
    rows = surface_rows(s, full, lambda z: host, 300.0, {1.5: (c, c)})
    assert all(abs(r["ratio"] - 1) < 2e-3 for r in rows), [(r["quantity"], r["ratio"]) for r in rows]
    # the log: one repeat, everything on one set of settings
    def log(*entries):  # (validation, settings, passed)
        return pd.DataFrame([dict(dict.fromkeys(VAL_COLS, ""), validation=v, attempt="1", settings=k, passed=str(p)) for v, k, p in entries], columns=VAL_COLS)

    def stops(fn, *args):
        try:
            fn(*args)
        except SystemExit:
            return True
        return False
    ok = [(v, "", True) for v in VALIDATIONS]
    one = dict(dt_safety=0.6)
    assert stops(run_settings, log()) and stops(run_settings, log(*ok[:3])) and run_settings(log(*ok)) == {}
    assert next_attempt("twin", {}, "", log(*ok[:3])) == 1 and stops(next_attempt, "twin", {}, "", log(*ok)), "not rerun after a pass"
    assert stops(next_attempt, "twin", one, "why", log(*ok[:3])), "--set needs a failure first"
    failed = log(*ok[:3], ("twin", "", False))
    assert stops(run_settings, failed) and stops(next_attempt, "twin", {}, "", failed) and stops(next_attempt, "twin", one, "", failed)
    assert next_attempt("twin", one, "why", failed) == 2
    again = log(*ok[:3], ("twin", "", False), ("twin", said(one), True))
    assert stops(run_settings, again), "the others must be rerun on the changed settings"
    assert stops(next_attempt, "surface", {}, "", again) and stops(next_attempt, "surface", dict(dt_safety=0.5), "why", again)
    assert next_attempt("surface", one, "why", again) == 2 and stops(next_attempt, "twin", one, "why", again)
    assert run_settings(log(*ok[:3], ("twin", "", False), *[(v, said(one), True) for v in VALIDATIONS])) == one
    assert stops(run_settings, log(*ok[:3], ("twin", "", False), ("twin", said(one), False)))
    # A change of settings logged by a run counts as the one repeat; --next follows the order and stops at a failure
    used = log(*ok[:3], ("twin", "", False), ("settings", said(one), ""))
    assert repeat(used) == said(one) and stops(check_settings, dict(dt_safety=0.5), used) and stops(check_settings, {}, used)

    def never(kind, name, changes):
        return False
    for table, done, want in ((log(), never, ("validate", "reference")), (log(*ok[:3]), never, ("run", "twin")),
                              (log(*ok[:3]), lambda k, n, c: n == "twin", ("validate", "twin")), (log(*ok), lambda k, n, c: n == "twin", ("run", "body")),
                              (log(*ok), lambda k, n, c: k == "run", ("measure", "")), (used, never, ("validate", "reference"))):
        ch, steps, nxt, stop = plan(table, done)
        assert steps[nxt][:2] == want and not stop and ch == (one if table is used else {}), (want, steps, nxt, stop)
    assert plan(log(*ok), lambda k, n, c: True)[2:] == (None, "") and [q[:2] for q in plan(log(), never)[1]] == list(ORDER)
    if not SMOKE:
        assert plan(failed, never)[2] is None and "one repeat" in plan(failed, never)[3]
        assert plan(log(*ok[:3], ("twin", "", False), ("twin", said(one), False)), never)[2] is None
    # The twin check's window ends where the edge wave begins; a receiver where the pulse's first half does not fit is left out
    sw = dict(dt=2e-3, nt=2000)
    tt = np.arange(sw["nt"]) * sw["dt"]
    clean = fd.ricker(tt, prod.F0, 2.0)
    edged = clean + fd.ricker(tt, prod.F0, 2.55)  # an edge wave as large as the pulse, 0.55 s behind it
    rows = pulse_rows(sw, np.column_stack([edged, edged]), np.column_stack([clean, clean]), ["a", "b"], [2.0, 2.0],
                      np.array([0.55, 0.3]) - 1 / prod.F0, 0.2, FREQS)
    judged = [r for r in rows if r["passed"] != ""]
    assert len(judged) == len(FREQS) + 1 and all(r["passed"] for r in judged) and [r["at"] for r in rows if r["quantity"] == "receiver_not_used"] == ["b"]
    old = [abs(fd.win_dft(edged, f, 1.5, 2.5, sw["dt"], gates.TAPER)) / abs(fd.win_dft(clean, f, 1.5, 2.5, sw["dt"], gates.TAPER)) for f in FREQS]
    assert max(abs(q - 1) for q in old) > gates.TOL, "the window as first built would have failed on this edge wave"
    assert stops(window_ends, [-0.1, -0.2], 1, "test") == (not SMOKE) and len(code_sha()["a8_gradient.py"]) == 64
    assert [k for _, k, _ in numerics(dict(dt_safety=0.6, sponge_m=2 * SPONGE_M), H)] == ["NBL", "DT_SAFETY"] and numerics(dict(sponge_m=2 * SPONGE_M), H)[0][2] == 2 * numerics({}, H)[0][2]
    # the bracket and the statements on made-up rows
    med, _, lowest = medians()
    ranges = np.arange(100.0, 1501.0, 100.0)
    base = pd.DataFrame([dict(case=CASE, azimuth=az, range_m=r, band=b, peak_window="rayleigh", peak_h_over_v=0.7, peak_phase_lag_deg=-90.0)
                         for az in prod.AZIMUTHS for r in ranges for b in BANDS])
    l_r = base[["azimuth", "range_m", "band"]].assign(L_r=0.1)
    with gen.swapped([(sys.modules[__name__], "FAR_M", smoothing.FAR_M)]):
        for scale, level, want in ((0.5, 0.2, "supports"), (0.9, 0.2, "untested"), (2.0, 0.2, "breaks")):
            d = bracket(base.assign(share=scale * base.band.map(med), share_smoothed=scale * base.band.map(med)), {b: level for b in BANDS}, l_r)
            assert (d.outcome_share == want).all() and np.allclose(d.share_low, d.share * 0.8 / 1.1) and np.allclose(d.share_high, d.share * 1.2 / 0.9)
            got = statements(d).set_index(["statement", "measure"]).loc[("below_median_beyond_1km", "share")]
            assert got.outcome == dict(supports="holds", untested="could not decide: every row untested", breaks="broken")[want], dict(got)
            qf = quality_factor(d[d.range_m >= FAR_M], {b: (300.0, 350.0) for b in BANDS})  # A quality factor only if a row breaks
            assert len(qf) == (want == "breaks") and (not qf or abs(scale * gen.attenuation.factor(1.5, FAR_M, qf[0]["value"], 300.0) - 1) < 1e-9), qf
        assert reading(10, 0, 3) == "holds in the rows tested; some untested" and lowest[BANDS[0]] <= med[BANDS[0]]
        # with no bracket, every sentence comes out as a8_generous.statements has it (its own rule)
        rng = np.random.default_rng(0)
        for scale in (0.02, 0.3, 1.5):
            v = scale * base.band.map(med) * rng.uniform(0.5, 1.5, len(base)) * np.sqrt(1000.0 / base.range_m)
            mine = statements(bracket(base.assign(share=v, share_smoothed=1.1 * v), {b: 0.0 for b in BANDS}, l_r.assign(L_r=0.0))).set_index(["statement", "measure"]).outcome
            with gen.swapped([(gen, "CASES", {CASE: dict(host=None)})]):
                theirs = gen.statements(base.assign(share=v, share_smoothed=1.1 * v, edge_flag=False)).set_index(["statement", "measure"]).outcome
            for key in mine.index:
                assert key[1] == "" or mine[key] == theirs[key], (scale, key, mine[key], theirs[key])
    print("a8_gradient selfcheck ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="A8 gradient column, case R5. Default: the size table.")
    ap.add_argument("--size", action="store_true")
    ap.add_argument("--validate", choices=VALIDATIONS)
    ap.add_argument("--run", choices=("twin", "body"))
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--next", action="store_true", help="The fixed order from where the log stands: print the plan, do the next step only")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help=f"the one repeat's changes: {', '.join(RETRY)}")
    ap.add_argument("--note", default="", help="why the repeat changes what it changes")
    ap.add_argument("--selfcheck", action="store_true")
    arg = ap.parse_args()
    change = {k: float(v) for k, v in (kv.split("=") for kv in arg.set) if k in RETRY or ap.error(f"--set {k}: one of {', '.join(RETRY)}")}
    if arg.selfcheck:
        _selfcheck()
    elif arg.next:
        if change:
            ap.error("--next takes its settings from the log; the one repeat is made by hand with --validate NAME --set ... --note")
        next_step()
    elif arg.validate:
        validate(arg.validate, change, arg.note)
    elif arg.run:
        run(arg.run, change, arg.note)
    elif arg.measure:
        measure()
    else:
        size()
