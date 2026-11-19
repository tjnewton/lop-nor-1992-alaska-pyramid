"""A8 generous end: four more simulations where detection is most favored.

Question: do the statements of Section 5 hold in the highest band of the slower ground (R1), for a body
of stone (R2), for the largest base (R3) and for all of these at once in soft ground (R4)?

Rules implemented here, none changed: the runs, bodies, materials, gates, the box of
R4, the statements at risk, gate frequencies, a run bound to its gate's settings, three
outcomes for a statement, the series judged at the
gate's frequencies, the soft-ground pulse, what "fits" means, two gates on one setting, the edge wave
measured. Tolerances are those of a8_gates. The smoothed share is that of a8_smoothing. The
attenuation formula is that of a8_attenuation.

Inputs: a8_production.py (set-up, run and measurement, by import, with the case passed as `spec`);
a8_gates.py (gates two and three); the committed a8_production_checks.csv (memory and speed of
the production runs), a8_smoothing.csv (shares of the production cases), a8_production.csv (R1's
comparison) and a5_dprk_site_spread.csv (the clutter, through a8_smoothing.clutter).

Method:
- Size. Grid step of R1 and R4: the largest that gives the ground's Rayleigh wave the production
  runs' points per wavelength at the top of the highest band measured. Memory: the largest bytes per
  grid point of the committed production runs, times the points. Time: the slowest committed run's
  points per second. Both are proportional models; they err high for grids larger than production's.
  A run or gate fits when its predicted memory is at most SAFETY of the budget.
- Edge-wave rule (a8_elastic3d, rule 4a), in the form a8_gates.sphere_setup applies to its own
  windows: the sheet's edge wave must reach the body's top later than the plane wave by the pulse's
  half-length plus what is left of the slow window after the ground's Rayleigh wave has come. The
  delay is that of the edge the tilted wave leaves first. R4's half-width is the smallest that meets
  the rule at every receiver and band, to the receiver spacing; if that exceeds the memory budget,
  the widest that fits. A row that breaks the rule carries edge_flag.
- Edge-wave level, for every case, production's included, from the run without the body:
  see `edge_level`.
- Gates. The fixed gate code runs unchanged with other module constants: the run's ground, stone, grid
  step and time step; only the frequencies the run measures; the pulse's peak frequency scaled by
  the run's highest frequency over that of the gates; source, sheet and receivers moved to the nearest nodes.
  The series' check against the weak-scattering limit is judged at the rows up to the gate's
  highest frequency. A sphere gate whose full box does not fit runs on the half box with the
  solver's mirror plane (the same problem; not ruled, see MIRROR_IF_TOO_BIG). A repeat changes
  numerical settings only (--set) and is logged with them.
- Run. A case runs only when the latest attempt of each of its gates passed, all on one set of
  settings, which the run takes. If they differ, the refusal names the command that runs the other
  gate on the common settings. Twin first, then the body, each in a fresh process.
- Measure. a8_production.measure and a8_smoothing.measure, unchanged. Added: a window from the
  ground's Rayleigh arrival (less half a period, as every production window opens) to the end of the
  slow window; the time of the slow window's peak after the incident P; the part of the slow window
  that held that peak, with its ratio of horizontal to vertical motion and phase lag; attenuated
  shares by formula.
- Statements. Each is tested for the unsmoothed and the smoothed share over the production cases and
  the new ones (forward azimuth, as a8_smoothing; the statement on "every simulated case" on all
  three azimuths). Outcomes: holds; broken; untested, when only rows
  with the edge-wave flag break it (an edge wave can only raise a peak). The rule tested is written
  in the table: several sentences give no number, and are tested against the production cases.

Outputs, in this folder (with AKP_A8_SMOKE=1 in $AKP_DATA/a8/generous_smoke): a8_generous_size.csv;
a8_generous_gates.csv and a8_generous_gate_checks.csv (every attempt); a8_generous.csv (a8_production.csv's
columns, the form akp/stage_p4/s9_census.detection_ranges reads, plus rows of the window `rayleigh`);
a8_generous_shares.csv; a8_generous_attenuation.csv (a8_attenuation.csv's columns); a8_generous_statements.csv;
a8_generous_convergence.csv (R1 against production); a8_generous_edge.csv; a8_generous_checks.csv.
Traces: $AKP_DATA/a8/generous.

Run, in this order, one command at a time, each under the memory guard:
  PYTHONPATH=. python -m akp.stage_a.a8_generous                       (--size: the size table; nothing heavy)
  OMP_NUM_THREADS=6 PYTHONPATH=. python scripts/mem_guard.py --frac 0.40 --every 0.5 -- \
      python -m akp.stage_a.a8_generous --gate sphere_thawed           (then --gate surface_soft, --gate sphere_soft)
      ... --gate NAME --set h=8 --note "why"                           (the one repeat after a failure)
      ... --run R1                                                     (then R2, R3, R4)
      ... --measure                                                    (saved traces only; all tables)
  R4, surface_soft and sphere_soft run only with --full-size; without it the module stops.
  AKP_A8_SMOKE=1 ...                                                   (miniature boxes, coarse gates: the pipeline only)

Self-check (--selfcheck, no simulation): constants are put back after a gate; a run is refused when a
gate is missing, failed, or passed on other settings than its sister gate, and the refusal names the
next command; a repeat needs a change and there is only one; the soft-ground pulse is scaled and the
series passes at the gate's frequencies; a wider box breaks the edge rule at fewer receivers; a statement comes out holds,
broken or untested on made-up rows; the thawed stone gate differs from gate two in the stone alone, and
its time step is R2's.
"""
from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from akp.stage_a import a8_elastic3d as fd  # first: sets the Devito environment
from akp.stage_a import a8_attenuated as attenuated
from akp.stage_a import a8_attenuation as attenuation
from akp.stage_a import a8_gates as gates
from akp.stage_a import a8_production as prod
from akp.stage_a import a8_smoothing as smoothing
from akp.stage_a import a8_sphere_series as series
from akp.stage_a.a3b4_rg import rayleigh_speed

HERE = Path(__file__).parent
SMOKE = prod.SMOKE
# Inputs fixed by which amends before any gate or run.
SOFT = (1500.0, 300.0, 1900.0)  # Soft saturated ground, P and S speed (m/s), density (kg/m3); an end-member
STONE = (5500.0, 3200.0, 2650.0)  # Kept: body of stone; S speed and density are round values without a page
BASE_MAX = 400.0  # m. Largest base of the trials' prior
TOP_MIN = 30.48  # m. Shallowest top of the trials' prior (100 ft)
F_SOFT = 2.0  # Hz. Highest frequency measured in soft ground
Q_D102 = (50.0, 20.0)  # The Q values shown beside the elastic share (both are in a8_attenuation.Q_VALUES)
BUDGET = 0.40  # Share of RAM one run may take
SAFETY = 0.9  # A run fits when its predicted memory is at most this share of the budget (the memory model is extrapolated)
MIRROR_IF_TOO_BIG = True  # not ruled: a sphere gate whose full box does not fit runs on the half box y >= 0 (a8_elastic3d's mirror plane)
EDGE_CASES = ("R4",)  # "Box of R4": the case whose box follows the edge-wave rule and whose flags enter the statements
HELD = ("R4", "surface_soft", "sphere_soft")  # these run only with --full-size
RETRY = dict(h=(gates, "H"), dt_safety=(fd, "DT_SAFETY"), nbl=(fd, "NBL"), sponge_a=(fd, "SPONGE_A"))  # What a repeat may change
SMOKE_COARSEN, SMOKE_GB = 4.0, 0.7  # smoke only: gates on a grid this much coarser (their verdicts mean nothing); the memory budget
GATE_COLS = gates.GATE_COLS + ["h_m", "dt_s", "changed", "note"]


def thawed(case):
    """(vp, vs, rho) of a production case's thawed ground."""
    vp = prod.CASES[case][0]
    return (vp, vp / prod.VP_VS, prod.RHO)


def ppw(host, bands, h):
    """Points per Rayleigh wavelength of the ground at the top of the highest band measured."""
    return rayleigh_speed(host[0], host[1]) / prod.BANDS[bands[-1]][1] / h


PPW_PROD = min(ppw(thawed(c), b, prod.H) for c, (_, _, b) in prod.CASES.items())  # the production runs' resolution (decision 8)


def step(host, bands):
    """The largest grid step that gives the Rayleigh wave the production runs' points per wavelength."""
    return rayleigh_speed(host[0], host[1]) / prod.BANDS[bands[-1]][1] / PPW_PROD


T1200, T850, ALL = thawed("thawed_1200"), thawed("thawed_850"), tuple(prod.BANDS)
SOFT_BANDS = tuple(b for b, (_, top) in prod.BANDS.items() if top <= F_SOFT)
BIG = dict(prod.BODY, base=BASE_MAX)  # height and top as production
CASES = {  #. A key left out takes the production value (a8_production.setup).
    "R1": dict(host=T850, bands=ALL, h=step(T850, ALL)),
    "R2": dict(host=T1200, bands=ALL, stone=STONE, gates=("sphere_thawed",)),
    "R3": dict(host=T1200, bands=ALL, body=BIG),
    "R4": dict(host=SOFT, bands=SOFT_BANDS, stone=STONE, body=dict(BIG, apex_depth=TOP_MIN), h=step(SOFT, SOFT_BANDS),
               gates=("surface_soft", "sphere_soft")),
}
GATES = {"sphere_thawed": ("sphere", "R2"), "surface_soft": ("surface", "R4"), "sphere_soft": ("sphere", "R4")}  # Kind, the run it precedes


def out_dir():
    if not SMOKE:
        return HERE
    from akp.paths import data_dir
    d = data_dir("a8") / "generous_smoke"
    d.mkdir(parents=True, exist_ok=True)
    return d


def archive(case, with_body):
    from akp.paths import data_dir
    d = data_dir("a8") / ("generous_smoke" if SMOKE else "generous")
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{case}_{'body' if with_body else 'twin'}.npz"


@contextmanager
def swapped(triples):
    """Module constants (module, name, value) set for one gate or run, then put back (as a8_attenuated swaps fd.simulate)."""
    old = [(m, k, getattr(m, k)) for m, k, _ in triples]
    for m, k, v in triples:
        setattr(m, k, v)
    try:
        yield
    finally:
        for m, k, v in old:
            setattr(m, k, v)


def numerics(changes):
    """The solver-wide settings among a repeat's changes; the grid step goes through the case or the gate."""
    return [(*RETRY[k], int(v) if k == "nbl" else v) for k, v in changes.items() if k != "h"]


# ---------------------------------------------------------------- size, memory, the edge-wave rule
@lru_cache(maxsize=1)
def production_log():
    """(bytes per grid point, points times steps per second) of the committed production runs: the worst of the eight."""
    c = pd.read_csv(HERE / "a8_production_checks.csv").set_index("check")
    mem, speed = [], []
    for case in prod.CASES:
        npts = np.prod([int(v) for v in c.value[f"{case}_grid_shape"].split("x")])
        steps = int(c.detail[f"{case}_time_step_s"].split()[0])
        for run in ("twin", "body"):
            mem.append(float(c.value[f"{case}_{run}_peak_rss_gb"]) * 1e9 / npts)
            speed.append(npts * steps / float(c.value[f"{case}_{run}_run_seconds"]))
    return max(mem), min(speed)


def predicted_gb(g):
    return production_log()[0] * np.prod(g["shape"]) / 1e9


def budget_gb(smoke=SMOKE):
    return SMOKE_GB if smoke else BUDGET * os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9


def fits(g, smoke=SMOKE):
    """Predicted memory at most SAFETY of the budget."""
    return bool(predicted_gb(g) <= SAFETY * budget_gb(smoke))


def edge_delay(s):
    """Seconds by which the sheet's edge wave reaches the body's top after the plane wave.

    The wave is tilted, so the edge it leaves first decides. With no tilt this is the solver's
    (sqrt(W^2 + d^2) - d) / vp, W the half-width and d the height of the top above the sheet."""
    vp, d = s["host"][0], prod.Z_SHEET - s["body"]["apex_depth"]
    return float(np.hypot(s["half"], d) / vp - s["half"] * s["p"] - d * np.sqrt(1 / vp**2 - s["p"] ** 2))


def edge_needed(s, band, r):
    """The delay the rule asks for at range r: the pulse's half-length, and the slow window's end less the Rayleigh arrival."""
    return 1 / prod.F0 + r / (0.85 * s["vs_slow"]) + 1.5 / np.mean(prod.BANDS[band]) - r / s["c_r"]


def rule_half(s):
    """Smallest half-width, from the production one up by the receiver spacing, that meets the rule everywhere."""
    need = max(edge_needed(s, b, r) for b in s["bands"] for r in s["ranges"])
    half = prod.RMAX + prod.MARGIN
    while edge_delay(dict(s, half=half)) < need:
        half += prod.DR
    return half


def box(c):
    """R4's half-width: the one the rule asks for, or the widest that fits the memory budget."""
    s = prod.setup("box", c)
    half = rule_half(s)
    while half > s["half"] and not fits(prod.setup("box", dict(c, margin=half - prod.RMAX))["g"]):
        half -= prod.DR
    return half


def spec(case, changes):
    """The case as a8_production takes it: the fixed inputs, the grid step its gate passed on, R4's box."""
    c = {k: v for k, v in CASES[case].items() if k != "gates"}
    if "h" in changes:
        c["h"] = changes["h"]
    if case in EDGE_CASES:
        c["margin"] = box(c) - prod.RMAX
    return c


def rules():
    """The fixed assertions on R1 and R4, made before anything is sized or run."""
    r1, r4 = prod.setup("R1", spec("R1", {})), CASES["R4"]
    p12, p85 = prod.setup("thawed_1200"), prod.setup("thawed_850")

    def ratios(s):  # every speed over the ground's P speed, and the density contrast
        return [v / s["host"][0] for v in (*s["host"][:2], *s["stone"][:2])] + [s["stone"][2] / s["host"][2]]
    assert np.allclose(ratios(r1), ratios(p12), rtol=1e-12) and r1["body"] == p12["body"], "R1: the speed ratios of production"
    assert np.isclose(r1["h"], prod.H * T850[0] / T1200[0], rtol=1e-12), "R1: the grid is refined in proportion"
    assert np.isclose(ppw(T850, ALL, r1["h"]), ppw(T1200, ALL, p12["h"]), rtol=1e-12), "R1: production's points per wavelength"
    # The rule says "the same ratio of time step to grid step". It is the same as production's in this ground;
    # against the faster ground, whose problem R1 scales, it is the Courant number that is the same.
    assert np.isclose(r1["dt"] / r1["h"], p85["dt"] / p85["h"], rtol=1e-12), "R1: time step over grid step, as production in this ground"
    assert np.isclose(r1["vp_fast"] * r1["dt"] / r1["h"], p12["vp_fast"] * p12["dt"] / p12["h"], rtol=1e-12), "R1: the Courant number"
    assert np.isclose(ppw(SOFT, SOFT_BANDS, r4["h"]), PPW_PROD, rtol=1e-12) and prod.BANDS[SOFT_BANDS[-1]][1] == F_SOFT, "R4: grid step"
    assert r4["body"]["height"] == r1["body"]["height"] == prod.BODY["height"] and set(Q_D102) <= set(attenuation.Q_VALUES)


def edge_text(s):
    """Per band, the nearest receiver whose slow window breaks the rule (the need grows with range)."""
    out = []
    for b in s["bands"]:
        bad = [r for r in s["ranges"] if edge_delay(s) < edge_needed(s, b, r)]
        out.append(f"{b}: " + (f"broken from {bad[0]:g} m" if bad else "met"))
    return "; ".join(out)


def size_row(name, kind, g, dt, nt, host, bands, runs, **more):
    gb, n = predicted_gb(g), int(np.prod(g["shape"]))
    return dict(name=name, kind=kind, h_m=round(g["h"], 4), points_per_rayleigh_wavelength=round(ppw(host, bands, g["h"]), 2),
                at_hz=prod.BANDS[bands[-1]][1], grid="x".join(str(v) for v in g["shape"]), points_millions=round(n / 1e6, 1),
                dt_ms=round(dt * 1e3, 4), steps=nt, predicted_gb=round(gb, 1), predicted_hours_each=round(n * nt / production_log()[1] / 3600, 2),
                simulations=runs, fits=fits(g), **more)


def size():
    """The size table: every case and gate, computed before anything runs. Production's rows are there for comparison."""
    rules()
    bpp, pps = production_log()
    rows = []
    for case in list(prod.CASES) + list(CASES):
        new = case in CASES
        changes, state = settings(case, quiet=True) if new else ({}, "")
        with swapped(numerics(changes)):
            s = prod.setup(case, spec(case, changes) if new else None)
        rows.append(size_row(case, "run" if new else "production, for comparison", s["g"], s["dt"], s["nt"], s["host"], s["bands"], 2,
                             half_width_m=s["half"], half_width_rule_m=rule_half(s), edge_delay_s=round(edge_delay(s), 2),
                             edge_rule=edge_text(s), note=state + ("; needs --full-size" if case in HELD else "")))
    for name, (kind, case) in GATES.items():
        with swapped(gate_settings(name, {})):
            s = gates.sphere_setup() if kind == "sphere" else gates.surface_setup()
            freqs = gates.FREQS if kind == "sphere" else gates.RG_FREQS
            bad = series_failures(gates.sphere_reference_checks(s) + series_rows()) if kind == "sphere" else ""
            note = f"wavelet peak {gates.F0:g} Hz; frequencies {' '.join(f'{f:g}' for f in freqs)} Hz" + ("; mirror plane y = 0" if s["g"]["sym"] else "")
        c = CASES[case]
        rows.append(size_row(name, f"gate before {case}", s["g"], s["dt"], s["nt"], c["host"], c["bands"], 2 if kind == "sphere" else 1,
                             note=note + ("; needs --full-size" if name in HELD else "")
                             + (f"; STOPS: the exact series fails its own checks here ({bad})" if bad else "")))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir() / "a8_generous_size.csv", index=False)
    print(f"memory model: {bpp:.0f} bytes per point; speed model: {pps / 1e6:.0f} million point-steps per second; "
          f"budget {budget_gb():.1f} GB, fits at {SAFETY:g} of it ({SAFETY * budget_gb():.1f} GB); production points per Rayleigh wavelength {PPW_PROD:.2f}")
    print(df.drop(columns=["edge_rule", "note"]).to_string(index=False))
    print(df[["name", "edge_rule", "note"]].to_string(index=False))


# ---------------------------------------------------------------- gates
def gate_settings(name, changes):
    """The constants one gate runs with: the fixed gate with its run's ground, stone, grid step and time step,
    the frequencies the run measures, the pulse scaled to them, and what a repeat changed."""
    kind, case = GATES[name]
    c = CASES[case]
    h0 = changes.get("h", c.get("h", prod.H))
    top = prod.BANDS[c["bands"][-1]][1]

    def build(h, sym):
        def node(v):  # The fixed gate has its source, sheet and receivers on nodes; a step that does not divide them would misplace them
            return round(v / h) * h
        return [(gates, "HOST", c["host"]), (gates, "STONE", c["stone"]), (gates, "WEAK", tuple((1 + gates.WEAK_REL) * v for v in c["host"])),
                (series, "HOST", c["host"]), (series, "STONE", c["stone"]), (gates, "H", h), (gates, "SYM_Y", sym),
                (gates, "DT_VP", max(c["host"][0], c["stone"][0])),  # the run's time step follows its fastest speed, the stone's
                (gates, "FREQS", tuple(f for f in gates.FREQS if f <= top)), (gates, "RG_FREQS", tuple(f for f in gates.RG_FREQS if f <= top)),
                (gates, "F0", gates.F0 * top / gates.FREQS[-1]),  # The pulse stands to the run's highest frequency as the fixed did to its own
                (gates, "SRC_DEPTH", node(gates.SRC_DEPTH)), (gates, "RANGES", tuple(node(r) for r in gates.RANGES)),
                (gates, "D_SHEET", node(gates.D_SHEET)),
                (gates, "SPHERE_MARGIN", float(np.ceil((gates.R + gates.SPHERE_MARGIN) / h) * h - gates.R))] + numerics(changes)
    full, sym = build(h0, False), False
    if kind == "sphere" and MIRROR_IF_TOO_BIG:  # judged at full size, so that the smoke gate has the real gate's layout
        with swapped(full):
            sym = not fits(gates.sphere_setup()["g"], smoke=False)
    return build(h0 * (SMOKE_COARSEN if SMOKE else 1), sym)


def gate_table():
    path = out_dir() / "a8_generous_gates.csv"
    return pd.read_csv(path, keep_default_na=False) if path.exists() else pd.DataFrame(columns=GATE_COLS)


def gate_state(name, table):
    """(not run | passed | failed, attempt, what that attempt changed) for the gate's latest attempt. Every row decides."""
    t = table[table.gate == name]
    if t.empty:
        return "not run", 0, {}
    last = t[t.attempt.astype(int) == t.attempt.astype(int).max()]
    changed = dict((k, float(v)) for k, v in (kv.split("=") for kv in str(last.changed.iloc[0]).split("; ") if kv))
    return ("passed" if (last.passed.astype(str) == "True").all() else "failed"), int(last.attempt.iloc[0]), changed


def said(changes):
    return "; ".join(f"{k}={v:g}" for k, v in changes.items()) or "the fixed settings"


def command(name, changes):
    """The command that runs a gate on given settings, as text for a refusal."""
    return (f"python -m akp.stage_a.a8_generous --gate {name}" + "".join(f" --set {k}={v:g}" for k, v in changes.items())
            + (" --full-size" if name in HELD else "") + " (under the memory guard, as the module docstring shows)")


def lead(case, table):
    """Two gates and one setting: (gate, settings) the run's other gates follow: the passed gate with the latest
    attempt, the first listed on a tie. (None, {}) when none has passed."""
    passed = [(at, -i, g, ch) for i, g in enumerate(CASES[case].get("gates", ())) for st, at, ch in [gate_state(g, table)] if st == "passed"]
    return max(passed, key=lambda q: q[:2])[2:] if passed else (None, {})


def next_attempt(name, changes, table):
    """A failed gate is repeated once, with numerical settings changed. A gate is not rerun after a pass.
But a gate is run on the settings its sister gate passed on, when they differ from its own."""
    state, attempt, own = gate_state(name, table)
    ref, common = lead(GATES[name][1], table)
    if ref not in (None, name) and common != own and state != "failed":
        if changes != common:
            raise SystemExit(f"{name}: {ref} passed on {said(common)}. This gate runs on the same settings. Run next: {command(name, common)}")
        return attempt + 1
    if state == "not run":
        if changes:
            raise SystemExit(f"{name}: the first attempt runs on the fixed settings; --set is for the repeat")
        return 1
    if state == "passed":
        raise SystemExit(f"{name}: attempt {attempt} passed; it is not run again")
    if attempt >= 2:
        raise SystemExit(f"{name}: attempt {attempt} failed and no attempt is left. Its run is not made, and the paper says the statement is untested for that case")
    if not changes:
        raise SystemExit(f"{name}: attempt 1 failed. The one repeat changes numerical settings only: --set {' | '.join(RETRY)}=VALUE, and --note")
    return 2


def run_settings(case, table):
    """The numerical settings a run takes from its gates. Stops if a gate has not passed."""
    states = {g: gate_state(g, table) for g in CASES[case].get("gates", ())}
    bad = [f"{g} {state}" for g, (state, _, _) in states.items() if state != "passed"]
    if bad:
        raise SystemExit(f"{case} is not run: gate {', '.join(bad)}")
    ref, common = lead(case, table)
    other = [g for g, (_, _, ch) in states.items() if ch != common]
    if other:  # Both gates must have passed on one set of settings
        raise SystemExit(f"{case} is not run: {ref} passed on {said(common)}, {', '.join(other)} on other settings. Run next: "
                         + "; then ".join(command(g, common) for g in other))
    return common


def settings(case, quiet=False):
    """(changes, note). In smoke a coarse gate's verdict means nothing, so a missing or failed gate is reported and passed over."""
    try:
        return run_settings(case, gate_table()), "gates passed" if CASES[case].get("gates") else "no gate"
    except SystemExit as e:
        if not (SMOKE or quiet):
            raise
        if not quiet:
            print(f"smoke only; at full size this stops the run: {e}")
        return {}, str(e)


def held(name, approved):
    if name in HELD and not (SMOKE or approved):
        raise SystemExit(f"{name} runs only with --full-size. Add it")


def series_rows():
    """The fixed first check of the series (against the weak-scattering limit) row by row, inside a gate's settings.
Every row is recorded; the rows up to the gate's highest frequency are judged."""
    top = max(gates.FREQS)
    return [dict(check=f"series_weak_vs_born_rel_err_at_kpa_{ka:g}", value=e, passed=bool(e <= series.BORN_TOL),
                 detail=f"{f:.2f} Hz; " + ("judged" if f <= top else f"not judged: above the gate's {top:g} Hz")) for ka, f, e in series.weak_rows(gates.A)]


def series_failures(checks):
    """The series' checks that stop a gate: all but the largest error over every row, which the rule replaces by the judged rows."""
    return ", ".join(f"{c['check']} {c['value']:.3g}" for c in checks if c["check"].startswith("series_") and not c["passed"]
                     and c["check"] != "series_weak_vs_born_max_rel_err" and "not judged" not in str(c.get("detail", "")))


def gate_member(name, changes, member):
    """One simulation of a gate, in its own process."""
    with swapped(gate_settings(name, changes)):
        return gates.run_surface() if member == "half_space" else gates.run_sphere(member)


def run_gate(name, changes, note, approved=False):
    held(name, approved)
    kind, case = GATES[name]
    attempt = next_attempt(name, changes, gate_table())
    group = f"generous_{name}" + ("_smoke" if SMOKE else "")

    def member(m):
        return gates.run_or_load(group, attempt, m, gate_member, (name, changes, m), False)
    with swapped(gate_settings(name, changes)):
        s = dict(gates.sphere_setup() if kind == "sphere" else gates.surface_setup(), f0=gates.F0)
        if not fits(s["g"]):
            raise SystemExit(f"{name}: predicted {predicted_gb(s['g']):.1f} GB, over {SAFETY:g} of the budget of {budget_gb():.1f} GB")
        if kind == "sphere":
            checks = gates.sphere_reference_checks(s) + series_rows()  # The series passes its own checks first, for this ground and stone
            if bad := series_failures(checks):
                gates.merge_csv(out_dir() / "a8_generous_gate_checks.csv", [dict(c, gate=name, attempt=0) for c in checks], gates.CHECK_COLS, [name], 0)
                msg = f"{name}: the exact series fails its own checks for this ground and stone ({bad}); the gate cannot be judged against it"
                if not SMOKE:
                    raise SystemExit(msg)
                print(f"smoke only; at full size this stops the gate: {msg}")
            bg, tr = member("bg"), member("stone")
            rows, more = gates.measure_sphere(bg, tr, "stone", s)
        else:
            tr, checks = member("half_space"), []
            rows, more = gates.measure_surface(tr, s)
        checks = gates.run_checks(name, tr, s["g"]) + checks + more
    for c in checks:
        if c["check"].endswith("_peak_rss_gb"):  # run_checks marks it against production's 8 GB
            c["passed"] = bool(c["value"] <= budget_gb())
    changed = "; ".join(f"{k}={v!r}" for k, v in changes.items())
    note = "; ".join(q for q in (note, f"wavelet peak {s['f0']:g} Hz", "mirror plane y = 0" if s["g"]["sym"] else "") if q)
    rows = [dict(r, gate=name, attempt=attempt, h_m=s["g"]["h"], dt_s=s["dt"], changed=changed, note=note) for r in rows]
    gates.merge_csv(out_dir() / "a8_generous_gates.csv", rows, GATE_COLS, [name], attempt)
    gates.merge_csv(out_dir() / "a8_generous_gate_checks.csv", [dict(c, gate=name, attempt=attempt) for c in checks], gates.CHECK_COLS, [name], attempt)
    failed = sum(not r["passed"] for r in rows)
    print(f"{name} attempt {attempt}: {'passed' if not failed else 'FAILED'} ({len(rows)} rows, {failed} outside tolerance)"
          + ("; smoke: coarse grid, the verdict means nothing" if SMOKE else ""))


# ---------------------------------------------------------------- runs
def run_member(case, with_body, changes):
    """One simulation of a case, in its own process, on the numerical settings its gate passed on."""
    with swapped(numerics(changes)):
        return prod.run_case(case, with_body, spec(case, changes))


def run(case, approved=False):
    held(case, approved)
    rules()
    changes, _ = settings(case)
    with swapped(numerics(changes)):
        g = prod.setup(case, spec(case, changes))["g"]
    if not fits(g):  # simplification: no range cut is built; all four fit. Add `rmax` to the spec when one does not
        raise SystemExit(f"{case}: predicted {predicted_gb(g):.1f} GB, over {SAFETY:g} of the budget of {budget_gb():.1f} GB. Cut the range, not the resolution")
    for with_body in (False, True):
        path = archive(case, with_body)
        if path.exists():
            continue
        out = attenuated.in_fresh_process(run_member, case, with_body, changes)
        np.savez_compressed(path, **out)
        print(f"{path.stem}: {out['npts'] / 1e6:.1f}M points, {out['nt']} steps, {out['seconds']:.0f} s, "
              f"{out['mpts_per_s']:.0f} Mpts/s, peak rss {out['rss_gb']:.2f} GB", flush=True)


# ---------------------------------------------------------------- measurement
def extras(case, s, runs):
    """Per receiver and band: rows of the window from the ground's Rayleigh arrival; the slow window's peak,
    its time after the incident P, and the part of the slow window that held it with its Rayleigh evidence."""
    twin, body = runs
    t, n = np.arange(s["nt"]) * s["dt"], len(s["ranges"])
    vp, vs = s["host"][:2]
    delay = edge_delay(s)
    wins, peaks = [], []
    for band in s["bands"]:
        T = 1 / np.mean(prod.BANDS[band])
        inc = prod.band_envelope(twin["vz"].astype(float), band, s["dt"])
        env = {c: prod.band_envelope(body[c].astype(float) - twin[c].astype(float), band, s["dt"]) for c in ("vx", "vy", "vz")}
        for a, (az, (ux, _)) in enumerate(prod.AZIMUTHS.items()):
            for i, r in enumerate(s["ranges"]):
                j = a * n + i
                ver, rad = env["vz"][:, j], (ux * env["vx"][:, j]) if ux else env["vy"][:, j]
                t_inc = s["t_surf"] + s["p"] * r * ux
                inc_peak = prod.peak(inc[:, j], t, t_inc - 1 / prod.F0 - T, t_inc + 1 / prod.F0 + T)  # as a8_production
                lo, hi = s["t0"] + r / s["vs_fast"] - T / 2, s["t0"] + r / (0.85 * s["vs_slow"]) + 1.5 * T  # its slow window
                starts = {"before_ground_p": lo, "host_p": s["t0"] + r / vp - T / 2, "ground_s": s["t0"] + r / vs - T / 2,
                          "rayleigh": s["t0"] + r / s["c_r"] - T / 2}  # each opens half a period early, as production's windows do
                hv, lag = prod.rayleigh_evidence(rad, ver, t, starts["rayleigh"], hi)
                for comp, tr in (("vertical", ver), ("radial", rad)):
                    wins.append(dict(case=case, azimuth=az, range_m=float(r), band=band, window="rayleigh", component=comp,
                                     numerical=prod.peak(tr, t, starts["rayleigh"], hi) / inc_peak, h_over_v=hv, phase_lag_deg=lag,
                                     t_lo_s=starts["rayleigh"], t_hi_s=hi))
                sel = (t >= lo) & (t <= hi)
                t_pk = t[sel][np.argmax(np.abs(ver[sel]))]
                w_lo, name = max((v, k) for k, v in starts.items() if v <= t_pk)
                w_hi = min([v for v in starts.values() if v > w_lo] + [hi])
                hv, lag = prod.rayleigh_evidence(rad, ver, t, w_lo, w_hi)
                peaks.append(dict(case=case, azimuth=az, range_m=float(r), band=band, slow_share=prod.peak(ver, t, lo, hi) / inc_peak,
                                  lag_after_p_s=round(float(t_pk - t_inc), 3), clutter_lag_s=smoothing.CLUTTER_LAG, peak_window=name,
                                  peak_h_over_v=hv, peak_phase_lag_deg=lag, edge_flag=bool(delay < edge_needed(s, band, r))))
    return wins, pd.DataFrame(peaks)


def edge_level(case, s, twin):
    """The edge wave where the body stands, measured in the run without the body, at the nearest receivers.

    level: the largest band envelope of the vertical trace after the incident window (the one the incident peak is
    taken in) and up to the end of that receiver's slow window, over the incident peak. At the nearest receivers
    that interval is short or empty. So the same is given over two longer intervals: up to the last moment an
    edge wave at the body can still send a Rayleigh wave into any receiver's slow window (..._rule_horizon, the
    time the edge-wave rule is about), and up to the end of the slow window of the farthest receiver
    (..._to_last_window: anything that comes later in the record). Every interval starts where the band filter's
    tail of the incident pulse is still large. So each level has beside it the value with the incident pulse cut
    out of the record before filtering (edge_only...): what arrives after the pulse.
    """
    t, n, r = np.arange(s["nt"]) * s["dt"], len(s["ranges"]), float(s["ranges"][0])
    vz = twin["vz"][:, [a * n for a in range(len(prod.AZIMUTHS))]].astype(float)
    rows = []
    for band in s["bands"]:
        T = 1 / np.mean(prod.BANDS[band])
        for k, (az, (ux, _)) in enumerate(prod.AZIMUTHS.items()):
            t_inc = s["t_surf"] + s["p"] * r * ux
            lo = t_inc + 1 / prod.F0 + T
            pulse = np.where(np.abs(t - t_inc) <= 1.5 / prod.F0, vz[:, k], 0.0)
            env, rest = (np.abs(prod.band_envelope(x, band, s["dt"])) for x in (vz[:, k], vz[:, k] - pulse))
            inc_peak = prod.peak(env, t, t_inc - 1 / prod.F0 - T, lo)
            row = dict(case=case, band=band, azimuth=az, range_m=r, edge_delay_s=round(edge_delay(s), 3))
            last = float(s["ranges"][-1])
            for name, hi in (("", s["t0"] + r / (0.85 * s["vs_slow"]) + 1.5 * T),
                             ("_rule_horizon", s["t0"] + edge_needed(s, band, last) - 1 / prod.F0),
                             ("_to_last_window", s["t0"] + last / (0.85 * s["vs_slow"]) + 1.5 * T)):
                sel = (t > lo) & (t <= hi)
                row.update({f"level{name}": env[sel].max() / inc_peak if sel.any() else np.nan,
                            f"edge_only{name}": rest[sel].max() / inc_peak if sel.any() else np.nan,
                            f"interval{name}_s": round(max(hi - lo, 0.0), 3)})
            row["edge_only_largest_at_s_after_incident"] = round(float(t[sel][np.argmax(rest[sel])] - t_inc), 3)
            rows.append(row)
    return rows


def measure(case):
    """One case from its two saved runs: (rows in a8_production's form, shares, attenuation rows, checks, edge-level rows)."""
    changes, state = settings(case)
    with swapped(numerics(changes)):
        sp = spec(case, changes)
        s = prod.setup(case, sp)
        runs = tuple(attenuated.load(archive(case, b)) for b in (False, True))
        rows, checks = prod.measure(case, sp, runs)
        shares = pd.DataFrame(smoothing.measure(case, sp, runs))
    wins, peaks = extras(case, s, runs)
    key = ["case", "azimuth", "range_m", "band"]
    slow = pd.DataFrame(rows).query("window == 'slow' and component == 'vertical'")[key + ["numerical"]]
    shares = shares.merge(slow, on=key, validate="one_to_one").merge(peaks, on=key, validate="one_to_one")
    repro = float(max((shares.share / shares.numerical - 1).abs().max(), (shares.slow_share / shares.numerical - 1).abs().max()))
    assert repro <= smoothing.REPRO_TOL, f"{case}: the share recomputed here differs from a8_production.measure ({repro:.1e}); nothing is reported"
    shares = shares.drop(columns=["numerical", "slow_share"])
    note = "by formula; rescues no statement" + ("; untested in soft ground" if s["host"] == SOFT else "")
    att = [dict(case=case, band=r.band, range_m=r.range_m, q=q, rayleigh_speed_ms=round(s["c_r"], 1), share_elastic=round(r.share, 5),
                factor=round(f, 5), share_attenuated=round(r.share * f, 5), note=note)
           for r in shares[shares.azimuth == "forward"].itertuples() for q in attenuation.Q_VALUES
           for f in [attenuation.factor(attenuation.CENTER_HZ[r.band], r.range_m, q, s["c_r"])]]
    for c in checks:  # two details of a8_production.measure speak of production's limits
        if c["check"].endswith("_peak_rss_gb"):
            c["detail"] = f"budget {budget_gb():.1f} GB"
        if c["check"].endswith("_bands_measured"):
            c["detail"] = "" if s["bands"] == ALL else f"bands up to {F_SOFT:g} Hz"
    nominal = np.vstack([np.column_stack([r * ux, r * uy, 0 * r]) for ux, uy in prod.AZIMUTHS.values() for r in [s["ranges"]]])
    checks += [dict(check=f"{case}_gates_and_settings", value="; ".join(f"{k}={v!r}" for k, v in changes.items()) or "fixed settings", detail=state),
               dict(check=f"{case}_share_recomputed_max_relative_difference", value=repro, detail="a8_smoothing's and this module's unsmoothed share against a8_production.measure"),
               dict(check=f"{case}_receiver_off_nominal_max_m", value=float(np.abs(s["nodes"] - nominal).max()),
                    detail="receivers sit on nodes; ranges in the tables are nominal"),
               dict(check=f"{case}_half_width_m", value=s["half"], detail=f"edge wave {edge_delay(s):.2f} s after the plane wave at the body's top; {edge_text(s)}"),
               dict(check=f"{case}_rows_with_edge_flag", value=int(shares.edge_flag.sum()), detail=f"of {len(shares)}; they enter the statements for {' '.join(EDGE_CASES)} only")]
    same = [k for k, (_, frozen, _) in prod.CASES.items() if not frozen and thawed(k) == s["host"] and prod.archive(k, False).exists()]
    if same and (old := prod.load(same[0], False))["dt"] == s["dt"] and old["nt"] == s["nt"] and old["vz"].shape == runs[0]["vz"].shape:
        checks.append(dict(check=f"{case}_twin_identical_to_production_twin", value=bool(all(np.array_equal(old[c], runs[0][c]) for c in ("vx", "vy", "vz"))),
                           detail=f"same grid and step as {same[0]}: its own twin must repeat the archived one bit for bit"))
    return rows + wins, shares, att, checks, edge_level(case, s, runs[0])


def where(bad, m, low=False):
    """Name the rows that break a statement: case, band, azimuth, ranges, the extreme value and the window that held its peak."""
    out = []
    for (case, band, az), g in bad.groupby(["case", "band", "azimuth"], sort=False):
        w = g.loc[g[m].idxmin() if low else g[m].idxmax()]
        text = f"{case} {band} Hz {az}, {g.range_m.min():g} to {g.range_m.max():g} m, {'least' if low else 'largest'} {w[m]:.3f} at {w.range_m:g} m"
        if isinstance(w.get("peak_window"), str):
            text += f" (peak in the {w.peak_window} part of the slow window, H/V {w.peak_h_over_v:.2f}, phase lag {w.peak_phase_lag_deg:.0f} deg)"
        out.append(text)
    return "; ".join(out)


def stiffness(host, stone):
    """Shear modulus of the body over the ground's."""
    return stone[2] * stone[1] ** 2 / (host[2] * host[1] ** 2)


def statements(new):
    """The fixed statements at risk (the census ones apart), for the unsmoothed and the smoothed share, over all cases.

`broken` when a row without the edge-wave flag breaks the statement; `untested` when only flagged rows
    do, since an edge wave can only raise a peak; `holds` otherwise, flagged rows below the level included.
    """
    clut = smoothing.clutter()
    every = pd.concat([pd.read_csv(HERE / "a8_smoothing.csv").assign(new=False, edge_flag=False), new.assign(new=True)], ignore_index=True)
    every["edge_flag"] = every.edge_flag.astype(bool) & every.case.isin(EDGE_CASES)
    d = every[every.azimuth == "forward"]  # the manuscript's numbers are forward ones; "every simulated case" is tested on all three azimuths
    med, p05, far, at = d.band.map(clut["median"]), d.band.map(clut["p05"]), d.range_m >= smoothing.FAR_M, d.range_m == smoothing.FAR_M
    here, gt = sorted(d.case[d.new].unique()), gate_table()

    def why(c):
        twice = any(gate_state(g, gt)[:2] == ("failed", 2) for g in CASES[c].get("gates", ()))
        return "gate failed twice: untested for this case" if twice else "not run yet"
    absent = "; ".join(f"{c}: {why(c)}" for c in CASES if c not in here)
    rows = []

    def add(name, sentence, rule, m, domain, high, was, now, low=None, d=d):
        """high: rows that break the statement from above (a flag makes them untested); low: rows that break it from below."""
        hard, soft = d[high & ~d.edge_flag], d[high & d.edge_flag]
        if low is not None and not len(hard):
            hard = d[low]
        outcome = "broken" if len(hard) else "untested" if len(soft) else "holds"
        text = [where(hard, m, low is not None and not high[hard.index].any())] if len(hard) else []
        text += ["untested, the rows carry the edge-wave flag: " + where(soft, m)] if len(soft) else []
        rows.append(dict(statement=name, measure=m, outcome=outcome, where="; ".join(text),
                         production=was, all_cases=now, new_rows=int((domain & d.new).sum()), new_rows_flagged=int((domain & d.new & d.edge_flag).sum()),
                         new_cases=" ".join(here), absent=absent, sentence=sentence, rule=rule))

    def top_two(ratio):
        return ", ".join(f"{c} {b}" for c, b in ratio.nlargest(2).index)
    for m in ("share", "share_smoothed"):
        v, o = d[m], d[~d.new]
        rel, far3 = every[m] / every.band.map(clut["median"]), every.range_m >= smoothing.FAR_M
        add("below_median_beyond_1km", "beyond 1 km the scattered wave stays below the median clutter in every simulated case",
            "share below its band's median clutter at every range of 1 km or more, on all three azimuths", m, far3, far3 & (rel >= 1),
            round(float(rel[far3 & ~every.new].max()), 3), round(float(rel[far3].max()), 3), d=every)
        pct = np.round(100 * v)  # the abstract gives whole percent
        lo0, hi0 = pct[at & ~d.new].min(), pct[at & ~d.new].max()
        add("abstract_range_at_1km", "sends a surface wave of ... to ... percent of the incident P to 1 km",
            "the share at 1 km, in whole percent, stays inside the range of the production cases", m, at, at & (pct > hi0),
            f"{lo0:g} to {hi0:g}", f"{pct[at].min():g} to {pct[at].max():g}", low=at & (pct < lo0))
        hi = pd.Series(False, index=d.index)
        for case in here:  # a stone body against the production body in the same ground, row by row
            same = [k for k, (_, frozen, _) in prod.CASES.items() if not frozen and thawed(k) == CASES[case]["host"]]
            if "stone" in CASES[case] and same:
                ref = o[o.case == same[0]].set_index(["band", "range_m"])[m]
                rows_c = d.case == case
                hi |= rows_c & (v > pd.Series([ref.get(k, np.inf) for k in zip(d.band, d.range_m)], index=d.index))
        add("contrast_is_the_extreme", "these are upper values in contrast ... the contrast is the extreme",
            "a body of stone sends no larger share than the production body in the same ground, band and range", m,
            d.new & d.case.isin([c for c in here if "stone" in CASES[c]]), hi, "", "")
        of = d[far & ~d.new]
        top = d.range_m.map(of.groupby("range_m")[m].max())
        add("few_percent_to_few_km", "the body sends a few percent at most to a few kilometers in these bands",
            "at each range of 1 km or more, no new case exceeds the largest production share (the sentence gives no number)", m, far,
            d.new & far & (v > top), round(float(of[m].max()), 3), round(float(v[far].max()), 3))
        ratio = (v / med)[far].groupby([d.case[far], d.band[far]]).max()
        old_ratio = ratio[[c in prod.CASES for c, _ in ratio.index]]
        add("the_two_closest", "in the two closest cases it is about half the clutter",
            "no new case and band comes nearer the median clutter, at 1 km or more, than the second of the production cases", m, far,
            d.new & far & (v / med > old_ratio.nlargest(2).iloc[-1]), top_two(old_ratio), top_two(ratio))
        for name, sentence, level in (("cleanest_twentieth_range", "against the cleanest twentieth of sites, those two cases reach the clutter out to ... km", p05),
                                      ("few_hundred_meters", "less than ordinary hills and valleys do except within a few hundred meters of it", med)):
            reach = float(o.range_m[(v >= level)[o.index]].max()) if (v >= level)[o.index].any() else 0.0
            reach_all = float(d.range_m[v >= level].max()) if (v >= level).any() else 0.0
            add(name, sentence, "no new case reaches the level (cleanest twentieth; median) farther out than the farthest production case", m,
                d.new, d.new & (v >= level) & (d.range_m > reach), f"{reach:g} m", f"{reach_all:g} m")
    mu = {c: stiffness(thawed(c), tuple(2.0 * q for q in thawed(c))) for c in prod.CASES}
    over = {c: stiffness(CASES[c]["host"], CASES[c]["stone"]) for c in here if "stone" in CASES[c]}
    over = {c: v for c, v in over.items() if v > max(mu.values()) * (1 + 1e-9)}
    rows.append(dict(statement="stiffest_contrast_considered", measure="inputs", outcome="broken" if over else "holds",
                     where="; ".join(f"{c}: shear modulus {v:.1f} times the ground's" for c, v in over.items()), production=round(max(mu.values()), 1),
                     all_cases=round(max([*mu.values(), *over.values()]), 1), new_rows="", new_rows_flagged="", new_cases=" ".join(here), absent=absent,
                     sentence="at the stiffest contrast considered", rule="no case that was run has a body stiffer against its ground than the production body"))
    return pd.DataFrame(rows)


def convergence(rows):
    """R1's two lower bands against the production run in the same ground, row by row."""
    pc = next(k for k, (_, frozen, _) in prod.CASES.items() if not frozen and thawed(k) == CASES["R1"]["host"])
    key = ["azimuth", "range_m", "band", "window", "component"]
    a = pd.read_csv(HERE / "a8_production.csv").query("case == @pc")[key + ["numerical"]].rename(columns={"numerical": pc})
    m = a.merge(rows[rows.case == "R1"][key + ["numerical"]].rename(columns={"numerical": "R1"}), on=key, validate="one_to_one")
    return m.assign(ratio=m.R1 / m[pc])


def measure_all():
    rules()
    done = [c for c in CASES if archive(c, False).exists() and archive(c, True).exists()]
    edge = [row for c in prod.CASES if prod.archive(c, False).exists() for row in edge_level(c, prod.setup(c), prod.load(c, False))]
    rows, shares, att, checks = [], [], [], []
    for case in done:
        r, s, a, c, e = measure(case)
        rows, shares, att, checks, edge = rows + r, shares + [s], att + a, checks + c, edge + e
    out = out_dir()
    edge = pd.DataFrame(edge)
    edge.round(4).to_csv(out / "a8_generous_edge.csv", index=False)
    print(edge.round(4).to_string(index=False))
    if not done:
        print(f"no new case has both of its runs saved: only a8_generous_edge.csv is written, to {out}")
        return
    rows, shares = pd.DataFrame(rows).reindex(columns=prod.COLS), pd.concat(shares, ignore_index=True)
    rows.to_csv(out / "a8_generous.csv", index=False)
    shares.round(6).to_csv(out / "a8_generous_shares.csv", index=False)
    pd.DataFrame(att).to_csv(out / "a8_generous_attenuation.csv", index=False)
    st = statements(shares)
    st.to_csv(out / "a8_generous_statements.csv", index=False)
    if "R1" in done:
        conv = convergence(rows)
        conv.to_csv(out / "a8_generous_convergence.csv", index=False)
        sv = conv[(conv.window == "slow") & (conv.component == "vertical")]
        checks.append(dict(check="R1_grid_convergence_slow_vertical_ratio_min_median_max", detail=f"{len(sv)} rows; all windows and components in a8_generous_convergence.csv",
                           value=" ".join(f"{q:.3f}" for q in (sv.ratio.min(), sv.ratio.median(), sv.ratio.max()))))
    pd.DataFrame(checks).reindex(columns=["check", "value", "detail"]).to_csv(out / "a8_generous_checks.csv", index=False)
    print(st[["statement", "measure", "outcome", "production", "all_cases", "new_rows", "where"]].to_string(index=False))
    print(f"wrote the tables of {done} to {out}")


# ---------------------------------------------------------------- self-check
def _selfcheck() -> None:
    """No simulation (module docstring)."""
    before = (gates.HOST, gates.H, gates.FREQS, fd.DT_SAFETY, gates.F0)
    with swapped(gate_settings("sphere_soft", dict(dt_safety=0.4))):
        assert gates.HOST == SOFT and fd.DT_SAFETY == 0.4 and max(gates.FREQS) <= F_SOFT and max(gates.RG_FREQS) <= F_SOFT
        assert gates.F0 == before[4] * F_SOFT / before[2][-1], "The pulse scales with the highest frequency R4 measures"
        judged = [c for c in series_rows() if "not judged" not in c["detail"]]
        assert judged and all(c["passed"] for c in judged) and not series_failures(series.checks(gates.A) + series_rows()), "The series, at the gate's frequencies"
    assert (gates.HOST, gates.H, gates.FREQS, fd.DT_SAFETY, gates.F0) == before
    if not SMOKE:  # the thawed stone gate is gate two with another stone, and its time step is R2's
        new = {(m.__name__, k): v for m, k, v in gate_settings("sphere_thawed", {})}
        diff = {k for (m, k), v in new.items() if getattr(gates if "gates" in m else series, k) != v}
        assert diff == {"STONE", "DT_VP"}, diff
        with swapped(gate_settings("sphere_thawed", {})):
            assert gates.sphere_setup()["dt"] == prod.setup("R2", spec("R2", {}))["dt"]

    def table(*gs):  # (gate, attempt, passed, changed), two rows each
        return pd.DataFrame([(*g, ) for g in gs for _ in range(2)], columns=["gate", "attempt", "passed", "changed"])

    def stops(fn, *args):
        try:
            fn(*args)
        except SystemExit:
            return True
        return False
    none, fail1, pass2 = table(), table(("sphere_thawed", 1, False, "")), table(("sphere_thawed", 1, False, ""), ("sphere_thawed", 2, True, "h=8.0"))
    assert stops(run_settings, "R2", none) and stops(run_settings, "R2", fail1) and run_settings("R2", pass2) == dict(h=8.0)
    assert run_settings("R1", none) == {} and spec("R2", dict(h=8.0))["h"] == 8.0
    assert stops(run_settings, "R4", table(("surface_soft", 1, True, ""))), "both gates of R4 must pass"
    # Two gates and one setting: the gate that passed on the fixed settings is run again on its sister's, then R4 takes them
    one = table(("surface_soft", 1, False, ""), ("surface_soft", 2, True, "h=8.0"), ("sphere_soft", 1, True, ""))
    both = table(("surface_soft", 1, False, ""), ("surface_soft", 2, True, "h=8.0"), ("sphere_soft", 1, True, ""), ("sphere_soft", 2, True, "h=8.0"))
    try:
        run_settings("R4", one)
        raise AssertionError("R4 must wait for sphere_soft on the changed settings")
    except SystemExit as e:
        assert "--gate sphere_soft --set h=8 --full-size" in str(e), e
    assert next_attempt("sphere_soft", dict(h=8.0), one) == 2 and stops(next_attempt, "sphere_soft", {}, one) and stops(next_attempt, "surface_soft", dict(h=8.0), one)
    assert run_settings("R4", both) == dict(h=8.0) and stops(next_attempt, "sphere_soft", dict(h=8.0), both)
    first = table(("surface_soft", 1, False, ""), ("surface_soft", 2, True, "h=8.0"))
    assert next_attempt("sphere_soft", dict(h=8.0), first) == 1 and stops(next_attempt, "sphere_soft", {}, first)
    assert next_attempt("sphere_thawed", {}, none) == 1 and next_attempt("sphere_thawed", dict(h=8.0), fail1) == 2
    assert stops(next_attempt, "sphere_thawed", dict(h=8.0), none) and stops(next_attempt, "sphere_thawed", {}, fail1)
    assert stops(next_attempt, "sphere_thawed", dict(h=8.0), pass2)
    assert stops(next_attempt, "sphere_thawed", dict(h=6.0), table(("sphere_thawed", 1, False, ""), ("sphere_thawed", 2, False, "h=8.0")))
    assert stops(held, "R4", False) == (not SMOKE) and not stops(held, "R4", True) and not stops(held, "R2", False)
    s = prod.setup("R4", {k: v for k, v in CASES["R4"].items() if k != "gates"})
    wide = dict(s, half=2 * s["half"])
    assert edge_delay(wide) > edge_delay(s) > 0 and edge_needed(s, "0.5-1", 3000.0) > edge_needed(s, "1-2", 3000.0) > edge_needed(s, "1-2", 100.0)
    assert sum(edge_delay(wide) < edge_needed(s, "1-2", r) for r in s["ranges"]) <= sum(edge_delay(s) < edge_needed(s, "1-2", r) for r in s["ranges"])
    dt = prod.setup("R2", spec("R2", {}))["dt"]
    with swapped(numerics(dict(dt_safety=fd.DT_SAFETY / 2))):  # a run takes its gate's time-step setting
        assert np.isclose(2 * prod.setup("R2", spec("R2", {}))["dt"], dt)
    # statements on made-up rows of R4 at 2 km, 1-2 Hz: below the median, above it, and above it with the edge flag
    med = float(smoothing.clutter().loc["1-2", "median"])
    for share, flag, want in ((0.5 * med, True, "holds"), (2 * med, False, "broken"), (2 * med, True, "untested")):
        row = pd.DataFrame([dict(case="R4", azimuth="forward", range_m=2000.0, band="1-2", share=share, share_smoothed=share, edge_flag=flag,
                                 peak_window="host_p", peak_h_over_v=1.5, peak_phase_lag_deg=10.0)])
        st = statements(row).set_index(["statement", "measure"])
        got = st.loc[("below_median_beyond_1km", "share")]
        assert got.outcome == want and (want == "holds" or "R4 1-2 Hz" in got["where"] and "host_p" in got["where"]), (want, dict(got))
        assert st.loc[("stiffest_contrast_considered", "inputs"), "outcome"] == "broken"
    print("a8_generous selfcheck ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="A8 generous end. Default: the size table.")
    ap.add_argument("--size", action="store_true")
    ap.add_argument("--gate", choices=list(GATES))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help=f"the repeat's changes: {', '.join(RETRY)}")
    ap.add_argument("--note", default="", help="why the repeat changes what it changes")
    ap.add_argument("--run", choices=list(CASES))
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--full-size", action="store_true", help="For R4 and the two soft-ground gates")
    ap.add_argument("--selfcheck", action="store_true")
    arg = ap.parse_args()
    if arg.selfcheck:
        _selfcheck()
    elif arg.gate:
        run_gate(arg.gate, {k: float(v) for k, v in (kv.split("=") for kv in arg.set) if k in RETRY or ap.error(f"--set {k}")},
                 arg.note, arg.full_size)
    elif arg.run:
        run(arg.run, arg.full_size)
    elif arg.measure:
        measure_all()
    else:
        size()
