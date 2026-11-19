"""A8 attenuation check: a gate for the attenuating solver, then one pair of runs.

Inputs: a8_elastic3d.py (option q: one standard linear solid, equal Q for P and S), a8_gates.py (the
half-space setup of gate three and its archived elastic run), a8_production.py (the case thawed_1200,
its measurement and its elastic results a8_production.csv).

Rules fixed before any number, not changed here:
- Q = 50 for P and S alike (the stage's upper sediment value), referenced to 3 Hz, the center of the
  stage's highest band. With one standard linear solid, Q is 50 at 3 Hz and rises away from it as
  Q (f / 3 + 3 / f) / 2: about 54 at 2 Hz, 52 at 4 Hz, 62 at 1.5 Hz and 106 at 0.75 Hz (checks). The
  phase speeds at 3 Hz are the elastic model's.
- Gate, before the pair: a buried explosive point source in a homogeneous half-space with the faster
  thawed host's properties. Setup: gate three's, unchanged (a8_gates.surface_setup: source 50 m deep,
  vertical velocity at 1800-3600 m). The elastic run of the identical setup is gate three's archived
  second attempt (`--prove` reruns it with today's solver and compares bit for bit). Measured: the
  peak envelope of the 2-4 Hz vertical motion in the Rayleigh window, attenuated run over elastic
  run, at 1800 m and at 3600 m (the nearest and farthest receivers; geometric spreading and the
  source cancel). Pass: the fall of that ratio between the two ranges is exp(-pi f dr / (Q U)), f =
  3 Hz, U the host's Rayleigh speed, within 20%. Every attempt is logged; after a failure genuine
  bugs are fixed and the gate is rerun once; after a second failure the pair is not run.
- The pair (only if the gate passes): thawed_1200 exactly as a8_production (its run_case, with the
  solver's q option switched on and nothing else changed), body and twin, measured by
  a8_production.measure for all three bands.
- Agreement rule (reported either way): in the highest band, beyond 1 km, the attenuated share
  against the elastic share times exp(-pi f r / (Q U)), within 20%. Evaluated on the slow window's
  vertical and radial rows at every azimuth.

Outputs: a8_attenuated_gates.csv (every attempt), a8_attenuated.csv (a8_production.csv's columns plus
`elastic`, the same row of a8_production.csv, and `ratio_to_elastic`), a8_attenuated_checks.csv.
Traces: $AKP_DATA/a8/attenuated.
Run: OMP_NUM_THREADS=6 PYTHONPATH=. python scripts/mem_guard.py --gb 8 --every 0.5 -- \
       python -m akp.stage_a.a8_attenuated [--attempt N] [--gate]   (runs what is not archived, gate first;
                                                                    --gate stops after the gate)
     python -m akp.stage_a.a8_attenuated --prove            (elastic path against gate three's archive)
     python -m akp.stage_a.a8_attenuated --measure          (saved traces only)
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import numpy as np

from akp.stage_a import a8_elastic3d as fd  # first: sets the Devito environment
from akp.stage_a import a8_gates as gates
from akp.stage_a import a8_production as prod

HERE = Path(__file__).parent
Q, F_REF, TOL = 50.0, 3.0, 0.20
QD = dict(q=Q, f0=F_REF)
CASE, BAND = "thawed_1200", "2-4"
R1, R2 = gates.RANGES[0], gates.RANGES[-1]
GATE_COLS = ["gate", "attempt", "quantity", "range_m", "attenuated", "elastic", "ratio", "predicted",
             "ratio_over_predicted", "tolerance", "passed"]
ATTEMPT_NOTES: dict[int, str] = {}  # attempt: what failed and what was fixed


def folder():
    from akp.paths import data_dir
    d = data_dir("a8") / "attenuated"
    d.mkdir(parents=True, exist_ok=True)
    return d


def in_fresh_process(fn, *args):
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(1) as pool:
        return pool.apply(fn, args)


def load(path):
    z = np.load(path)
    return {k: (z[k] if z[k].ndim else z[k].item()) for k in z.files}


# ---------------------------------------------------------------- the gate
def run_half_space(attenuated=True):
    """Gate three's half-space run (a8_gates.run_surface), with the q option on or off."""
    orig = fd.simulate
    fd.simulate = partial(orig, q=QD) if attenuated else orig
    try:
        return gates.run_surface()
    finally:
        fd.simulate = orig


def prove_elastic():
    """Today's solver without q against gate three's archived run: are the traces identical, bit for bit?"""
    out = in_fresh_process(run_half_space, False)
    ref = np.load(gates.archive("surface", 2, "half_space"))
    same = bool(np.array_equal(out["vx"], ref["vx"]) and np.array_equal(out["vz"], ref["vz"]))
    np.savez(folder() / "elastic_identity.npz", identical=same, samples=out["vz"].size + out["vx"].size)
    print("elastic path reproduces gate three's archive bit for bit:", same)
    return same


def measure_gate(attempt):
    s = gates.surface_setup()
    att, el = load(folder() / f"gate_att{attempt}.npz"), load(gates.archive("surface", 2, "half_space"))
    assert att["dt"] == el["dt"] == s["dt"] and att["nt"] == el["nt"]
    t = np.arange(s["nt"]) * s["dt"]
    half = 1 / gates.F0 + 0.35
    f = float(np.mean(prod.BANDS[BAND]))
    env = {k: np.abs(prod.band_envelope(run["vz"].astype(float), BAND, s["dt"])) for k, run in (("att", att), ("el", el))}
    rows, share = [], {}
    for j, r in enumerate(gates.RANGES):
        lo, hi = s["t0"] + r / s["c_r"] - half, s["t0"] + r / s["c_r"] + half
        pa, pe = prod.peak(env["att"][:, j], t, lo, hi), prod.peak(env["el"][:, j], t, lo, hi)
        share[r] = pa / pe
        pred = float(np.exp(-np.pi * f * r / (Q * s["c_r"])))
        rows.append(dict(quantity="vz_peak_envelope_2-4Hz", range_m=r, attenuated=pa, elastic=pe, ratio=pa / pe,
                         predicted=pred, ratio_over_predicted=pa / pe / pred, tolerance="", passed=""))
    fall, pred = share[R2] / share[R1], float(np.exp(-np.pi * f * (R2 - R1) / (Q * s["c_r"])))
    rows.append(dict(quantity="fall_of_ratio_between_ranges", range_m=f"{R1:g}-{R2:g}", attenuated="", elastic="",
                     ratio=fall, predicted=pred, ratio_over_predicted=fall / pred, tolerance=TOL,
                     passed=bool(abs(fall / pred - 1) <= TOL)))
    q_seen = -np.pi * f * (R2 - R1) / (s["c_r"] * np.log(fall))
    checks = [dict(check=f"gate_attempt{attempt}_fall_over_predicted", value=fall / pred, detail=f"Tolerance {TOL}; passed {rows[-1]['passed']}"),
              dict(check=f"gate_attempt{attempt}_apparent_q", value=float(q_seen), detail="the Q that the measured fall implies in the formula"),
              dict(check=f"gate_attempt{attempt}_run_seconds", value=round(float(att["seconds"]), 1),
                   detail=f"{att['npts'] / 1e6:.1f}M points, {int(att['nt'])} steps, {float(att['mpts_per_s']):.0f} Mpts/s; elastic run {float(el['seconds']):.0f} s"),
              dict(check=f"gate_attempt{attempt}_peak_rss_gb", value=round(float(att["rss_gb"]), 2),
                   detail=f"{att['rss_gb'] * 1e9 / att['npts']:.0f} bytes per point; elastic run {float(el['rss_gb']):.2f} GB")]
    return [dict(gate="attenuation", attempt=attempt, **r) for r in rows], checks, bool(rows[-1]["passed"])


# ---------------------------------------------------------------- the pair
def run_member(with_body):
    """a8_production.run_case for the case, unchanged, with the solver's q option switched on."""
    orig = fd.simulate
    fd.simulate = partial(orig, q=QD)
    try:
        return prod.run_case(CASE, with_body)
    finally:
        fd.simulate = orig


def member_path(case, with_body):
    return folder() / f"{case}_{'body' if with_body else 'twin'}.npz"


def measure_pair():
    """a8_production.measure on the attenuated traces, with the elastic value of each row beside it."""
    import pandas as pd
    orig = prod.archive
    prod.archive = member_path
    try:
        rows, run_checks = prod.measure(CASE)
    finally:
        prod.archive = orig
    key = ["case", "azimuth", "range_m", "band", "window", "component"]
    df = pd.DataFrame(rows).reindex(columns=prod.COLS)
    el = pd.read_csv(HERE / "a8_production.csv")[key + ["numerical"]].rename(columns={"numerical": "elastic"})
    df = df.merge(el, on=key, how="left", validate="one_to_one")
    assert df.elastic.notna().all()
    df["ratio_to_elastic"] = df.numerical / df.elastic
    u = prod.setup(CASE)["c_r"]
    slow = df[df.window == "slow"]
    checks = [c for c in run_checks if any(k in c["check"] for k in ("run_seconds", "peak_rss", "field_left", "grid_shape",
                                                                    "time_step", "incident_peak_min"))]
    for band in prod.CASES[CASE][2]:
        f = float(np.mean(prod.BANDS[band]))
        for r in (1000.0, 2000.0, 3000.0):
            formula = float(np.exp(-np.pi * f * r / (Q * u)))
            for az in prod.AZIMUTHS:
                row = slow[(slow.band == band) & (slow.range_m == r) & (slow.azimuth == az) & (slow.component == "vertical")].iloc[0]
                checks.append(dict(check=f"slow_vertical_ratio_over_formula_{band}_{az}_{r:g}m", value=row.ratio_to_elastic / formula,
                                   detail=f"attenuated share {row.numerical:.4f}, elastic {row.elastic:.4f}, ratio {row.ratio_to_elastic:.3f}, "
                                          f"formula exp(-pi f r / (Q U)) {formula:.3f} at f {f} Hz"))
    top = slow[(slow.band == BAND) & (slow.range_m > 1000.0)]
    dev = top.ratio_to_elastic / np.exp(-np.pi * 3.0 * top.range_m / (Q * u))
    checks += [dict(check="d81_agreement_max_abs_dev", value=float(np.abs(dev - 1).max()),
                    detail=f"highest band, beyond 1 km, slow window, vertical and radial, three azimuths ({len(top)} rows); "
                           f"rule: within {TOL}; met: {bool(np.abs(dev - 1).max() <= TOL)}"),
               dict(check="d81_agreement_median_simulated_over_table", value=float(dev.median()),
                    detail=f"range of simulated over table: {dev.min():.3f} to {dev.max():.3f}; vertical rows alone: "
                           f"{dev[top.component == 'vertical'].min():.3f} to {dev[top.component == 'vertical'].max():.3f}")]
    return df, checks


def common_checks():
    c = fd.sls_constants(Q, F_REF)
    qf, cf = fd.sls_q([0.75, 1.5, 2.0, 3.0, 4.0], Q, F_REF)
    rows = [dict(check="scheme", value="one standard linear solid, memory variables, equal Q for P and S",
                 detail=f"Q {Q} at {F_REF} Hz; tau_sigma {c['tau_s']:.5f} s, tau_epsilon {c['tau_e']:.5f} s; relaxed moduli scaled by "
                        f"{c['kappa']:.5f} so the phase speeds at {F_REF} Hz are the elastic model's")]
    rows += [dict(check=f"q_at_{f:g}hz", value=float(q), detail=f"phase speed over the stated speed {v:.4f}")
             for f, q, v in zip([0.75, 1.5, 2.0, 3.0, 4.0], qf, cf)]
    p = folder() / "elastic_identity.npz"
    if p.exists():
        z = np.load(p)
        rows.append(dict(check="elastic_path_bit_identical", value=bool(z["identical"]),
                         detail=f"solver without q against gate three's archived traces ({int(z['samples'])} samples)"))
    rows += [dict(check=f"gate_attempt{a}_diagnosis", value="", detail=note) for a, note in ATTEMPT_NOTES.items()]
    return rows


def main(attempt=None, measure_only=False, gate_only=False):
    import pandas as pd
    done = sorted(int(p.stem.split("att")[-1]) for p in folder().glob("gate_att*.npz") if not p.name.startswith("._"))
    attempt = attempt or (done[-1] if done else 1)
    gate_path = folder() / f"gate_att{attempt}.npz"
    if not gate_path.exists() and not measure_only:
        if attempt > 2:
            raise SystemExit("The gate is rerun once only")
        out = in_fresh_process(run_half_space, True)
        np.savez_compressed(gate_path, **out)
    rows, checks, passed = measure_gate(attempt)
    gates.merge_csv(HERE / "a8_attenuated_gates.csv", rows, GATE_COLS, ["attenuation"], attempt)
    for r in rows:
        print(r["quantity"], r["range_m"], f"ratio {r['ratio']:.4f} predicted {r['predicted']:.4f}", r["passed"])
    checks = common_checks() + checks
    if passed:
        for with_body in (False, True):
            path = member_path(CASE, with_body)
            if not path.exists() and not (measure_only or gate_only):
                out = in_fresh_process(run_member, with_body)
                np.savez_compressed(path, **out)
                print(f"{path.stem}: {out['npts'] / 1e6:.1f}M points, {out['nt']} steps, {out['seconds']:.0f} s, "
                      f"{out['mpts_per_s']:.0f} Mpts/s, peak rss {out['rss_gb']:.2f} GB", flush=True)
        if member_path(CASE, False).exists() and member_path(CASE, True).exists():
            df, pair_checks = measure_pair()
            df.to_csv(HERE / "a8_attenuated.csv", index=False)
            checks += pair_checks
    else:
        checks.append(dict(check="pair", value="not run", detail="the gate has not passed"))
    pd.DataFrame(checks).reindex(columns=["check", "value", "detail"]).to_csv(HERE / "a8_attenuated_checks.csv", index=False)


def _selfcheck() -> None:
    """No simulation: the solid's constants, the step weights and the gate's prediction."""
    c = fd.sls_constants(Q, F_REF, dt=1.5e-3)
    assert abs(c["tau_s"] * c["tau_e"] * (2 * np.pi * F_REF) ** 2 - 1) < 1e-12  # Q is least at the reference frequency
    q3, v3 = fd.sls_q(F_REF, Q, F_REF)
    assert abs(q3 - Q) < 1e-9 and abs(v3 - 1) < 1e-12
    # the stepped equations at steady state: a constant elastic stress rate E gives r -> -kappa (ratio - 1) E,
    # so the stress rate tends to the relaxed one, kappa E
    r = 0.0
    for _ in range(20000):
        r = c["c1"] * r + c["c2"] * 1.0
    assert abs(c["g_e"] + c["g_r"] * r - c["kappa"]) < 1e-9
    # a plane wave in this solid loses exp(-pi f x / (Q c)) over x, to first order in 1 / Q
    w = 2 * np.pi * F_REF
    m = c["kappa"] * (1 + 1j * w * c["tau_e"]) / (1 + 1j * w * c["tau_s"])
    alpha = w * abs(np.imag(1 / np.sqrt(m)))  # per unit of x / c
    assert abs(alpha / (np.pi * F_REF / Q) - 1) < 2e-3
    s = gates.surface_setup()
    assert abs(np.exp(-np.pi * 3.0 * (R2 - R1) / (Q * s["c_r"])) - 0.576) < 2e-3
    print("a8_attenuated selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    elif "--prove" in sys.argv:
        prove_elastic()
    else:
        arg = dict(zip(sys.argv[1:], sys.argv[2:]))
        main(attempt=int(arg["--attempt"]) if "--attempt" in arg else None, measure_only="--measure" in sys.argv,
             gate_only="--gate" in sys.argv)
