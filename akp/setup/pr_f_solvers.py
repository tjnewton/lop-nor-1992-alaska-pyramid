"""S5-f: osx-arm64 smoke tests for the forward-modeling candidates (CPU only; the GPU stays free).

- SPECFEM2D (spectral-element elastic waves): repo default example, built serial in env `akp-specfem`
  at $SPECFEM2D_HOME (default ~/src/specfem2d). Pass = both binaries exit 0 and seismograms are finite.
- Devito (finite-difference codegen): 2-D acoustic point-source run. Pass = finite, energy spread.
- SimPEG (potential fields / EM): signed gravity of a small buried dense cube vs the point-mass formula,
  and the sign flip for a low-density cube.
These are install smoke tests; the A3 solver validation (analytic scattering) is a separate gate.
Writes pr_f_solvers.csv. Run: python -m akp.setup.pr_f_solvers
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from akp.paths import data_dir, data_glob

OUT = Path(__file__).with_suffix(".csv")
SPECFEM = Path(os.environ.get("SPECFEM2D_HOME", Path.home() / "src" / "specfem2d"))


def specfem2d() -> dict:
    sha = subprocess.run(["git", "-C", str(SPECFEM), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    run = data_dir("probes/specfem2d_smoke")
    shutil.rmtree(run / "DATA", ignore_errors=True)
    shutil.rmtree(run / "OUTPUT_FILES", ignore_errors=True)
    shutil.copytree(SPECFEM / "DATA", run / "DATA")
    # the default Par_file points (via ./DATA/../EXAMPLES/) at an example's interfaces file; copy it in
    par = run / "DATA" / "Par_file"
    txt = par.read_text()
    rel = next(ln.split("=", 1)[1].split()[0] for ln in txt.splitlines() if ln.startswith("interfacesfile"))
    src = (SPECFEM / "DATA" / rel).resolve()
    shutil.copy(src, run / "DATA" / src.name)
    par.write_text(txt.replace(rel, src.name))
    (run / "OUTPUT_FILES").mkdir()
    t0 = time.perf_counter()
    codes = []
    for exe in ("xmeshfem2D", "xspecfem2D"):
        r = subprocess.run(["conda", "run", "-n", "akp-specfem", str(SPECFEM / "bin" / exe)],
                           cwd=run, capture_output=True, text=True)
        codes.append(r.returncode)
    wall = time.perf_counter() - t0
    traces = data_glob(run / "OUTPUT_FILES", "*.sem*")
    # SPECFEM2D exits 0 even on fatal errors: pass needs seismograms AND no Error_message file
    errors = data_glob(run / "OUTPUT_FILES", "Error_message*")
    finite = bool(traces) and not errors and all(np.isfinite(np.loadtxt(t)).all() for t in traces)
    peak = max((np.abs(np.loadtxt(t)[:, 1]).max() for t in traces), default=float("nan"))
    return dict(solver="specfem2d", version=sha[:12], ok=codes == [0, 0] and finite, wall_s=round(wall, 1),
                metric=f"{len(traces)} seismograms, peak |u|={peak:.3e}", note=f"exit codes {codes}; error files {len(errors)}; default DATA/Par_file")


def devito() -> dict:
    import devito as dv
    grid = dv.Grid(shape=(101, 101), extent=(1000.0, 1000.0))  # 10 m spacing
    u = dv.TimeFunction(name="u", grid=grid, time_order=2, space_order=4)
    v = 1500.0  # m/s
    op = dv.Operator([dv.Eq(u.forward, dv.solve(u.dt2 - v**2 * u.laplace, u.forward))])
    u.data[:, 50, 50] = 1.0
    t0 = time.perf_counter()
    op.apply(time_M=300, dt=0.002)  # CFL: v*dt/h = 0.3
    wall = time.perf_counter() - t0
    a = u.data[0]
    spread = float((np.abs(a) > 1e-3 * np.abs(a).max()).mean())
    ok = bool(np.isfinite(a).all() and spread > 0.05)
    return dict(solver="devito", version=dv.__version__, ok=ok, wall_s=round(wall, 2),
                metric=f"fraction of grid excited={spread:.3f}", note=f"language={dv.configuration['language']}")


def simpeg_gravity() -> dict:
    import discretize
    import simpeg
    from simpeg import maps
    from simpeg.potential_fields import gravity
    h = 10.0
    mesh = discretize.TensorMesh([[(h, 40)], [(h, 40)], [(h, 30)]], origin="CCN")  # 400 x 400 x 300 m, top at z=0
    cc = mesh.cell_centers
    block = (np.abs(cc[:, 0]) < h) & (np.abs(cc[:, 1]) < h) & (np.abs(cc[:, 2] + 100.0) < h)  # 20 m cube, center 100 m deep
    rho = np.where(block, 1.0, 0.0)  # g/cc contrast
    rx = gravity.receivers.Point(np.array([[0.0, 0.0, 1.0]]), components="gz")
    survey = gravity.survey.Survey(gravity.sources.SourceField(receiver_list=[rx]))
    sim = gravity.simulation.Simulation3DIntegral(mesh=mesh, survey=survey, rhoMap=maps.IdentityMap(nP=mesh.n_cells),
                                                  store_sensitivities="forward_only", engine="geoana")
    t0 = time.perf_counter()
    gz = float(sim.dpred(rho)[0])  # mGal
    gz_void = float(sim.dpred(-rho)[0])  # H2-like low-density body
    wall = time.perf_counter() - t0
    mass = 1000.0 * block.sum() * h**3  # kg (1 g/cc = 1000 kg/m3)
    r = 101.0  # receiver 1 m above surface, cube center 100 m deep
    # SimPEG gz is z-up: attraction toward a dense body below is NEGATIVE. Compare SIGNED values,
    # since the sign is what separates dense (H3) from low-density (H2) bodies.
    gz_point = -6.674e-11 * mass / r**2 * 1e5  # mGal
    rel = abs(gz - gz_point) / abs(gz_point)
    ok = bool(rel < 0.01 and gz < 0 < gz_void)
    return dict(solver="simpeg", version=simpeg.__version__, ok=ok, wall_s=round(wall, 3),
                metric=f"gz={gz:.6f} mGal vs point mass {gz_point:.6f} (rel err {rel:.2e}); void gz={gz_void:.6f}",
                note=f"discretize {discretize.__version__}; TensorMesh integral, engine geoana")


def main() -> None:
    rows = []
    for fn in (specfem2d, devito, simpeg_gravity):
        try:
            rows.append(fn())
        except Exception as exc:  # record the failure; a failed install is a finding
            rows.append(dict(solver=fn.__name__, version="", ok=False, wall_s="", metric="", note=f"{type(exc).__name__}: {exc}"))
        print(rows[-1])
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
