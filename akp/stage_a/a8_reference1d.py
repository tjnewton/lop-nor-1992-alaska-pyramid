"""A8 one-dimensional reference: a layered column, exact, for the checks of case R5.

Question: what do a plane P wave and a Rayleigh wave do in a stack of flat layers, so that the
three-dimensional run in a column (a8_gradient.py) can be checked against something exact?

Rules implemented here, none changed: a reference written for this purpose, the
reference checks itself first: the exact Rayleigh speed of a half-space, and mode speeds against a
second, independent thin-layer calculation within half a percent, the reference's own
integral of vertical slowness is the timing the wrapper is held to.

Inputs: a stack (`stack`): thicknesses of homogeneous layers, and P speed, S speed and density of
each layer and of the half-space under them. A free surface on top. numpy and scipy only.

Method. Motion and stress are written as the real vector (r1, r2, r3, r4) of a wave
exp(i (k x - w t)), z down: u_x = r1, u_z = i r2, tau_zx = r3, tau_zz = i r4 (the convention of
a8_gates). In a homogeneous layer the vector at the bottom is a matrix times the vector at the top;
the matrix is built from cosh and sinh(x) / x of the vertical wavenumbers, which are real and
smooth whether a wave propagates or decays.
- No matrix is multiplied by another. The vectors at all interfaces are the unknowns of one banded
  system: the surface is free (two rows), each layer links two interfaces (four rows), and the
  half-space holds only decaying or outgoing waves (four rows, two more unknowns). Rounding cannot
  grow from layer to layer, as it does in a product of layer matrices, as long as each layer is
  thin against the decay length; `stack` and `refine` keep them thin.
- Plane P wave (`plane_response`): the solver's source sheet is a plane of nodes that takes the
  stress rate s(t) in the three normal stresses. Across it u_z jumps by -S h / (lambda + 2 mu) and
  tau_zx by -i k 2 mu S h / (lambda + 2 mu) (S the time integral of s, h the grid step); it
  radiates P only. The system is solved with that jump for each frequency, k = w p.
- Rayleigh fundamental mode (`fundamental`): the determinant of the same system without a source
  is real below the half-space's S speed. Its sign comes from the banded factorization. The phase
  speed is scanned upward from below the slowest S speed in steps of half a percent; the first
  change of sign is closed by bisection. Group speed: the difference of w over the difference of k
  between two frequencies a quarter of a percent apart.
- Thin layers (`tlm_modes`), the second method: linear finite elements in depth with a rigid base
  far below the stack; a matrix eigenproblem in k^2, solved near the largest k. It shares no
  formula with the determinant. It also gives the mode shapes, the group speed (from the matrices)
  and the energy integral I1.
- Shallow explosion (`modal_response`): the far-field vertical velocity as the sum over the modes
  whose phase speed is below `c_max`, each r2(z_r) (k r1(z_s) + r2'(z_s)) / (8 c U I1)
  sqrt(2 / (pi k r)) exp(i (k r + pi / 4)) per unit moment rate (the modal form quoted in
  a8_gates; a lead, not opened). Limits: far field only; modes that turn below the stack are left
  out; only the fundamental mode's term has an exact check (the half-space).

Outputs: functions only; `checks` returns the self-check rows.
Run: PYTHONPATH=. python -m akp.stage_a.a8_reference1d   (prints the rows; stops if one fails)

What the self-check proves: the banded determinant has the sign of the dense one; both mode
calculations give the exact Rayleigh speed of a half-space; the explosion's fundamental term
equals the half-space's closed form; the plane-wave response of a uniform half-space equals the
analytic free-surface response, at the surface and at depth; the vertical travel time is the sum
over layers; and, on the R5 column (taken from a8_gradient when run as a module), the two mode
calculations agree within half a percent at 1, 1.5, 2, 3 and 4 Hz.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded
from scipy.linalg.lapack import dgbtrf
from scipy.sparse import bmat, diags
from scipy.sparse.linalg import LinearOperator, eigs, splu

MODE_TOL = 0.005  # The two mode calculations agree within half a percent
MODE_FREQS = (1.0, 1.5, 2.0, 3.0, 4.0)  # Hz. As the brief names them: where the two calculations are compared
KL = 5  # bandwidth of the system below and above the diagonal


def stack(profile, edges):
    """Layers between the depths `edges` (m, from 0) with the profile's values at their middles, over a
    half-space with its values at the last depth. profile(z) -> (vp, vs, rho), vectorized."""
    edges = np.asarray(edges, float)
    z = np.append(0.5 * (edges[:-1] + edges[1:]), edges[-1])
    vp, vs, rho = (np.broadcast_to(np.asarray(q, float), z.shape).copy() for q in profile(z))
    return dict(d=np.diff(edges), vp=vp, vs=vs, rho=rho)


def refine(st, thick):
    """The same stack with every layer cut into equal parts no thicker than `thick`."""
    n = np.maximum(1, np.ceil(st["d"] / thick - 1e-9).astype(int))
    rep = np.append(n, 1)
    return dict(d=np.repeat(st["d"] / n, n), **{q: np.repeat(st[q], rep) for q in ("vp", "vs", "rho")})


def edges(st):
    return np.concatenate([[0.0], np.cumsum(st["d"])])


def vertical_time(st, p, z0, z1):
    """Travel time (s) of a plane P wave of ray parameter p (s/m) between depths z0 < z1: the sum over layers of
    thickness times vertical slowness. Below the stack the half-space counts."""
    e = edges(st)
    top, bot = np.append(e[:-1], e[-1]), np.append(e[1:], np.inf)
    part = np.clip(np.minimum(bot, z1) - np.maximum(top, z0), 0.0, None)
    return float(np.sum(part * np.sqrt(1 / st["vp"] ** 2 - p**2)))


# ---------------------------------------------------------------- the banded system
def _cs(nu2, d):
    """cosh(nu d) and sinh(nu d) / nu for real nu^2 of either sign (cos and sin(x) / x when it is negative)."""
    x = np.sqrt(nu2.astype(complex)) * d
    safe = np.where(x == 0, 1.0, x)
    return np.cosh(x).real, (np.where(x == 0, 1.0, np.sinh(safe) / safe) * d).real


def _layers(k, w, d, vp, vs, rho, sc):
    """The matrix (n, 4, 4) that takes (r1, r2, r3 / sc, r4 / sc) from the top of each layer to depth d below it."""
    mu, na2, nb2 = rho * vs**2, k * k - (w / vp) ** 2, k * k - (w / vs) ** 2
    gam, m2 = rho * w * w - 2 * mu * k * k, 2 * mu * k
    ca, sa = _cs(na2, d)
    cb, sb = _cs(nb2, d)
    o, kk = np.zeros_like(d), np.full_like(d, k)
    # rows r1..r4 against the potentials' values and slopes at the top (f, f', g, g'): r1 = k f - g', r2 = -f' + k g,
    # r3 = 2 mu k f' + gam g, r4 = gam f + 2 mu k g'
    t = np.array([[k * ca, k * sa, -nb2 * sb, -cb], [-na2 * sa, -ca, k * cb, k * sb],
                  [m2 * na2 * sa, m2 * ca, gam * cb, gam * sb], [gam * ca, gam * sa, m2 * nb2 * sb, m2 * cb]])
    inv = np.array([[m2, o, o, o + 1], [o, -gam, kk, o], [o, m2, o + 1, o], [-gam, o, o, kk]]) / (rho * w * w)
    scale = np.array([1.0, 1.0, sc, sc])
    return np.einsum("imn,mjn->nij", t, inv) * scale[None, None, :] / scale[None, :, None]


def _system(k, w, st, dtype=float):
    """(banded matrix, stress scale, the layer matrices). Unknowns: the vector at each interface, then two for the half-space."""
    d, vp, vs, rho = st["d"], st["vp"], st["vs"], st["rho"]
    n = len(d)
    sc = rho[0] * vs[0] * w  # stresses are counted in units of this, so that every entry is of order one
    lay = _layers(k, w, d, vp[:-1], vs[:-1], rho[:-1], sc)
    ab = np.zeros((2 * KL + 1, 4 * n + 6), dtype)

    def put(rows, cols, vals):
        ab[KL + rows - cols, cols] = vals
    j, i, m = np.ogrid[:n, :4, :4]
    put(2 + 4 * j + i, 4 * j + m, -lay)  # each layer: vector below less the matrix times the vector above
    r = 2 + np.arange(4 * n)
    put(r, r + 2, 1.0)
    put(np.arange(2), np.arange(2, 4), 1.0)  # free surface: both tractions vanish
    nu = [np.sqrt(complex(k * k - (w / v) ** 2)) for v in (vp[-1], vs[-1])]
    na, nb = (q.real - 1j * abs(q.imag) for q in nu)  # decaying, or going down when the wave propagates
    mu = rho[-1] * vs[-1] ** 2
    gam, m2 = rho[-1] * w * w - 2 * mu * k * k, 2 * mu * k
    half = np.array([[k, nb], [na, k], [-m2 * na / sc, gam / sc], [gam / sc, -m2 * nb / sc]])
    if dtype is float:
        assert abs(half.imag).max() == 0, "the phase speed must be below the half-space's S speed"
        half = half.real
    rb = 2 + 4 * n + np.arange(4)
    put(rb, 4 * n + np.arange(4), 1.0)
    put(rb[:, None], 4 * n + 4 + np.arange(2)[None, :], -half)
    return ab, sc, lay


def plane_response(st, p, z_src, z_rec, freqs):
    """Velocity spectra (vx, vz), each (frequencies, depths), per unit s h, of the stack under the solver's source sheet.

    s is the stress rate the sheet puts into the three normal stresses of one plane of nodes, h the grid step, the
    sheet lies on the interface at depth z_src, and the wave is exp(i w (p x - t)): multiply by the spectrum of s
    taken with exp(+i w t), by h and by exp(i w p x). Receivers lie above the sheet."""
    e = edges(st)
    js = int(np.argmin(np.abs(e - z_src)))
    assert abs(e[js] - z_src) < 1e-6 * max(1.0, z_src) and js > 0, "the sheet must lie on an interface below the surface"
    z_rec = np.atleast_1d(np.asarray(z_rec, float))
    assert z_rec.max() < z_src
    jr = np.searchsorted(e, z_rec, side="right") - 1
    n = len(st["d"])
    a, b = min(js, n) - 1, min(js, n)  # the layers above and below the sheet (the half-space when it lies at the bottom)
    l2m = 0.5 * sum(st["rho"][q] * st["vp"][q] ** 2 for q in (a, b))
    mu = 0.5 * sum(st["rho"][q] * st["vs"][q] ** 2 for q in (a, b))
    vx, vz = (np.zeros((len(freqs), len(z_rec)), complex) for _ in range(2))
    for q, f in enumerate(freqs):
        w = 2 * np.pi * f
        k = w * p
        ab, sc, lay = _system(k, w, st, complex)
        jump = np.array([0.0, 1 / l2m, -2 * mu * k / l2m / sc, 0.0])  # per unit i S h (module docstring)
        rhs = np.zeros(ab.shape[1], complex)
        if js < n:
            rhs[2 + 4 * js:6 + 4 * js] = lay[js] @ jump
        else:
            rhs[2 + 4 * n:6 + 4 * n] = -jump
        y = solve_banded((KL, KL), ab, rhs)[:4 * n + 4].reshape(n + 1, 4)
        part = _layers(k, w, z_rec - e[jr], st["vp"][jr], st["vs"][jr], st["rho"][jr], sc)
        r = np.einsum("nij,nj->ni", part, y[jr])
        vx[q], vz[q] = 1j * r[:, 0], -r[:, 1]  # v = -i w u, with S = s / (-i w)
    return vx, vz


def free_surface_response(vp, vs, rho, p, z_src, f):
    """The same at the surface of a uniform half-space, in closed form: an upgoing P wave of displacement
    S h / (2 (lambda + 2 mu) cos i) along its ray, times the free surface's response to P."""
    ea, eb = np.sqrt(1 / vp**2 - p**2), np.sqrt(1 / vs**2 - p**2)
    den = vs**2 * ((1 / vs**2 - 2 * p * p) ** 2 + 4 * p * p * ea * eb)
    ph = np.exp(2j * np.pi * f * z_src * ea) / (rho * vp**2)
    return -ph * 2 * p * eb / den, ph * (1 / vs**2 - 2 * p * p) / den


# ---------------------------------------------------------------- the Rayleigh fundamental mode, by the determinant
def secular_sign(c, f, st):
    """Sign of the determinant of the source-free system at phase speed c (m/s) and frequency f (Hz)."""
    w = 2 * np.pi * f
    ab = _system(w / c, w, st)[0]
    lu, piv, info = dgbtrf(np.vstack([np.zeros((KL, ab.shape[1])), ab]), KL, KL)
    assert info >= 0
    swaps = np.count_nonzero(piv != np.arange(len(piv)))
    return float(np.prod(np.sign(lu[2 * KL])) * (-1) ** swaps)


def rayleigh_exact(vp, vs):
    """Rayleigh speed of a uniform half-space: the root of the Rayleigh equation."""
    from scipy.optimize import brentq
    return vs * brentq(lambda x: (2 - x * x) ** 2 - 4 * np.sqrt(1 - x * x * (vs / vp) ** 2) * np.sqrt(1 - x * x), 0.5, 0.999999)


def phase_speed(f, st, start=None, step=0.005):
    """Phase speed (m/s) of the fundamental mode: the first change of sign above `start` (default: under the
    slowest S speed), closed by bisection."""
    lo = start or 0.8 * st["vs"].min()
    top = 0.9999 * st["vs"][-1]
    s_lo = secular_sign(lo, f, st)
    while True:
        hi = min(lo * (1 + step), top)
        if secular_sign(hi, f, st) != s_lo:
            break
        if hi >= top:
            raise ValueError(f"no mode below the half-space's S speed at {f} Hz")
        lo = hi
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if secular_sign(mid, f, st) == s_lo else (lo, mid)
    return 0.5 * (lo + hi)


def fundamental(f, st, df=0.0025):
    """(phase speed, group speed) of the fundamental mode at f, m/s."""
    c = phase_speed(f, st)
    fa, fb = f * (1 - df), f * (1 + df)
    ca, cb = (phase_speed(q, st, start=0.97 * c, step=0.002) for q in (fa, fb))
    return c, (fb - fa) / (fb / cb - fa / ca)


# ---------------------------------------------------------------- thin layers: the second method
def _tri(a00, a01, a10, a11):
    """Sum of two-by-two element matrices along a chain of nodes; the last node (the rigid base) is dropped."""
    n = len(a00)
    main = np.zeros(n + 1)
    main[:-1] += a00
    main[1:] += a11
    return diags([a10[:-1], main[:-1], a01[:-1]], [-1, 0, 1], format="csc")


def tlm_modes(st, f, c_max=None, base=4.0, per_wavelength=400.0):
    """Rayleigh modes with phase speed below c_max (default: the fundamental only), slowest first, by linear
    finite elements in depth, `per_wavelength` of them in the slowest S wavelength (linear elements converge slowly
    when the P speed is several times the S speed: at 400 the half-space speed is exact to 3 parts in 100,000 and
    the explosion's term to 2 in 1000). The half-space is meshed down to a rigid base `base` of its S wavelengths
    under the stack, with elements that grow. Returns dicts(c, U, k, z, r1, r2, i1): speeds (m/s), the nodes' depths and
    the mode shape there (u_x = r1, u_z = i r2), and I1 = half the integral of rho (r1^2 + r2^2)."""
    w = 2 * np.pi * f
    fine = refine(st, st["vs"].min() / f / per_wavelength)
    grow = fine["d"][-1] * 1.15 ** np.arange(1, 400)
    grow = grow[:np.searchsorted(np.cumsum(grow), base * st["vs"][-1] / f) + 1]
    h = np.concatenate([fine["d"], grow])
    vp, vs, rho = (np.concatenate([fine[q][:-1], np.full(len(grow), fine[q][-1])]) for q in ("vp", "vs", "rho"))
    mu = rho * vs**2
    l2m, lam = rho * vp**2, rho * vp**2 - 2 * mu

    def mass(v):  # integral of N_i N_j
        return _tri(v * h / 3, v * h / 6, v * h / 6, v * h / 3)

    def stiff(v):  # integral of N_i' N_j'
        return _tri(v / h, -v / h, -v / h, v / h)
    ax, az, m = mass(l2m), mass(mu), mass(rho)
    cx, cz = stiff(mu) - w * w * m, stiff(l2m) - w * w * m
    bxz = _tri(0.5 * (mu - lam), 0.5 * (lam + mu), -0.5 * (lam + mu), 0.5 * (lam - mu))  # lam N_i N_j' - mu N_i' N_j
    # (k^2 A + k B + C) phi = 0 with x-rows times k: L v = k^2 R v for v = (k r1, r2)
    left = bmat([[-cx, None], [-bxz.T, -cz]], format="csc")
    right = bmat([[ax, bxz], [None, az]], format="csc")
    sigma = (w / (0.8 * st["vs"].min())) ** 2  # above every mode's k^2: the nearest eigenvalue is the fundamental's
    lu = splu((left - sigma * right).tocsc())
    op = LinearOperator(left.shape, matvec=lambda x: lu.solve(right @ x), dtype=float)
    nev = 4
    while True:
        th, vec = eigs(op, k=nev, which="LM", v0=np.ones(left.shape[0]), tol=1e-12)
        k2 = sigma + 1 / th
        real = (np.abs(k2.imag) < 1e-8 * np.abs(k2)) & (k2.real > 0)
        if c_max is None or nev >= left.shape[0] - 2 or np.abs(k2 - sigma).max() > sigma - (w / c_max) ** 2:
            break  # the nearest `nev` eigenvalues reach past c_max: every slower mode is among them
        nev *= 2
    n = left.shape[0] // 2
    z = np.concatenate([[0.0], np.cumsum(h)])[:n]
    out = []
    for q in np.flatnonzero(real):
        k = float(np.sqrt(k2[q].real))
        v = vec[:, q]
        v = (v / v[np.argmax(np.abs(v))]).real
        r1, r2 = v[:n] / k, v[n:]
        i1 = 0.5 * (r1 @ (m @ r1) + r2 @ (m @ r2))
        u = (2 * k * (r1 @ (ax @ r1) + r2 @ (az @ r2)) + 2 * r1 @ (bxz @ r2)) / (4 * w * i1)  # d w / d k of the quadratic form
        out.append(dict(c=w / k, U=float(u), k=k, z=z, r1=r1, r2=r2, i1=float(i1)))
    out.sort(key=lambda q: q["c"])
    return [q for q in out if c_max is None or q["c"] < c_max] if c_max else out[:1]


def excitation(mode, z_src, z_rec):
    """Vertical far-field term of one mode for an explosion: r2(z_rec) (k r1 + r2')(z_src) / (8 c U I1)."""
    z, r1, r2 = mode["z"], mode["r1"], mode["r2"]
    div = mode["k"] * np.interp(z_src, z, r1) + np.interp(z_src, z, np.gradient(r2, z))
    return float(np.interp(z_rec, z, r2) * div / (8 * mode["c"] * mode["U"] * mode["i1"]))


def modal_response(st, freqs, z_src, z_rec, ranges, c_max):
    """Vertical velocity spectrum (frequencies, ranges) per unit moment rate of an explosion at depth z_src: the
    far-field sum over the modes slower than c_max (module docstring), for exp(-i w t). Also the modes of each frequency."""
    ranges = np.asarray(ranges, float)
    out, kept = np.zeros((len(freqs), len(ranges)), complex), []
    for q, f in enumerate(freqs):
        modes = tlm_modes(st, f, c_max)
        kept.append(modes)
        for m in modes:
            out[q] += excitation(m, z_src, z_rec) * np.sqrt(2 / (np.pi * m["k"] * ranges)) * np.exp(1j * (m["k"] * ranges + np.pi / 4))
    return out, kept


def half_space_excitation(vp, vs, rho, f, z_src, z_rec):
    """`excitation` of the Rayleigh wave of a uniform half-space, in closed form (the mode shape is two exponentials)."""
    from scipy.integrate import quad
    w = 2 * np.pi * f
    c = rayleigh_exact(vp, vs)
    k = w / c
    na, nb, mu = np.sqrt(k * k - (w / vp) ** 2), np.sqrt(k * k - (w / vs) ** 2), rho * vs**2
    b = 2 * mu * k * na / (rho * w * w - 2 * mu * k * k)  # tau_zx = 0 at the surface

    def r1(z):
        return k * np.exp(-na * z) + b * nb * np.exp(-nb * z)

    def r2(z):
        return na * np.exp(-na * z) + k * b * np.exp(-nb * z)
    i1 = 0.5 * rho * quad(lambda z: r1(z) ** 2 + r2(z) ** 2, 0, 40 / k, limit=200)[0]
    return r2(z_rec) * (w / vp) ** 2 * np.exp(-na * z_src) / (8 * c * c * i1)  # k r1 + r2' is the P part alone; U = c


# ---------------------------------------------------------------- self-check
def mode_rows(st, freqs=MODE_FREQS):
    """The two mode calculations on one stack: rows of (frequency, c and U by the determinant, c and U by thin layers)."""
    rows = []
    for f in freqs:
        c, u = fundamental(f, st)
        t = tlm_modes(st, f)[0]
        rows.append(dict(freq_hz=f, c_determinant=c, c_thin_layers=t["c"], U_determinant=u, U_thin_layers=t["U"]))
    return rows


def checks(column=None):
    """Self-check rows dict(check, value, reference, tolerance, passed). column: (profile, layer thickness, depth)
    of the column whose two mode calculations are compared (R5's, from a8_gradient)."""
    rows = []

    def add(check, value, reference, tol):
        rows.append(dict(check=check, value=float(value), reference=float(reference), tolerance=tol,
                         passed=bool(abs(value / reference - 1) <= tol)))
    soft = (1500.0, 300.0, 1900.0)  # a soft half-space with a P speed five times its S speed, as the R5 column's top
    uni = stack(lambda z: soft, np.arange(0.0, 601.0, 5.0))
    # the banded determinant against the dense one
    small = stack(lambda z: (1500.0 + 2 * z, 300.0 + z, 1900.0), np.arange(0.0, 101.0, 10.0))
    w = 2 * np.pi * 3.0
    same = 0
    for c in (250.0, 290.0, 330.0, 370.0):
        ab = _system(w / c, w, small)[0]
        n = ab.shape[1]
        dense = np.zeros((n, n))
        for off in range(-KL, KL + 1):
            i = np.arange(max(0, -off), min(n, n - off))
            dense[i, i + off] = ab[KL - off, i + off]
        same += np.linalg.slogdet(dense)[0] == secular_sign(c, 3.0, small)
    add("banded_determinant_sign_equals_dense_of_4", same, 4, 0.0)
    # the exact Rayleigh speed of a half-space, by both methods; no dispersion
    exact = rayleigh_exact(*soft[:2])
    for f in (1.0, 4.0):
        c, u = fundamental(f, uni)
        t = tlm_modes(uni, f)[0]
        add(f"half_space_rayleigh_speed_determinant_{f:g}hz", c, exact, 1e-6)
        add(f"half_space_group_speed_determinant_{f:g}hz", u, exact, 1e-4)
        add(f"half_space_rayleigh_speed_thin_layers_{f:g}hz", t["c"], exact, 1e-3)
        add(f"half_space_group_speed_thin_layers_{f:g}hz", t["U"], exact, 2e-3)
        add(f"half_space_explosion_term_thin_layers_{f:g}hz", excitation(t, 10.0, 2.5), half_space_excitation(*soft, f, 10.0, 2.5), 0.01)
    # the plane-wave response of a uniform half-space: the free surface, and the standing wave under it at p = 0
    fr = np.array([0.5, 1.5, 4.0])
    for p in (5.87e-5, 2.0e-4):
        vx, vz = plane_response(uni, p, 400.0, [0.0], fr)
        ax, az = free_surface_response(*soft, p, 400.0, fr)
        add(f"uniform_half_space_surface_vz_p_{p * 1e3:g}_s_per_km_max_rel_err_plus_1", 1 + np.abs(vz[:, 0] / az - 1).max(), 1.0, 1e-8)
        add(f"uniform_half_space_surface_vx_p_{p * 1e3:g}_s_per_km_max_rel_err_plus_1", 1 + np.abs(vx[:, 0] / ax - 1).max(), 1.0, 1e-8)
    vz = plane_response(uni, 0.0, 400.0, [0.0, 137.0], fr)[1]
    add("uniform_half_space_vz_at_depth_p_0_max_rel_err_plus_1", 1 + np.abs(vz[:, 1] / (vz[:, 0] * np.cos(2 * np.pi * fr * 137.0 / soft[0])) - 1).max(), 1.0, 1e-8)
    add("vertical_time_sum_over_layers", vertical_time(small, 1e-4, 12.0, 250.0),
        sum(d * np.sqrt(1 / v**2 - 1e-8) for d, v in zip([8.0] + [10.0] * 8 + [150.0], small["vp"][1:])), 1e-12)
    # two layers over a stiff half-space: the two calculations on a strong contrast
    two = refine(dict(d=np.array([20.0, 60.0]), vp=np.array([1500.0, 1700.0, 2600.0]), vs=np.array([300.0, 450.0, 1100.0]),
                      rho=np.array([1900.0, 2000.0, 2200.0])), 2.5)
    for r in mode_rows(two, (1.5, 3.0)):
        add(f"layered_phase_speed_thin_layers_over_determinant_{r['freq_hz']:g}hz", r["c_thin_layers"], r["c_determinant"], MODE_TOL)
        add(f"layered_group_speed_thin_layers_over_determinant_{r['freq_hz']:g}hz", r["U_thin_layers"], r["U_determinant"], MODE_TOL)
    if column is not None:  # The reference's mode speeds against the thin-layer ones, on the column of the run
        profile, thick, depth = column
        st = stack(profile, np.arange(0.0, depth + 0.5 * thick, thick))
        for r in mode_rows(st):
            add(f"column_phase_speed_thin_layers_over_determinant_{r['freq_hz']:g}hz", r["c_thin_layers"], r["c_determinant"], MODE_TOL)
            add(f"column_group_speed_thin_layers_over_determinant_{r['freq_hz']:g}hz", r["U_thin_layers"], r["U_determinant"], MODE_TOL)
    return rows


if __name__ == "__main__":
    from akp.stage_a import a8_gradient  # the R5 column lives in the wrapper; everything above runs without it
    out = checks(a8_gradient.reference_column())
    for row in out:
        print(f"{row['check']:<72s} {row['value']:.9g}  against {row['reference']:.9g}  tol {row['tolerance']:g}  {'ok' if row['passed'] else 'FAILED'}")
    assert all(row["passed"] for row in out), [row["check"] for row in out if not row["passed"]]
    print("a8_reference1d selfcheck ok")
