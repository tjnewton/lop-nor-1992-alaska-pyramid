"""Exact series for a plane P wave on an elastic sphere in an elastic full space (gate two reference).

Inputs: host (vp, vs, rho), inclusion (vp, vs, rho) welded at r = a, frequency, scattering angle theta
(from the incident propagation direction) and, optionally, a finite distance r.

Method. Time factor exp(-i w t); axisymmetric about the incidence axis. Displacement from two scalar
potentials, u = grad(phi) + curl curl (r_vec psi), with phi a Helmholtz solution at k_p and psi at
k_s. For one order n (angular factors P_n(cos theta) for u_r and sigma_rr, dP_n/dtheta for u_theta
and sigma_rtheta), with x = k r and z = z_n(x) a spherical Bessel function:
    phi term:  u_r = k z',            u_theta = z / r,
               sigma_rr = -lam k^2 z + 2 mu k^2 z'',     sigma_rtheta = 2 mu (k z' / r - z / r^2)
    psi term:  u_r = n (n + 1) z / r, u_theta = (z + x z') / r,
               sigma_rr = 2 mu n (n + 1) (k z' / r - z / r^2),
               sigma_rtheta = mu ((2 n (n + 1) - 2 - x^2) z - 2 x z') / r^2
(z'' from Bessel's equation). Incident phi = exp(i k_p z) / (i k_p) (unit displacement) =
sum (2n + 1) i^n j_n(k_p r) P_n / (i k_p); scattered terms use h_n^(1), interior terms j_n with the
inclusion's wavenumbers. Continuity of u_r, u_theta, sigma_rr, sigma_rtheta at r = a gives a 4 x 4
system per order (2 x 2 for n = 0). Then, exactly at any r > a,
    div u_sc  = -k_p^2 phi_sc,            |curl u_sc| = k_s^2 |d psi_sc / d theta|,
so the amplitudes the FD measures (a8_gates) are
    A_P = |div u_sc| / k_p = k_p |sum a_n h_n(k_p r) P_n|,
    A_S = |curl u_sc| / k_s = k_s |sum b_n h_n(k_s r) dP_n/dtheta|,
which tend to |f_P| / r and |f_S| / r (far-field displacement amplitudes per unit incident
displacement) as r grows.

Source of the formulas: derived here from the potentials above, as a3b2_cylinder.py was for the
circle. The classical treatment is Ying & Truell (1956, J. Appl. Phys. 27, 1086) for the cavity and
rigid sphere, extended to an elastic sphere by Einspruch, Witterholt & Truell (1960, J. Appl. Phys.
31, 806); see also Korneev & Johnson (1993, Geophys. J. Int. 115, 230). These are leads: the
papers were not opened, and the code rests on the derivation and on the three checks below.

Internal checks (requires the first two):
  1. weak contrast -> the three-dimensional Born expression for a sphere (swe.scattering pattern
     times the form factor 3 j_1(q a) / (q a)), P and S;
  2. low frequency, strong contrast -> the Rayleigh (quasi-static) limit: Born with the bulk and
     shear perturbations replaced by their Eshelby strain-concentrated values (as a3b_quasistatic.py);
  3. energy: the optical theorem, (4 pi / k_p) Im f_P(0) = sigma_P + sigma_S, at strong contrast and
     k_p a of order one. An algebra error that cancels at weak contrast would fail this.

Outputs: functions only; `checks()` returns the rows a8_gates.py writes to a8_checks.csv.
Run: python -m akp.stage_a.a8_sphere_series  (self-check)
"""
from __future__ import annotations

import numpy as np
from scipy.special import lpmv, spherical_jn, spherical_yn

from swe.scattering import born_radiation_pattern

BORN_TOL, RAYLEIGH_TOL, ENERGY_TOL = 0.01, 0.01, 1e-6  # the series' own checks (not gate tolerances)
NULL_FRAC = 0.10  # angles where Born is under this share of its maximum are left out of check 1 (nulls)


def _lame(vp, vs, rho):
    return rho * (vp**2 - 2 * vs**2), rho * vs**2


def _z(n, x, kind):
    """(z_n, z_n') for j (kind 'j') or h^(1) (kind 'h')."""
    z, zp = spherical_jn(n, x), spherical_jn(n, x, derivative=True)
    if kind == "h":
        z = z + 1j * spherical_yn(n, x)
        zp = zp + 1j * spherical_yn(n, x, derivative=True)
    return z, zp


def _cols(n, k, r, kind, lam, mu, pot):
    """(u_r, u_theta, sigma_rr, sigma_rtheta) of one potential term at radius r, angular factor dropped."""
    x = k * r
    z, zp = _z(n, x, kind)
    if pot == "phi":
        zpp = -2 * zp / x - (1 - n * (n + 1) / x**2) * z
        return np.array([k * zp, z / r, -lam * k**2 * z + 2 * mu * k**2 * zpp, 2 * mu * (k * zp / r - z / r**2)],
                        dtype=complex)
    nn = n * (n + 1)
    return np.array([nn * z / r, (z + x * zp) / r, 2 * mu * nn * (k * zp / r - z / r**2),
                     mu * ((2 * nn - 2 - x**2) * z - 2 * x * zp) / r**2], dtype=complex)


def coefficients(*, freq, a, host, inclusion):
    """Scattered coefficients (a_n, b_n, k_p, k_s); host and inclusion are (vp, vs, rho)."""
    w = 2 * np.pi * freq
    (vp, vs, rho), (vp2, vs2, rho2) = host, inclusion
    kp, ks = w / vp, w / vs
    lam, mu = _lame(vp, vs, rho)
    lam2, mu2 = _lame(vp2, vs2, rho2)
    scale = np.array([1.0, 1.0, 1.0 / (mu * kp), 1.0 / (mu * kp)])  # stress rows to displacement units
    A, B = [], []
    for n in range(int(2 * ks * a) + 16):
        inc = (2 * n + 1) * 1j**n / (1j * kp) * _cols(n, kp, a, "j", lam, mu, "phi")
        M = np.column_stack([_cols(n, kp, a, "h", lam, mu, "phi"), _cols(n, ks, a, "h", lam, mu, "psi"),
                             -_cols(n, w / vp2, a, "j", lam2, mu2, "phi"),
                             -_cols(n, w / vs2, a, "j", lam2, mu2, "psi")]) * scale[:, None]
        rhs = -inc * scale
        if n == 0:  # no theta dependence: rows u_r, sigma_rr; unknowns a_0 and the inner phi
            A.append(np.linalg.solve(M[[0, 2]][:, [0, 2]], rhs[[0, 2]])[0])
            B.append(0.0)
            continue
        sol = np.linalg.solve(M, rhs)
        A.append(sol[0])
        B.append(sol[1])
    return np.array(A), np.array(B, dtype=complex), kp, ks


def amplitudes(*, freq, a, host, inclusion, theta, r=None):
    """(A_P, A_S) at angles theta. r=None: the far-field |f_P|, |f_S| (multiply by 1 / r for an amplitude).

    With r (m): A_P = |div u_sc| / k_p and A_S = |curl u_sc| / k_s at that distance, per unit incident
    displacement. Complex values are returned (take abs for the amplitude).
    """
    A, B, kp, ks = coefficients(freq=freq, a=a, host=host, inclusion=inclusion)
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    n = np.arange(len(A))
    c = np.cos(theta)
    Pn = np.array([lpmv(0, m, c) for m in n])
    dPn = np.array([lpmv(1, m, c) for m in n])  # dP_n / dtheta (Condon-Shortley sign)
    if r is None:
        radP = radS = (-1j) ** n
    else:
        radP = kp * (spherical_jn(n, kp * r) + 1j * spherical_yn(n, kp * r))
        radS = ks * (spherical_jn(n, ks * r) + 1j * spherical_yn(n, ks * r))
    return (A * radP) @ Pn, (B * radS) @ dPn


def displacement(*, freq, a, host, inclusion, points, axis):
    """Scattered displacement vectors (m, 3) at points (m, 3) about the center, incident along unit `axis`.

    Used by a8_gates' self-check to synthesize traces: the div and curl of this field, taken with the
    receiver stencil, must give back `amplitudes`.
    """
    A, B, kp, ks = coefficients(freq=freq, a=a, host=host, inclusion=inclusion)
    points, axis = np.asarray(points, float), np.asarray(axis, float)
    r = np.linalg.norm(points, axis=1)
    rhat = points / r[:, None]
    ct = np.clip(rhat @ axis, -1, 1)
    st = np.sqrt(1 - ct**2)
    n = np.arange(len(A))[:, None]
    hp, hpp = _z(n, kp * r, "h")
    hs, hsp = _z(n, ks * r, "h")
    ur = np.sum((A[:, None] * kp * hpp + B[:, None] * n * (n + 1) * hs / r) * lpmv(0, n, ct), axis=0)
    ut = np.sum((A[:, None] * hp / r + B[:, None] * (hs + ks * r * hsp) / r) * lpmv(1, n, ct), axis=0)
    that = (ct[:, None] * rhat - axis) / np.maximum(st, 1e-12)[:, None]  # theta unit vector; u_theta = 0 on the axis
    return ur[:, None] * rhat + ut[:, None] * that


def perturbations(host, inclusion, *, eshelby=False):
    """(drho, dlam, dmu): exact parameter differences, or with the Eshelby strain concentration."""
    (vp, vs, rho), (vp2, vs2, rho2) = host, inclusion
    lam, mu = _lame(vp, vs, rho)
    lam2, mu2 = _lame(vp2, vs2, rho2)
    K, dK, dmu = lam + 2 * mu / 3, (lam2 - lam) + 2 * (mu2 - mu) / 3, mu2 - mu
    if eshelby:
        dK = dK / (1 + 3 * K / (3 * K + 4 * mu) * dK / K)
        dmu = dmu / (1 + 6 * (K + 2 * mu) / (5 * (3 * K + 4 * mu)) * dmu / mu)
    return rho2 - rho, dK - 2 * dmu / 3, dmu


def born_sphere(*, freq, a, host, inclusion, theta, eshelby=False):
    """Far-field Born |f_P|, |f_S| for a sphere (signed real values), from exact or Eshelby perturbations."""
    vp, vs, rho = host
    drho, dlam, dmu = perturbations(host, inclusion, eshelby=eshelby)
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    kp, ks = 2 * np.pi * freq / vp, 2 * np.pi * freq / vs
    c, s = np.cos(theta), np.sin(theta)
    pp = drho / rho * c - (dlam + 2 * dmu * c**2) / (rho * vp**2)
    ps = drho / rho * s - 2 * dmu * s * c / (rho * vp * vs)
    vol = 4 * np.pi * a**3 / 3

    def form(q):
        x = np.maximum(q * a, 1e-9)
        return 3 * spherical_jn(1, x) / x
    return (kp**2 * vol * pp * form(2 * kp * np.sin(theta / 2)) / (4 * np.pi),
            ks**2 * vol * ps * form(np.sqrt(kp**2 + ks**2 - 2 * kp * ks * c)) / (4 * np.pi))


def energy_defect(*, freq, a, host, inclusion):
    """Optical theorem: (extinction - scattered) / extinction cross-section."""
    vp, vs, _ = host
    A, B, kp, _ = coefficients(freq=freq, a=a, host=host, inclusion=inclusion)
    n = np.arange(len(A))
    ext = 4 * np.pi / kp * np.imag(np.sum(A * (-1j) ** n))  # P_n(1) = 1
    # int P_n^2 sin = 2 / (2n+1); int (dP_n/dtheta)^2 sin = 2 n (n+1) / (2n+1); S flux carries vs / vp
    sca = 2 * np.pi * np.sum((np.abs(A) ** 2 + vs / vp * n * (n + 1) * np.abs(B) ** 2) * 2 / (2 * n + 1))
    return float((ext - sca) / ext)


HOST = (1200.0, 1200.0 / 1.8, 2000.0)  # the A3-0 floor host, vp/vs and density as A3a
STONE = tuple(2.0 * v for v in HOST)  # A3's dense, stiff body: all perturbations +100%
CHECK_THETA = np.radians([0.0, 30.0, 90.0, 150.0, 180.0])


def weak_rows(a: float = 100.0):
    """Check 1 row by row: (k_p a, frequency in Hz, largest relative error of the series against Born, P and S)."""
    kp1 = 2 * np.pi / HOST[0]
    weak = tuple(v * (1 + 1e-4) for v in HOST)
    out = []
    for ka in (0.2, 0.5, 1.0, 2.0, 4.0):
        f = ka / (kp1 * a)
        fp, fs = amplitudes(freq=f, a=a, host=HOST, inclusion=weak, theta=CHECK_THETA)
        bp, bs = born_sphere(freq=f, a=a, host=HOST, inclusion=weak, theta=CHECK_THETA)
        errs = []
        for num, born in ((fp, bp), (fs[1:4], bs[1:4])):  # S vanishes on the axis
            keep = np.abs(born) >= NULL_FRAC * np.abs(born).max()  # a ratio at a Born null says nothing
            errs += list(np.abs(np.abs(num) / np.abs(born) - 1)[keep])
        out.append((ka, float(f), float(max(errs))))
    return out


def checks(a: float = 100.0):
    """Rows (check, value, tol, passed) for the three internal checks."""
    rows = []
    kp1 = 2 * np.pi / HOST[0]
    weak = tuple(v * (1 + 1e-4) for v in HOST)
    errs = [e for _, _, e in weak_rows(a)]  # 1. weak contrast -> Born, P and S
    rows.append(dict(check="series_weak_vs_born_max_rel_err", value=float(max(errs)), tol=BORN_TOL,
                     passed=bool(max(errs) <= BORN_TOL)))
    errs = []
    for inc in (STONE, tuple(0.1 * v for v in HOST)):  # 2. Rayleigh limit, stone and a near-void
        f = 0.01 / (kp1 * a)
        fp, fs = amplitudes(freq=f, a=a, host=HOST, inclusion=inc, theta=CHECK_THETA)
        bp, bs = born_sphere(freq=f, a=a, host=HOST, inclusion=inc, theta=CHECK_THETA, eshelby=True)
        errs += list(np.abs(np.abs(fp) / np.abs(bp) - 1)) + list(np.abs(np.abs(fs[1:4]) / np.abs(bs[1:4]) - 1))
    rows.append(dict(check="series_rayleigh_limit_max_rel_err", value=float(max(errs)), tol=RAYLEIGH_TOL,
                     passed=bool(max(errs) <= RAYLEIGH_TOL)))
    defect = max(abs(energy_defect(freq=ka / (kp1 * a), a=a, host=HOST, inclusion=inc))
                 for ka in (0.5, 1.0, 2.0, 4.0) for inc in (STONE, tuple(0.1 * v for v in HOST)))
    rows.append(dict(check="series_optical_theorem_max_defect", value=float(defect), tol=ENERGY_TOL,
                     passed=bool(defect <= ENERGY_TOL)))
    # finite r tends to the far field
    f = 1.0 / (kp1 * a)
    fp, fs = amplitudes(freq=f, a=a, host=HOST, inclusion=STONE, theta=CHECK_THETA[1:4])
    rp, rs = amplitudes(freq=f, a=a, host=HOST, inclusion=STONE, theta=CHECK_THETA[1:4], r=1e6 * a)
    err = float(max(np.abs(np.abs(rp) * 1e6 * a / np.abs(fp) - 1).max(), np.abs(np.abs(rs) * 1e6 * a / np.abs(fs) - 1).max()))
    rows.append(dict(check="series_finite_r_to_far_field_max_rel_err", value=err, tol=1e-3, passed=bool(err <= 1e-3)))
    # the linearized pattern in swe.scattering is the same Born at weak contrast
    r = 1e-4
    pat = born_radiation_pattern(CHECK_THETA, vp=HOST[0], vs=HOST[1], density=HOST[2], dvp=r * HOST[0],
                                 dvs=r * HOST[1], drho=r * HOST[2])
    bp, _ = born_sphere(freq=1e-3, a=a, host=HOST, inclusion=weak, theta=CHECK_THETA)
    mine = bp * 4 * np.pi / ((2 * np.pi * 1e-3 / HOST[0]) ** 2 * 4 * np.pi * a**3 / 3)
    err = float(np.abs(mine / pat - 1).max())
    rows.append(dict(check="born_pattern_vs_swe_scattering_max_rel_err", value=err, tol=1e-3, passed=bool(err <= 1e-3)))
    return rows


def _selfcheck() -> None:
    for row in checks():
        print(row)
        assert row["passed"], row
    print("a8_sphere_series selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
