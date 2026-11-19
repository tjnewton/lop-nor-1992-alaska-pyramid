"""Exact 2-D P-SV scattering of a plane P wave by a circular cylinder: benchmark for A3b-2 (gate i-b).

Series solution (Pao & Mow type; lead, to be opened before citing). Incident potential
phi = exp(i k x) = sum eps_n i^n J_n(k r) cos(n theta); scattered phi_s = sum A_n H_n(k r) cos,
psi_s = sum B_n H_n(k_s r) sin; inside a welded inclusion, J_n terms with the inclusion's
wavenumbers. Boundary conditions at r = a: traction-free (void), or continuity of u_r, u_theta,
sigma_rr, sigma_rtheta (inclusion). Far-field scattered P over incident displacement:
|sum A_n (-i)^n cos(n theta)| sqrt(2 / (pi k r)), the same normalization as a3b2_numerical's 2-D
Born, so C_exact = exact / Born. Scattered S over incident displacement (mode="PS"):
(k_s / k_p) |sum B_n (-i)^n sin(n theta)| sqrt(2 / (pi k_s r)), against the 2-D Born P->S with
the same Green's-function factor.

Adapted from an independent check script, with the derivation re-checked:
displacements u = grad phi + curl(psi z); stresses from the Bessel equation for Z''.
Run: python -m akp.stage_a.a3b2_cylinder  (self-check)
"""
from __future__ import annotations

import numpy as np
from scipy.special import h1vp, hankel1, j1, jv, jvp

from swe.scattering import born_radiation_pattern


def _lame(vp, vs, rho):
    return rho * (vp**2 - 2 * vs**2), rho * vs**2


def _cols(n, k, r, Z, Zp, lam, mu, kind):
    """(u_r, u_theta, sigma_rr, sigma_rtheta) of one potential term, angular factor dropped."""
    x = k * r
    z, zp = Z(n, x), Zp(n, x)
    zpp = -zp / x - (1 - n**2 / x**2) * z
    F, Fp, Fpp = z, k * zp, k**2 * zpp
    if kind == "phi":  # phi = F cos(n theta)
        ur, ut = Fp, -(n / r) * F
        srr = -lam * k**2 * F + 2 * mu * Fpp
        srt = 2 * mu * n * (F / r**2 - Fp / r)
    else:  # psi = F sin(n theta)
        ur, ut = (n / r) * F, -Fp
        srr = 2 * mu * n * (Fp / r - F / r**2)
        srt = mu * (2 * Fp / r + k**2 * F - 2 * n**2 * F / r**2)
    return np.array([ur, ut, srr, srt], dtype=complex)


def coefficients(*, freq, a, vp, vs, rho, inclusion=None):
    """Scattered-P and scattered-S coefficients (A_n, B_n, k_p, k_s); inclusion = (vp', vs', rho') or None (void)."""
    w = 2 * np.pi * freq
    kp, ks = w / vp, w / vs
    lam, mu = _lame(vp, vs, rho)
    A, B = [], []
    for n in range(int(2 * kp * a) + 21):
        eps = 1 if n == 0 else 2
        inc = eps * (1j**n) * _cols(n, kp, a, jv, jvp, lam, mu, "phi")
        sP = _cols(n, kp, a, hankel1, h1vp, lam, mu, "phi")
        sS = _cols(n, ks, a, hankel1, h1vp, lam, mu, "psi")
        if inclusion is None:
            if n == 0:  # psi decouples for n = 0; only sigma_rr = 0
                A.append(-inc[2] / sP[2])
                B.append(0.0)
                continue
            sol = np.linalg.solve(np.column_stack([sP[2:], sS[2:]]), -inc[2:])
        else:
            vp2, vs2, r2 = inclusion
            l2, m2 = _lame(vp2, vs2, r2)
            iP = _cols(n, w / vp2, a, jv, jvp, l2, m2, "phi")
            iS = _cols(n, w / vs2, a, jv, jvp, l2, m2, "psi")
            M, rhs = np.column_stack([sP, sS, -iP, -iS]), -inc
            if n == 0:  # rows u_r, sigma_rr; unknowns A_0, inner phi
                M, rhs = M[[0, 2]][:, [0, 2]], rhs[[0, 2]]
                A.append(np.linalg.solve(M, rhs)[0])
                B.append(0.0)
                continue
            sol = np.linalg.solve(M, rhs)
        A.append(sol[0])
        B.append(sol[1])
    return np.array(A), np.array(B, dtype=complex), kp, ks


def point_coefficients(*, freq, vp, vs, rho, ka=0.01, rel=1e-4, n_max=2):
    """Linear point-scatterer coefficients (a_n, b_n, k_p, k_s) per unit area and unit relative contrast.

    The exact series for a small circle (k_p a = ka) at a small contrast (vp, vs, rho all scaled by
    1 + rel), divided by pi a^2 rel. Terms above n_max vanish as (ka)^(2 n). A body's linear Born field
    at any distance is the sum of a_n H_n(k_p R) cos(n theta) and b_n H_n(k_s R) sin(n theta) over its
    cells, each with its area and the incident phase at the cell.
    """
    a = ka * vp / (2 * np.pi * freq)
    A, B, kp, ks = coefficients(freq=freq, a=a, vp=vp, vs=vs, rho=rho,
                                inclusion=(vp * (1 + rel), vs * (1 + rel), rho * (1 + rel)))
    scale = np.pi * a**2 * rel
    return A[: n_max + 1] / scale, B[: n_max + 1] / scale, kp, ks


def c_exact(*, freq, a, theta, vp, vs, rho, rel_born, inclusion=None, mode="PP", r=None):
    """Exact amplitude (P->P or P->S) over linearized far-field 2-D Born (relative perturbation rel_born).

    r=None: the far-field limit, the Green's factor sqrt(2 / (pi k r)) dropped from both. With r
    (m), the exact field is evaluated at that distance (H_n(k r), not its asymptote), which is what
    the FD div / curl measures (|div u| / k_p = k_p |phi|, |curl u| / k_s = k_s |psi|), and the far-field
    Born carries its Green's factor. Returns (C, exact).
    """
    A, B, kp, ks = coefficients(freq=freq, a=a, vp=vp, vs=vs, rho=rho, inclusion=inclusion)
    theta = np.atleast_1d(theta)
    n = np.arange(len(A))
    k_out = kp if mode == "PP" else ks
    if r is None:
        radial, green = ((-1j) ** n)[:, None], 1.0
    else:
        radial, green = (hankel1(n, k_out * r) * np.exp(-1j * k_out * r))[:, None], np.sqrt(2 / (np.pi * k_out * r))
    if mode == "PP":
        exact = np.abs(np.sum(A[:, None] * radial * np.cos(np.outer(n, theta)), axis=0))
    else:
        exact = ks / kp * np.abs(np.sum(B[:, None] * radial * np.sin(np.outer(n, theta)), axis=0))
    qa = np.sqrt(kp**2 + k_out**2 - 2 * kp * k_out * np.cos(theta)) * a
    F = np.where(qa > 1e-9, 2 * j1(qa) / np.maximum(qa, 1e-300), 1.0)
    pat = born_radiation_pattern(theta, vp=vp, vs=vs, density=rho, dvp=rel_born * vp, dvs=rel_born * vs,
                                 drho=rel_born * rho, mode=mode)
    born = k_out**2 * np.pi * a**2 * np.abs(pat * F) / 4 * green
    return exact / born, exact


def _selfcheck() -> None:
    vp, vs, rho, a = 2000.0, 2000.0 / 1.8, 2000.0, 100.0
    th = np.radians([75.0, 90.0, 120.0])
    for ka in (0.05, 0.5, 1.5):  # weak contrast -> Born
        f = ka * vp / (2 * np.pi * a)
        r = 1e-4
        c, _ = c_exact(freq=f, a=a, theta=th, vp=vp, vs=vs, rho=rho, rel_born=r,
                       inclusion=(vp * (1 + r), vs * (1 + r), rho * (1 + r)))
        assert np.allclose(c, 1.0, atol=2e-3), (ka, c)
        cs, _ = c_exact(freq=f, a=a, theta=th, vp=vp, vs=vs, rho=rho, rel_born=r, mode="PS",
                        inclusion=(vp * (1 + r), vs * (1 + r), rho * (1 + r)))
        assert np.allclose(cs, 1.0, atol=2e-3), (ka, cs)
    for mode in ("PP", "PS"):  # finite r far away tends to the far-field value
        f = 0.5 * vp / (2 * np.pi * a)
        cf, _ = c_exact(freq=f, a=a, theta=th, vp=vp, vs=vs, rho=rho, rel_born=-0.9, mode=mode)
        cr, _ = c_exact(freq=f, a=a, theta=th, vp=vp, vs=vs, rho=rho, rel_born=-0.9, mode=mode, r=1e7 * a)
        assert np.allclose(cf, cr, rtol=1e-3), (mode, cf, cr)
    pa, pb, _, _ = point_coefficients(freq=2.0, vp=vp, vs=vs, rho=rho)  # converged in ka and rel
    qa, qb, _, _ = point_coefficients(freq=2.0, vp=vp, vs=vs, rho=rho, ka=0.005, rel=5e-5)
    assert np.allclose(pa, qa, rtol=1e-3) and np.allclose(pb[1:], qb[1:], rtol=1e-3), (pa, qa, pb, qb)
    # a near-void (10% properties) tends to the void as the contrast grows
    f = 0.5 * vp / (2 * np.pi * a)
    cv, _ = c_exact(freq=f, a=a, theta=th, vp=vp, vs=vs, rho=rho, rel_born=-0.9)
    cn, _ = c_exact(freq=f, a=a, theta=th, vp=vp, vs=vs, rho=rho, rel_born=-0.9,
                    inclusion=(0.01 * vp, 0.01 * vs, 0.01 * rho))
    assert np.allclose(cv, cn, rtol=0.02), (cv, cn)
    print("a3b2_cylinder selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
