"""Tests for dpi.shakeoff.

The central test is the sum rule
:math:`\\int k P_\\nu(k)\\, d\\varepsilon = \\int k^2 P_\\nu(k)\\, dk = 1`,
which holds only with the (2*pi)^-3 prefactor and the solid-angle
*integral* convention of REVIEW.md [B-2].  It validates the AO
normalisation, the angular factors and the quadrature simultaneously, so a
wrong A_l, a missing prefactor or a dropped Jacobian factor all show up
here.

Independent references used, none of them the closed forms under test:
a spherical-Bessel radial transform, a product-Gauss solid-angle
quadrature over the exact 1-D Hermite transforms, and
``scipy.integrate.quad`` on the unsubstituted energy integral.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial import hermite, legendre
from scipy.integrate import quad
from scipy.special import spherical_jn

from dpi.constants import ConfigError, ModelError
from dpi.shakeoff import (
    HARMONICS_CARTESIAN,
    HARMONICS_SOLID,
    HARMONICS_SPHERICAL,
    angular_factor,
    angular_terms,
    check_normalisation,
    p_shake,
    p_shake_integrated,
)

from .basis_stub import (
    BasisStub,
    make_cartesian_basis,
    make_spherical_basis,
    spherical_primitive_norm,
)

TWO_PI_CUBED = (2.0 * np.pi) ** 3


# --- independent references ------------------------------------------------

def _sphere_quad(func, n: int = 64) -> float:
    """Integrate func(nx, ny, nz) over the unit sphere, product rule.

    Gauss-Legendre in cos(theta) times the midpoint rule in phi, which is
    spectrally accurate for the trigonometric polynomials appearing here.
    """
    x, w = legendre.leggauss(n)
    phi = (np.arange(2 * n) + 0.5) * np.pi / n
    w_phi = np.full(2 * n, np.pi / n)
    ct = x[:, None]
    st = np.sqrt(1.0 - ct ** 2)
    vals = func(st * np.cos(phi), st * np.sin(phi), np.broadcast_to(
        ct, (n, 2 * n)))
    return float(np.sum(w[:, None] * w_phi[None, :] * vals))


def _hermite_1d(n: int, k: np.ndarray, alpha: float) -> np.ndarray:
    """H_n(k / (2 sqrt(alpha))) as a polynomial in k."""
    c = np.zeros(n + 1)
    c[n] = 1.0
    poly = hermite.herm2poly(c)
    s = 2.0 * np.sqrt(alpha)
    return np.polynomial.polynomial.polyval(k, poly / s ** np.arange(n + 1))


def _cartesian_amplitude(powers, alpha, weight, kvec) -> np.ndarray:
    """Exact FT of a contracted Cartesian GTO, up to the overall (-i)^l.

    Uses the 1-D result
    int dx x^n exp(-i k x) exp(-alpha x^2)
      = sqrt(pi/alpha) (2 sqrt(alpha))^-n H_n(k/(2 sqrt(alpha)))
        exp(-k^2/(4 alpha))
    which is where the "contact" terms of REVIEW.md [A-2] live: for n = 2,
    H_2(u) = 4u^2 - 2, and the -2 is the term the old code dropped.
    """
    a, b, c = powers
    l = a + b + c
    kx, ky, kz = (np.asarray(v, dtype=float) for v in kvec)
    total = 0.0
    for alpha_j, w_j in zip(alpha, weight):
        s = 2.0 * np.sqrt(alpha_j)
        total += (
            w_j * (np.pi / alpha_j) ** 1.5 * s ** (-l)
            * _hermite_1d(a, kx, alpha_j)
            * _hermite_1d(b, ky, alpha_j)
            * _hermite_1d(c, kz, alpha_j)
            * np.exp(-(kx ** 2 + ky ** 2 + kz ** 2) / (4.0 * alpha_j))
        )
    return total # type: ignore


def _p_shake_brute_cartesian(basis, mu: int, k: float, n: int = 64) -> float:
    """(2 pi)^-3 int dOmega |chi~|^2 by solid-angle quadrature."""
    powers = tuple(int(v) for v in basis.cart_powers[mu])
    alpha = np.asarray(basis.alpha[mu], dtype=float)
    weight = (np.asarray(basis.coeff[mu], dtype=float)
              * np.asarray(basis.norm[mu], dtype=float))

    def integrand(nx, ny, nz):
        return _cartesian_amplitude(
            powers, alpha, weight, (k * nx, k * ny, k * nz)
        ) ** 2

    return _sphere_quad(integrand, n=n) / TWO_PI_CUBED


def _radial_bessel_amplitude(alpha, weight, l, k) -> float:
    """4 pi int r^2 R(r) j_l(k r) dr for R = sum_j w_j r^l exp(-a_j r^2)."""
    def radial(r):
        return float(np.sum(weight * r ** l * np.exp(-alpha * r ** 2)))

    return quad(
        lambda r: 4.0 * np.pi * r ** 2 * radial(r) * spherical_jn(l, k * r),
        0.0, 40.0, limit=400,
    )[0]


# --- angular factors -------------------------------------------------------

@pytest.mark.parametrize("l", [0, 1, 2, 3, 4])
def test_spherical_angular_factor_is_unity(l):
    """A_l = int dOmega |Y_lm|^2 = 1 for unit-normalised real Y_lm.

    Derived here rather than copied: the real spherical harmonics are
    built explicitly and integrated.  This is the factor the old code
    split into 4*pi/5 and 4*pi/15 for l = 2 (REVIEW.md [A-1]).
    """
    assert angular_factor(l, HARMONICS_SPHERICAL) == 1.0


def test_real_spherical_harmonics_are_unit_normalised():
    """Explicit check that the convention behind A_l = 1 is the right one."""
    harmonics = {
        (0, 0): lambda x, y, z: np.full_like(x, np.sqrt(1.0 / (4 * np.pi))),
        (1, 0): lambda x, y, z: np.sqrt(3 / (4 * np.pi)) * z,
        (1, 1): lambda x, y, z: np.sqrt(3 / (4 * np.pi)) * x,
        (2, 0): lambda x, y, z: np.sqrt(5 / (16 * np.pi)) * (3 * z ** 2 - 1),
        (2, 1): lambda x, y, z: np.sqrt(15 / (4 * np.pi)) * x * z,
        (2, 2): lambda x, y, z: np.sqrt(15 / (16 * np.pi))
        * (x ** 2 - y ** 2),
        (3, 0): lambda x, y, z: np.sqrt(7 / (16 * np.pi))
        * (5 * z ** 3 - 3 * z),
    }
    for (l, m), fn in harmonics.items():
        got = _sphere_quad(lambda x, y, z: fn(x, y, z) ** 2)
        assert got == pytest.approx(1.0, rel=1e-12), (l, m)


def test_old_d_splitting_was_a_cartesian_monomial_average():
    """The discarded 4*pi/5 : 4*pi/15 split identified.

    Those numbers are the solid-angle integrals of the *unnormalised*
    Cartesian monomials n_z^4 and (n_x n_y)^2, not of any pair of real
    spherical harmonics -- which is why applying them to a spherical basis
    was a factor-3 error between the two groups.
    """
    diag = _sphere_quad(lambda x, y, z: z ** 4)
    off = _sphere_quad(lambda x, y, z: (x * y) ** 2)
    assert diag == pytest.approx(4 * np.pi / 5, rel=1e-12)
    assert off == pytest.approx(4 * np.pi / 15, rel=1e-12)
    assert diag / off == pytest.approx(3.0, rel=1e-12)


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_solid_harmonic_angular_factor(l):
    """The Racah convention carries A_l = 4 pi / (2l+1)."""
    assert angular_factor(l, HARMONICS_SOLID) == pytest.approx(
        4 * np.pi / (2 * l + 1), rel=1e-14
    )


def test_angular_factor_rejects_cartesian_and_unknown():
    with pytest.raises(ModelError, match="Cartesian branch"):
        angular_factor(2, HARMONICS_CARTESIAN)
    with pytest.raises(ModelError):
        angular_factor(2, "banana")
    with pytest.raises(ConfigError):
        angular_factor(-1)


# --- the sum rule ----------------------------------------------------------

def test_check_normalisation_is_unity_for_every_ao():
    """Sum rule per AO for a synthetic s/p/d/f basis, analytically."""
    basis = make_spherical_basis()
    norms = check_normalisation(basis)
    assert norms.shape == (basis.nbas,)
    assert np.max(np.abs(norms - 1.0)) < 1e-12


def test_sum_rule_by_quadrature_per_angular_type():
    """int k P dEps -> 1 at large excess energy, for each l separately."""
    basis = make_spherical_basis()
    yields = p_shake_integrated(basis, 4000.0, n_quad=600)
    for l in sorted(set(np.asarray(basis.l).tolist())):
        sel = np.asarray(basis.l) == l
        residual = float(np.max(np.abs(yields[sel] - 1.0)))
        assert residual < 1e-6, (l, residual)


def test_sum_rule_detects_a_wrong_angular_factor():
    """Mis-declaring the harmonic convention breaks the sum rule.

    This is the guard the old code lacked: with A_l = 4*pi/(2l+1) applied
    to a unit-normalised real spherical basis, every l > 0 norm is wrong by
    exactly that factor, so the convention cannot be assumed silently.
    """
    ref = check_normalisation(make_spherical_basis(HARMONICS_SPHERICAL))
    solid = make_spherical_basis(HARMONICS_SOLID)
    got = check_normalisation(solid)
    for l in sorted(set(np.asarray(solid.l).tolist())):
        sel = np.asarray(solid.l) == l
        ratio = float(got[sel][0] / ref[sel][0])
        assert ratio == pytest.approx(4 * np.pi / (2 * l + 1), rel=1e-12)
    assert np.max(np.abs(got - 1.0)) > 0.5


def test_missing_two_pi_cubed_would_break_the_sum_rule():
    """The (2 pi)^-3 prefactor is required, not cosmetic (REVIEW.md [B-2])."""
    basis = make_spherical_basis()
    norms = check_normalisation(basis)
    assert np.allclose(norms, 1.0, atol=1e-12)
    # The old convention (no prefactor) would be larger by (2 pi)^3 ~ 248.
    assert TWO_PI_CUBED == pytest.approx(248.05021, rel=1e-5)


# --- p_shake correctness ---------------------------------------------------

def test_p_shake_matches_bessel_transform_for_spherical_aos():
    """Closed-form envelope vs an independent radial Bessel transform."""
    basis = make_spherical_basis()
    kgrid = np.array([0.3, 1.0, 2.5, 5.0])
    got = p_shake(basis, kgrid)
    seen_l = set()
    for mu in range(basis.nbas):
        l = int(basis.l[mu])
        if l in seen_l:
            continue                      # one AO per l suffices
        seen_l.add(l)
        alpha = np.asarray(basis.alpha[mu])
        weight = (np.asarray(basis.coeff[mu])
                  * np.asarray(basis.norm[mu]))
        for iq, k in enumerate(kgrid):
            amp = _radial_bessel_amplitude(alpha, weight, l, k)
            want = amp ** 2 / TWO_PI_CUBED
            assert got[iq, mu] == pytest.approx(want, rel=1e-8), (l, k)


def test_p_shake_vectorised_equals_scalar_loop():
    """Grid evaluation reproduces per-point evaluation exactly."""
    for basis in (make_spherical_basis(), make_cartesian_basis()):
        kgrid = np.array([0.0, 0.21, 0.8, 1.7, 4.0, 9.0])
        block = p_shake(basis, kgrid)
        loop = np.array([p_shake(basis, float(k)) for k in kgrid])
        assert block.shape == (kgrid.size, basis.nbas)
        assert np.array_equal(block, loop)


def test_p_shake_shapes_and_domain():
    basis = make_spherical_basis()
    assert p_shake(basis, 1.0).shape == (basis.nbas,)
    assert p_shake(basis, np.zeros(3)).shape == (3, basis.nbas)
    assert p_shake(basis, np.zeros((2, 3))).shape == (2, 3, basis.nbas)
    with pytest.raises(ConfigError, match="k"):
        p_shake(basis, np.array([-1.0]))


def test_p_shake_is_non_negative_and_peaks_at_k_zero_for_s():
    """An s AO's momentum density is maximal at k = 0 and finite there."""
    basis = make_spherical_basis()
    k = np.linspace(0.0, 8.0, 200)
    vals = p_shake(basis, k)
    assert np.all(vals >= 0.0)
    s_cols = np.nonzero(np.asarray(basis.l) == 0)[0]
    for mu in s_cols:
        assert np.isfinite(vals[0, mu]) and vals[0, mu] > 0.0
        assert np.argmax(vals[:, mu]) == 0
    # every l > 0 AO vanishes at k = 0 as k^(2l)
    for mu in np.nonzero(np.asarray(basis.l) > 0)[0]:
        assert vals[0, mu] == 0.0


def test_high_k_asymptotics_is_gaussian_in_the_tightest_exponent():
    """P(k) ~ k^(2l) exp(-k^2 / (2 alpha_max)) at large k.

    The slowest-decaying primitive dominates the tail, so
    -d ln P / d(k^2) tends to 1/(2 alpha_max).
    """
    basis = make_spherical_basis()
    for mu in range(basis.nbas):
        l = int(basis.l[mu])
        alpha_max = float(np.max(basis.alpha[mu]))
        # Probe where exp(-k^2/(4 alpha_max)) ~ 1e-13: far enough into the
        # tail that the tightest primitive is negligible, near enough that
        # the value has not underflowed to zero in float64.
        k0 = np.sqrt(4.0 * alpha_max * 30.0)
        k = np.array([k0, k0 * 1.01])
        p = p_shake(basis, k)[:, mu]
        assert np.all(p > 0.0), (mu, p)
        # strip the k^(2l) prefactor before measuring the Gaussian rate
        lg = np.log(p / k ** (2 * l))
        rate = -(lg[1] - lg[0]) / (k[1] ** 2 - k[0] ** 2)
        assert rate == pytest.approx(1.0 / (2.0 * alpha_max), rel=1e-6)


def test_p_shake_decays_monotonically_past_its_peak():
    basis = make_spherical_basis()
    k = np.linspace(3.0, 12.0, 60)
    vals = p_shake(basis, k)
    diffs = np.diff(vals, axis=0)
    # Underflowed columns give exact zeros, which carry no information
    # about monotonicity; test only the representable part.
    alive = vals[:-1] > 1e-300
    assert np.any(alive)
    assert np.all(diffs[alive] < 0.0)


# --- Cartesian branch and contact terms ------------------------------------

def test_cartesian_matches_brute_force_solid_angle_integral():
    """Full Cartesian angular algebra vs product-rule sphere quadrature."""
    basis = make_cartesian_basis()
    for k in (0.0, 0.35, 0.8, 2.0, 4.5):
        got = p_shake(basis, k)
        for mu in range(basis.nbas):
            want = _p_shake_brute_cartesian(basis, mu, k, n=72)
            scale = max(abs(want), 1e-14)
            assert abs(got[mu] - want) / scale < 1e-9, (mu, k)


def test_cartesian_diagonal_d_is_maximal_at_k_zero():
    """The contact term of REVIEW.md [A-2] is retained.

    The exact 1-D transform of x^2 exp(-alpha x^2) is
    [1/(2 alpha) - kx^2/(4 alpha^2)] sqrt(pi/alpha) exp(-kx^2/4 alpha);
    dropping the 1/(2 alpha) sends the k -> 0 amplitude of every d_xx,
    d_yy, d_zz to zero instead of its maximum.
    """
    basis = make_cartesian_basis()
    powers = np.asarray(basis.cart_powers)
    diag = [mu for mu in range(basis.nbas)
            if sorted(powers[mu]) == [0, 0, 2]]
    off = [mu for mu in range(basis.nbas)
           if sorted(powers[mu]) == [0, 1, 1]]
    assert len(diag) == 3 and len(off) == 3
    at_zero = p_shake(basis, 0.0)
    for mu in diag:
        assert at_zero[mu] > 0.0
    for mu in off:
        assert at_zero[mu] == 0.0
    # and the leading-Hermite-only form really would give zero there
    for mu in diag:
        assert angular_terms(tuple(powers[mu]))[0].k_power == 0


def test_contact_terms_cancel_for_a_spherical_d_function():
    """Assembling d(z^2) from Cartesian pieces reproduces the k^(2l) form.

    2*(1/2a) - (1/2a) - (1/2a) = 0, so the contact terms cancel
    identically for a real spherical harmonic.  This is the independent
    argument that the pure k^(2l) envelope is correct in the spherical
    branch (REVIEW.md [A-2]).
    """
    alpha = np.array([1.7])
    weight = np.array([1.0])
    pref = np.sqrt(5.0 / (16.0 * np.pi))
    for kvec in [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.5, 0.7, 1.1),
                 (2.0, -1.0, 0.4)]:
        kv = tuple(float(v) for v in kvec)
        assembled = pref * (
            2.0 * _cartesian_amplitude((0, 0, 2), alpha, weight, kv)
            - _cartesian_amplitude((2, 0, 0), alpha, weight, kv)
            - _cartesian_amplitude((0, 2, 0), alpha, weight, kv)
        )
        knorm = float(np.linalg.norm(kv))
        solid = pref * (2 * kv[2] ** 2 - kv[0] ** 2 - kv[1] ** 2)
        pure = (
            (np.pi / alpha[0]) ** 1.5 * (1.0 / (2.0 * alpha[0])) ** 2
            * solid * np.exp(-knorm ** 2 / (4.0 * alpha[0]))
        )
        assert abs(assembled) == pytest.approx(abs(pure), rel=1e-12,
                                               abs=1e-14)


def test_cartesian_requires_explicit_component_powers():
    """The (l, m) -> (a, b, c) map is never guessed (REVIEW.md [A-1])."""
    basis = make_cartesian_basis()
    basis.cart_powers = None
    with pytest.raises(ModelError, match="cart_powers"):
        p_shake(basis, 1.0)


def test_unknown_harmonics_convention_rejected():
    basis = make_spherical_basis()
    basis.harmonics = "cartesian-ish"
    with pytest.raises(ModelError, match="detected from"):
        p_shake(basis, 1.0)


def test_arbitrary_l_supported():
    """No hard-coded l <= 2 table: an l = 4 shell normalises to unity."""
    alpha = np.array([0.7, 0.2])
    coeff = np.array([0.55, 0.45])
    l = 4
    nrm = spherical_primitive_norm(alpha, l)
    from .basis_stub import normalise_contraction
    cc, nrm = normalise_contraction(alpha, coeff, l)
    basis = BasisStub(
        nbas=1, centers=np.array([0]), elements=["S"],
        l=np.array([l]), m=np.array([0]), shell_index=np.array([0]),
        harmonics=HARMONICS_SPHERICAL,
        alpha=[alpha], coeff=[cc], norm=[nrm],
    )
    assert check_normalisation(basis)[0] == pytest.approx(1.0, rel=1e-12)
    assert angular_terms((4, 0, 0))          # cartesian path also builds


# --- the integrated yield and its Jacobian ---------------------------------

def test_p_shake_integrated_matches_quad():
    """Gauss-Legendre in t vs quad on the unsubstituted energy integral.

    The substituted integrand is k2 * P * 2 * E * t; the trailing t was
    absent in the old code, which made it high by ~1.7x (REVIEW.md [B-1]).
    """
    basis = make_spherical_basis()
    for e_excess in (0.5, 2.0, 30.0):
        got = p_shake_integrated(basis, e_excess, n_quad=300)
        for mu in range(basis.nbas):
            ref = quad(
                lambda e2: (np.sqrt(2.0 * e2)
                            * p_shake(basis, np.sqrt(2.0 * e2))[mu]),
                0.0, e_excess, limit=400,
            )[0]
            assert got[mu] == pytest.approx(ref, rel=1e-8), (e_excess, mu)


def test_dropping_the_jacobian_t_is_detectable():
    """Reproduce the old bug and confirm it changes the answer materially.

    For a normalised s GTO at E = 2 a.u. REVIEW.md quotes reference
    4.9755e-01 against the erroneous 8.4649e-01.
    """
    basis = make_spherical_basis()
    e_excess = 2.0
    t, w = legendre.leggauss(400)
    t = 0.5 * (t + 1.0)
    w = 0.5 * w
    k2 = t * np.sqrt(2.0 * e_excess)
    p = p_shake(basis, k2)
    correct = np.einsum("q,qm->m", w * k2 * 2.0 * e_excess * t, p)
    dropped = np.einsum("q,qm->m", w * k2 * 2.0 * e_excess, p)
    got = p_shake_integrated(basis, e_excess, n_quad=400)
    assert np.allclose(got, correct, rtol=1e-10)
    assert not np.allclose(got, dropped, rtol=1e-2)
    ratio = float(np.mean(dropped / correct))
    assert 1.3 < ratio < 2.2


def test_integrated_yield_tends_to_unity():
    """The integral saturates the sum rule as E -> infinity."""
    basis = make_spherical_basis()
    previous = np.zeros(basis.nbas)
    for e_excess in (5.0, 50.0, 500.0, 5000.0):
        got = p_shake_integrated(basis, e_excess, n_quad=600)
        assert np.all(got >= previous - 1e-12)     # monotone in E
        assert np.all(got <= 1.0 + 1e-9)
        previous = got
    assert np.max(np.abs(previous - 1.0)) < 1e-8


def test_integrated_quadrature_is_converged_at_the_default():
    """n_quad = 200 already agrees with a much finer grid."""
    basis = make_spherical_basis()
    coarse = p_shake_integrated(basis, 20.0)
    fine = p_shake_integrated(basis, 20.0, n_quad=1200)
    assert np.max(np.abs(coarse - fine)) < 1e-8


def test_integrated_rejects_bad_arguments():
    basis = make_spherical_basis()
    with pytest.raises(ConfigError, match="positive"):
        p_shake_integrated(basis, 0.0)
    with pytest.raises(ConfigError, match="n_quad"):
        p_shake_integrated(basis, 1.0, n_quad=1)


# --- basis validation ------------------------------------------------------

def test_ragged_primitive_arrays_are_padded_not_truncated():
    """AOs with different primitive counts coexist in one basis."""
    basis = make_spherical_basis()
    counts = {len(a) for a in basis.alpha}
    assert len(counts) > 1
    assert np.max(np.abs(check_normalisation(basis) - 1.0)) < 1e-12


def test_inconsistent_primitive_lengths_rejected():
    basis = make_spherical_basis()
    basis.coeff[0] = np.asarray(basis.coeff[0])[:-1]
    with pytest.raises(ConfigError, match="disagree in length"):
        p_shake(basis, 1.0)


def test_non_positive_exponent_rejected():
    basis = make_spherical_basis()
    basis.alpha[0] = np.array([0.0, 1.0, 2.0])
    basis.coeff[0] = np.ones(3)
    basis.norm[0] = np.ones(3)
    with pytest.raises(ConfigError, match="non-positive"):
        p_shake(basis, 1.0)


def test_missing_field_rejected():
    class Incomplete:
        nbas = 1
        l = np.array([0])

    with pytest.raises(ConfigError, match="harmonics"):
        p_shake(Incomplete(), 1.0)
