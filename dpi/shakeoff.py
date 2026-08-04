"""Shake-off probability densities from GTO Fourier transforms.

The sudden change of mean field on removal of a core electron projects a
bound electron in AO :math:`\\chi_\\nu` onto the continuum.  With plane-wave
final states, the differential probability for it to emerge with momentum
:math:`k` is the spherically averaged momentum density of that AO.

Convention
----------
The Fourier transform is

.. math:: \\tilde\\chi_\\nu(\\mathbf{k})
          = \\int d^3r\\, e^{-i\\mathbf{k}\\cdot\\mathbf{r}}
            \\chi_\\nu(\\mathbf{r})

for which Parseval reads
:math:`\\int d^3k\\,(2\\pi)^{-3} |\\tilde\\chi|^2 = \\int d^3r |\\chi|^2`.  The
shake-off density is defined (REVIEW.md [B-2]) as the solid-angle
**integral**, not the average:

.. math:: P_\\nu(k) = (2\\pi)^{-3} \\int d\\Omega_k\\,
          |\\tilde\\chi_\\nu(\\mathbf{k})|^2 .

Only with the integral -- and only with the :math:`(2\\pi)^{-3}` present --
does the sum rule

.. math:: \\int_0^\\infty k\\, P_\\nu(k)\\, d\\varepsilon
          = \\int_0^\\infty k^2 P_\\nu(k)\\, dk
          = \\langle\\chi_\\nu|\\chi_\\nu\\rangle = 1

hold for a normalised AO, which is what makes :func:`check_normalisation`
a genuine unit test rather than a self-consistency check.  The previous
implementation omitted :math:`(2\\pi)^{-3}` entirely, so its output was
larger than the notes' quantity by :math:`(2\\pi)^3 \\approx 248`; since
that is a constant it cancelled in every ratio, but it means the old
output was not on an absolute scale.

Angular factors
---------------
For a contracted GTO the transform separates into a radial envelope times
an angular factor.  Writing :math:`s_j = 2\\sqrt{\\alpha_j}` and
:math:`w_j = d_j N_j`:

**Real spherical harmonics.**  With unit-normalised real
:math:`Y_{lm}` (so :math:`\\int d\\Omega |Y_{lm}|^2 = 1`),

.. math:: \\tilde\\chi_\\nu(\\mathbf{k}) = (-i)^l\\, Y_{lm}(\\hat k)\\, k^l\\,
          F_\\nu(k), \\qquad
          F_\\nu(k) = \\sum_j w_j \\left(\\frac{\\pi}{\\alpha_j}\\right)^{3/2}
          \\left(\\frac{1}{2\\alpha_j}\\right)^{l} e^{-k^2/4\\alpha_j}

so :math:`P_\\nu(k) = (2\\pi)^{-3} A_l k^{2l} |F_\\nu(k)|^2` with **one**
factor per :math:`l`,

.. math:: A_l = \\int d\\Omega |Y_{lm}(\\hat k)|^2 = 1 \\quad
          \\text{for every } l, m .

This is verified numerically in ``tests/test_shakeoff.py`` rather than
asserted.  The previous implementation split :math:`l = 2` into
:math:`4\\pi/5` for "diagonal" and :math:`4\\pi/15` for "off-diagonal"
components; those are the solid-angle averages of the *unnormalised*
Cartesian monomials :math:`n_z^4` and :math:`(n_x n_y)^2`, and the
splitting is meaningless for real spherical harmonics, which all share one
angular factor (REVIEW.md [A-1]).  If a basis instead carries Racah-
normalised solid harmonics, set ``harmonics='solid'``, for which
:math:`A_l = 4\\pi/(2l+1)`; :func:`check_normalisation` detects a wrong
choice immediately, because the momentum norm then misses unity by exactly
that ratio.

**Cartesian components.**  The pure :math:`k^{2l}` form is *wrong* for
diagonal Cartesian functions (REVIEW.md [A-2]).  The exact 1-D transform of
:math:`x^2 e^{-\\alpha x^2}` is

.. math:: \\left[\\frac{1}{2\\alpha} - \\frac{k_x^2}{4\\alpha^2}\\right]
          \\sqrt{\\pi/\\alpha}\\; e^{-k_x^2/4\\alpha},

and dropping the :math:`1/(2\\alpha)` "contact" term sends the
:math:`k \\to 0` amplitude of every :math:`d_{xx}, d_{yy}, d_{zz}` to zero
instead of its maximum.  This module therefore builds the Cartesian branch
from the exact Hermite-polynomial transform,

.. math:: \\tilde\\chi(\\mathbf{k}) = \\sum_j w_j (\\pi/\\alpha_j)^{3/2}
          s_j^{-l} H_a(k_x/s_j) H_b(k_y/s_j) H_c(k_z/s_j)
          e^{-k^2/4\\alpha_j}

(up to the overall :math:`(-i)^l`, which cancels in
:math:`|\\tilde\\chi|^2`), and performs the solid-angle integral exactly
term by term using
:math:`\\int d\\Omega\\, n_x^p n_y^q n_z^r =
4\\pi (p-1)!!(q-1)!!(r-1)!! / (p+q+r+1)!!` for even :math:`p,q,r` and zero
otherwise.  For a real spherical harmonic assembled from its Cartesian
components the contact terms cancel identically, which is an independent
argument that the pure :math:`k^{2l}` form is right in the spherical
branch; ``tests/test_shakeoff.py`` demonstrates that cancellation.

Both branches are implemented for arbitrary :math:`l`; there is no
hard-coded :math:`l \\le 2` table.

Units
-----
:math:`k` and all energies are in atomic units.  :math:`P_\\nu(k)` has
dimensions of (a.u.)\\ :sup:`-3`, i.e. inverse momentum cubed, so that
:math:`k P_\\nu` integrated over an energy is dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.polynomial import hermite
from scipy.special import gamma as _gamma

from .constants import ConfigError, ModelError

__all__ = [
    "HARMONICS_SPHERICAL",
    "HARMONICS_SOLID",
    "HARMONICS_CARTESIAN",
    "angular_factor",
    "p_shake",
    "p_shake_integrated",
    "check_normalisation",
    "AngularTerm",
    "angular_terms",
]

HARMONICS_SPHERICAL = "spherical"
HARMONICS_SOLID = "solid"
HARMONICS_CARTESIAN = "cartesian"

_TWO_PI_CUBED = (2.0 * np.pi) ** 3


def _double_factorial(n: int) -> float:
    """``n!!`` with the standard ``(-1)!! = 1``.

    ``scipy.special.factorial2`` returns 0 for negative arguments, which
    would silently zero every :math:`\\int d\\Omega\\, n_x^p n_y^q n_z^r`
    that has an exponent equal to zero -- that is, almost all of them.
    """
    if n <= 0:
        return 1.0
    out = 1.0
    while n > 1:
        out *= float(n)
        n -= 2
    return out


def _omega_moment(p: int, q: int, r: int) -> float:
    """:math:`\\int d\\Omega\\, n_x^p n_y^q n_z^r` on the unit sphere.

    Zero unless all three exponents are even.

    Parameters
    ----------
    p, q, r : int
        Exponents of the Cartesian direction cosines.

    Returns
    -------
    float
        The solid-angle integral, dimensionless.
    """
    if p % 2 or q % 2 or r % 2:
        return 0.0
    return (
        4.0 * np.pi
        * _double_factorial(p - 1)
        * _double_factorial(q - 1)
        * _double_factorial(r - 1)
        / _double_factorial(p + q + r + 1)
    )


def angular_factor(l: int, harmonics: str = HARMONICS_SPHERICAL) -> float:
    """Angular factor :math:`A_l` of the spherical-type branches.

    Parameters
    ----------
    l : int
        Angular momentum.
    harmonics : str
        ``'spherical'`` for unit-normalised real :math:`Y_{lm}`
        (:math:`A_l = 1`), or ``'solid'`` for Racah-normalised solid
        harmonics (:math:`A_l = 4\\pi/(2l+1)`).

    Returns
    -------
    float
        :math:`A_l`, dimensionless.
    """
    if l < 0:
        raise ConfigError(f"angular momentum must be >= 0, got l={l}")
    if harmonics == HARMONICS_SPHERICAL:
        return 1.0
    if harmonics == HARMONICS_SOLID:
        return 4.0 * np.pi / (2.0 * l + 1.0)
    raise ModelError(
        f"angular_factor is defined for harmonics "
        f"{HARMONICS_SPHERICAL!r} and {HARMONICS_SOLID!r}, not "
        f"{harmonics!r}; the Cartesian branch needs the full term "
        "expansion of angular_terms()"
    )


@dataclass(frozen=True)
class AngularTerm:
    """One term of the exact solid-angle integral of a Cartesian GTO.

    The integral is a finite sum
    :math:`\\sum_T M_T k^{P_T} G_{q_1^T}(k) G_{q_2^T}(k)` where

    .. math:: G_q(k) = \\sum_j w_j (\\pi/\\alpha_j)^{3/2} s_j^{-l-q}
              e^{-k^2/4\\alpha_j}.

    Because the term factorises into two independent envelope sums, no
    primitive-pair array is ever formed: cost is
    ``nterms * nk * nprim``, not ``nterms * nk * nprim**2``.

    Attributes
    ----------
    moment : float
        :math:`M_T`, the solid-angle moment times the Hermite coefficients.
    q_left, q_right : int
        Extra powers of :math:`s_j^{-1}` carried by the two envelopes.
    k_power : int
        :math:`P_T`, the power of :math:`k`.
    """

    moment: float
    q_left: int
    q_right: int
    k_power: int


@lru_cache(maxsize=None)
def _hermite_coeffs(n: int) -> tuple[float, ...]:
    """Coefficients of the physicists' :math:`H_n`, ascending powers."""
    c = np.zeros(n + 1)
    c[n] = 1.0
    return tuple(float(v) for v in hermite.herm2poly(c))


@lru_cache(maxsize=None)
def angular_terms(powers: tuple[int, int, int]) -> tuple[AngularTerm, ...]:
    """Exact solid-angle expansion for one Cartesian component.

    Expands
    :math:`\\int d\\Omega_k |H_a(k_x/s)H_b(k_y/s)H_c(k_z/s)|^2` into the
    factorised form of :class:`AngularTerm`, retaining the contact terms
    that the previous implementation dropped (REVIEW.md [A-2]).

    Parameters
    ----------
    powers : tuple of int
        Cartesian exponents ``(a, b, c)`` of the AO,
        :math:`x^a y^b z^c e^{-\\alpha r^2}`.

    Returns
    -------
    tuple of AngularTerm
        Terms with a non-vanishing solid-angle moment.
    """
    a, b, c = (int(v) for v in powers)
    if min(a, b, c) < 0:
        raise ConfigError(f"Cartesian powers must be >= 0, got {powers}")
    ca, cb, cc = (_hermite_coeffs(a), _hermite_coeffs(b),
                  _hermite_coeffs(c))

    merged: dict[tuple[int, int, int], float] = {}
    for i1, v1 in enumerate(ca):
        if v1 == 0.0:
            continue
        for i2, v2 in enumerate(ca):
            if v2 == 0.0:
                continue
            for j1, u1 in enumerate(cb):
                if u1 == 0.0:
                    continue
                for j2, u2 in enumerate(cb):
                    if u2 == 0.0:
                        continue
                    for k1, t1 in enumerate(cc):
                        if t1 == 0.0:
                            continue
                        for k2, t2 in enumerate(cc):
                            if t2 == 0.0:
                                continue
                            px, py, pz = i1 + i2, j1 + j2, k1 + k2
                            mom = _omega_moment(px, py, pz)
                            if mom == 0.0:
                                continue
                            key = (i1 + j1 + k1, i2 + j2 + k2,
                                   px + py + pz)
                            merged[key] = merged.get(key, 0.0) + (
                                v1 * v2 * u1 * u2 * t1 * t2 * mom
                            )
    return tuple(
        AngularTerm(moment=m, q_left=q1, q_right=q2, k_power=p)
        for (q1, q2, p), m in sorted(merged.items()) if m != 0.0
    )


@dataclass(frozen=True)
class _Prepared:
    """Padded primitive data and per-AO angular expansion, built once."""

    nbas: int
    harmonics: str
    alpha: np.ndarray          # (nbas, npmax), 1.0 in pad slots
    weight: np.ndarray         # (nbas, npmax) = coeff * norm, 0 in pad
    l: np.ndarray              # (nbas,)
    terms: tuple[tuple[AngularTerm, ...], ...]
    qmax: int


def _prepare(basis) -> _Prepared:
    """Validate a BasisSet and pad its ragged primitive arrays.

    Consumes ``nbas``, ``l``, ``harmonics``, ``alpha``, ``coeff``,
    ``norm``, and -- for the Cartesian branch only -- ``cart_powers``.
    Padded slots carry ``alpha = 1`` (never zero: it appears in
    denominators) and ``weight = 0``, so they contribute nothing.
    """
    for attr in ("nbas", "l", "harmonics", "alpha", "coeff", "norm"):
        if not hasattr(basis, attr):
            raise ConfigError(
                f"BasisSet is missing the {attr!r} field required by "
                "dpi.shakeoff (SPEC.md section 2)"
            )
    harmonics = str(basis.harmonics)
    known = (HARMONICS_SPHERICAL, HARMONICS_SOLID, HARMONICS_CARTESIAN)
    if harmonics not in known:
        raise ModelError(
            f"BasisSet.harmonics is {harmonics!r}; dpi.shakeoff "
            f"implements {known}.  The convention must be detected from "
            "the file, not assumed (REVIEW.md [A-1])"
        )

    nbas = int(basis.nbas)
    l_arr = np.asarray(basis.l, dtype=int)
    if l_arr.size != nbas:
        raise ConfigError(
            f"BasisSet.l has {l_arr.size} entries but nbas is {nbas}"
        )
    if np.any(l_arr < 0):
        raise ConfigError("BasisSet.l contains a negative angular momentum")

    alphas = [np.atleast_1d(np.asarray(a, dtype=float))
              for a in basis.alpha]
    coeffs = [np.atleast_1d(np.asarray(c, dtype=float))
              for c in basis.coeff]
    norms = [np.atleast_1d(np.asarray(n, dtype=float))
             for n in basis.norm]
    for name, seq in (("alpha", alphas), ("coeff", coeffs),
                      ("norm", norms)):
        if len(seq) != nbas:
            raise ConfigError(
                f"BasisSet.{name} has {len(seq)} entries but nbas is "
                f"{nbas}"
            )
    for mu in range(nbas):
        n_a, n_c, n_n = alphas[mu].size, coeffs[mu].size, norms[mu].size
        if not (n_a == n_c == n_n):
            raise ConfigError(
                f"AO {mu}: primitive arrays disagree in length "
                f"(alpha {n_a}, coeff {n_c}, norm {n_n})"
            )
        if n_a == 0:
            raise ConfigError(f"AO {mu} has no primitives")
        if np.any(alphas[mu] <= 0.0):
            raise ConfigError(
                f"AO {mu} has a non-positive Gaussian exponent "
                f"{alphas[mu].min()}"
            )

    npmax = max(a.size for a in alphas)
    alpha_pad = np.ones((nbas, npmax), dtype=float)
    weight_pad = np.zeros((nbas, npmax), dtype=float)
    for mu in range(nbas):
        n = alphas[mu].size
        alpha_pad[mu, :n] = alphas[mu]
        weight_pad[mu, :n] = coeffs[mu] * norms[mu]

    if harmonics == HARMONICS_CARTESIAN:
        # A spherical-harmonic reader may legitimately carry the field as
        # None rather than omit it, so absent and None must behave alike.
        if getattr(basis, "cart_powers", None) is None:
            raise ModelError(
                "harmonics='cartesian' requires BasisSet.cart_powers, a "
                "(nbas, 3) array of Cartesian exponents (a, b, c) per AO. "
                "dpi.shakeoff will not guess the component ordering from "
                "(l, m): mis-assigning it is exactly the failure mode of "
                "REVIEW.md [A-1], where a missing d-component map "
                "silently gave s-function normalisations to two AOs"
            )
        cart = np.asarray(basis.cart_powers, dtype=int)
        if cart.shape != (nbas, 3):
            raise ConfigError(
                f"BasisSet.cart_powers has shape {cart.shape}, expected "
                f"({nbas}, 3)"
            )
        bad = np.nonzero(cart.sum(axis=1) != l_arr)[0]
        if bad.size:
            mu = int(bad[0])
            raise ConfigError(
                f"AO {mu}: cart_powers {tuple(cart[mu])} sum to "
                f"{int(cart[mu].sum())} but l is {int(l_arr[mu])}"
            )
        terms = tuple(angular_terms(tuple(int(v) for v in cart[mu]))
                      for mu in range(nbas))
    else:
        # A spherical harmonic is exactly the *leading* Hermite term of the
        # Cartesian expansion: the contact terms cancel identically (see
        # the module docstring and REVIEW.md [A-2]).  So its single term
        # has the same envelope powers q = l as the Cartesian leading term
        # and carries the leading Hermite coefficient 2^l on each side,
        # 4^l in the product, with the solid-angle moment replaced by A_l.
        # That reproduces the (1/(2 alpha))^l = 2^l s^(-2l) of the envelope
        # F_nu(k) quoted in the module docstring.
        terms = tuple(
            (AngularTerm(
                moment=(angular_factor(int(l_arr[mu]), harmonics)
                        * 4.0 ** int(l_arr[mu])),
                q_left=int(l_arr[mu]), q_right=int(l_arr[mu]),
                k_power=2 * int(l_arr[mu]),
            ),)
            for mu in range(nbas)
        )

    qmax = max((max(t.q_left, t.q_right) for ts in terms for t in ts),
               default=0)
    return _Prepared(
        nbas=nbas, harmonics=harmonics, alpha=alpha_pad,
        weight=weight_pad, l=l_arr, terms=terms, qmax=qmax,
    )


def _envelopes(prep: _Prepared, k: np.ndarray) -> np.ndarray:
    """Envelope sums :math:`G_q(k)` for every AO.

    Parameters
    ----------
    prep : _Prepared
    k : ndarray, shape (nk,)
        Momentum magnitudes, atomic units.

    Returns
    -------
    ndarray, shape (qmax+1, nk, nbas)
        :math:`G_q(k)` in Mb-free atomic units.
    """
    s = 2.0 * np.sqrt(prep.alpha)                      # (nbas, npmax)
    base = (prep.weight * (np.pi / prep.alpha) ** 1.5
            * s ** (-prep.l[:, None]))                 # (nbas, npmax)
    # exp(-k^2 / (4 alpha)) for every (k, AO, primitive) at once.
    expo = np.exp(-k[:, None, None] ** 2
                  / (4.0 * prep.alpha)[None, :, :])    # (nk, nbas, npmax)
    out = np.empty((prep.qmax + 1, k.size, prep.nbas), dtype=float)
    for q in range(prep.qmax + 1):
        out[q] = np.einsum(
            "kmp,mp->km", expo, base * s ** (-q), optimize=True
        )
    return out


def p_shake(basis, k: float | np.ndarray) -> np.ndarray:
    """Differential shake-off probability density per AO.

    Implements
    :math:`P_\\nu(k) = (2\\pi)^{-3}\\int d\\Omega_k |\\tilde\\chi_\\nu(k)|^2`
    -- the solid-angle integral, per REVIEW.md [B-2] -- exactly, for real
    spherical, Racah solid, or Cartesian GTOs, at arbitrary :math:`l`.
    Vectorised over both AOs and :math:`k`: the driver calls it once with
    the whole quadrature grid, replacing the previous implementation's
    Python loop over AOs inside a loop over quadrature points
    (REVIEW.md [C-4]).

    Parameters
    ----------
    basis : BasisSet
        Uses ``nbas``, ``l``, ``harmonics``, ``alpha``, ``coeff``,
        ``norm``, and ``cart_powers`` when ``harmonics == 'cartesian'``.
    k : float or ndarray
        Electron momentum magnitude, atomic units.  Must be
        non-negative.

    Returns
    -------
    ndarray
        Shape ``(nbas,)`` for scalar ``k``, ``(nk, nbas)`` for array
        ``k``.  Units: inverse momentum cubed, atomic units, such that
        :math:`\\int k^2 P_\\nu\\,dk = \\langle\\chi_\\nu|\\chi_\\nu\\rangle`.
    """
    prep = _prepare(basis)
    k_in = np.asarray(k, dtype=float)
    scalar = k_in.ndim == 0
    k_arr = np.atleast_1d(k_in).ravel()
    if np.any(k_arr < 0.0):
        raise ConfigError(
            f"p_shake needs |k| >= 0, got a minimum of {k_arr.min()}"
        )

    G = _envelopes(prep, k_arr)
    out = np.zeros((k_arr.size, prep.nbas), dtype=float)
    # k**0 is 1 even at k = 0, so the s-function value at threshold is
    # finite and no special case is needed there.
    kpow_cache: dict[int, np.ndarray] = {}
    for mu, terms in enumerate(prep.terms):
        acc = np.zeros(k_arr.size, dtype=float)
        for t in terms:
            kp = kpow_cache.get(t.k_power)
            if kp is None:
                kp = k_arr ** t.k_power
                kpow_cache[t.k_power] = kp
            acc += t.moment * kp * G[t.q_left, :, mu] * G[t.q_right, :, mu]
        out[:, mu] = acc
    out /= _TWO_PI_CUBED

    if scalar:
        return out[0]
    return out.reshape(k_in.shape + (prep.nbas,))


def p_shake_integrated(
    basis, e_excess_au: float, n_quad: int = 200
) -> np.ndarray:
    """Integrated shake-off yield per AO up to an excess energy.

    Evaluates

    .. math:: \\int_0^{E} k_2 P_\\nu(k_2)\\, d\\varepsilon_2, \\qquad
              k_2 = \\sqrt{2\\varepsilon_2}

    on the substitution used throughout the pipeline (SPEC.md section 7):
    :math:`\\varepsilon_2 = E t^2`, so :math:`k_2 = t\\sqrt{2E}` and
    :math:`d\\varepsilon_2 = 2Et\\,dt`, with Gauss--Legendre nodes in
    :math:`t` on :math:`[0, 1]`.  The substitution removes the integrable
    :math:`k_2 \\to 0` singularity.

    The trailing :math:`t` of the Jacobian is easy to lose -- the previous
    implementation did, and was high by ~1.7x (REVIEW.md [B-1]).  The
    integrand here is therefore written as one expression,
    ``k2 * P * 2 * E * t``, and ``tests/test_shakeoff.py`` checks it
    against ``scipy.integrate.quad`` on the unsubstituted integral.

    Parameters
    ----------
    basis : BasisSet
    e_excess_au : float
        Excess energy :math:`E = \\varepsilon_1 + \\varepsilon_2`, Hartree.
        Must be positive.
    n_quad : int
        Number of Gauss--Legendre nodes.

    Returns
    -------
    ndarray, shape (nbas,)
        Dimensionless integrated yield per AO.  Tends to
        :math:`\\langle\\chi_\\nu|\\chi_\\nu\\rangle = 1` as
        :math:`E \\to \\infty`.
    """
    e_excess = float(e_excess_au)
    if not e_excess > 0.0:
        raise ConfigError(
            f"e_excess_au must be positive, got {e_excess}"
        )
    if int(n_quad) < 2:
        raise ConfigError(f"n_quad must be >= 2, got {n_quad}")

    t, w = np.polynomial.legendre.leggauss(int(n_quad))
    t = 0.5 * (t + 1.0)
    w = 0.5 * w
    k2 = t * np.sqrt(2.0 * e_excess)
    p = p_shake(basis, k2)                       # (nq, nbas)
    jac = 2.0 * e_excess * t                     # d eps2 / dt
    return np.einsum("q,qm->m", w * k2 * jac, p, optimize=True)


def check_normalisation(basis) -> np.ndarray:
    """Momentum-space norm of every AO, analytically.

    Returns :math:`\\int_0^\\infty k^2 P_\\nu(k)\\, dk`, which by Parseval
    equals :math:`\\langle\\chi_\\nu|\\chi_\\nu\\rangle` and is therefore
    :math:`1` for a correctly normalised contracted AO.  Evaluated in
    closed form from
    :math:`\\int_0^\\infty k^{2+P} e^{-\\beta k^2} dk =
    \\Gamma(\\frac{P+3}{2}) / (2\\beta^{(P+3)/2})`, so it is independent of
    the quadrature used by :func:`p_shake_integrated` and can serve as its
    reference.  A departure from unity means the primitive normalisations,
    the contraction coefficients, or the assumed harmonic convention are
    inconsistent -- it is the test that catches a wrong ``harmonics``
    setting, which changes the result by exactly
    :math:`4\\pi/(2l+1)`.

    Unlike the rest of the module this forms the primitive-pair sum
    explicitly; it is a diagnostic called once, never inside a
    quadrature loop.

    Parameters
    ----------
    basis : BasisSet

    Returns
    -------
    ndarray, shape (nbas,)
        Dimensionless momentum-space norms, one per AO.
    """
    prep = _prepare(basis)
    s = 2.0 * np.sqrt(prep.alpha)
    base = (prep.weight * (np.pi / prep.alpha) ** 1.5
            * s ** (-prep.l[:, None]))
    inv4 = 1.0 / (4.0 * prep.alpha)
    beta = inv4[:, :, None] + inv4[:, None, :]        # (nbas, np, np)

    out = np.zeros(prep.nbas, dtype=float)
    for mu, terms in enumerate(prep.terms):
        acc = 0.0
        for t in terms:
            half = 0.5 * (t.k_power + 3.0)
            pair = np.outer(base[mu] * s[mu] ** (-t.q_left),
                            base[mu] * s[mu] ** (-t.q_right))
            acc += t.moment * _gamma(half) * float(
                np.sum(pair / beta[mu] ** half)
            ) / 2.0
        out[mu] = acc / _TWO_PI_CUBED
    return out
