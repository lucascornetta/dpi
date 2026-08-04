"""A minimal BasisSet stand-in for testing atomic_sigma and shakeoff.

``dpi.molcas_io`` owns the real reader; this stub exposes only the fields
that the two modules under test consume, so the tests are self-contained
and do not need an OpenMolcas HDF5 file.

Fields provided (SPEC.md section 2)
-----------------------------------
``nbas``        number of contracted AOs
``centers``     0-based atom index per AO
``elements``    element symbol per AO
``l``           angular momentum per AO
``m``           component index within the shell, per AO
``shell_index`` sequential index of the contracted shell within its
                ``(center, l)`` group
``harmonics``   ``'spherical'`` | ``'solid'`` | ``'cartesian'``
``alpha``       per AO, the primitive Gaussian exponents
``coeff``       per AO, the contraction coefficients
``norm``        per AO, the primitive normalisation constants
``cart_powers`` (Cartesian bases only) ``(nbas, 3)`` exponents (a, b, c)

Consumed by ``atomic_sigma.SigmaBuilder``: ``nbas``, ``elements``, ``l``,
``shell_index``.  Consumed by ``dpi.shakeoff``: ``nbas``, ``l``,
``harmonics``, ``alpha``, ``coeff``, ``norm``, ``cart_powers``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import gamma


@dataclass
class BasisStub:
    """Duck-typed BasisSet with per-AO primitive data."""

    nbas: int
    centers: np.ndarray
    elements: list[str]
    l: np.ndarray
    m: np.ndarray
    shell_index: np.ndarray
    harmonics: str
    alpha: list[np.ndarray]
    coeff: list[np.ndarray]
    norm: list[np.ndarray]
    cart_powers: np.ndarray | None = None


def spherical_primitive_norm(alpha: np.ndarray, l: int) -> np.ndarray:
    """Normalisation of a single spherical GTO primitive.

    Returns ``N`` such that ``int |N r^l exp(-alpha r^2)|^2 r^2 dr = 1``,
    i.e. the radial part is unit-normalised and the angular part is a
    unit-normalised real spherical harmonic.

    Parameters
    ----------
    alpha : ndarray
        Gaussian exponents, atomic units.
    l : int
        Angular momentum.

    Returns
    -------
    ndarray
        Primitive normalisation constants.
    """
    alpha = np.asarray(alpha, dtype=float)
    return np.sqrt(2.0 * (2.0 * alpha) ** (l + 1.5) / gamma(l + 1.5))


def normalise_contraction(alpha: np.ndarray, coeff: np.ndarray,
                          l: int) -> tuple[np.ndarray, np.ndarray]:
    """Rescale contraction coefficients so the contracted AO has norm 1.

    Parameters
    ----------
    alpha, coeff : ndarray
        Primitive exponents and raw contraction coefficients.
    l : int
        Angular momentum.

    Returns
    -------
    coeff : ndarray
        Rescaled coefficients.
    norm : ndarray
        Primitive normalisation constants.
    """
    alpha = np.asarray(alpha, dtype=float)
    coeff = np.asarray(coeff, dtype=float)
    nrm = spherical_primitive_norm(alpha, l)
    a = alpha[:, None] + alpha[None, :]
    w = (coeff * nrm)[:, None] * (coeff * nrm)[None, :]
    overlap = float(np.sum(w * gamma(l + 1.5) / (2.0 * a ** (l + 1.5))))
    return coeff / np.sqrt(overlap), nrm


# A small contracted set with s, p, d and f shells, loosely modelled on
# cc-pVDZ contraction patterns.  Exponents are arbitrary but realistic;
# nothing in the tests depends on their values.
_SHELLS = [
    (0, [13.01, 1.962, 0.4446], [0.019685, 0.137977, 0.478148]),
    (0, [0.1220], [1.0]),
    (1, [9.439, 2.002, 0.5456], [0.038090, 0.209480, 0.508440]),
    (1, [0.1517], [1.0]),
    (2, [1.7000], [1.0]),
    (3, [0.8000, 0.2500], [0.6, 0.5]),
]


def make_spherical_basis(harmonics: str = "spherical") -> BasisStub:
    """Synthetic normalised spherical basis with l = 0, 1, 2, 3 shells.

    Parameters
    ----------
    harmonics : str
        ``'spherical'`` (unit-normalised real ``Y_lm``) or ``'solid'``.

    Returns
    -------
    BasisStub
    """
    centers, elements, ls, ms, shell_idx = [], [], [], [], []
    alphas, coeffs, norms = [], [], []
    per_l_count: dict[int, int] = {}
    for l, a, c in _SHELLS:
        a = np.asarray(a, dtype=float)
        cc, nrm = normalise_contraction(a, np.asarray(c, dtype=float), l)
        idx = per_l_count.get(l, 0)
        per_l_count[l] = idx + 1
        for m in range(2 * l + 1):
            centers.append(0)
            elements.append("S")
            ls.append(l)
            ms.append(m)
            shell_idx.append(idx)
            alphas.append(a.copy())
            coeffs.append(cc.copy())
            norms.append(nrm.copy())
    n = len(ls)
    return BasisStub(
        nbas=n,
        centers=np.asarray(centers, dtype=int),
        elements=elements,
        l=np.asarray(ls, dtype=int),
        m=np.asarray(ms, dtype=int),
        shell_index=np.asarray(shell_idx, dtype=int),
        harmonics=harmonics,
        alpha=alphas, coeff=coeffs, norm=norms,
    )


_CART_D = [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1),
           (0, 0, 2)]


def make_cartesian_basis() -> BasisStub:
    """Synthetic Cartesian basis: one s, one p set, one full d set.

    The primitive normalisations are *not* the Cartesian ones, so the
    momentum norms of this basis are not unity; it exists to exercise the
    Cartesian angular algebra (contact terms), not the normalisation.

    Returns
    -------
    BasisStub
    """
    shells = [
        (0, [(0, 0, 0)], np.array([1.7, 0.4]), np.array([0.6, 0.5])),
        (1, [(1, 0, 0), (0, 1, 0), (0, 0, 1)], np.array([0.9]),
         np.array([1.0])),
        (2, _CART_D, np.array([1.7, 0.4]), np.array([0.6, 0.5])),
    ]
    centers, elements, ls, ms, shell_idx = [], [], [], [], []
    alphas, coeffs, norms, powers = [], [], [], []
    per_l_count: dict[int, int] = {}
    for l, comps, a, c in shells:
        idx = per_l_count.get(l, 0)
        per_l_count[l] = idx + 1
        for m, p in enumerate(comps):
            centers.append(0)
            elements.append("S")
            ls.append(l)
            ms.append(m)
            shell_idx.append(idx)
            alphas.append(a.copy())
            coeffs.append(c.copy())
            norms.append(np.ones_like(a))
            powers.append(p)
    n = len(ls)
    return BasisStub(
        nbas=n,
        centers=np.asarray(centers, dtype=int),
        elements=elements,
        l=np.asarray(ls, dtype=int),
        m=np.asarray(ms, dtype=int),
        shell_index=np.asarray(shell_idx, dtype=int),
        harmonics="cartesian",
        alpha=alphas, coeff=coeffs, norm=norms,
        cart_powers=np.asarray(powers, dtype=int),
    )


def make_sf6_like_basis() -> BasisStub:
    """AO layout of one S plus six F atoms in cc-pVDZ (nbas = 102).

    Only the fields ``SigmaBuilder`` consumes are meaningful here; the
    primitive data is a placeholder single exponent per AO.

    Returns
    -------
    BasisStub
    """
    # cc-pVDZ contracted shells: S has 4s3p1d, F has 3s2p1d.
    layout = [("S", [(0, 4), (1, 3), (2, 1)])]
    layout += [("F", [(0, 3), (1, 2), (2, 1)])] * 6

    centers, elements, ls, ms, shell_idx = [], [], [], [], []
    alphas, coeffs, norms = [], [], []
    for atom, shells in enumerate(layout):
        elem, groups = shells
        for l, nshell in groups:
            for idx in range(nshell):
                for m in range(2 * l + 1):
                    centers.append(atom)
                    elements.append(elem)
                    ls.append(l)
                    ms.append(m)
                    shell_idx.append(idx)
                    alphas.append(np.array([1.0]))
                    coeffs.append(np.array([1.0]))
                    norms.append(spherical_primitive_norm(
                        np.array([1.0]), l))
    n = len(ls)
    return BasisStub(
        nbas=n,
        centers=np.asarray(centers, dtype=int),
        elements=elements,
        l=np.asarray(ls, dtype=int),
        m=np.asarray(ms, dtype=int),
        shell_index=np.asarray(shell_idx, dtype=int),
        harmonics="spherical",
        alpha=alphas, coeff=coeffs, norm=norms,
    )
