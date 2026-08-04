"""Dyson amplitudes between a closed-shell neutral and a relaxed dication.

The neutral is RHF with ``n`` doubly occupied spatial orbitals
``phi^neu_q``; the core-valence dication is OSRHF with its **own** relaxed
set ``phi^dic_k``.  The two sets are not orthogonal,

.. code::

    Q[k, q] = <phi^dic_k | phi^neu_q>,

and it is precisely that non-orthogonality -- orbital relaxation -- that
makes the double-photoionization amplitude non-zero.  For orthogonal sets
``Q`` would be a rectangular slice of the identity, the amplitudes would
collapse to Kronecker deltas (:func:`frozen_objects`) and no
two-electron transition would be driven by a one-body operator.

Generalized Slater-Condon rules for determinants built from
non-orthogonal orbitals reduce the transition matrix elements to minors of
``Q``.  With 1-based orbital labels,

.. code::

    d^(q)_i    = (-1)^(n+q)      det(Q with row i and column q deleted)
    D^(qq')_ij = (-1)^(q+q'+1)   det(Q with rows i,j and columns q,q'
                                     deleted)

Since ``Q`` restricted to the alpha-occupied dication rows already has
row ``i`` absent, ``Q_a`` is ``(n-1) x n`` and the one-electron amplitude
is a plain cofactor vector of ``Q_a``; likewise the doubly-occupied block
``Q_ij`` is ``(n-2) x n`` and the two-electron amplitude is its matrix of
``2 x 2`` complementary minors.

**Cost.** Evaluated literally, one state needs ``n`` determinants of order
``n-1``, ``n(n-1)/2`` of order ``n-2`` and ``(n-1) n(n-1)/2`` more for the
indirect coefficients -- of order 20 000 dense determinants at ``n = 35``,
per state, per spin channel.  Two identities collapse each family to one
decomposition:

* the cofactor vector of an ``(n-1) x n`` matrix is parallel to its right
  null vector;
* the matrix of ``2 x 2`` complementary minors of an ``(n-2) x n`` matrix
  is parallel to ``u ^ v = outer(u, v) - outer(v, u)``, where ``u, v``
  span its two-dimensional right null space.

In both cases the proportionality constant has modulus
``prod(singular values)``, but its **sign is not fixed by the
decomposition**: the SVD chooses the phase of a null vector arbitrarily,
and for the wedge the phase of a two-dimensional null basis is an
arbitrary ``O(2)`` frame.  The relative signs of ``d_i``, ``d_j``,
``d2_ij`` and ``lam_i`` inside one state are physical -- they enter the
``C_ij`` interference term, which is *linear* in each -- so the gauge is
fixed here by evaluating exactly **one** reference minor per object as an
explicit determinant and matching, choosing the reference index at the
largest null-space component for numerical safety.  Consequently every
object belonging to one state must be produced by a single
:func:`build_state_objects` call, which is what shares the gauge.

Units: ``Q``, the coefficients and the overlap are dimensionless, so the
amplitudes are dimensionless; dipole integrals and hence
:func:`lambda_coefficients` are in bohr.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .constants import ModelError
from .molcas_io import HoleAssignment

__all__ = [
    "DysonObjects",
    "build_Q",
    "cofactor_vector",
    "minor_matrix",
    "dyson_amplitudes",
    "two_electron_amplitudes",
    "two_electron_amplitudes_ij",
    "det_s_beta",
    "lambda_coefficients",
    "to_ao",
    "spectroscopic_factor",
    "frozen_objects",
    "build_state_objects",
    "dyson_amplitudes_bruteforce",
    "two_electron_amplitudes_bruteforce",
    "lambda_coefficients_bruteforce",
]

_XYZ = ("x", "y", "z")


# ── Q matrices ──────────────────────────────────────────────────────────────

def build_Q(
    C_neu: np.ndarray,
    C_dic: np.ndarray,
    S_ao: np.ndarray,
    n_neu_occ: int,
    mo_indices: Sequence[int],
) -> np.ndarray:
    """Overlap block between selected dication orbitals and neutral ones.

    Parameters
    ----------
    C_neu : ndarray, shape (nbas, nmo)
        Neutral MO coefficients, dimensionless; column = MO.
    C_dic : ndarray, shape (nbas, nmo)
        Dication MO coefficients in the same AO basis, dimensionless.
    S_ao : ndarray, shape (nbas, nbas)
        AO overlap, dimensionless.
    n_neu_occ : int
        Number ``n`` of doubly occupied neutral spatial MOs; the columns
        of the result are neutral MOs ``0 .. n-1``.
    mo_indices : sequence of int
        0-based *dication* MO columns occupied in the spin sector being
        built, ascending.  Length ``n-1`` for a dication spin sector.

    Returns
    -------
    ndarray, shape (len(mo_indices), n_neu_occ)
        ``Q[k, q] = <phi^dic_{mo_indices[k]} | phi^neu_q>``.

    Notes
    -----
    Rows are dication orbitals and columns neutral orbitals; the matrix is
    not square and not symmetric, and transposing it silently produces a
    plausible-looking but wrong amplitude, so the orientation is asserted
    against ``n_neu_occ`` here rather than trusted downstream.
    """
    C_neu = np.asarray(C_neu, dtype=float)
    C_dic = np.asarray(C_dic, dtype=float)
    S_ao = np.asarray(S_ao, dtype=float)
    idx = np.asarray(mo_indices, dtype=int)
    if C_neu.ndim != 2 or C_dic.ndim != 2:
        raise ModelError(
            f"C_neu and C_dic must be 2-D (nbas, nmo); got "
            f"{C_neu.shape} and {C_dic.shape}"
        )
    nbas = C_neu.shape[0]
    if C_dic.shape[0] != nbas or S_ao.shape != (nbas, nbas):
        raise ModelError(
            f"AO dimensions disagree: C_neu {C_neu.shape}, C_dic "
            f"{C_dic.shape}, S_ao {S_ao.shape}"
        )
    if not 1 <= n_neu_occ <= C_neu.shape[1]:
        raise ModelError(
            f"n_neu_occ={n_neu_occ} outside 1..nmo={C_neu.shape[1]}"
        )
    if idx.size and (idx.min() < 0 or idx.max() >= C_dic.shape[1]):
        raise ModelError(
            f"mo_indices out of range 0..{C_dic.shape[1] - 1}: "
            f"{tuple(int(v) for v in idx)}"
        )
    if np.any(np.diff(idx) <= 0):
        raise ModelError(
            f"mo_indices must be strictly ascending, got "
            f"{tuple(int(v) for v in idx)}; the row order sets the sign of "
            f"every determinant built from Q"
        )
    return C_dic[:, idx].T @ S_ao @ C_neu[:, :n_neu_occ]


# ── null-space fast path ────────────────────────────────────────────────────

def _prod_singular(a: np.ndarray) -> np.ndarray:
    """Product of the singular values, batched over leading axes."""
    s = np.linalg.svd(a, compute_uv=False)
    return np.prod(s, axis=-1)


def cofactor_vector(
    A: np.ndarray, check: bool = True
) -> np.ndarray:
    """Signed cofactor vector of an ``(m, m+1)`` matrix, batched.

    Implements ``c[q] = (-1)^(m+1+q+1) det(A with column q deleted)`` for
    0-based ``q`` -- i.e. the ``(-1)^(n+q)`` rule of the module docstring
    written with 1-based ``n = m+1`` and ``q``.

    Parameters
    ----------
    A : ndarray, shape (..., m, m+1)
        One or a batch of matrices, dimensionless.
    check : bool
        Verify that the reference-matched scale agrees in modulus with
        ``prod(singular values)``.  This is the guard that catches a
        rank-deficient or badly scaled ``Q``, where the null vector is not
        unique and the fast path is meaningless.

    Returns
    -------
    ndarray, shape (..., m+1)

    Notes
    -----
    The right null vector of ``A`` is parallel to ``c``; the SVD supplies
    its direction but not its sign or scale, so both are fixed by
    evaluating one explicit determinant.  The reference column is the one
    with the largest ``|null component|``, which is at least
    ``1/sqrt(m+1)`` and therefore never a division by a small number.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim < 2:
        raise ModelError(f"cofactor_vector needs a 2-D matrix, got {A.shape}")
    m, ncol = A.shape[-2], A.shape[-1]
    if ncol != m + 1:
        raise ModelError(
            f"cofactor_vector expects shape (..., m, m+1); got "
            f"{A.shape[-2:]}"
        )
    batch = A.shape[:-2]
    flat = A.reshape((-1, m, ncol))
    nb = flat.shape[0]

    if m == 0:
        # A 0 x 1 matrix: the single empty minor is det([]) = 1, and the
        # sign is (-1)^(1+1) = +1.
        return np.ones(batch + (1,))

    _, s, vt = np.linalg.svd(flat)
    v = vt[:, -1, :]                                   # (nb, ncol)
    qref = np.argmax(np.abs(v), axis=1)                # (nb,)

    keep = np.empty((nb, m), dtype=np.int64)
    for b in range(nb):
        keep[b] = np.delete(np.arange(ncol), qref[b])
    ref_minor = np.take_along_axis(flat, keep[:, None, :], axis=2)
    ref = np.linalg.det(ref_minor)
    # 1-based n = ncol and q = qref+1, so the phase is (-1)^(ncol+qref+1).
    ref = ref * (-1.0) ** (ncol + qref + 1)

    vref = np.take_along_axis(v, qref[:, None], axis=1)[:, 0]
    scale = ref / vref
    out = scale[:, None] * v

    if check:
        # The null-space identity assumes a unique null direction. When A is
        # numerically rank-deficient by more than one, the direction is not
        # determined and the shortcut silently returns a wrong vector, so we
        # detect that by comparing |scale| against prod(singular values) --
        # the two agree identically for a well-conditioned A.
        #
        # This is NOT hypothetical: the Lambda construction replaces a row of
        # Q by a dipole row, and real SF6 matrices come out rank-deficient
        # there. The right response is to fall back to explicit cofactors for
        # the offending batch elements -- they are always correct, just
        # slower -- rather than to refuse the calculation. Only the affected
        # elements pay the cost.
        expect = np.prod(s, axis=1)
        bad = np.flatnonzero(
            np.abs(np.abs(scale) - expect) > 1e-6 * np.maximum(expect, 1e-30))
        for b in bad:
            cols = np.arange(ncol)
            for q in range(ncol):
                minor = flat[b][:, cols != q]
                out[b, q] = ((-1.0) ** (ncol + q + 1)
                             * np.linalg.det(minor))

    return out.reshape(batch + (ncol,))


def minor_matrix(A: np.ndarray, check: bool = True) -> np.ndarray:
    """Signed ``2 x 2``-complementary-minor matrix of an ``(m, m+2)`` matrix.

    Implements ``P[q, q'] = (-1)^(q+q'+1) det(A with columns q, q'
    deleted)`` with 1-based ``q, q'`` -- the ``D^(qq')`` phase of the
    module docstring -- and returns it antisymmetrised, ``P[q', q] =
    -P[q, q']``.

    Parameters
    ----------
    A : ndarray, shape (m, m+2)
        Dimensionless.
    check : bool
        As in :func:`cofactor_vector`.

    Returns
    -------
    ndarray, shape (m+2, m+2)
        Antisymmetric to machine precision by construction: it is built
        from a wedge product, not symmetrised after the fact.

    Notes
    -----
    ``P`` is parallel to ``u ^ v`` for any basis ``u, v`` of the
    two-dimensional right null space of ``A``.  Rotating that basis
    multiplies the wedge by ``det`` of the ``2 x 2`` rotation, so an
    ``SO(2)`` change of frame leaves it invariant and only the overall
    sign of an ``O(2)`` frame remains free -- again fixed against one
    explicit determinant, at the largest wedge component.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ModelError(f"minor_matrix needs a 2-D matrix, got {A.shape}")
    m, ncol = A.shape
    if ncol != m + 2:
        raise ModelError(
            f"minor_matrix expects shape (m, m+2); got {A.shape}"
        )
    if m == 0:
        # A 0 x 2 matrix: the single empty minor is 1 and the phase for
        # (q, q') = (1, 2) is (-1)^(1+2+1) = +1.
        return np.array([[0.0, 1.0], [-1.0, 0.0]])

    _, s, vt = np.linalg.svd(A)
    u, v = vt[-2], vt[-1]
    W = np.outer(u, v) - np.outer(v, u)

    flat = int(np.argmax(np.abs(W)))
    q, qp = divmod(flat, ncol)
    keep = np.delete(np.arange(ncol), [q, qp])
    ref = np.linalg.det(A[:, keep]) * (-1.0) ** ((q + 1) + (qp + 1) + 1)
    scale = ref / W[q, qp]
    if check:
        expect = float(np.prod(s))
        if abs(abs(scale) - expect) > 1e-6 * max(expect, 1e-30):
            raise ModelError(
                f"two-electron gauge check failed: reference minor implies "
                f"|scale|={abs(scale):.6e} but prod(singular values)="
                f"{expect:.6e}.  The doubly-occupied block is close to "
                f"rank deficient (two smallest singular values "
                f"{s[-2]:.3e}, {s[-1]:.3e})"
            )
    return scale * W


def dyson_amplitudes(Q: np.ndarray) -> np.ndarray:
    """One-electron Dyson amplitudes ``d^(q)_i`` in the neutral MO basis.

    Parameters
    ----------
    Q : ndarray, shape (n-1, n)
        Spin-sector overlap block from :func:`build_Q`, dimensionless.
        Row ``i`` of the full ``n x n`` overlap is already absent, so this
        is the alpha block for hole ``i`` (or the beta block for hole
        ``j``).

    Returns
    -------
    ndarray, shape (n,)
        ``d^(q)_i``, dimensionless, in the basis of the ``n`` occupied
        neutral MOs.  Convert to the AO basis with :func:`to_ao`.
    """
    return cofactor_vector(Q)


def two_electron_amplitudes(
    Q: np.ndarray, delete_row: int
) -> np.ndarray:
    """Two-electron Dyson amplitudes from an ``(n-1) x n`` block.

    Parameters
    ----------
    Q : ndarray, shape (n-1, n)
        A spin-sector block, dimensionless.
    delete_row : int
        Row of ``Q`` to remove, leaving the ``(n-2) x n`` doubly-occupied
        block.  Negative indices count from the end.  Removing the row of
        the *other* hole is what makes the result the two-hole object; see
        :func:`build_state_objects`, which locates that row from the hole
        assignment instead of assuming a position.

    Returns
    -------
    ndarray, shape (n, n)
        ``D^(qq')_ij``, antisymmetric, dimensionless.
    """
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2:
        raise ModelError(
            f"two_electron_amplitudes needs a 2-D block, got {Q.shape}"
        )
    nrow = Q.shape[0]
    r = delete_row + nrow if delete_row < 0 else delete_row
    if not 0 <= r < nrow:
        raise ModelError(
            f"delete_row={delete_row} outside 0..{nrow - 1}"
        )
    return minor_matrix(np.delete(Q, r, axis=0))


def two_electron_amplitudes_ij(Q_a: np.ndarray) -> np.ndarray:
    """Two-electron amplitudes for the canonical hole pair ``(i, j)``.

    Parameters
    ----------
    Q_a : ndarray, shape (n-1, n)
        The alpha block, whose rows are the alpha-occupied dication MOs in
        ascending order.  Hole ``i`` is already absent from those rows.

    Returns
    -------
    ndarray, shape (n, n)
        ``D^(qq')_ij``, antisymmetric.

    Notes
    -----
    This convenience wrapper drops the **last** row, which is hole ``j``'s
    row only when ``j`` is the highest-numbered alpha-occupied orbital.
    That holds for a core-valence state whose valence hole is the HOMO,
    which is the common case and the one the old code assumed, but it is
    not general: if any doubly occupied orbital lies above hole ``j``, the
    last row belongs to that orbital instead and the result is a different
    physical object.  :func:`build_state_objects` therefore does not use
    this function; it locates hole ``j``'s row from the
    :class:`~dpi.molcas_io.HoleAssignment` and calls
    :func:`two_electron_amplitudes` with that index.
    """
    return two_electron_amplitudes(Q_a, -1)


# ── beta-sector determinant ─────────────────────────────────────────────────

def det_s_beta(
    Q_b: np.ndarray,
    C_dic_a: np.ndarray,
    S_ao: np.ndarray,
    C_neu: np.ndarray,
    n_neu_occ: int,
    alpha_idx: Sequence[int],
    beta_idx: Sequence[int] | None = None,
) -> float:
    """Full ``n x n`` beta-sector determinant for the alpha-alpha channel.

    Parameters
    ----------
    Q_b : ndarray, shape (n-1, n)
        The beta block, rows = beta-occupied dication MOs ascending.
    C_dic_a : ndarray, shape (nbas,) or (nbas, nmo)
        The alpha singly-occupied dication orbital, either as its own
        coefficient column or as the full dication coefficient matrix, in
        which case the column is selected using the index sets.
        Dimensionless.
    S_ao : ndarray, shape (nbas, nbas)
    C_neu : ndarray, shape (nbas, nmo)
    n_neu_occ : int
        ``n``.
    alpha_idx : sequence of int
        Alpha-occupied dication MO columns, ascending.
    beta_idx : sequence of int, optional
        Beta-occupied dication MO columns, ascending.  Supplying it is
        what allows the canonical row order to be *asserted* rather than
        assumed; see Notes.

    Returns
    -------
    float
        ``det(S^beta)``, dimensionless.

    Notes
    -----
    For the alpha-alpha channel both continuum electrons leave the alpha
    sector, so the beta sector contributes an unreduced ``n x n``
    determinant: the ``n-1`` beta-occupied dication orbitals plus the
    alpha singly-occupied one, against the ``n`` occupied neutral
    orbitals.  Building it from ``Q_a[:, :n-1]`` instead -- the fallback
    the old code deprecated -- gives a near-zero value and would suppress
    ``G_ij``.

    The row order is not a free choice.  ``G_ij`` goes as ``det^2`` and is
    immune, but ``C_ij`` is **linear** in ``det(S^beta)``, so a
    transposition of two rows flips the sign of a term that enters the
    triplet amplitude as ``-4 C_ij`` (REVIEW.md [B-4], where a
    ``vstack([Q_b, row_i])`` had put the holes in the order
    ``(phi_j, phi_i)``).  The convention pinned here is: rows in
    ascending dication MO index.  Because the alpha and beta occupied
    sets differ only by the two holes and their union is the full occupied
    set, that ordering is simply ``0 .. n-1`` and it places the holes in
    the canonical ``(phi_i, phi_j)`` sense, ``i < j``.  When ``beta_idx``
    is given the reconstruction is checked against a direct build from the
    coefficients and raises on mismatch; when it is omitted, hole ``j`` is
    identified as the one alpha index whose row is absent from ``Q_b``,
    which requires the full ``C_dic``.
    """
    Q_b = np.asarray(Q_b, dtype=float)
    S_ao = np.asarray(S_ao, dtype=float)
    C_neu = np.asarray(C_neu, dtype=float)
    C_dic_a = np.asarray(C_dic_a, dtype=float)
    n = int(n_neu_occ)
    a_idx = [int(v) for v in alpha_idx]
    if Q_b.shape != (n - 1, n):
        raise ModelError(
            f"Q_b has shape {Q_b.shape}, expected {(n - 1, n)}"
        )
    if len(a_idx) != n - 1:
        raise ModelError(
            f"alpha_idx has {len(a_idx)} entries, expected n-1={n - 1}"
        )

    if beta_idx is not None:
        b_idx = [int(v) for v in beta_idx]
        if len(b_idx) != n - 1:
            raise ModelError(
                f"beta_idx has {len(b_idx)} entries, expected n-1={n - 1}"
            )
        missing_from_beta = sorted(set(a_idx) - set(b_idx))
        if len(missing_from_beta) != 1:
            raise ModelError(
                f"alpha_idx and beta_idx must differ by exactly one MO "
                f"each (the two holes); alpha-only entries are "
                f"{missing_from_beta}"
            )
        hole_j = missing_from_beta[0]
    else:
        if C_dic_a.ndim != 2:
            raise ModelError(
                "with beta_idx omitted, C_dic_a must be the full "
                "(nbas, nmo) dication coefficient matrix so that the beta "
                "index set can be recovered by matching Q_b's rows"
            )
        # Recover beta_idx by matching each row of Q_b against the overlap
        # row of every dication orbital.  The alternative -- inferring the
        # insertion position from alpha_idx alone -- is off by one whenever
        # hole_i < hole_j, which is always, so it is not attempted.
        all_rows = C_dic_a.T @ S_ao @ C_neu[:, :n]     # (nmo, n)
        scale = max(1.0, float(np.abs(Q_b).max()))
        b_idx = []
        for k in range(Q_b.shape[0]):
            resid = np.abs(all_rows - Q_b[k][None, :]).max(axis=1)
            hits = np.flatnonzero(resid < 1e-8 * scale)
            if hits.size != 1:
                raise ModelError(
                    f"row {k} of Q_b matches {hits.size} dication orbitals "
                    f"(expected exactly 1); pass beta_idx explicitly"
                )
            b_idx.append(int(hits[0]))
        missing_from_beta = sorted(set(a_idx) - set(b_idx))
        if len(missing_from_beta) != 1:
            raise ModelError(
                f"the recovered beta set {b_idx} and alpha_idx differ by "
                f"{len(missing_from_beta)} orbitals, expected exactly one "
                f"(hole j); alpha-only entries are {missing_from_beta}"
            )
        hole_j = missing_from_beta[0]

    if hole_j in b_idx:
        raise ModelError(
            f"hole j = {hole_j} is also listed in beta_idx; the index sets "
            f"are inconsistent"
        )
    row_j = _alpha_singly_row(C_dic_a, hole_j, S_ao, C_neu, n)

    # Canonical construction: insert hole j's row so that the row order is
    # ascending in dication MO index.
    insert_at = int(np.searchsorted(b_idx, hole_j))
    order = b_idx[:insert_at] + [hole_j] + b_idx[insert_at:]
    S_beta = np.insert(Q_b, insert_at, row_j, axis=0)
    if S_beta.shape != (n, n):
        raise ModelError(
            f"S^beta has shape {S_beta.shape}, expected {(n, n)}"
        )
    if sorted(order) != order:
        raise ModelError(
            f"canonical row order violated: rows would be {order}, which "
            f"is not ascending in dication MO index"
        )
    if C_dic_a.ndim == 2:
        direct = C_dic_a[:, order].T @ S_ao @ C_neu[:, :n]
        err = float(np.abs(direct - S_beta).max())
        tol = 1e-8 * max(1.0, float(np.abs(direct).max()))
        if err > tol:
            raise ModelError(
                f"S^beta assembled from Q_b disagrees with a direct build "
                f"from the coefficients by {err:.3e} (tolerance {tol:.3e}); "
                f"the row order of Q_b is not ascending"
            )
    return float(np.linalg.det(S_beta))


def _alpha_singly_row(
    C_dic_a: np.ndarray, hole_j: int, S_ao: np.ndarray,
    C_neu: np.ndarray, n: int,
) -> np.ndarray:
    """``<phi^dic_{hole_j} | phi^neu_q>`` for ``q = 0 .. n-1``."""
    if C_dic_a.ndim == 1:
        col = C_dic_a
    elif C_dic_a.ndim == 2:
        if not 0 <= hole_j < C_dic_a.shape[1]:
            raise ModelError(
                f"hole j = {hole_j} outside the dication MO range "
                f"0..{C_dic_a.shape[1] - 1}"
            )
        col = C_dic_a[:, hole_j]
    else:
        raise ModelError(
            f"C_dic_a must be 1-D or 2-D, got shape {C_dic_a.shape}"
        )
    return col @ S_ao @ C_neu[:, :n]


# ── indirect (Lambda) coefficients ──────────────────────────────────────────

def lambda_coefficients(
    Q_a: np.ndarray,
    alpha_idx: Sequence[int],
    C_neu: np.ndarray,
    C_dic: np.ndarray,
    n_neu_occ: int,
    dipole_ao: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Indirect Dyson coefficients ``lam_i`` in the AO basis (Eq. 95).

    Parameters
    ----------
    Q_a : ndarray, shape (n-1, n)
        The spin-sector block for this hole, dimensionless.
    alpha_idx : sequence of int
        The dication MO columns forming ``Q_a``'s rows, ascending; row
        ``p`` of ``Q_a`` is dication orbital ``alpha_idx[p]``.
    C_neu, C_dic : ndarray, shape (nbas, nmo)
        Dimensionless.
    n_neu_occ : int
        ``n``.
    dipole_ao : mapping
        ``{'x', 'y', 'z'} -> (nbas, nbas)`` AO dipole matrices in bohr,
        as returned by :func:`dpi.molcas_io.read_ao_dipole` with
        ``one_centre=True``.  The one-centre form is required for
        consistency with the atomic cross sections; see that function.

    Returns
    -------
    ndarray, shape (nbas, 3)
        ``lam_i`` in the AO basis, bohr.  The three columns are the
        ``x``, ``y``, ``z`` polarisation components, kept separate so the
        isotropic average can be taken at the amplitude level rather than
        on a pre-averaged dipole, which would be a different (and wrong)
        average.

    Notes
    -----
    The indirect pathway is a bound-to-bound dipole excitation inside the
    determinant overlap, accompanied by the ejection of one electron.  For
    a one-body operator acting between non-orthogonal determinants, the
    generalized Slater-Condon result is a sum of the cofactor vectors of
    ``Q_a`` with **one row at a time replaced** by the corresponding
    dipole row,

    .. code::

        lam^(q)_i = sum_p  cofactor_q( Q_a with row p -> M[p, :] )
        M[p, q']  = <phi^dic_{alpha_idx[p]} | eps . r | phi^neu_q'>

    which is the multilinearity-in-rows form: it is the first-order term
    of ``cofactor(Q_a + t M)`` in ``t``.  Written this way the row stays
    in place, so every sign is inherited from the same ``(-1)^(n+q)``
    cofactor convention as ``d_i`` and no separate Laplace phase has to be
    introduced -- the ``(-1)^p`` factors of the equivalent expansion into
    ``(n-2)``-order minors are generated automatically.

    Expanding each replaced-row cofactor by its own Laplace sum over the
    dipole row is what produced the ``(n-1) n(n-1)/2`` determinants of the
    old implementation.  Here the ``n-1`` matrices are formed at once and
    handed to :func:`cofactor_vector` as a single batch, so the whole
    family costs one batched SVD and one batched determinant.

    Caution.  The manuscript's Eq. (95) was available to this
    implementation only through its description, not verbatim.  The
    structure above is the unique one-body generalized Slater-Condon
    result for this matrix element and reproduces the brute-force
    row-replacement sum exactly (see ``tests/test_dyson.py``), but the
    placement of the dipole indices should be confirmed against the
    equation before the indirect terms are switched on for production --
    they are off by default (REVIEW.md [P-4]).
    """
    Q_a = np.asarray(Q_a, dtype=float)
    C_neu = np.asarray(C_neu, dtype=float)
    C_dic = np.asarray(C_dic, dtype=float)
    n = int(n_neu_occ)
    idx = np.asarray(alpha_idx, dtype=int)
    if Q_a.shape != (n - 1, n):
        raise ModelError(
            f"Q_a has shape {Q_a.shape}, expected {(n - 1, n)}"
        )
    if idx.size != n - 1:
        raise ModelError(
            f"alpha_idx has {idx.size} entries, expected n-1={n - 1}"
        )
    missing = [c for c in _XYZ if c not in dipole_ao]
    if missing:
        raise ModelError(
            f"dipole_ao is missing component(s) {missing}; expected keys "
            f"{_XYZ}"
        )

    nbas = C_neu.shape[0]
    lam_ao = np.empty((nbas, 3))
    occ = C_neu[:, :n]
    for axis, comp in enumerate(_XYZ):
        D = np.asarray(dipole_ao[comp], dtype=float)
        if D.shape != (nbas, nbas):
            raise ModelError(
                f"dipole_ao[{comp!r}] has shape {D.shape}, expected "
                f"{(nbas, nbas)}"
            )
        M = C_dic[:, idx].T @ D @ occ                   # (n-1, n), bohr
        # One matrix per replaced row, stacked into a single batch.
        stack = np.repeat(Q_a[None, :, :], n - 1, axis=0)
        rows = np.arange(n - 1)
        stack[rows, rows, :] = M
        lam_mo = cofactor_vector(stack).sum(axis=0)     # (n,)
        lam_ao[:, axis] = occ @ lam_mo
    return lam_ao


# ── projection and diagnostics ──────────────────────────────────────────────

def to_ao(
    d_mo: np.ndarray, C_neu: np.ndarray, n_neu_occ: int
) -> np.ndarray:
    """Project an MO-basis amplitude onto the AO basis.

    Parameters
    ----------
    d_mo : ndarray, shape (n,) or (n, n)
        Amplitude in the basis of the ``n`` occupied neutral MOs,
        dimensionless.  A matrix is transformed on both indices, which is
        what turns ``D^(qq')`` into the AO-basis ``d2_ij``.
    C_neu : ndarray, shape (nbas, nmo)
    n_neu_occ : int
        ``n``.

    Returns
    -------
    ndarray, shape (nbas,) or (nbas, nbas)
        Same units as ``d_mo``.
    """
    d_mo = np.asarray(d_mo, dtype=float)
    occ = np.asarray(C_neu, dtype=float)[:, :n_neu_occ]
    if d_mo.ndim == 1:
        if d_mo.shape[0] != occ.shape[1]:
            raise ModelError(
                f"d_mo has length {d_mo.shape[0]}, expected "
                f"n_neu_occ={occ.shape[1]}"
            )
        return occ @ d_mo
    if d_mo.ndim == 2:
        if d_mo.shape != (occ.shape[1], occ.shape[1]):
            raise ModelError(
                f"d_mo has shape {d_mo.shape}, expected "
                f"{(occ.shape[1], occ.shape[1])}"
            )
        return occ @ d_mo @ occ.T
    raise ModelError(f"d_mo must be 1-D or 2-D, got shape {d_mo.shape}")


def spectroscopic_factor(
    d_mo: np.ndarray, d_ao: np.ndarray, S_ao: np.ndarray
) -> tuple[float, float]:
    """Spectroscopic factor by the MO and the AO route.

    Parameters
    ----------
    d_mo : ndarray, shape (n,)
        MO-basis Dyson amplitude, dimensionless.
    d_ao : ndarray, shape (nbas,)
        The same amplitude in the AO basis.
    S_ao : ndarray, shape (nbas, nbas)
        AO overlap.

    Returns
    -------
    (float, float)
        ``(sum_q d_mo[q]**2, d_ao @ S_ao @ d_ao)``, both dimensionless.

    Notes
    -----
    The two must agree, because ``d_ao = C_occ d_mo`` and the neutral MOs
    are orthonormal in the AO metric, ``C_occ^T S_ao C_occ = 1``.
    Returning both is a live check on the orbital set and the overlap:
    note #2 writes the spectroscopic factor as the bare ``sum_mu
    (d^mu)^2``, which omits the AO metric and is simply a different
    number in a non-orthogonal basis (REVIEW.md [D-3]).  The value is also
    the model's own measure of how far a state sits from the
    frozen-orbital limit, where it is exactly 1.
    """
    d_mo = np.asarray(d_mo, dtype=float)
    d_ao = np.asarray(d_ao, dtype=float)
    S_ao = np.asarray(S_ao, dtype=float)
    mo_route = float(np.dot(d_mo, d_mo))
    ao_route = float(d_ao @ S_ao @ d_ao)
    return mo_route, ao_route


# ── container ───────────────────────────────────────────────────────────────

@dataclass
class DysonObjects:
    """Everything one core-valence dication state needs downstream.

    All arrays are dimensionless except :attr:`lam_i` / :attr:`lam_j`,
    which carry the dipole's bohr.  A single instance is produced by one
    :func:`build_state_objects` call, which is what guarantees that the
    relative signs are physically meaningful.

    Attributes
    ----------
    d_i, d_j : ndarray, shape (nbas,)
        One-electron Dyson orbitals in the AO basis for the hole missing
        from the alpha and from the beta sector respectively.
    lam_i, lam_j : ndarray, shape (nbas, 3), or None
        Indirect coefficients, columns ``x, y, z``; ``None`` when no
        dipole integrals were supplied.
    d2_ij : ndarray, shape (nbas, nbas), or None
        Two-electron Dyson amplitude in the AO basis, antisymmetric.
    det_sb : float or None
        ``det(S^beta)`` for the alpha-alpha channel.
    p_i, p_j : float
        Spectroscopic factors (AO route, i.e. with the AO metric).
    d_i_mo, d_j_mo : ndarray, shape (n,), or None
        The same one-electron amplitudes in the occupied-MO basis, kept so
        that the MO route of :func:`spectroscopic_factor` and the frozen
        limit's Kronecker-delta form remain checkable after persistence.
    frozen : DysonObjects or None
        The frozen-orbital-limit counterpart of this state.
    meta : dict
        JSON-serialisable provenance: hole indices, index sets, the
        spectroscopic factors by both routes, any occupation-pattern
        approximation, and whether a dipole was supplied.
    """

    d_i: np.ndarray
    d_j: np.ndarray
    lam_i: np.ndarray | None = None
    lam_j: np.ndarray | None = None
    d2_ij: np.ndarray | None = None
    det_sb: float | None = None
    p_i: float = float("nan")
    p_j: float = float("nan")
    d_i_mo: np.ndarray | None = None
    d_j_mo: np.ndarray | None = None
    frozen: "DysonObjects | None" = None
    meta: dict[str, Any] = field(default_factory=dict)

    _ARRAYS = ("d_i", "d_j", "lam_i", "lam_j", "d2_ij", "d_i_mo", "d_j_mo")
    _SCALARS = ("det_sb", "p_i", "p_j")

    def _flatten(self, prefix: str = "") -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name in self._ARRAYS:
            val = getattr(self, name)
            if val is not None:
                out[prefix + name] = np.asarray(val)
        for name in self._SCALARS:
            val = getattr(self, name)
            if val is not None:
                out[prefix + name] = np.asarray(float(val))
        out[prefix + "meta_json"] = np.array(
            json.dumps(self.meta, sort_keys=True, default=str)
        )
        if self.frozen is not None:
            out.update(self.frozen._flatten(prefix + "frozen_"))
        return out

    def save(self, path: str | os.PathLike) -> str:
        """Write the state to a single compressed ``.npz``.

        Parameters
        ----------
        path : str or path-like
            Destination; ``.npz`` is appended if absent.

        Returns
        -------
        str
            The path written.

        Notes
        -----
        One file per state, replacing the old pair of a hand-rolled INPORB
        plus a ``_d2ij.npy`` sidecar.  That pair could not express
        ``det_sb`` or the indirect coefficients, forced a text round trip
        through fixed-width Fortran fields, and let the two halves fall
        out of step.  An INPORB writer is still worth keeping for
        visualising a Dyson orbital, but it is not the persistence format.
        """
        p = os.fspath(path)
        if not p.endswith(".npz"):
            p += ".npz"
        np.savez_compressed(p, **self._flatten()) # type: ignore
        return p

    @classmethod
    def _unflatten(
        cls, data: Mapping[str, np.ndarray], prefix: str = ""
    ) -> "DysonObjects":
        kw: dict[str, Any] = {}
        for name in cls._ARRAYS:
            key = prefix + name
            if key in data:
                kw[name] = np.asarray(data[key])
        for name in cls._SCALARS:
            key = prefix + name
            if key in data:
                kw[name] = float(np.asarray(data[key]).reshape(()))
        mk = prefix + "meta_json"
        kw["meta"] = json.loads(str(np.asarray(data[mk]).reshape(())[()])) \
            if mk in data else {}
        if prefix + "frozen_d_i" in data:
            kw["frozen"] = cls._unflatten(data, prefix + "frozen_")
        return cls(**kw)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "DysonObjects":
        """Read a state written by :meth:`save`."""
        with np.load(os.fspath(path), allow_pickle=False) as fh:
            data = {k: fh[k] for k in fh.files}
        return cls._unflatten(data)


# ── frozen-orbital limit ────────────────────────────────────────────────────

def frozen_objects(
    C_neu: np.ndarray,
    S_ao: np.ndarray,
    n_neu_occ: int,
    hole_i: int,
    hole_j: int,
    dipole_ao: Mapping[str, np.ndarray] | None = None,
    meta: Mapping[str, Any] | None = None,
    hole_vectors: tuple[np.ndarray, np.ndarray] | None = None,
) -> DysonObjects:
    """Dyson objects in the frozen-orbital limit (note #4).

    Parameters
    ----------
    C_neu : ndarray, shape (nbas, nmo)
    S_ao : ndarray, shape (nbas, nbas)
    n_neu_occ : int
        ``n``.
    hole_i, hole_j : int
        0-based neutral MO columns of the two holes, ``hole_i < hole_j``.
    dipole_ao : mapping, optional
        As in :func:`lambda_coefficients`.
    meta : mapping, optional
        Extra provenance merged into :attr:`DysonObjects.meta`.

    Returns
    -------
    DysonObjects
        With ``frozen=None`` (it is its own limit).

    Notes
    -----
    Setting the dication orbitals equal to the neutral ones makes ``Q`` a
    slice of the identity, and the minors collapse to

    .. code::

        d_i^q     -> delta_{qi}
        D_ij^{pq} -> delta_{pi} delta_{qj} - delta_{pj} delta_{qi}
        det(S^beta) -> 1

    so in the AO basis ``d_i = C_neu[:, i]`` and ``d2_ij =
    outer(C_i, C_j) - outer(C_j, C_i)``, and both spectroscopic factors
    are exactly 1.

    One phase caveat, verified numerically in ``tests/test_dyson.py``.
    Note #4's ``delta_{qi}`` is the cofactor result only up to an overall
    sign: evaluating the ``(-1)^(n+q)`` cofactor rule on an identity slice
    gives support on ``q = i`` alone, as it must, but with the value
    ``(-1)^(n+i)`` rather than ``+1``.  This function implements note #4's
    convention literally, so a relaxed build taken continuously to zero
    relaxation approaches these objects up to one overall sign per object.
    That is not a discrepancy to be reconciled -- the overall sign of a
    Dyson amplitude is unphysical, as SPEC.md section 1 states -- and only
    the *relative* signs within a state are meaningful.  It does mean a
    continuity check against the relaxed build must be written up to a
    per-object sign, and that the frozen and relaxed objects should not be
    mixed inside one expression.

    This limit is a diagnostic, not a vanishing baseline.  The bullet
    under manuscript Eq. (116) claims the two-electron and indirect
    blocks all vanish here; they do not (REVIEW.md [P-5]).  ``D_ij`` and
    ``S_ij`` survive because two distinct MOs share AO support, ``G_ij``
    is manifestly non-zero, and the indirect term survives because the
    dipole between two occupied orbitals is non-zero.  The underlying
    reason is the plane-wave continuum: plane waves are not orthogonal to
    the bound orbitals, so the cancellation that should send DPI to zero
    in the independent-electron limit is lost term by term.  What the
    limit *is* good for is bounding relaxation: the difference between a
    relaxed quantity and its frozen counterpart is the part of the
    intensity that orbital relaxation is responsible for.
    """
    C_neu = np.asarray(C_neu, dtype=float)
    S_ao = np.asarray(S_ao, dtype=float)
    n = int(n_neu_occ)
    if not 0 <= hole_i < n or not 0 <= hole_j < n:
        raise ModelError(
            f"holes ({hole_i}, {hole_j}) must lie in 0..n-1 with n={n}"
        )
    if hole_i == hole_j:
        raise ModelError(
            f"the two holes coincide (both {hole_i}); a dication needs two "
            f"distinct orbitals"
        )
    if hole_vectors is None:
        # Note #4 read literally: the hole is one canonical neutral MO, so
        # the Dyson amplitude is a Kronecker delta on that column.
        d_i_mo = np.zeros(n)
        d_i_mo[hole_i] = 1.0
        d_j_mo = np.zeros(n)
        d_j_mo[hole_j] = 1.0
    else:
        # A hole that is a LOCALIZED combination of degenerate neutral MOs.
        # This is the normal situation at a core edge of a symmetric molecule:
        # the SF6 F1s hole is localized on one fluorine, so it spans all six
        # degenerate neutral F1s MOs with a dominant weight of only ~0.47.
        # Forcing delta_{qi} onto its largest single component would discard
        # more than half of the hole, so the frozen amplitude is instead the
        # projection of the localized hole orbital onto the neutral occupied
        # space, `d^q = <phi^neu_q | h>`.  This reduces exactly to delta_{qi}
        # when the hole coincides with a single neutral MO, so it strictly
        # generalises note #4 rather than departing from it.
        d_i_mo = np.asarray(hole_vectors[0], dtype=float)
        d_j_mo = np.asarray(hole_vectors[1], dtype=float)
        for name, vec in (("hole_i", d_i_mo), ("hole_j", d_j_mo)):
            if vec.shape != (n,):
                raise ModelError(
                    f"hole_vectors[{name}] has shape {vec.shape}, expected "
                    f"({n},): one coefficient per occupied neutral MO"
                )

    d2_mo = np.outer(d_i_mo, d_j_mo) - np.outer(d_j_mo, d_i_mo)

    d_i = to_ao(d_i_mo, C_neu, n)
    d_j = to_ao(d_j_mo, C_neu, n)
    d2_ij = to_ao(d2_mo, C_neu, n)
    p_i = spectroscopic_factor(d_i_mo, d_i, S_ao)
    p_j = spectroscopic_factor(d_j_mo, d_j, S_ao)

    lam_i = lam_j = None
    if dipole_ao is not None:
        eye = np.eye(n)
        alpha_idx = tuple(q for q in range(n) if q != hole_i)
        beta_idx = tuple(q for q in range(n) if q != hole_j)
        Q_a = eye[list(alpha_idx), :]
        Q_b = eye[list(beta_idx), :]
        lam_i = lambda_coefficients(
            Q_a, alpha_idx, C_neu, C_neu, n, dipole_ao
        )
        lam_j = lambda_coefficients(
            Q_b, beta_idx, C_neu, C_neu, n, dipole_ao
        )

    info = {
        "limit": "frozen-orbital (note #4)",
        "hole_i": int(hole_i),
        "hole_j": int(hole_j),
        "n_neu_occ": n,
        "p_i_mo_route": p_i[0],
        "p_i_ao_route": p_i[1],
        "p_j_mo_route": p_j[0],
        "p_j_ao_route": p_j[1],
        "det_sb": 1.0,
        "has_dipole": dipole_ao is not None,
    }
    if meta:
        info.update(dict(meta))
    return DysonObjects(
        d_i=d_i,
        d_j=d_j,
        lam_i=lam_i,
        lam_j=lam_j,
        d2_ij=d2_ij,
        det_sb=1.0,
        p_i=p_i[1],
        p_j=p_j[1],
        d_i_mo=d_i_mo,
        d_j_mo=d_j_mo,
        frozen=None,
        meta=info,
    )


# ── one-call state builder ──────────────────────────────────────────────────

def build_state_objects(
    C_neu: np.ndarray,
    C_dic: np.ndarray,
    S_ao: np.ndarray,
    n_neu_occ: int,
    holes: HoleAssignment,
    dipole_ao: Mapping[str, np.ndarray] | None = None,
    include_frozen: bool = True,
    meta: Mapping[str, Any] | None = None,
) -> DysonObjects:
    """Build every Dyson object of one dication state in one gauge.

    Parameters
    ----------
    C_neu : ndarray, shape (nbas, nmo)
        Neutral RHF coefficients, dimensionless.
    C_dic : ndarray, shape (nbas, nmo)
        Relaxed dication coefficients in the same AO basis.
    S_ao : ndarray, shape (nbas, nbas)
        AO overlap, dimensionless.
    n_neu_occ : int
        ``n``, the number of doubly occupied neutral spatial MOs.
    holes : HoleAssignment
        From :func:`dpi.molcas_io.rohf_hole_indices`.  Hole ``i`` is the
        orbital missing from the alpha sector, hole ``j`` from the beta
        sector.
    dipole_ao : mapping, optional
        One-centre AO dipole matrices in bohr.  When omitted,
        :attr:`DysonObjects.lam_i` and ``lam_j`` are ``None``.
    include_frozen : bool
        Also attach the frozen-orbital limit, for the relaxation
        diagnostic.
    meta : mapping, optional
        Extra provenance merged into the returned object's ``meta``.

    Returns
    -------
    DysonObjects
        Fully populated, with all relative signs mutually consistent.

    Notes
    -----
    Single-call construction is a correctness requirement, not a
    convenience.  Each fast-path object fixes its own gauge against an
    explicitly evaluated reference minor of the *same* ``Q`` built from
    the *same* coefficient matrices, so the signs of ``d_i``, ``d_j``,
    ``d2_ij``, ``lam_i`` and ``det_sb`` are mutually meaningful.  Building
    them in separate calls from re-read inputs -- the old code invoked a
    subprocess per state and passed results back through a text file --
    leaves each sign correct in isolation while giving no guarantee about
    the products that enter ``C_ij``.

    The doubly-occupied block is obtained by deleting hole ``j``'s row
    from ``Q_a`` *by index*, never by position: ``Q_a[:-1]`` is only that
    block when no doubly occupied orbital lies above hole ``j``.  As a
    check, the same block is rebuilt by deleting hole ``i``'s row from
    ``Q_b`` and the two are compared -- they must be identical, since both
    are the overlap of the ``n-2`` doubly occupied dication orbitals with
    the neutral occupied set.

    One residual freedom is not removable here, and it has a consequence
    worth stating precisely.  The phase of each dication orbital is
    arbitrary: replacing ``phi^dic_p`` by ``-phi^dic_p`` describes the
    same physical state.  Each object above is a determinant over a set of
    dication rows, so it changes sign exactly when ``p`` belongs to its
    row set:

    ======== ==================== =============================
    object   row set              flips when
    ======== ==================== =============================
    ``d_i``  alpha-occupied       ``p != hole_i``
    ``d_j``  beta-occupied        ``p != hole_j``
    ``d2_ij`` doubly occupied     ``p not in (hole_i, hole_j)``
    ``det_sb`` all ``n``          always
    ======== ==================== =============================

    Terms quadratic in the amplitudes are therefore invariant, as they
    must be: ``F ~ d_i^2 d_j^2`` and ``G_ij ~ det_sb^2 d2_ij^2`` pick up
    an even number of sign changes for every choice of ``p`` (verified in
    ``tests/test_dyson.py``).  The cross term of SPEC.md section 6,

    .. code::

        c_cross = det_sb * ( (sqrt(sigma) * d_i) @ d2_ij @ sqrt(k2*p) +
                             (sqrt(sigma) * d_j) @ d2_ij @ sqrt(k2*p) )

    carries an **odd** number of dication row sets -- one each of
    ``det_sb``, ``d`` and ``d2_ij`` -- and is not invariant: flipping a
    doubly occupied orbital reverses its sign, and flipping a hole orbital
    changes it by a factor that is not even ``+-1``, because the two
    summands transform differently.  Since ``c_cross`` enters the triplet
    amplitude as ``-4 c_cross``, its contribution is then set by an
    arbitrary phase convention rather than by the state.  This module
    cannot fix that -- the expression lives in :mod:`dpi.amplitudes` -- so
    it does the two things it can: it makes the gauge *deterministic*
    (identical inputs give identical signs, every time, because the
    reference minor is an explicit determinant of the given
    coefficients), and it records the parity table above so the defect is
    not rediscovered numerically.  Restoring the ``d_j^nu`` factor that
    makes the term dimensionally parallel to ``F = absorb_i * shake_j``
    also makes it exactly invariant; that is reported as an objection to
    SPEC.md section 6 rather than silently implemented here.
    """
    C_neu = np.asarray(C_neu, dtype=float)
    C_dic = np.asarray(C_dic, dtype=float)
    S_ao = np.asarray(S_ao, dtype=float)
    n = int(n_neu_occ)
    if not isinstance(holes, HoleAssignment):
        raise ModelError(
            f"holes must be a HoleAssignment, got {type(holes).__name__}"
        )
    if len(holes.alpha_idx) != n - 1 or len(holes.beta_idx) != n - 1:
        raise ModelError(
            f"HoleAssignment index sets have lengths "
            f"({len(holes.alpha_idx)}, {len(holes.beta_idx)}), expected "
            f"n-1={n - 1} each"
        )
    if holes.hole_i in holes.alpha_idx:
        raise ModelError(
            f"hole i = {holes.hole_i} appears in alpha_idx; by convention "
            f"hole i is the orbital *missing* from the alpha sector"
        )
    if holes.hole_j in holes.beta_idx:
        raise ModelError(
            f"hole j = {holes.hole_j} appears in beta_idx; by convention "
            f"hole j is the orbital *missing* from the beta sector"
        )
    if holes.hole_i >= holes.hole_j:
        raise ModelError(
            f"holes must satisfy hole_i < hole_j, got "
            f"({holes.hole_i}, {holes.hole_j}); the canonical ordering "
            f"fixes the sign of det(S^beta), which C_ij is linear in"
        )

    Q_a = build_Q(C_neu, C_dic, S_ao, n, holes.alpha_idx)
    Q_b = build_Q(C_neu, C_dic, S_ao, n, holes.beta_idx)

    d_i_mo = dyson_amplitudes(Q_a)
    d_j_mo = dyson_amplitudes(Q_b)
    d_i = to_ao(d_i_mo, C_neu, n)
    d_j = to_ao(d_j_mo, C_neu, n)

    row_j = int(np.searchsorted(np.asarray(holes.alpha_idx), holes.hole_j))
    if holes.alpha_idx[row_j] != holes.hole_j:
        raise ModelError(
            f"hole j = {holes.hole_j} is not among the alpha-occupied "
            f"orbitals {holes.alpha_idx}; the dication would not have a "
            f"singly occupied alpha orbital"
        )
    Q_ij = np.delete(Q_a, row_j, axis=0)
    row_i = int(np.searchsorted(np.asarray(holes.beta_idx), holes.hole_i))
    Q_ij_check = np.delete(Q_b, row_i, axis=0)
    err = float(np.abs(Q_ij - Q_ij_check).max()) if Q_ij.size else 0.0
    tol = 1e-10 * max(1.0, float(np.abs(Q_a).max()))
    if err > tol:
        raise ModelError(
            f"the doubly-occupied block differs by {err:.3e} depending on "
            f"whether it is cut from Q_a or Q_b (tolerance {tol:.3e}); the "
            f"alpha and beta index sets are not consistent with one pair "
            f"of holes"
        )
    d2_mo = minor_matrix(Q_ij)
    d2_ij = to_ao(d2_mo, C_neu, n)

    det_sb = det_s_beta(
        Q_b, C_dic, S_ao, C_neu, n, holes.alpha_idx, holes.beta_idx
    )

    lam_i = lam_j = None
    if dipole_ao is not None:
        lam_i = lambda_coefficients(
            Q_a, holes.alpha_idx, C_neu, C_dic, n, dipole_ao
        )
        lam_j = lambda_coefficients(
            Q_b, holes.beta_idx, C_neu, C_dic, n, dipole_ao
        )

    p_i = spectroscopic_factor(d_i_mo, d_i, S_ao)
    p_j = spectroscopic_factor(d_j_mo, d_j, S_ao)

    info = {
        "hole_i": int(holes.hole_i),
        "hole_j": int(holes.hole_j),
        "alpha_idx": [int(v) for v in holes.alpha_idx],
        "beta_idx": [int(v) for v in holes.beta_idx],
        "n_neu_occ": n,
        "n_doubly": int(holes.n_doubly),
        "n_singly": int(holes.n_singly),
        "approximation": holes.approximation,
        "p_i_mo_route": p_i[0],
        "p_i_ao_route": p_i[1],
        "p_j_mo_route": p_j[0],
        "p_j_ao_route": p_j[1],
        "det_sb": float(det_sb),
        "has_dipole": dipole_ao is not None,
        "antisymmetry_residual": float(
            np.abs(d2_ij + d2_ij.T).max()
        ),
    }
    if meta:
        info.update(dict(meta))

    frozen = None
    if include_frozen:
        # `holes.hole_i/hole_j` index the DICATION orbital set, not the
        # neutral one, so they must not be handed to frozen_objects as
        # neutral MO columns -- on the real SF6 F1s file that substituted two
        # F2p valence MOs (#34, #35) for the actual S1s + F1s holes and made
        # the frozen reference meaningless (it came out 3e-4 of the relaxed
        # intensity).
        #
        # The correct frozen counterpart of a relaxed hole is that hole
        # orbital expanded in the neutral occupied set: h_q = <phi^neu_q|h>.
        # For a core hole on a symmetric molecule this is genuinely spread
        # over the degenerate manifold -- the SF6 F1s hole puts only 0.47 of
        # its weight on its single largest neutral MO -- so picking the
        # argmax column would discard half of it.
        occ = C_neu[:, :n]
        hole_i_vec = occ.T @ S_ao @ C_dic[:, holes.hole_i]
        hole_j_vec = occ.T @ S_ao @ C_dic[:, holes.hole_j]
        # Report the dominant neutral MO for provenance, and how localized
        # the hole is, so a user can see whether the frozen reference is
        # comparing like with like.
        arg_i = int(np.argmax(np.abs(hole_i_vec)))
        arg_j = int(np.argmax(np.abs(hole_j_vec)))
        frozen = frozen_objects(
            C_neu, S_ao, n, arg_i, arg_j, dipole_ao,
            meta={
                "parent_state": info.get("label"),
                "hole_i_dominant_neutral_mo": arg_i + 1,
                "hole_j_dominant_neutral_mo": arg_j + 1,
                "hole_i_dominant_weight": float(hole_i_vec[arg_i] ** 2),
                "hole_j_dominant_weight": float(hole_j_vec[arg_j] ** 2),
                "hole_i_norm_in_occ": float(hole_i_vec @ hole_i_vec),
                "hole_j_norm_in_occ": float(hole_j_vec @ hole_j_vec),
            },
            hole_vectors=(hole_i_vec, hole_j_vec),
        )
    return DysonObjects(
        d_i=d_i,
        d_j=d_j,
        lam_i=lam_i,
        lam_j=lam_j,
        d2_ij=d2_ij,
        det_sb=float(det_sb),
        p_i=p_i[1],
        p_j=p_j[1],
        d_i_mo=d_i_mo,
        d_j_mo=d_j_mo,
        frozen=frozen,
        meta=info,
    )


# ── brute-force reference implementations (used by the tests) ───────────────

def dyson_amplitudes_bruteforce(Q: np.ndarray) -> np.ndarray:
    """``d^(q)_i`` by ``n`` explicit determinants of order ``n-1``.

    The literal transcription of ``d^(q)_i = (-1)^(n+q) det(Q with column
    q deleted)``, kept as the reference the fast path is validated
    against.  Cost ``O(n^4)``; not for production use.
    """
    Q = np.asarray(Q, dtype=float)
    nrow, n = Q.shape
    if nrow != n - 1:
        raise ModelError(f"expected an (n-1, n) block, got {Q.shape}")
    out = np.empty(n)
    for q in range(n):
        out[q] = (-1.0) ** (n + q + 1) * np.linalg.det(
            np.delete(Q, q, axis=1)
        )
    return out


def two_electron_amplitudes_bruteforce(Q_ij: np.ndarray) -> np.ndarray:
    """``D^(qq')_ij`` by ``n(n-1)/2`` explicit determinants of order ``n-2``.

    Literal transcription of ``D^(qq')_ij = (-1)^(q+q'+1) det(Q with
    columns q, q' deleted)``.  Cost ``O(n^5)``; reference only.
    """
    Q_ij = np.asarray(Q_ij, dtype=float)
    nrow, n = Q_ij.shape
    if nrow != n - 2:
        raise ModelError(f"expected an (n-2, n) block, got {Q_ij.shape}")
    out = np.zeros((n, n))
    for q in range(n):
        for qp in range(q + 1, n):
            v = (-1.0) ** ((q + 1) + (qp + 1) + 1) * np.linalg.det(
                Q_ij[:, np.delete(np.arange(n), [q, qp])]
            )
            out[q, qp] = v
            out[qp, q] = -v
    return out


def lambda_coefficients_bruteforce(
    Q_a: np.ndarray, M: np.ndarray
) -> np.ndarray:
    """``lam^(q)_i`` in the MO basis by explicit row-replacement cofactors.

    Parameters
    ----------
    Q_a : ndarray, shape (n-1, n)
    M : ndarray, shape (n-1, n)
        Dipole rows ``M[p, q'] = <phi^dic_p | eps . r | phi^neu_q'>`` for
        one polarisation, bohr.

    Returns
    -------
    ndarray, shape (n,)
        ``sum_p cofactor(Q_a with row p replaced by M[p, :])``, bohr.
        Reference for :func:`lambda_coefficients`; cost ``O(n^5)``.
    """
    Q_a = np.asarray(Q_a, dtype=float)
    M = np.asarray(M, dtype=float)
    nrow, n = Q_a.shape
    if M.shape != Q_a.shape:
        raise ModelError(
            f"M has shape {M.shape}, expected {Q_a.shape}"
        )
    out = np.zeros(n)
    for p in range(nrow):
        A = Q_a.copy()
        A[p, :] = M[p, :]
        out += dyson_amplitudes_bruteforce(A)
    return out
