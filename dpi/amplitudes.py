"""DPÌ-model building blocks and spin-recoupled squared DPI amplitudes.

This module is the physics-assembly layer.  It reads no files and computes
no Dyson orbitals: it consumes the arrays produced by :mod:`dpi.dyson`,
:mod:`dpi.atomic_sigma` and :mod:`dpi.shakeoff` and returns the squared
amplitude ``A_f`` of the singly-differential DPI cross section on the whole
energy-sharing quadrature grid at once.

Equation numbers throughout this module refer to **dpi_notes_revised.tex**
(the 47-page revision, not the earlier manuscript, whose numbering differs
because the i=j sections were inserted).  Notation follows SPEC.md
section 6, which renames the notes' overloaded symbols::

    absorb_i    D_i     Eq. 103     photoabsorption block of hole i
    shake_i     S_i     Eq. 106     shake-off block of hole i
    indirect_i  B_i     Eq. 110     indirect (bound-bound dipole) block
    absorb_ij   D_ij    Eq. 112     cross-Dyson photoabsorption bilinear
    shake_ij    S_ij    Eq. 113     cross-Dyson shake-off bilinear
    g_aa        G_ij    Eq. 130    alpha-alpha / beta-beta two-continuum term
    c_cross     C_ij    Eq. 142    one-electron / two-electron interference

Units.  ``sigma_grid`` is in Mb (never converted, see SPEC.md section 1),
``pshake_grid`` and ``k2`` are in atomic units, and every block and
amplitude therefore carries the mixed unit Mb*a.u.  The output is a
*relative* intensity; see :func:`dpi.spectrum.prefactor` and REVIEW.md
[P-6] for why an absolute Mb cross section is not currently attainable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np

from .constants import CHANNEL_LABELS, ModelError

__all__ = [
    "TermSwitches",
    "Blocks",
    "build_blocks",
    "amplitude_ls",
    "amplitude_jj",
    "amplitude",
    "term_breakdown",
    "TERM_ORDER",
    "TERM_DOC",
]


# Canonical order of the additive terms of A_f.  report.py writes the
# per-term integrals in this order, so it is part of the file format.
TERM_ORDER: tuple[str, ...] = (
    "direct",
    "cross_dyson",
    "indirect",
    "indirect_cross",
    "aa_bb",
    "c_cross",
    "dir_ind_interference",
)

TERM_DOC: dict[str, str] = {
    "direct": "2*(F_ij + F_ji), F_ij = absorb_i * shake_j",
    "cross_dyson": "-4*X_ij (singlet) / +4*X_ij (triplet), "
                   "X_ij = absorb_ij * shake_ij",
    "indirect": "2*(Fin_ij + Fin_ji), Fin_ij = indirect_i * shake_j",
    "indirect_cross": "-4*Y_ij (singlet) / +4*Y_ij (triplet), "
                      "Y_ij = indirect_ij * shake_ij",
    "aa_bb": "+4*g_aa, the two-continuum alpha-alpha/beta-beta channel "
             "(triplet only)",
    "c_cross": "-4*c_cross, one-/two-electron Dyson interference "
               "(triplet only)",
    "dir_ind_interference": "direct-indirect mechanism interference X_f, "
                            "sign undetermined, off by default",
}


@dataclass(frozen=True)
class TermSwitches:
    """Which terms of ``A_f`` are evaluated, and two labelled diagnostics.

    Defaults reproduce note #2's production model.  Every term is
    separately switchable *and* separately reported by
    :func:`term_breakdown`, so the open question of REVIEW.md [P-4] --
    whether the indirect terms really are negligible once the one-centre
    dipole of Eq. (146) is used -- can be settled from a single run rather
    than argued from Table 3.

    Attributes
    ----------
    direct:
        Include ``2*(F_ij + F_ji)``.
    cross_dyson:
        Include the ``+-4*X_ij`` cross-Dyson term (and ``+-4*Y_ij`` when
        ``indirect`` is also on).
    indirect:
        Include the indirect blocks ``B_i`` (off by default, [P-4]).
    aa_bb:
        Include ``+4*g_aa`` in the triplet.
    dir_ind_interference:
        Include the direct/indirect mechanism interference ``X_f``
        (notes Eqs. (121) and (128)).  Off by
        default, and the reason is now stronger than the caution this knob
        was built for: Remark 6 of the notes proves
        ``X_f == 0`` identically in the one-centre approximation, Eq.
        (122).  Every one of the sixteen monomials of
        ``2 Re[A_dir A_ind*]`` puts one dipole leg and one overlap-type leg
        on the same electron, which the parity selection rule
        Eq. (148) kills AO-diagonally.  So no
        undetermined sign enters the model at any point, and switching this
        on only probes the *inter*-AO terms that Eq. (98)
        discards -- which this implementation does not compute.  ``dir_ind_sign``
        therefore selects the sign of a term that should evaluate to zero;
        a non-zero result is a bug report, not a physical contribution.
    c_cross:
        Include ``-4*c_cross`` in the triplet.
    dir_ind_sign:
        Documented sign of ``X_f``.  Must be exactly ``+1.0`` or
        ``-1.0``; only consulted when ``dir_ind_interference`` is on.
    spin_degeneracy_factor:
        Diagnostic multiplier on ``A(S=1)``.  **Physics default 1.0.**  No
        ``(2S+1)`` factor belongs here: Eq. (86) already sums the three
        ``M_S^dic`` substates coherently with their Clebsch-Gordan
        weights, and ``1 (x) 1`` contains exactly one ``S_tot = 0``
        state, so a further factor 3 double-counts (ruling [P-2]).  The
        knob exists only so that the note-#2 suggestion can be tested
        numerically and rejected on the record.
    """

    direct: bool = True
    cross_dyson: bool = True
    indirect: bool = False
    aa_bb: bool = True
    dir_ind_interference: bool = False
    c_cross: bool = True
    dir_ind_sign: float = 1.0
    spin_degeneracy_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.dir_ind_sign not in (1.0, -1.0):
            raise ModelError(
                "TermSwitches.dir_ind_sign must be exactly +1.0 or -1.0, "
                f"got {self.dir_ind_sign!r}; the sign of the "
                "direct-indirect interference is an explicit modelling "
                "choice, not a scale factor."
            )
        if self.spin_degeneracy_factor <= 0.0:
            raise ModelError(
                "TermSwitches.spin_degeneracy_factor must be positive, got "
                f"{self.spin_degeneracy_factor!r}."
            )

    def active(self) -> tuple[str, ...]:
        """Names of the terms currently switched on, in TERM_ORDER."""
        flags = {
            "direct": self.direct,
            "cross_dyson": self.cross_dyson,
            "indirect": self.indirect,
            "indirect_cross": self.indirect and self.cross_dyson,
            "aa_bb": self.aa_bb,
            "c_cross": self.c_cross,
            "dir_ind_interference": self.dir_ind_interference,
        }
        return tuple(name for name in TERM_ORDER if flags[name])

    def as_dict(self) -> dict[str, Any]:
        """Flat mapping for the self-documenting output-file headers."""
        return {
            "direct": self.direct,
            "cross_dyson": self.cross_dyson,
            "indirect": self.indirect,
            "aa_bb": self.aa_bb,
            "dir_ind_interference": self.dir_ind_interference,
            "c_cross": self.c_cross,
            "dir_ind_sign": self.dir_ind_sign,
            "spin_degeneracy_factor": self.spin_degeneracy_factor,
        }


@dataclass
class Blocks:
    """Building blocks of ``A_f``, each of shape ``(nq,)``.

    One entry per energy-sharing quadrature point.  ``None`` marks a block
    that could not be built because its Dyson input was absent (e.g. no
    ``lam_i`` for the indirect blocks); requesting a term whose block is
    ``None`` raises :class:`~dpi.constants.ModelError` rather than
    silently contributing zero.

    All blocks are in Mb*a.u.  ``mixed_i`` / ``mixed_j`` are not in the
    manuscript's list; they carry the amplitude-level direct/indirect
    overlap needed by the optional ``X_f`` term and are documented in
    :func:`build_blocks`.
    """

    absorb_i: np.ndarray
    absorb_j: np.ndarray
    absorb_ij: np.ndarray
    shake_i: np.ndarray
    shake_j: np.ndarray
    shake_ij: np.ndarray
    indirect_i: np.ndarray | None = None
    indirect_j: np.ndarray | None = None
    indirect_ij: np.ndarray | None = None
    g_aa: np.ndarray | None = None
    c_cross: np.ndarray | None = None
    mixed_i: np.ndarray | None = None
    mixed_j: np.ndarray | None = None
    nq: int = 0

    def __post_init__(self) -> None:
        if self.nq == 0:
            self.nq = int(np.asarray(self.absorb_i).shape[0])


def _as_2d(name: str, arr: Any, nq: int, nbas: int) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.shape != (nq, nbas):
        raise ModelError(
            f"amplitudes.build_blocks: {name} has shape {a.shape}, "
            f"expected (nq, nbas) = ({nq}, {nbas})."
        )
    return a


def _as_1d(name: str, arr: Any, nbas: int) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.shape != (nbas,):
        raise ModelError(
            f"amplitudes.build_blocks: {name} has shape {a.shape}, "
            f"expected (nbas,) = ({nbas},)."
        )
    return a


def _iso_square(lam_a: np.ndarray, lam_b: np.ndarray) -> np.ndarray:
    """Isotropic polarisation average ``(1/3) sum_alpha lam_a . lam_b``.

    Parameters
    ----------
    lam_a, lam_b:
        ``(nbas, 3)`` indirect Dyson coefficients, columns x, y, z, a.u.

    Returns
    -------
    ndarray
        ``(nbas,)``.  For ``lam_a is lam_b`` this is the manuscript's
        ``(Lam_i^nu)^2``; the 1/3 is the average over photon polarisation
        directions for an isotropic (gas-phase, randomly oriented) sample.
    """
    return np.einsum("na,na->n", lam_a, lam_b) / 3.0


def build_blocks(
    dyson: Any,
    sigma_grid: np.ndarray,
    pshake_grid: np.ndarray,
    k2: np.ndarray,
    terms: TermSwitches,
) -> Blocks:
    """Evaluate every block of ``A_f`` on the whole quadrature grid.

    Parameters
    ----------
    dyson:
        A :class:`dpi.dyson.DysonObjects`-like object.  Required
        attributes: ``d_i``, ``d_j`` (``(nbas,)``, AO basis, a.u.).
        Optional: ``lam_i``, ``lam_j`` (``(nbas, 3)``, a.u.), ``d2_ij``
        (``(nbas, nbas)``, antisymmetric, a.u.), ``det_sb`` (float, the
        beta-sector determinant ``det(S^beta)``, dimensionless).
    sigma_grid:
        ``(nq, nbas)`` atomic subshell cross sections already evaluated at
        the per-AO argument ``eps1 + I_mu`` (Remark 1), in Mb.
    pshake_grid:
        ``(nq, nbas)`` shake-off probability densities ``P^shake_nu(k2)``,
        a.u.
    k2:
        ``(nq,)`` slow-electron momentum, a.u.
    terms:
        Which blocks are needed.  Blocks whose terms are all off are not
        computed, which is the whole point of switching a term off.

    Returns
    -------
    Blocks
        Blocks in Mb*a.u.

    Notes
    -----
    Implemented equations, with ``w = k2 * P^shake`` the shake-off weight::

        absorb_i  = sum_mu (d_i^mu)^2 sigma_mu           (Eq. 103)
        shake_i   = sum_nu (d_i^nu)^2 w_nu               (Eq. 106)
        indirect_i= sum_nu (Lam_i^nu)^2 w_nu             (Eq. 110)
        absorb_ij = sum_mu d_i^mu d_j^mu sigma_mu        (Eq. 112)
        shake_ij  = sum_nu d_i^nu d_j^nu w_nu            (Eq. 113)
        g_aa      = det_sb^2 sum_{mu,nu} (D_ij^{mu,nu})^2 sigma_mu w_nu
        c_cross   = det_sb [ (sqrt(sigma) o d_i) D_ij (d_j o sqrt(w))
                             + (i<->j) ]                   (Eq. 142, made
                                                            gauge covariant)

    ``g_aa`` is not an independent integral.  All three Dyson objects are
    minors of the *same* n x n overlap matrix ``Q[p,q] = <phi^dic_p|phi^neu_q>``
    -- ``d_i`` deletes row i, ``d_j`` deletes row j, ``d2_ij`` deletes both,
    and ``det_sb = det(Q)`` deletes nothing (verified: ``det(Q)`` and
    ``det_sb`` agree to a ratio of ``1.0000000000`` on the real SF6 F1s CV
    state).  Minors of one matrix are not independent, and Jacobi's identity
    on complementary minors, notes Eq. (132), pins them:

        det_sb * d2_ij = -(d_i (x) d_j - d_j (x) d_i)

    which is notes Eq. (134) in the AO basis
    (Eq. (133) in the MO basis).  Verified to a relative
    ``9e-16`` over all 35x35 index pairs on the real state and to ``<2e-15``
    for random non-orthogonal sets at ``n = 5,8,12,20,35``
    (``tests/test_dyson.py``).

    Substituting it cancels ``det_sb^2`` and expands the square --- notes
    Eqs. (135) to (138) --- giving
    Eq. (131):

        g_aa == absorb_i*shake_j + absorb_j*shake_i - 2*absorb_ij*shake_ij
             == F_ij + F_ji - 2 D_ij S_ij

    to ``1e-14``.  So ``aa_bb`` changes the *coefficients* of blocks the
    model already has, not the information content: with ``aa_bb`` and
    ``cross_dyson`` both on, the triplet's spin-summed content collapses to
    ``6(F_ij + F_ji) - 4 D_ij S_ij``, notes Eq. (139).
    Computing it from ``d2_ij`` is still a valid and well-conditioned route;
    it simply cannot disagree, and a disagreement is a bug.

    Two consequences worth stating.  First, the note-#2 argument that
    ``G_ij`` is "not negligibly suppressed" because
    ``|det(S_beta)| ~ 0.91-0.93`` is beside the point: ``det_sb`` cancels
    identically, so ``G_ij``'s size is set by ``F_ij + F_ji`` and by how
    nearly parallel ``d_i`` and ``d_j`` are, not by the overlap determinant.
    (The argument does still hold for ``c_cross``, which is *linear* in
    ``det_sb`` with no cancellation.)  Second, on the real F1s CV state
    ``D_ij S_ij ~ 4.9e-11`` while ``F_ij + F_ji = 8.48795973e-2``, so
    ``g_aa`` equals the triplet direct term to eight digits -- which is why
    the measured breakdown shows ``aa_bb`` at exactly ``2.00000000x`` the
    triplet ``direct`` entry, notes Eq. (140).  That is
    the identity, not a bug, and any departure from 2 measures
    ``D_ij S_ij`` -- the AO-basis non-orthogonality of the two Dyson
    orbitals -- and nothing else.

    ``g_aa`` uses the *diagonal* form of note #2, ruling [P-3]: only the
    diagonal form is consistent with the one-centre approximation used
    everywhere else, and the manuscript's four-index version is not
    dimensionally parallel to ``D_i S_j``.  ``sigma_mu`` is present, which
    the ``sigma``/``sudden`` path of the old code dropped (bug [A-4]).

    ``d2_ij`` is used as the *full* antisymmetric matrix, not its upper
    triangle.  Because ``(D^{nu,mu})^2 = (D^{mu,nu})^2``, the matrix
    ``d2_ij**2`` is symmetric and the full contraction already counts both
    AO orderings -- mu absorbs while nu shakes off, and the reverse -- so
    no explicit factor of two is applied.

    ``c_cross`` is a **joint** contraction over ``(mu, nu)``.  The old code
    factorised it into a product of two independent sums, ``(sum_nu
    D^{mu,nu})`` times ``(sum_mu D^{mu,nu})``, which on a random
    antisymmetric block overestimates it by an order of magnitude and, as
    it enters with a minus sign, biased every triplet intensity (bug
    [A-3]).

    ``mixed_i = sum_nu d_i^nu |Lam_i^nu| sqrt(sigma_nu w_nu)`` is the
    amplitude-level overlap of the direct and indirect mechanisms, built
    with the *magnitude* ``|Lam_i^nu| = sqrt((1/3) sum_alpha
    (Lam_i^{nu,alpha})^2)``.  The isotropic average cannot fix the
    relative phase of a scalar against a vector component, which is
    exactly why the sign of ``X_f`` is undetermined; it is supplied
    explicitly by ``terms.dir_ind_sign``.
    """
    d_i = np.asarray(dyson.d_i, dtype=float).ravel()
    nbas = d_i.size
    d_j = _as_1d("dyson.d_j", dyson.d_j, nbas)
    k2 = np.asarray(k2, dtype=float).ravel()
    nq = k2.size
    sigma_grid = _as_2d("sigma_grid", sigma_grid, nq, nbas)
    pshake_grid = _as_2d("pshake_grid", pshake_grid, nq, nbas)

    if np.any(k2 < 0.0):
        raise ModelError(
            "amplitudes.build_blocks: k2 contains negative momenta; the "
            "energy-sharing substitution of spectrum.integrate_state "
            "guarantees k2 >= 0."
        )

    # w_nu(eps2) = k2 * P^shake_nu(k2).  Eq. (96) normalises the integral
    # of this weight to one, so it is non-negative by construction; a
    # negative entry means the shake-off module produced an unphysical
    # density and must not be silently square-rooted.
    w = k2[:, None] * pshake_grid
    w_min = float(w.min()) if w.size else 0.0
    if w_min < 0.0:
        scale = float(np.abs(w).max())
        if w_min < -1e-12 * max(scale, 1.0):
            raise ModelError(
                "amplitudes.build_blocks: k2 * p_shake has a negative "
                f"entry ({w_min:.3e}); the shake-off probability density "
                "must be non-negative."
            )
        w = np.clip(w, 0.0, None)

    di2 = d_i * d_i
    dj2 = d_j * d_j
    dij = d_i * d_j

    absorb_i = sigma_grid @ di2
    absorb_j = sigma_grid @ dj2
    absorb_ij = sigma_grid @ dij
    shake_i = w @ di2
    shake_j = w @ dj2
    shake_ij = w @ dij

    blk = Blocks(
        absorb_i=absorb_i,
        absorb_j=absorb_j,
        absorb_ij=absorb_ij,
        shake_i=shake_i,
        shake_j=shake_j,
        shake_ij=shake_ij,
        nq=nq,
    )

    need_indirect = terms.indirect or terms.dir_ind_interference
    lam_i = getattr(dyson, "lam_i", None)
    lam_j = getattr(dyson, "lam_j", None)
    if need_indirect and (lam_i is None or lam_j is None):
        raise ModelError(
            "amplitudes.build_blocks: an indirect term is switched on but "
            "dyson.lam_i / dyson.lam_j is None; dyson.lambda_coefficients "
            "must be evaluated for this state, or switch indirect and "
            "dir_ind_interference off."
        )
    if need_indirect:
        lam_i = np.asarray(lam_i, dtype=float)
        lam_j = np.asarray(lam_j, dtype=float)
        for nm, arr in (("dyson.lam_i", lam_i), ("dyson.lam_j", lam_j)):
            if arr.shape != (nbas, 3):
                raise ModelError(
                    f"amplitudes.build_blocks: {nm} has shape "
                    f"{arr.shape}, expected (nbas, 3) = ({nbas}, 3)."
                )
        lam2_i = _iso_square(lam_i, lam_i)
        lam2_j = _iso_square(lam_j, lam_j)
        lam2_ij = _iso_square(lam_i, lam_j)
        blk.indirect_i = w @ lam2_i
        blk.indirect_j = w @ lam2_j
        blk.indirect_ij = w @ lam2_ij

        if terms.dir_ind_interference:
            sqrt_sw = np.sqrt(sigma_grid * w)
            blk.mixed_i = sqrt_sw @ (d_i * np.sqrt(lam2_i))
            blk.mixed_j = sqrt_sw @ (d_j * np.sqrt(lam2_j))

    need_two_electron = terms.aa_bb or terms.c_cross
    d2_ij = getattr(dyson, "d2_ij", None)
    det_sb = getattr(dyson, "det_sb", None)
    if need_two_electron and (d2_ij is None or det_sb is None):
        raise ModelError(
            "amplitudes.build_blocks: aa_bb or c_cross is switched on but "
            "dyson.d2_ij / dyson.det_sb is None; two_electron_amplitudes_ij "
            "and det_s_beta must be evaluated for this state."
        )
    if need_two_electron:
        d2 = np.asarray(d2_ij, dtype=float)
        if d2.shape != (nbas, nbas):
            raise ModelError(
                f"amplitudes.build_blocks: dyson.d2_ij has shape "
                f"{d2.shape}, expected ({nbas}, {nbas})."
            )
        det_sb = float(det_sb) # type: ignore
        if terms.aa_bb:
            # sigma_q . (d2**2) . w_q at every grid point q at once.
            blk.g_aa = det_sb**2 * np.einsum(
                "qm,mn,qn->q", sigma_grid, d2 * d2, w, optimize=True
            )
        if terms.c_cross:
            # Gauge-covariant form.  Notes Eq. (142) reads
            #
            #   C_ij = det_sb sum_{mu,nu} D_ij^{mu,nu} d_i^mu
            #          [sigma_mu k2 P_nu]^(1/2)  +  (i<->j)
            #
            # which carries the dication orbital set an ODD number of times
            # (once each in det_sb, d_i and D_ij).  Every Dyson object is a
            # determinant over a set of dication rows and changes sign when
            # any orbital in that set is rephased phi_p -> -phi_p, an
            # unphysical operation that no observable may see.  Measured on
            # a 6-orbital test case: F and G are invariant to 1e-12 under
            # all such flips, while the Eq. (142) expression returns -1.0000
            # times its value for a flip of a doubly occupied orbital and
            # 0.6361 for a flip of a singly occupied one -- not even a sign,
            # because its two summands transform differently.  The term
            # would then contribute to the triplet according to an arbitrary
            # convention in the orbital file.
            #
            # Restoring the second one-electron Dyson factor on the nu leg
            # makes each summand quadratic in every row set, hence exactly
            # invariant (verified: ratio +1.0000 for every flip).  It is
            # also the form dimensionally parallel to the direct term
            # F_ij = absorb_i * shake_j, whose nu leg likewise carries a
            # Dyson coefficient.  See REVIEW.md [A-5].
            a_i = np.sqrt(sigma_grid) * d_i
            a_j = np.sqrt(sigma_grid) * d_j
            b_i = np.sqrt(w) * d_i
            b_j = np.sqrt(w) * d_j
            blk.c_cross = det_sb * (
                np.einsum("qm,mn,qn->q", a_i, d2, b_j, optimize=True)
                + np.einsum("qm,mn,qn->q", a_j, d2, b_i, optimize=True)
            )

    return blk


def _require(name: str, block: np.ndarray | None, term: str) -> np.ndarray:
    if block is None:
        raise ModelError(
            f"amplitudes: term '{term}' is switched on but block "
            f"'{name}' was not built; pass the same TermSwitches to "
            "build_blocks and to the amplitude functions."
        )
    return block


def _ls_breakdown(
    blk: Blocks, spin: str, terms: TermSwitches
) -> dict[str, np.ndarray]:
    """Per-term arrays of A(S=0) or A(S=1); their sum is the amplitude."""
    if spin == "singlet":
        cross_sign = -1.0
        overall = 1.0
    elif spin == "triplet":
        # The triplet is symmetric under 1<->2 so its cross terms add; the
        # 1/3 is the Clebsch-Gordan weight of Eq. (86) (ruling [P-1]).
        cross_sign = +1.0
        overall = terms.spin_degeneracy_factor / 3.0
    else:
        raise ModelError(
            f"amplitudes: unknown LS spin channel {spin!r}; expected "
            "'singlet' or 'triplet'."
        )

    zero = np.zeros(blk.nq)
    out: dict[str, np.ndarray] = {}

    if terms.direct:
        f_ij = blk.absorb_i * blk.shake_j
        f_ji = blk.absorb_j * blk.shake_i
        out["direct"] = 2.0 * (f_ij + f_ji)
    if terms.cross_dyson:
        out["cross_dyson"] = cross_sign * 4.0 * (blk.absorb_ij * blk.shake_ij)
    if terms.indirect:
        b_i = _require("indirect_i", blk.indirect_i, "indirect")
        b_j = _require("indirect_j", blk.indirect_j, "indirect")
        out["indirect"] = 2.0 * (b_i * blk.shake_j + b_j * blk.shake_i)
        if terms.cross_dyson:
            b_ij = _require("indirect_ij", blk.indirect_ij, "indirect_cross")
            out["indirect_cross"] = cross_sign * 4.0 * (b_ij * blk.shake_ij)
    if terms.dir_ind_interference:
        m_i = _require("mixed_i", blk.mixed_i, "dir_ind_interference")
        m_j = _require("mixed_j", blk.mixed_j, "dir_ind_interference")
        # X_f, notes Eqs. (121) and (128).  Remark
        # 6 proves X_f == 0 identically in the one-centre
        # approximation, Eq. (122): every monomial of
        # 2 Re[A_dir A_ind*] carries one dipole and one overlap leg on the
        # same electron, which Eq. (148) kills
        # AO-diagonally.  The sign switch therefore selects the sign of a
        # vanishing quantity; a non-zero value here means an AO-diagonal
        # leg survived that should not have.
        out["dir_ind_interference"] = (
            terms.dir_ind_sign * 2.0
            * (m_i * blk.shake_j + m_j * blk.shake_i)
        )

    if spin == "triplet":
        if terms.aa_bb:
            out["aa_bb"] = 4.0 * _require("g_aa", blk.g_aa, "aa_bb")
        if terms.c_cross:
            out["c_cross"] = -4.0 * _require(
                "c_cross", blk.c_cross, "c_cross"
            )
    else:
        # The alpha-alpha / beta-beta two-continuum channel and its
        # interference exist only for S_dic = 1: a singlet dication core
        # cannot be reached by two same-spin continuum electrons.
        if terms.aa_bb:
            out["aa_bb"] = zero.copy()
        if terms.c_cross:
            out["c_cross"] = zero.copy()

    if overall != 1.0:
        out = {k: overall * v for k, v in out.items()}
    return {k: out[k] for k in TERM_ORDER if k in out}


def term_breakdown(
    blk: Blocks, spin: str, terms: TermSwitches,
    blk_triplet: Blocks | None = None,
) -> dict[str, np.ndarray]:
    """Additive decomposition of the squared amplitude of one channel.

    Parameters
    ----------
    blk:
        Blocks from :func:`build_blocks`, Mb*a.u.
    spin:
        Channel identifier: ``'singlet'``, ``'triplet'``, ``'jc32'`` or
        ``'jc12'``.  (The parameter keeps the name fixed by SPEC.md
        section 6 although it also accepts the two jj channels.)
    terms:
        The same switches that were passed to :func:`build_blocks`.

    Returns
    -------
    dict[str, ndarray]
        Keys drawn from :data:`TERM_ORDER`, values ``(nq,)`` in Mb*a.u.
        The sum over keys equals :func:`amplitude` for the same arguments
        to machine precision; this identity is a test.
    """
    if spin in ("singlet", "triplet"):
        return _ls_breakdown(blk, spin, terms)
    if spin in ("jc32", "jc12"):
        # A(jC=3/2) = A(S=0) + A(S=1) and A(jC=1/2) = 0.5 * the same sum,
        # so the two jj bands are identical in shape and differ only by
        # the 2:1 statistical factor (note #3 and its supplement).
        scale = 1.0 if spin == "jc32" else 0.5
        # As in amplitude_jj: the S=1 piece belongs to a different dication
        # state with its own orbitals, so it needs its own Blocks.
        s0 = _ls_breakdown(blk, "singlet", terms)
        s1 = _ls_breakdown(blk if blk_triplet is None else blk_triplet,
                           "triplet", terms)
        keys = [k for k in TERM_ORDER if k in s0 or k in s1]
        zero = np.zeros(blk.nq)
        return {
            k: scale * (s0.get(k, zero) + s1.get(k, zero)) for k in keys
        }
    raise ModelError(
        f"amplitudes.term_breakdown: unknown channel {spin!r}; expected "
        f"one of {tuple(CHANNEL_LABELS)}."
    )


def amplitude_ls(blk: Blocks, spin: str, terms: TermSwitches) -> np.ndarray:
    """Spin-recoupled squared amplitude ``A_f(eps2)`` for an LS channel.

    Parameters
    ----------
    blk:
        Blocks from :func:`build_blocks`, Mb*a.u.
    spin:
        ``'singlet'`` (``S_dic = 0``) or ``'triplet'`` (``S_dic = 1``).
    terms:
        The switches used to build ``blk``.

    Returns
    -------
    ndarray
        ``(nq,)`` squared amplitude in Mb*a.u.

    Notes
    -----
    With ``F_ij = D_i S_j``, ``X_ij = D_ij S_ij``, ``Fin_ij = B_i S_j``
    and ``Y_ij = B_ij S_ij``::

        A(S=0) =   2*(F_ij + F_ji) - 4*X_ij
                 [+ 2*(Fin_ij + Fin_ji) - 4*Y_ij]

        A(S=1) = [ 2*(F_ij + F_ji) + 4*X_ij
                 [+ 2*(Fin_ij + Fin_ji) + 4*Y_ij]
                 + 4*g_aa - 4*c_cross ] / 3

    The singlet is the combination that is antisymmetric under exchange of
    the two continuum electrons, so its cross term *subtracts*; the
    triplet is symmetric, so its cross term *adds* and it additionally
    carries the same-spin two-continuum channel ``g_aa`` and that
    channel's interference ``c_cross``.  No ``(2S+1)`` degeneracy factor
    is applied anywhere (ruling [P-2]); see
    ``TermSwitches.spin_degeneracy_factor``.

    A squared amplitude that comes out negative signals a breakdown of the
    model, not of the arithmetic, and is reported rather than clamped (bug
    [B-3]); see :class:`dpi.spectrum.StateResult`.  Which channel can do
    this is worth stating precisely, because REVIEW.md [B-3] attributes it
    to the singlet's ``-4*X_ij`` and that attribution does not survive:

    * With ``direct``, ``cross_dyson`` and ``indirect`` on, ``A(S=0)`` is
      **non-negative by construction**.  ``sigma_mu >= 0`` and
      ``w_nu >= 0`` make ``D`` and ``S`` inner products of positive
      measure, so Cauchy-Schwarz gives ``D_ij^2 <= D_i D_j`` and
      ``S_ij^2 <= S_i S_j``, whence ``4|X_ij| <= 4 sqrt(D_i D_j S_i S_j)
      <= 2(D_i S_j + D_j S_i)`` by AM-GM.  Equality needs ``d_i``
      parallel to ``d_j``, where the singlet vanishes identically.  The
      indirect pair obeys the same chain in the ``(nbas, 3)`` inner
      product, and the two brackets are separately non-negative.
    * ``A(S=1)`` *can* go negative: ``-4*c_cross`` is a signed joint
      contraction of the antisymmetric ``d2_ij`` with no positivity
      bound, and it grows linearly in ``d2_ij`` while the direct term does
      not depend on it at all.  This is the regime in which the
      one-centre truncation of Eq. (142) fails, and it is exactly what a
      reported negative intensity is for.
    * ``A(S=0)`` can also go negative once ``dir_ind_interference`` is
      enabled, since the sign of ``X_f`` is a free choice.

    So the guard against clamping matters, but the channel it protects is
    the triplet.
    """
    parts = _ls_breakdown(blk, spin, terms)
    if not parts:
        return np.zeros(blk.nq)
    return np.sum(np.stack(list(parts.values())), axis=0)


def amplitude_jj(blk: Blocks, jc: str, terms: TermSwitches,
                 blk_triplet: Blocks | None = None) -> np.ndarray:
    """Squared amplitude ``A_f(eps2)`` for a jj-coupled 2p core hole.

    Parameters
    ----------
    blk:
        Blocks from :func:`build_blocks`, Mb*a.u.
    jc:
        ``'3/2'`` or ``'jc32'`` for ``j_C = 3/2``; ``'1/2'`` or
        ``'jc12'`` for ``j_C = 1/2``.
    terms:
        The switches used to build ``blk``.
    blk_triplet:
        Blocks built from the **triplet** (``S_dic = 1``) dication orbital
        set.  When given, ``blk`` supplies the ``A(S=0)`` piece and this
        argument the ``A(S=1)`` piece.  When ``None`` both are taken from
        ``blk``, which is correct only if the two multiplets happen to share
        an orbital set -- see the warning in the Notes.

    Returns
    -------
    ndarray
        ``(nq,)`` squared amplitude in Mb*a.u.

    Notes
    -----
    From the enumeration of the 30 non-vanishing ``(S_tot, M_S) = (0, 0)``
    projections in the note-#3 supplement::

        A(jC=3/2) =       A(S=0) + A(S=1)
        A(jC=1/2) = 0.5 * [ A(S=0) + A(S=1) ]

    The 2:1 branching is therefore independent of ``F`` and ``G``, and the
    two spin-orbit bands have *identical* shape as functions of the energy
    sharing.  Any measured shape difference beyond the ``E_excess``
    difference between the two multiplets falsifies the spectator
    approximation for the valence hole, so the two peaks must be obtained
    by running the pipeline twice at the two energy sets rather than by
    scaling one result.

    **Both LS amplitudes are needed for each j_C peak, and they belong to
    two physically distinct dication states.**  The ``S_dic = 0`` and
    ``S_dic = 1`` states have different OSRHF orbitals, hence different
    Dyson objects, so evaluating both from one ``Blocks`` is wrong.  With
    real SF6 orbital sets that error moved the branching ratio from the
    exact 2.0000 to ~200 and mis-scaled the individual peaks by +221% and
    -97%.  The driver therefore builds two Dyson objects per jj state and
    passes the triplet one here.
    """
    key = {"3/2": "jc32", "jc32": "jc32", "1/2": "jc12", "jc12": "jc12"}.get(
        str(jc)
    )
    if key is None:
        raise ModelError(
            f"amplitudes.amplitude_jj: unknown core j_C {jc!r}; expected "
            "'3/2' or '1/2'."
        )
    scale = 1.0 if key == "jc32" else 0.5
    # A(S=0) and A(S=1) are properties of DIFFERENT dication states, each
    # with its own relaxed OSRHF orbital set, so in general they must be
    # evaluated from different Blocks. `blk_triplet` carries the S=1 set.
    # Passing a single `blk` (the historical behaviour) silently evaluates
    # the S=1 amplitude on the singlet's orbitals; with real SF6 orbital
    # sets that mis-scaled the branching ratio from 2.0 to ~200.
    blk_s = blk
    blk_t = blk if blk_triplet is None else blk_triplet
    return scale * (
        amplitude_ls(blk_s, "singlet", terms)
        + amplitude_ls(blk_t, "triplet", terms)
    )


def amplitude(blk: Blocks, channel: str, terms: TermSwitches,
              blk_triplet: Blocks | None = None) -> np.ndarray:
    """Dispatch to :func:`amplitude_ls` or :func:`amplitude_jj`.

    Parameters
    ----------
    blk:
        Blocks from :func:`build_blocks`, Mb*a.u.
    channel:
        One of ``'singlet'``, ``'triplet'``, ``'jc32'``, ``'jc12'``.
    terms:
        The switches used to build ``blk``.

    Returns
    -------
    ndarray
        ``(nq,)`` squared amplitude in Mb*a.u.
    """
    if channel in ("singlet", "triplet"):
        return amplitude_ls(blk, channel, terms)
    if channel in ("jc32", "jc12"):
        return amplitude_jj(blk, channel, terms, blk_triplet)
    raise ModelError(
        f"amplitudes.amplitude: unknown channel {channel!r}; expected one "
        f"of {tuple(CHANNEL_LABELS)}."
    )


def switches_for(terms: TermSwitches, **overrides: Any) -> TermSwitches:
    """Copy of ``terms`` with individual switches overridden.

    Convenience for the term-by-term diagnostic runs of [P-4]: the
    dataclass is frozen so that a switch set cannot be mutated behind the
    back of the header written into an output file.
    """
    return replace(terms, **overrides)


def blocks_summary(blk: Blocks) -> Mapping[str, float]:
    """Grid maxima of every built block, for the run log.

    Returns
    -------
    Mapping[str, float]
        Block name -> max absolute value over the quadrature grid,
        Mb*a.u.  Absent blocks are omitted.
    """
    names = (
        "absorb_i", "absorb_j", "absorb_ij",
        "shake_i", "shake_j", "shake_ij",
        "indirect_i", "indirect_j", "indirect_ij",
        "g_aa", "c_cross", "mixed_i", "mixed_j",
    )
    out: dict[str, float] = {}
    for name in names:
        arr = getattr(blk, name)
        if arr is not None:
            out[name] = float(np.max(np.abs(arr)))
    return out
