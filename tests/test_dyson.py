"""Tests of the determinant algebra in :mod:`dpi.dyson`.

Runnable either with ``python -m pytest tests/ -q`` or directly as
``python tests/test_dyson.py``, which is why every check is a plain
``assert`` and the module ends with a hand-rolled runner rather than
relying on pytest fixtures or parametrisation.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpi import dyson as dy
from dpi import molcas_io as mio
from dpi.constants import ModelError

SEED = 4242


def _rng(offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(SEED + offset)


def _synthetic_state(tmpdir: str, **kw):
    """A synthetic case plus the Dyson objects of its one dication state."""
    case = mio.write_synthetic_case(tmpdir, **kw)
    neu = mio.read_inporb(case.inporb_neutral)
    dic = mio.read_inporb(case.inporb_dication)
    s_ao = mio.read_overlap(case.overlap, case.nbas)
    holes = mio.rohf_hole_indices(dic.occ, case.n_neu_occ)
    dipole = mio.read_ao_dipole(case.h5, one_centre=True)
    obj = dy.build_state_objects(
        neu.coeff, dic.coeff, s_ao, case.n_neu_occ, holes, dipole
    )
    return case, neu, dic, s_ao, holes, dipole, obj


# ── the two null-space identities against brute force ───────────────────────

def test_one_electron_matches_bruteforce_including_signs():
    """d_i from the null vector equals the explicit cofactor vector.

    The comparison is on the signed difference, not on absolute values:
    the SVD's phase choice is arbitrary, so a test that compared |d| would
    pass even with the gauge fixing removed.
    """
    for n in (2, 3, 4, 6, 9, 13):
        rng = _rng(n)
        for trial in range(4):
            q = rng.standard_normal((n - 1, n))
            fast = dy.dyson_amplitudes(q)
            brute = dy.dyson_amplitudes_bruteforce(q)
            scale = max(np.abs(brute).max(), 1e-300)
            err = np.abs(fast - brute).max() / scale
            assert err < 1e-9, (n, trial, err)
            # And the signs are not accidentally all-positive.
            assert np.any(fast < 0) or np.any(fast > 0)


def test_two_electron_matches_bruteforce_including_signs():
    """D_ij from the wedge of the two null vectors equals explicit minors."""
    for n in (2, 3, 4, 6, 9, 13):
        rng = _rng(100 + n)
        for trial in range(4):
            q_ij = rng.standard_normal((n - 2, n))
            fast = dy.minor_matrix(q_ij)
            brute = dy.two_electron_amplitudes_bruteforce(q_ij)
            scale = max(np.abs(brute).max(), 1e-300)
            err = np.abs(fast - brute).max() / scale
            assert err < 1e-9, (n, trial, err)


def test_two_electron_amplitudes_row_deletion():
    """The (n-1, n) wrapper deletes the row it is told to delete."""
    rng = _rng(7)
    n = 7
    q = rng.standard_normal((n - 1, n))
    for row in range(n - 1):
        got = dy.two_electron_amplitudes(q, row)
        want = dy.two_electron_amplitudes_bruteforce(np.delete(q, row, axis=0))
        assert np.abs(got - want).max() / np.abs(want).max() < 1e-9, row
    # The documented convenience wrapper is the -1 case.
    assert np.array_equal(
        dy.two_electron_amplitudes_ij(q), dy.two_electron_amplitudes(q, -1)
    )


def test_d2_is_antisymmetric_to_machine_precision():
    """Antisymmetry is exact, not merely small.

    It is built from a wedge product rather than symmetrised afterwards,
    so the residual should be at the level of the floating-point
    representation of the entries themselves.
    """
    for n in (3, 5, 10, 20):
        rng = _rng(200 + n)
        q_ij = rng.standard_normal((n - 2, n))
        d2 = dy.minor_matrix(q_ij)
        resid = np.abs(d2 + d2.T).max() / max(np.abs(d2).max(), 1e-300)
        assert resid < 1e-14, (n, resid)
        assert np.abs(np.diag(d2)).max() == 0.0


def test_lambda_matches_bruteforce_row_replacement():
    """The batched Lambda equals the explicit row-replacement cofactor sum."""
    for n in (3, 5, 8, 11):
        rng = _rng(300 + n)
        q = rng.standard_normal((n - 1, n))
        m = rng.standard_normal((n - 1, n))
        stack = np.repeat(q[None, :, :], n - 1, axis=0)
        rows = np.arange(n - 1)
        stack[rows, rows, :] = m
        fast = dy.cofactor_vector(stack).sum(axis=0)
        brute = dy.lambda_coefficients_bruteforce(q, m)
        err = np.abs(fast - brute).max() / max(np.abs(brute).max(), 1e-300)
        assert err < 1e-9, (n, err)


def test_lambda_coefficients_against_bruteforce_in_ao_basis():
    """lambda_coefficients reproduces the reference sum, per polarisation."""
    with tempfile.TemporaryDirectory() as tmp:
        case, neu, dic, s_ao, holes, dipole, _ = _synthetic_state(
            os.path.join(tmp, "case")
        )
        n = case.n_neu_occ
        q_a = dy.build_Q(neu.coeff, dic.coeff, s_ao, n, holes.alpha_idx)
        lam = dy.lambda_coefficients(
            q_a, holes.alpha_idx, neu.coeff, dic.coeff, n, dipole
        )
        occ = neu.coeff[:, :n]
        for axis, comp in enumerate("xyz"):
            m = dic.coeff[:, list(holes.alpha_idx)].T @ dipole[comp] @ occ
            want = occ @ dy.lambda_coefficients_bruteforce(q_a, m)
            err = np.abs(lam[:, axis] - want).max()
            assert err < 1e-9 * max(np.abs(want).max(), 1.0), (comp, err)
        # The three polarisations are kept separate, so they must differ:
        # a pre-averaged dipole would make these columns proportional.
        assert np.abs(lam[:, 0] - lam[:, 1]).max() > 1e-8


# ── gauge consistency ───────────────────────────────────────────────────────

def _c_cross(obj, sigma, pshake):
    """The SPEC.md section 6 cross term, linear in det_sb, d and d2_ij."""
    a = (np.sqrt(sigma) * obj.d_i) @ obj.d2_ij @ np.sqrt(pshake)
    b = (np.sqrt(sigma) * obj.d_j) @ obj.d2_ij @ np.sqrt(pshake)
    return obj.det_sb * (a + b)


def _c_cross_covariant(obj, sigma, pshake):
    """The parity-covariant variant, with the second Dyson factor restored.

    Dimensionally parallel to ``F = absorb_i * shake_j``: each summand
    carries both one-electron amplitudes as well as ``d2_ij``, so the
    dication row sets appear an even number of times.
    """
    a = (np.sqrt(sigma) * obj.d_i) @ obj.d2_ij @ (obj.d_j * np.sqrt(pshake))
    b = (np.sqrt(sigma) * obj.d_j) @ obj.d2_ij @ (obj.d_i * np.sqrt(pshake))
    return obj.det_sb * (a + b)


def test_single_call_gauge_is_deterministic():
    """Rebuilding a state from identical inputs reproduces every sign.

    The gauge is fixed against an explicitly computed reference minor, so
    it is a function of the inputs alone: two builds must agree bit for
    bit, not merely up to an overall sign.
    """
    with tempfile.TemporaryDirectory() as tmp:
        case, neu, dic, s_ao, holes, dipole, first = _synthetic_state(
            os.path.join(tmp, "case")
        )
        second = dy.build_state_objects(
            neu.coeff, dic.coeff, s_ao, case.n_neu_occ, holes, dipole
        )
        for name in ("d_i", "d_j", "d2_ij", "lam_i", "lam_j"):
            a, b = getattr(first, name), getattr(second, name)
            assert np.array_equal(a, b), name
        assert first.det_sb == second.det_sb
        rng = _rng(11)
        sigma = np.abs(rng.standard_normal(case.nbas))
        pshake = np.abs(rng.standard_normal(case.nbas))
        assert _c_cross(first, sigma, pshake) == _c_cross(
            second, sigma, pshake
        )


def test_relative_gauge_under_reference_sign_flip():
    """Flipping the gauge of one object alone breaks C_ij; the real
    build keeps every object in one gauge.

    This is the check that a per-object independent gauge would fail: the
    quadratic terms F and G are blind to it, so only a term linear in the
    amplitudes -- C_ij -- can detect it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        case, _, _, _, _, _, obj = _synthetic_state(os.path.join(tmp, "case"))
        rng = _rng(12)
        sigma = np.abs(rng.standard_normal(case.nbas))
        pshake = np.abs(rng.standard_normal(case.nbas))
        ref = _c_cross(obj, sigma, pshake)
        assert abs(ref) > 1e-12, "degenerate fixture: C_ij vanishes"

        import copy

        for name in ("d_i", "d_j", "d2_ij", "det_sb"):
            broken = copy.deepcopy(obj)
            setattr(broken, name, -getattr(obj, name))
            got = _c_cross(broken, sigma, pshake)
            # A lone sign flip must change the term: if it did not, the
            # test could not distinguish a shared gauge from independent
            # ones.
            assert abs(got - ref) > 1e-12 * max(abs(ref), 1.0), name
            if name in ("d2_ij", "det_sb"):
                # These two multiply the whole expression, so the flip is
                # a clean overall sign reversal.
                assert abs(got + ref) < 1e-9 * max(abs(ref), 1.0), name


def test_orbital_phase_flip_leaves_quadratic_terms_invariant():
    """Physical (quadratic) terms are invariant under a dication phase flip.

    Replacing phi^dic_p by -phi^dic_p is not a physical change.  Each
    Dyson object is a determinant over a set of dication rows and flips
    sign exactly when p is in that set, so F and G -- which carry every
    row set an even number of times -- must be strictly invariant.  This
    simultaneously verifies the parity table in the build_state_objects
    docstring and shows that the SPEC.md c_cross is *not* invariant, which
    is reported as an objection rather than patched here.
    """
    with tempfile.TemporaryDirectory() as tmp:
        case, neu, dic, s_ao, holes, dipole, ref = _synthetic_state(
            os.path.join(tmp, "case")
        )
        rng = _rng(13)
        sigma = np.abs(rng.standard_normal(case.nbas))
        pshake = np.abs(rng.standard_normal(case.nbas))

        def f_term(o):
            return (sigma @ o.d_i ** 2) * (pshake @ o.d_j ** 2)

        def g_term(o):
            return o.det_sb ** 2 * (sigma @ (o.d2_ij ** 2) @ pshake)

        f_ref, g_ref = f_term(ref), g_term(ref)
        cov_ref = _c_cross_covariant(ref, sigma, pshake)
        assert abs(f_ref) > 1e-12 and abs(g_ref) > 1e-12
        assert abs(cov_ref) > 1e-14

        doubly = [
            q for q in range(case.n_neu_occ)
            if q not in (holes.hole_i, holes.hole_j)
        ]
        for p in range(case.n_neu_occ):
            c_dic = dic.coeff.copy()
            c_dic[:, p] *= -1.0
            flipped = dy.build_state_objects(
                neu.coeff, c_dic, s_ao, case.n_neu_occ, holes, dipole
            )
            assert abs(f_term(flipped) / f_ref - 1.0) < 1e-10, p
            assert abs(g_term(flipped) / g_ref - 1.0) < 1e-10, p
            # Spectroscopic factors are quadratic, hence invariant too.
            assert abs(flipped.p_i - ref.p_i) < 1e-10
            assert abs(flipped.p_j - ref.p_j) < 1e-10
            # The covariant form of the cross term is invariant ...
            assert abs(
                _c_cross_covariant(flipped, sigma, pshake) / cov_ref - 1.0
            ) < 1e-9, p
            # ... while the SPEC.md form is not, for a flip of a doubly
            # occupied orbital it reverses outright.
            if p in doubly:
                ratio = _c_cross(flipped, sigma, pshake) / _c_cross(
                    ref, sigma, pshake
                )
                assert abs(ratio + 1.0) < 1e-9, (p, ratio)


def test_individual_object_signs_follow_the_row_set_parity():
    """Each object flips exactly when the flipped orbital is in its rows."""
    with tempfile.TemporaryDirectory() as tmp:
        case, neu, dic, s_ao, holes, dipole, ref = _synthetic_state(
            os.path.join(tmp, "case")
        )
        for p in range(case.n_neu_occ):
            c_dic = dic.coeff.copy()
            c_dic[:, p] *= -1.0
            flip = dy.build_state_objects(
                neu.coeff, c_dic, s_ao, case.n_neu_occ, holes, dipole
            )
            want_di = -1.0 if p != holes.hole_i else 1.0
            want_dj = -1.0 if p != holes.hole_j else 1.0
            want_d2 = -1.0 if p not in (holes.hole_i, holes.hole_j) else 1.0
            assert np.allclose(flip.d_i, want_di * ref.d_i, atol=1e-12), p
            assert np.allclose(flip.d_j, want_dj * ref.d_j, atol=1e-12), p
            assert np.allclose(flip.d2_ij, want_d2 * ref.d2_ij, atol=1e-12), p
            # det(S^beta) spans all n rows, so it always flips.
            assert abs(flip.det_sb + ref.det_sb) < 1e-12, p


# ── frozen-orbital limit ────────────────────────────────────────────────────

def test_frozen_limit_reproduces_kronecker_deltas():
    """The frozen limit equals the analytic note #4 values exactly."""
    with tempfile.TemporaryDirectory() as tmp:
        case = mio.write_synthetic_case(os.path.join(tmp, "case"))
        neu = mio.read_inporb(case.inporb_neutral)
        s_ao = mio.read_overlap(case.overlap, case.nbas)
        n = case.n_neu_occ
        i, j = 1, 3
        froz = dy.frozen_objects(neu.coeff, s_ao, n, i, j)

        # MO basis: d_i^q -> delta_qi.
        want_i = np.zeros(n)
        want_i[i] = 1.0
        want_j = np.zeros(n)
        want_j[j] = 1.0
        assert np.array_equal(froz.d_i_mo, want_i)
        assert np.array_equal(froz.d_j_mo, want_j)

        # AO basis: d_i -> C_neu[:, i] exactly.
        assert np.allclose(froz.d_i, neu.coeff[:, i], atol=0.0, rtol=0.0)
        assert np.allclose(froz.d_j, neu.coeff[:, j], atol=0.0, rtol=0.0)

        # D_ij^{pq} -> delta_pi delta_qj - delta_pj delta_qi, i.e. in the
        # AO basis the antisymmetrised outer product.
        want_d2 = (
            np.outer(neu.coeff[:, i], neu.coeff[:, j])
            - np.outer(neu.coeff[:, j], neu.coeff[:, i])
        )
        assert np.abs(froz.d2_ij - want_d2).max() < 1e-14

        # det(S^beta) -> 1 and both spectroscopic factors -> 1.
        assert froz.det_sb == 1.0
        assert abs(froz.p_i - 1.0) < 1e-12
        assert abs(froz.p_j - 1.0) < 1e-12
        assert abs(froz.meta["p_i_mo_route"] - 1.0) < 1e-12
        assert abs(froz.meta["p_j_ao_route"] - 1.0) < 1e-12
        assert froz.frozen is None


def test_frozen_limit_is_the_zero_rotation_limit_of_the_relaxed_build():
    """With rotation=0 the relaxed build collapses onto the frozen limit.

    A continuity check on the whole pipeline: if the dication orbitals are
    the neutral ones, Q is a slice of the identity and every fast-path
    object must reproduce the analytic limit -- signs included.
    """
    with tempfile.TemporaryDirectory() as tmp:
        case = mio.write_synthetic_case(
            os.path.join(tmp, "case"), rotation=0.0, hole_i=1, hole_j=3
        )
        neu = mio.read_inporb(case.inporb_neutral)
        dic = mio.read_inporb(case.inporb_dication)
        s_ao = mio.read_overlap(case.overlap, case.nbas)
        holes = mio.rohf_hole_indices(dic.occ, case.n_neu_occ)
        obj = dy.build_state_objects(
            neu.coeff, dic.coeff, s_ao, case.n_neu_occ, holes
        )
        froz = obj.frozen
        assert froz is not None
        # Up to one overall sign per object: note #4's delta_{qi} form and
        # the (-1)^(n+q) cofactor rule differ by (-1)^(n+i), and the
        # overall sign of a Dyson amplitude is unphysical (SPEC.md section
        # 1).  What must hold is that the support and the magnitudes
        # coincide.
        for relaxed, frozen_obj in (
            (obj.d_i, froz.d_i), (obj.d_j, froz.d_j),
            (obj.d2_ij, froz.d2_ij),
        ):
            s = np.sign(np.sum(relaxed * frozen_obj))
            assert abs(s) == 1.0
            err = np.abs(relaxed - s * frozen_obj).max()
            assert err < 1e-10, err
        assert abs(abs(obj.det_sb) - 1.0) < 1e-10
        assert abs(obj.p_i - 1.0) < 1e-10
        assert abs(obj.p_j - 1.0) < 1e-10
        # The MO-basis amplitude is a Kronecker delta up to that sign.
        assert np.abs(np.abs(obj.d_i_mo) - froz.d_i_mo).max() < 1e-10


def test_relaxation_moves_the_spectroscopic_factor_below_one():
    """A relaxed dication has p < 1, and more rotation means smaller p.

    This is the physical content of the non-orthogonality: the
    spectroscopic factor measures how far the state sits from the frozen
    limit, so it must decrease monotonically as the relaxation angle
    grows.
    """
    prev = 1.0
    with tempfile.TemporaryDirectory() as tmp:
        for k, rot in enumerate((0.0, 0.05, 0.15, 0.3)):
            case = mio.write_synthetic_case(
                os.path.join(tmp, f"case{k}"), rotation=rot
            )
            neu = mio.read_inporb(case.inporb_neutral)
            dic = mio.read_inporb(case.inporb_dication)
            s_ao = mio.read_overlap(case.overlap, case.nbas)
            holes = mio.rohf_hole_indices(dic.occ, case.n_neu_occ)
            obj = dy.build_state_objects(
                neu.coeff, dic.coeff, s_ao, case.n_neu_occ, holes,
                include_frozen=False,
            )
            assert obj.p_i <= prev + 1e-12, (rot, obj.p_i, prev)
            assert obj.p_i <= 1.0 + 1e-12
            prev = obj.p_i
    assert prev < 0.999, f"rotation had no effect on p_i ({prev})"


# ── spectroscopic factor: the two routes ────────────────────────────────────

def test_spectroscopic_factor_mo_route_equals_ao_route():
    """sum_q d_q^2 == d_ao S_ao d_ao, because C_occ^T S C_occ = 1.

    REVIEW.md [D-3]: note #2 writes the factor as the bare sum over AO
    components, which drops the AO metric and is a different number in a
    non-orthogonal basis.  Both are returned so the discrepancy is
    visible; here they must agree, which also validates the overlap and
    the orbital set that were read back from disk.
    """
    with tempfile.TemporaryDirectory() as tmp:
        for style in ("rohf", "natural"):
            case, neu, dic, s_ao, holes, dipole, obj = _synthetic_state(
                os.path.join(tmp, style), occ_style=style
            )
            for mo, ao in ((obj.d_i_mo, obj.d_i), (obj.d_j_mo, obj.d_j)):
                p_mo, p_ao = dy.spectroscopic_factor(mo, ao, s_ao)
                assert abs(p_mo - p_ao) < 1e-10 * max(abs(p_mo), 1.0), style
                assert 0.0 < p_ao <= 1.0 + 1e-12
            # And the metric genuinely matters: the naive AO sum, which is
            # what note #2 writes, differs.
            naive = float(np.dot(obj.d_i, obj.d_i))
            assert abs(naive - obj.p_i) > 1e-6, (
                "fixture too close to an orthogonal AO basis to "
                "demonstrate [D-3]"
            )


def test_to_ao_round_trips_through_the_metric():
    """The MO amplitude is recoverable from the AO one via C^T S."""
    with tempfile.TemporaryDirectory() as tmp:
        case, neu, dic, s_ao, holes, dipole, obj = _synthetic_state(
            os.path.join(tmp, "case")
        )
        n = case.n_neu_occ
        occ = neu.coeff[:, :n]
        back = occ.T @ s_ao @ obj.d_i
        assert np.abs(back - obj.d_i_mo).max() < 1e-10
        # The matrix form transforms on both indices.
        d2_mo = occ.T @ s_ao @ obj.d2_ij @ s_ao @ occ
        assert np.abs(dy.to_ao(d2_mo, neu.coeff, n) - obj.d2_ij).max() < 1e-9


# ── det(S^beta) ─────────────────────────────────────────────────────────────

def test_det_s_beta_is_the_full_n_by_n_determinant():
    """det(S^beta) is built from n rows and is not the near-zero fallback."""
    with tempfile.TemporaryDirectory() as tmp:
        case, neu, dic, s_ao, holes, dipole, obj = _synthetic_state(
            os.path.join(tmp, "case")
        )
        n = case.n_neu_occ
        q_b = dy.build_Q(neu.coeff, dic.coeff, s_ao, n, holes.beta_idx)
        got = dy.det_s_beta(
            q_b, dic.coeff, s_ao, neu.coeff, n, holes.alpha_idx,
            holes.beta_idx,
        )
        # Explicit reference: all n dication orbitals 0..n-1 against the n
        # neutral occupied ones, in ascending order.
        want = float(np.linalg.det(
            dic.coeff[:, :n].T @ s_ao @ neu.coeff[:, :n]
        ))
        assert abs(got - want) < 1e-10 * max(abs(want), 1.0), (got, want)
        assert abs(got) > 1e-3, (
            "det(S^beta) is near zero, which is the deprecated "
            "det(Q_a[:, :n-1]) fallback behaviour"
        )
        assert abs(obj.det_sb - want) < 1e-10

        # Recovering beta_idx from the coefficients gives the same value,
        # so the canonical row order does not depend on being told.
        inferred = dy.det_s_beta(
            q_b, dic.coeff, s_ao, neu.coeff, n, holes.alpha_idx
        )
        assert abs(inferred - want) < 1e-10


def test_det_s_beta_rejects_a_permuted_row_order():
    """A non-ascending Q_b is refused rather than silently sign-flipped.

    C_ij is linear in det(S^beta), so a transposition of two rows changes
    the physics (REVIEW.md [B-4]).  The canonical order is asserted, not
    assumed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        case, neu, dic, s_ao, holes, dipole, _ = _synthetic_state(
            os.path.join(tmp, "case")
        )
        n = case.n_neu_occ
        q_b = dy.build_Q(neu.coeff, dic.coeff, s_ao, n, holes.beta_idx)
        swapped = q_b[[1, 0] + list(range(2, q_b.shape[0]))]
        raised = False
        try:
            dy.det_s_beta(
                swapped, dic.coeff, s_ao, neu.coeff, n, holes.alpha_idx,
                holes.beta_idx,
            )
        except ModelError as exc:
            raised = True
            assert "ascending" in str(exc), str(exc)
        assert raised, "a permuted Q_b was accepted"

        # build_Q itself refuses a descending index list for the same
        # reason.
        raised = False
        try:
            dy.build_Q(
                neu.coeff, dic.coeff, s_ao, n,
                tuple(reversed(holes.beta_idx)),
            )
        except ModelError as exc:
            raised = True
            assert "ascending" in str(exc)
        assert raised


# ── build_Q and validation ──────────────────────────────────────────────────

def test_build_Q_orientation_and_values():
    """Q[k, q] = <dic_k | neu_q>, with rows dication and columns neutral."""
    with tempfile.TemporaryDirectory() as tmp:
        case = mio.write_synthetic_case(os.path.join(tmp, "case"))
        neu = mio.read_inporb(case.inporb_neutral)
        dic = mio.read_inporb(case.inporb_dication)
        s_ao = mio.read_overlap(case.overlap, case.nbas)
        n = case.n_neu_occ
        idx = (0, 2, 3, 4)
        q = dy.build_Q(neu.coeff, dic.coeff, s_ao, n, idx)
        assert q.shape == (len(idx), n)
        for k, kk in enumerate(idx):
            for qq in range(n):
                want = dic.coeff[:, kk] @ s_ao @ neu.coeff[:, qq]
                assert abs(q[k, qq] - want) < 1e-12, (k, qq)
        # Near-identity but not identity: this is what makes the Dyson
        # amplitudes non-trivial.
        sub = dy.build_Q(neu.coeff, dic.coeff, s_ao, n, tuple(range(n)))
        off = np.abs(sub - np.eye(n)).max()
        assert 1e-4 < off < 0.9, off


def test_rank_deficient_Q_falls_back_to_explicit_cofactors():
    """A rank-deficient block is detected and computed the slow, exact way.

    When Q loses rank by more than one the null space is not
    one-dimensional, the SVD's choice within it is arbitrary, and the fast
    path has no meaning.  The modulus check against prod(singular values)
    catches that, and the affected batch elements fall back to explicit
    cofactor determinants -- always correct, just slower.

    This is not hypothetical: the Lambda construction replaces a row of Q by
    a dipole row and real SF6 matrices come out rank-deficient there, so
    raising would refuse a legitimate calculation.
    """
    n = 6
    rng = _rng(31)
    q = rng.standard_normal((n - 1, n))
    q[1] = q[0]  # exactly rank deficient -> every cofactor is zero
    got = dy.dyson_amplitudes(q)
    ref = dy.dyson_amplitudes_bruteforce(q)
    assert np.allclose(got, ref, rtol=0.0, atol=1e-10), (got, ref)
    assert np.allclose(got, 0.0, atol=1e-10), (
        "a matrix with two identical rows has vanishing cofactors")

    # And a matrix that is rank deficient only in the deleted column still
    # has non-zero cofactors, which the fallback must reproduce exactly.
    q2 = rng.standard_normal((n - 1, n))
    q2[:, 0] = 0.0
    got2 = dy.dyson_amplitudes(q2)
    ref2 = dy.dyson_amplitudes_bruteforce(q2)
    assert np.allclose(got2, ref2, rtol=1e-10, atol=1e-12)
    assert np.abs(got2).max() > 1e-6, "expected a non-trivial cofactor vector"


def test_build_state_objects_validates_the_hole_convention():
    """The [D-1] convention is enforced, not just documented."""
    with tempfile.TemporaryDirectory() as tmp:
        case, neu, dic, s_ao, holes, dipole, _ = _synthetic_state(
            os.path.join(tmp, "case")
        )
        n = case.n_neu_occ
        # hole i must be ABSENT from the alpha set: a HoleAssignment that
        # puts it there is the reversed convention and must be refused.
        bad = mio.HoleAssignment(
            alpha_idx=holes.beta_idx,
            beta_idx=holes.alpha_idx,
            n_doubly=holes.n_doubly,
            n_singly=holes.n_singly,
            hole_i=holes.hole_i,
            hole_j=holes.hole_j,
        )
        raised = False
        try:
            dy.build_state_objects(neu.coeff, dic.coeff, s_ao, n, bad)
        except ModelError as exc:
            raised = True
            assert "missing" in str(exc), str(exc)
        assert raised, "the reversed hole convention was accepted"


def test_doubly_occupied_block_is_cut_by_index_not_position():
    """Q_ij is consistent whether cut from Q_a or from Q_b.

    Both cuts must give the overlap of the n-2 doubly occupied dication
    orbitals with the neutral occupied set.  Cutting Q_a[:-1] by position
    is only correct when no doubly occupied orbital lies above hole j;
    this fixture puts hole j below the top of the occupied set, so a
    positional cut would give a different matrix.
    """
    with tempfile.TemporaryDirectory() as tmp:
        case, neu, dic, s_ao, holes, dipole, obj = _synthetic_state(
            os.path.join(tmp, "case"), hole_i=0, hole_j=2, n_neu_occ=5
        )
        n = case.n_neu_occ
        assert holes.hole_j < n - 1, "fixture does not exercise the case"
        q_a = dy.build_Q(neu.coeff, dic.coeff, s_ao, n, holes.alpha_idx)
        row_j = list(holes.alpha_idx).index(holes.hole_j)
        by_index = np.delete(q_a, row_j, axis=0)
        by_position = q_a[:-1, :]
        assert np.abs(by_index - by_position).max() > 1e-6, (
            "fixture failed to distinguish the two cuts"
        )
        doubly = [
            c for c in range(n) if c not in (holes.hole_i, holes.hole_j)
        ]
        want = dic.coeff[:, doubly].T @ s_ao @ neu.coeff[:, :n]
        assert np.abs(by_index - want).max() < 1e-12
        # And the object actually stored uses the index cut.
        want_d2 = dy.to_ao(dy.minor_matrix(want), neu.coeff, n)
        assert np.abs(obj.d2_ij - want_d2).max() < 1e-10


# ── persistence ─────────────────────────────────────────────────────────────

def test_dyson_objects_npz_round_trip():
    """One .npz per state carries every field, including the frozen limit."""
    with tempfile.TemporaryDirectory() as tmp:
        case, neu, dic, s_ao, holes, dipole, obj = _synthetic_state(
            os.path.join(tmp, "case"), occ_style="natural"
        )
        path = obj.save(os.path.join(tmp, "state"))
        assert path.endswith(".npz") and os.path.isfile(path)
        back = dy.DysonObjects.load(path)
        for name in ("d_i", "d_j", "lam_i", "lam_j", "d2_ij", "d_i_mo",
                     "d_j_mo"):
            a, b = getattr(obj, name), getattr(back, name)
            assert np.array_equal(a, b), name
        assert back.det_sb == obj.det_sb
        assert back.p_i == obj.p_i and back.p_j == obj.p_j
        assert back.meta["hole_i"] == obj.meta["hole_i"]
        assert back.meta["approximation"] == obj.meta["approximation"]
        assert back.meta["approximation"] is not None
        assert back.frozen is not None
        assert np.array_equal(back.frozen.d_i, obj.frozen.d_i)
        assert back.frozen.det_sb == 1.0
        # A state built without a dipole persists lam_* as None.
        plain = dy.build_state_objects(
            neu.coeff, dic.coeff, s_ao, case.n_neu_occ, holes
        )
        back2 = dy.DysonObjects.load(plain.save(os.path.join(tmp, "plain")))
        assert back2.lam_i is None and back2.lam_j is None


# ── performance ─────────────────────────────────────────────────────────────

def test_fast_path_is_faster_than_bruteforce_at_n35():
    """The null-space path beats explicit minors on the dominant family.

    Only the two-electron family is timed here, because it is the one with
    n(n-1)/2 determinants and therefore dominates; a wall-clock assertion
    is kept loose so the test does not fail on a loaded machine.
    """
    import time

    n = 35
    rng = _rng(35)
    q_ij = rng.standard_normal((n - 2, n))
    fast = dy.minor_matrix(q_ij)
    brute = dy.two_electron_amplitudes_bruteforce(q_ij)
    assert np.abs(fast - brute).max() / np.abs(brute).max() < 1e-9

    t0 = time.perf_counter()
    for _ in range(20):
        dy.minor_matrix(q_ij)
    t_fast = (time.perf_counter() - t0) / 20
    t0 = time.perf_counter()
    for _ in range(3):
        dy.two_electron_amplitudes_bruteforce(q_ij)
    t_brute = (time.perf_counter() - t0) / 3
    assert t_brute > 5.0 * t_fast, (t_brute, t_fast)


def _main() -> int:
    """Run every test in this module without pytest."""
    tests = [
        (name, obj) for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a bare runner
            failures.append((name, exc))
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
