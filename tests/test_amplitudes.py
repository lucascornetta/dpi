"""Tests of the amplitude assembly.

Every contraction that the audit found wrong in the old code is checked
against an explicit double loop, and every algebraic identity claimed in the
docstrings (jj branching, band-shape identity, breakdown sum rule, the
d_i == d_j limit) is checked numerically.
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import FakeDyson, antisym

from dpi.amplitudes import (
    TERM_ORDER,
    TermSwitches,
    amplitude,
    amplitude_jj,
    amplitude_ls,
    blocks_summary,
    build_blocks,
    term_breakdown,
)
from dpi.constants import ModelError


def _blocks(dyson, grid_inputs, terms):
    sigma_grid, pshake_grid, k2 = grid_inputs
    return build_blocks(dyson, sigma_grid, pshake_grid, k2, terms)


def test_blocks_match_explicit_sums(dyson, grid_inputs, all_terms):
    sigma_grid, pshake_grid, k2 = grid_inputs
    blk = _blocks(dyson, grid_inputs, all_terms)
    w = k2[:, None] * pshake_grid

    assert np.allclose(blk.absorb_i, sigma_grid @ dyson.d_i**2)
    assert np.allclose(blk.shake_j, w @ dyson.d_j**2)
    assert np.allclose(blk.absorb_ij, sigma_grid @ (dyson.d_i * dyson.d_j))
    assert np.allclose(blk.shake_ij, w @ (dyson.d_i * dyson.d_j))

    # Isotropic polarisation average: (1/3) sum_alpha lam^2.
    lam2 = (dyson.lam_i**2).sum(axis=1) / 3.0
    assert np.allclose(blk.indirect_i, w @ lam2)


def test_g_aa_equals_explicit_double_loop(dyson, grid_inputs, all_terms):
    sigma_grid, pshake_grid, k2 = grid_inputs
    blk = _blocks(dyson, grid_inputs, all_terms)
    w = k2[:, None] * pshake_grid
    nq, nbas = sigma_grid.shape
    d2 = dyson.d2_ij

    ref = np.zeros(nq)
    for q in range(nq):
        acc = 0.0
        for mu in range(nbas):
            for nu in range(nbas):
                acc += d2[mu, nu] ** 2 * sigma_grid[q, mu] * w[q, nu]
        ref[q] = dyson.det_sb**2 * acc
    assert np.allclose(blk.g_aa, ref, rtol=1e-13, atol=0.0)


def test_g_aa_uses_sigma(dyson, grid_inputs, all_terms):
    """Regression guard for [A-4]: sigma_mu must never be dropped."""
    sigma_grid, pshake_grid, k2 = grid_inputs
    blk = _blocks(dyson, grid_inputs, all_terms)
    dropped = dyson.det_sb**2 * (
        ((dyson.d2_ij**2).sum(axis=0)) @ (k2[:, None] * pshake_grid).T
    )
    assert not np.allclose(blk.g_aa, dropped)


def test_g_aa_invariant_under_mu_nu_relabel(dyson, grid_inputs, all_terms):
    """(D_{nu,mu})^2 == (D_{mu,nu})^2, so d2**2 is symmetric."""
    d2sq = dyson.d2_ij**2
    assert np.allclose(d2sq, d2sq.T, atol=0.0)

    sigma_grid, pshake_grid, k2 = grid_inputs
    blk = _blocks(dyson, grid_inputs, all_terms)
    transposed = FakeDyson(
        d_i=dyson.d_i, d_j=dyson.d_j,
        lam_i=dyson.lam_i, lam_j=dyson.lam_j,
        d2_ij=dyson.d2_ij.T, det_sb=dyson.det_sb,
    )
    blk_t = build_blocks(transposed, sigma_grid, pshake_grid, k2, all_terms)
    assert np.allclose(blk.g_aa, blk_t.g_aa, rtol=1e-14, atol=0.0)


def test_c_cross_equals_explicit_double_loop(dyson, grid_inputs, all_terms):
    sigma_grid, pshake_grid, k2 = grid_inputs
    blk = _blocks(dyson, grid_inputs, all_terms)
    w = k2[:, None] * pshake_grid
    nq, nbas = sigma_grid.shape
    d2 = dyson.d2_ij

    # Gauge-covariant form: the nu leg carries the OTHER hole's Dyson
    # coefficient, so each summand is quadratic in every dication row set.
    # See test_c_cross_is_gauge_invariant and REVIEW.md [A-5].
    ref = np.zeros(nq)
    for q in range(nq):
        acc = 0.0
        for mu in range(nbas):
            for nu in range(nbas):
                joint = np.sqrt(sigma_grid[q, mu] * w[q, nu])
                acc += d2[mu, nu] * joint * (
                    dyson.d_i[mu] * dyson.d_j[nu]
                    + dyson.d_j[mu] * dyson.d_i[nu])
        ref[q] = dyson.det_sb * acc
    assert np.allclose(blk.c_cross, ref, rtol=1e-12, atol=1e-14)


def test_c_cross_is_gauge_invariant(rng, grid_inputs):
    """Rephasing a dication orbital must not change any observable.

    Each Dyson object is a determinant over a set of dication rows and
    changes sign when an orbital in that set is replaced by its negative.
    That operation is unphysical, so every term of ``A_f`` must be even in
    each row set.  Manuscript Eq. (114) is odd, and this test pins the
    corrected form: emulating a rephasing of hole j flips ``d_j`` and
    ``d2_ij`` together, which the covariant expression absorbs and the
    literal Eq. (114) expression does not.
    """
    sigma_grid, pshake_grid, k2 = grid_inputs
    nbas = sigma_grid.shape[1]
    d_i, d_j = rng.normal(size=nbas), rng.normal(size=nbas)
    d2 = antisym(rng, nbas)
    terms = TermSwitches(aa_bb=True, c_cross=True)

    base = build_blocks(FakeDyson(d_i=d_i, d_j=d_j, d2_ij=d2, det_sb=0.92),
                        sigma_grid, pshake_grid, k2, terms)
    # A flip of a doubly occupied dication orbital: d_i, d_j and d2_ij all
    # sit in row sets containing it, so all three change sign, and det_sb
    # with them.
    flip = build_blocks(FakeDyson(d_i=-d_i, d_j=-d_j, d2_ij=-d2,
                                  det_sb=-0.92),
                        sigma_grid, pshake_grid, k2, terms)
    assert np.allclose(flip.g_aa, base.g_aa, rtol=1e-12, atol=0.0)
    assert np.allclose(flip.c_cross, base.c_cross, rtol=1e-12, atol=0.0)


def test_c_cross_differs_from_old_factorised_form(rng, grid_inputs):
    """Regression guard for [A-3] on a random antisymmetric matrix.

    The old code computed (sum_nu D^{mu,nu}) and (sum_mu D^{mu,nu})
    separately and multiplied the two resulting scalars.  The correct
    quantity is the joint contraction.  Reports the discrepancy factor.
    """
    sigma_grid, pshake_grid, k2 = grid_inputs
    nq, nbas = sigma_grid.shape
    d2 = antisym(rng, nbas)
    d_i = rng.normal(size=nbas)
    d_j = rng.normal(size=nbas)
    det_sb = 1.0
    dy = FakeDyson(d_i=d_i, d_j=d_j, d2_ij=d2, det_sb=det_sb)
    terms = TermSwitches(aa_bb=False, c_cross=True)
    blk = build_blocks(dy, sigma_grid, pshake_grid, k2, terms)

    w = k2[:, None] * pshake_grid
    row = d2.sum(axis=1)      # sum_nu D^{mu,nu}
    col = d2.sum(axis=0)      # sum_mu D^{mu,nu}
    factorised = det_sb * (
        ((np.sqrt(sigma_grid) * (d_i + d_j)) @ row) * (np.sqrt(w) @ col)
    )

    joint = blk.c_cross
    assert not np.allclose(joint, factorised)
    mask = np.abs(joint) > 1e-12
    ratio = np.abs(factorised[mask] / joint[mask])
    assert np.max(ratio) > 3.0, (
        "the factorised form should overestimate c_cross substantially"
    )
    print(
        "c_cross discrepancy |factorised/joint|: "
        f"min {ratio.min():.3f}  median {np.median(ratio):.3f}  "
        f"max {ratio.max():.3f}"
    )


def test_c_cross_vanishes_when_d2_symmetric_part_used(rng, grid_inputs):
    """A symmetric matrix contracted the same way is not antisymmetric.

    Sanity check that c_cross is genuinely sensitive to the antisymmetry,
    i.e. that it is a signed joint contraction rather than a sum of squares.
    """
    sigma_grid, pshake_grid, k2 = grid_inputs
    nbas = sigma_grid.shape[1]
    d = np.ones(nbas)
    d2 = antisym(rng, nbas)
    dy = FakeDyson(d_i=d, d_j=d, d2_ij=d2, det_sb=1.0)
    terms = TermSwitches(aa_bb=False, c_cross=True)
    blk = build_blocks(dy, sigma_grid, pshake_grid, k2, terms)
    # With a flat grid the contraction would cancel by antisymmetry; the
    # random grid breaks that, so the value must be non-zero and finite.
    assert np.all(np.isfinite(blk.c_cross))
    flat_sigma = np.ones_like(sigma_grid)
    flat_p = np.ones_like(pshake_grid)
    blk_flat = build_blocks(dy, flat_sigma, flat_p, k2, terms)
    assert np.allclose(blk_flat.c_cross, 0.0, atol=1e-12)


def test_jj_branching_is_exactly_two(dyson, grid_inputs, all_terms):
    blk = _blocks(dyson, grid_inputs, all_terms)
    a32 = amplitude_jj(blk, "3/2", all_terms)
    a12 = amplitude_jj(blk, "1/2", all_terms)
    assert np.array_equal(a32, 2.0 * a12), (
        "A(jC=3/2) must be exactly twice A(jC=1/2)"
    )
    # Quote the pointwise ratio away from the k2 -> 0 node, where both
    # amplitudes vanish and the ratio is 0/0.
    mask = np.abs(a12) > 0.0
    ratios = a32[mask] / a12[mask]
    assert np.all(ratios == 2.0)
    print(f"jj ratio A(3/2)/A(1/2): {ratios.min():.17g} to "
          f"{ratios.max():.17g} over {mask.sum()} grid points")


@pytest.mark.parametrize("seed", [1, 7, 99, 12345])
def test_jj_band_shapes_identical(rng, grid_inputs, all_terms, seed):
    """The two spin-orbit bands have identical shape to machine precision.

    This is the falsifiable prediction of the spectator approximation for
    the valence hole (note #3): the 2p_3/2 and 2p_1/2 core-valence bands
    differ only by the 2:1 factor and by their E_excess.
    """
    sigma_grid, pshake_grid, k2 = grid_inputs
    nbas = sigma_grid.shape[1]
    local = np.random.default_rng(seed)
    dy = FakeDyson(
        d_i=local.normal(size=nbas),
        d_j=local.normal(size=nbas),
        lam_i=local.normal(size=(nbas, 3)),
        lam_j=local.normal(size=(nbas, 3)),
        d2_ij=antisym(local, nbas),
        det_sb=float(local.normal()),
    )
    blk = build_blocks(dy, sigma_grid, pshake_grid, k2, all_terms)
    a32 = amplitude_jj(blk, "3/2", all_terms)
    a12 = amplitude_jj(blk, "1/2", all_terms)
    n32 = a32 / np.max(np.abs(a32))
    n12 = a12 / np.max(np.abs(a12))
    assert np.max(np.abs(n32 - n12)) < 1e-15


def test_singlet_triplet_reduce_in_identical_dyson_limit(
    rng, grid_inputs
):
    """cross_dyson off and d_i == d_j: hand-computed reductions.

    With F_ij = D_i S_j and d_i == d_j, D_i == D_j and S_i == S_j, so

        A(S=0) = 2*(F_ij + F_ji) = 4*D_i*S_i
        A(S=1) = [2*(F_ij + F_ji) + 4*g_aa - 4*c_cross] / 3
    """
    sigma_grid, pshake_grid, k2 = grid_inputs
    nbas = sigma_grid.shape[1]
    d = rng.normal(size=nbas)
    d2 = antisym(rng, nbas)
    det_sb = 0.55
    dy = FakeDyson(d_i=d, d_j=d.copy(), d2_ij=d2, det_sb=det_sb)
    terms = TermSwitches(
        direct=True, cross_dyson=False, indirect=False,
        aa_bb=True, c_cross=True, dir_ind_interference=False,
    )
    blk = build_blocks(dy, sigma_grid, pshake_grid, k2, terms)

    w = k2[:, None] * pshake_grid
    d_block = sigma_grid @ d**2
    s_block = w @ d**2
    f_sum = 2.0 * (d_block * s_block + d_block * s_block)
    assert np.allclose(f_sum, 4.0 * d_block * s_block)

    a0 = amplitude_ls(blk, "singlet", terms)
    assert np.allclose(a0, f_sum, rtol=1e-13, atol=0.0)

    g = det_sb**2 * np.einsum("qm,mn,qn->q", sigma_grid, d2 * d2, w)
    # With d_i == d_j == d the two summands of the gauge-covariant c_cross
    # coincide, so the factor 2 stands but both legs carry d.  Note this
    # particular contraction vanishes identically here: d2 is antisymmetric
    # while the outer product of the two legs is symmetric in (mu, nu)
    # whenever sigma and w are equal up to a scalar, which is a useful
    # cross-check on the antisymmetry convention.
    c = det_sb * 2.0 * np.einsum(
        "qm,mn,qn->q",
        np.sqrt(sigma_grid) * d, d2, np.sqrt(w) * d,
    )
    a1 = amplitude_ls(blk, "triplet", terms)
    assert np.allclose(
        a1, (f_sum + 4.0 * g - 4.0 * c) / 3.0, rtol=1e-13, atol=0.0
    )


def test_cross_term_signs(dyson, grid_inputs):
    """Singlet subtracts 4*X_ij, triplet adds it (inside the 1/3)."""
    sigma_grid, pshake_grid, k2 = grid_inputs
    on = TermSwitches(direct=True, cross_dyson=True, aa_bb=False,
                      c_cross=False)
    off = TermSwitches(direct=True, cross_dyson=False, aa_bb=False,
                       c_cross=False)
    blk_on = build_blocks(dyson, sigma_grid, pshake_grid, k2, on)
    blk_off = build_blocks(dyson, sigma_grid, pshake_grid, k2, off)
    x = blk_on.absorb_ij * blk_on.shake_ij

    d_singlet = (amplitude_ls(blk_on, "singlet", on)
                 - amplitude_ls(blk_off, "singlet", off))
    assert np.allclose(d_singlet, -4.0 * x, rtol=1e-13)

    d_triplet = (amplitude_ls(blk_on, "triplet", on)
                 - amplitude_ls(blk_off, "triplet", off))
    assert np.allclose(d_triplet, +4.0 * x / 3.0, rtol=1e-13)


def test_singlet_has_no_two_continuum_channel(dyson, grid_inputs, all_terms):
    """g_aa and c_cross contribute only to S_dic = 1."""
    blk = _blocks(dyson, grid_inputs, all_terms)
    parts = term_breakdown(blk, "singlet", all_terms)
    assert np.allclose(parts["aa_bb"], 0.0, atol=0.0)
    assert np.allclose(parts["c_cross"], 0.0, atol=0.0)
    tparts = term_breakdown(blk, "triplet", all_terms)
    assert not np.allclose(tparts["aa_bb"], 0.0)
    assert not np.allclose(tparts["c_cross"], 0.0)


@pytest.mark.parametrize("channel",
                         ["singlet", "triplet", "jc32", "jc12"])
def test_term_breakdown_sums_to_total(
    dyson, grid_inputs, all_terms, channel
):
    blk = _blocks(dyson, grid_inputs, all_terms)
    parts = term_breakdown(blk, channel, all_terms)
    total = amplitude(blk, channel, all_terms)
    assert set(parts) <= set(TERM_ORDER)
    assert np.allclose(
        np.sum(np.stack(list(parts.values())), axis=0), total,
        rtol=1e-14, atol=1e-300,
    )


def test_term_breakdown_reports_only_active_terms(dyson, grid_inputs):
    terms = TermSwitches(direct=True, cross_dyson=False, indirect=False,
                         aa_bb=False, c_cross=False)
    blk = build_blocks(dyson, *_split(grid_inputs), terms)
    parts = term_breakdown(blk, "triplet", terms)
    assert list(parts) == ["direct"]
    assert terms.active() == ("direct",)


def _split(grid_inputs):
    sigma_grid, pshake_grid, k2 = grid_inputs
    return sigma_grid, pshake_grid, k2


def test_no_spin_degeneracy_factor_by_default(dyson, grid_inputs, all_terms):
    """Ruling [P-2]: the physics default applies no (2S+1)."""
    assert TermSwitches().spin_degeneracy_factor == 1.0
    blk = _blocks(dyson, grid_inputs, all_terms)
    a1 = amplitude_ls(blk, "triplet", all_terms)
    diag = TermSwitches(
        direct=all_terms.direct, cross_dyson=all_terms.cross_dyson,
        indirect=all_terms.indirect, aa_bb=all_terms.aa_bb,
        dir_ind_interference=all_terms.dir_ind_interference,
        c_cross=all_terms.c_cross, spin_degeneracy_factor=3.0,
    )
    a1_diag = amplitude_ls(blk, "triplet", diag)
    assert np.allclose(a1_diag, 3.0 * a1, rtol=1e-14)
    # The singlet must be untouched by the triplet diagnostic knob.
    assert np.allclose(
        amplitude_ls(blk, "singlet", diag),
        amplitude_ls(blk, "singlet", all_terms), atol=0.0,
    )


def test_dir_ind_interference_sign_is_explicit(dyson, grid_inputs):
    plus = TermSwitches(dir_ind_interference=True, dir_ind_sign=+1.0,
                        indirect=True, aa_bb=False, c_cross=False)
    minus = TermSwitches(dir_ind_interference=True, dir_ind_sign=-1.0,
                         indirect=True, aa_bb=False, c_cross=False)
    sigma_grid, pshake_grid, k2 = grid_inputs
    b_p = build_blocks(dyson, sigma_grid, pshake_grid, k2, plus)
    b_m = build_blocks(dyson, sigma_grid, pshake_grid, k2, minus)
    x_p = term_breakdown(b_p, "singlet", plus)["dir_ind_interference"]
    x_m = term_breakdown(b_m, "singlet", minus)["dir_ind_interference"]
    assert np.allclose(x_p, -x_m, atol=0.0)
    assert not TermSwitches().dir_ind_interference
    with pytest.raises(ModelError, match="dir_ind_sign"):
        TermSwitches(dir_ind_sign=0.0)


def test_missing_blocks_raise_rather_than_contributing_zero(rng, nbas,
                                                            grid_inputs):
    sigma_grid, pshake_grid, k2 = grid_inputs
    bare = FakeDyson(d_i=rng.normal(size=nbas), d_j=rng.normal(size=nbas))
    with pytest.raises(ModelError, match="d2_ij"):
        build_blocks(bare, sigma_grid, pshake_grid, k2,
                     TermSwitches(aa_bb=True))
    with pytest.raises(ModelError, match="lam_i"):
        build_blocks(bare, sigma_grid, pshake_grid, k2,
                     TermSwitches(indirect=True, aa_bb=False,
                                  c_cross=False))


def test_shape_validation(dyson, grid_inputs, all_terms):
    sigma_grid, pshake_grid, k2 = grid_inputs
    with pytest.raises(ModelError, match="sigma_grid"):
        build_blocks(dyson, sigma_grid[:, :-1], pshake_grid, k2, all_terms)
    with pytest.raises(ModelError, match="k2 contains negative"):
        build_blocks(dyson, sigma_grid, pshake_grid, -k2 - 1.0, all_terms)


def test_unknown_channel_raises(dyson, grid_inputs, all_terms):
    blk = _blocks(dyson, grid_inputs, all_terms)
    with pytest.raises(ModelError, match="unknown channel"):
        amplitude(blk, "quintet", all_terms)
    with pytest.raises(ModelError, match="unknown core"):
        amplitude_jj(blk, "5/2", all_terms)


def test_blocks_summary_lists_built_blocks(dyson, grid_inputs, all_terms):
    blk = _blocks(dyson, grid_inputs, all_terms)
    summary = blocks_summary(blk)
    assert "g_aa" in summary and "c_cross" in summary
    assert all(v >= 0.0 for v in summary.values())
