"""Tests of the energy-sharing quadrature and the broadening."""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import FakeConfig, FakeDyson, antisym

from dpi import amplitudes as amp
from dpi import spectrum
from dpi.amplitudes import TermSwitches
from dpi.constants import ModelError, au_to_ev, ev_to_au


def _flat_case(nbas=6, omega_ev=100.0, dip_ev=40.0, seed=3):
    """A case whose integrand is constant in eps2, so the integral is exact.

    sigma and P^shake are taken energy independent, so every block is
    constant *except* for the factor k2 in the shake-off weight.  With
    k2 = t*sqrt(2E) the integrand of every product of one absorb block and
    one shake block is therefore proportional to t, and

        integral_0^1 (c*t) * (2E t) dt = 2*c*E/3

    which is checked below.
    """
    rng = np.random.default_rng(seed)
    d_i = rng.normal(size=nbas)
    d_j = rng.normal(size=nbas)
    sigma_ao = rng.uniform(0.2, 1.5, size=nbas)
    pshake_ao = rng.uniform(0.05, 0.4, size=nbas)
    dy = FakeDyson(
        d_i=d_i, d_j=d_j,
        d2_ij=antisym(rng, nbas), det_sb=0.61,
        p_i=0.88, p_j=0.79,
        meta={"index": 0, "label": "T1", "e_dication_ev": dip_ev},
    )
    terms = TermSwitches(direct=True, cross_dyson=False, indirect=False,
                         aa_bb=False, c_cross=False)
    cfg = FakeConfig(omega_ev=omega_ev, terms=terms,
                     sigma_ao=sigma_ao, pshake_ao=pshake_ao)
    return dy, cfg, dip_ev, sigma_ao, pshake_ao


def test_grid_substitution_relations():
    e_exc_ev = 60.0
    g = spectrum.quadrature_grid(e_exc_ev, n_quad=32)
    e_exc_au = ev_to_au(e_exc_ev)
    assert np.allclose(ev_to_au(g.eps2_ev), e_exc_au * g.t**2)
    assert np.allclose(g.eps1_ev + g.eps2_ev, e_exc_ev)
    assert np.allclose(g.k2_au, g.t * np.sqrt(2.0 * e_exc_au))
    assert np.allclose(g.jacobian_au, 2.0 * e_exc_au * g.t)
    # The measure integrates t -> eps2 exactly: integral d(eps2) = E_excess.
    assert np.isclose(float(np.sum(g.measure_au)), e_exc_au, rtol=1e-14)


def test_integral_of_constant_blocks_is_analytic():
    """Exact answer for the direct term with energy-independent inputs."""
    dy, cfg, dip_ev, sigma_ao, pshake_ao = _flat_case()
    res = spectrum.integrate_state(dy, cfg, dip_ev, "singlet", n_quad=64)

    e_exc_au = ev_to_au(cfg.omega_ev - dip_ev)
    # D_i, D_j are constant; S_i = k2 * (sum_nu d^2 P) = t*sqrt(2E)*s_i.
    d_i = float(sigma_ao @ dy.d_i**2)
    d_j = float(sigma_ao @ dy.d_j**2)
    s_i = float(pshake_ao @ dy.d_i**2)
    s_j = float(pshake_ao @ dy.d_j**2)
    c = 2.0 * (d_i * s_j + d_j * s_i) * np.sqrt(2.0 * e_exc_au)
    exact = c * 2.0 * e_exc_au / 3.0

    assert np.isclose(res.intensity, exact, rtol=1e-13), (
        f"quadrature {res.intensity!r} vs exact {exact!r}"
    )
    print(f"exact {exact:.15e}  quadrature {res.intensity:.15e}  "
          f"rel err {abs(res.intensity / exact - 1.0):.3e}")


class RealisticConfig:
    """Physically shaped inputs, so the integrand is not a polynomial in t.

    ``sigma_mu`` is a power law in ``eps1 + I_mu`` with a spread of AO
    thresholds, and ``P^shake_nu(k)`` is the ``k^(2l) exp(-k^2/2alpha)``
    momentum density of a GTO.  The flat case above is integrated exactly
    by any ``n_quad`` because its integrand is linear in ``t``, so it tests
    the Jacobian but not the convergence; this one does.
    """

    def __init__(self, omega_ev, terms, nbas=10, seed=7):
        r = np.random.default_rng(seed)
        self.omega_ev = float(omega_ev)
        self.terms = terms
        self.i_mu_ev = r.uniform(20.0, 700.0, size=nbas)
        self.sigma0 = r.uniform(0.05, 2.0, size=nbas)
        self.alpha = r.uniform(0.3, 6.0, size=nbas)
        self.l = r.integers(0, 3, size=nbas)
        self.display_shift_ev = 0.0

    def sigma_at_eps1_grid(self, eps1_ev):
        e = np.asarray(eps1_ev, float)[:, None] + self.i_mu_ev[None, :]
        return self.sigma0[None, :] * (e / 100.0) ** (-2.5)

    def p_shake_at_k(self, k_au):
        k = np.asarray(k_au, float)[:, None]
        return k ** (2 * self.l[None, :]) * np.exp(
            -(k**2) / (2.0 * self.alpha[None, :])
        )


def _realistic_case(nbas=10, seed=13):
    r = np.random.default_rng(seed)
    a = r.normal(size=(nbas, nbas))
    dy = FakeDyson(
        d_i=r.normal(size=nbas), d_j=r.normal(size=nbas),
        d2_ij=a - a.T, det_sb=0.63, p_i=0.9, p_j=0.85,
        meta={"index": 0, "label": "real"},
    )
    cfg = RealisticConfig(770.0, TermSwitches(), nbas=nbas)
    return dy, cfg, 40.0


def test_convergence_in_n_quad_flat_case():
    """The linear-in-t integrand is exact at any order."""
    dy, cfg, dip_ev, *_ = _flat_case()
    for n in (4, 8, 200):
        conv = spectrum.convergence(dy, cfg, dip_ev, "singlet", n_quad=n)
        assert conv["rel_residual"] < 1e-13


def test_convergence_in_n_quad_realistic_integrand():
    """n_quad = 200 is justified: the residual falls to ~1e-11 or below."""
    dy, cfg, dip_ev = _realistic_case()
    ref = spectrum.integrate_state(
        dy, cfg, dip_ev, "triplet", n_quad=2000
    ).intensity
    errs = []
    for n in (10, 25, 50, 100, 200):
        val = spectrum.integrate_state(
            dy, cfg, dip_ev, "triplet", n_quad=n
        ).intensity
        errs.append(abs(val / ref - 1.0))
        print(f"n_quad={n:4d}  intensity {val:.12e}  rel err {errs[-1]:.3e}")
    assert errs[0] > 1e-3, "the test integrand must be genuinely non-trivial"
    assert all(b <= a * 1.5 for a, b in zip(errs, errs[1:])), (
        f"residual not decreasing: {errs}"
    )
    assert errs[-1] < 1e-10

    conv = spectrum.convergence(dy, cfg, dip_ev, "triplet", n_quad=200)
    assert conv["rel_residual"] < 1e-10
    print(f"convergence() at n_quad=200: "
          f"rel_residual = {conv['rel_residual']:.3e}")


def test_closed_channel_returns_flagged_zero():
    dy, cfg, _, *_ = _flat_case(omega_ev=30.0, dip_ev=40.0)
    res = spectrum.integrate_state(dy, cfg, 40.0, "triplet", n_quad=16)
    assert res.open is False
    assert res.intensity == 0.0
    assert res.e_excess_ev < 0.0
    assert set(res.terms) == set(cfg.terms.active())
    assert all(v == 0.0 for v in res.terms.values())


def test_zero_excess_is_closed():
    dy, cfg, _, *_ = _flat_case(omega_ev=40.0, dip_ev=40.0)
    res = spectrum.integrate_state(dy, cfg, 40.0, "singlet", n_quad=16)
    assert res.open is False and res.intensity == 0.0


def test_excess_uses_state_dip_not_display_shift():
    """The cosmetic display shift must never enter E_excess."""
    dy, cfg, dip_ev, *_ = _flat_case()
    plain = spectrum.integrate_state(dy, cfg, dip_ev, "singlet", n_quad=32)
    cfg_shifted = FakeConfig(
        omega_ev=cfg.omega_ev, terms=cfg.terms,
        sigma_ao=cfg.sigma_ao, pshake_ao=cfg.pshake_ao,
        display_shift_ev=-12.5,
    )
    dy_noshift = FakeDyson(
        d_i=dy.d_i, d_j=dy.d_j, d2_ij=dy.d2_ij, det_sb=dy.det_sb,
        p_i=dy.p_i, p_j=dy.p_j,
        meta={"index": 0, "label": "T1", "e_dication_ev": dip_ev},
    )
    shifted = spectrum.integrate_state(
        dy_noshift, cfg_shifted, dip_ev, "singlet", n_quad=32
    )
    assert shifted.e_excess_ev == plain.e_excess_ev
    assert shifted.intensity == pytest.approx(plain.intensity, rel=1e-15)
    assert shifted.e_shifted_ev == pytest.approx(dip_ev - 12.5)


def test_per_term_integrals_sum_to_intensity():
    nbas = 7
    rng = np.random.default_rng(11)
    dy = FakeDyson(
        d_i=rng.normal(size=nbas), d_j=rng.normal(size=nbas),
        lam_i=rng.normal(size=(nbas, 3)), lam_j=rng.normal(size=(nbas, 3)),
        d2_ij=antisym(rng, nbas), det_sb=0.42,
        meta={"index": 1},
    )
    terms = TermSwitches(direct=True, cross_dyson=True, indirect=True,
                         aa_bb=True, c_cross=True,
                         dir_ind_interference=True)
    cfg = FakeConfig(omega_ev=120.0, terms=terms,
                     sigma_ao=rng.uniform(0.1, 1.0, size=nbas),
                     pshake_ao=rng.uniform(0.02, 0.3, size=nbas))
    # A jj channel needs both multiplicity orbital sets; a second FakeDyson
    # stands in for the S_dic = 1 solution.
    dy_t = FakeDyson(
        d_i=rng.normal(size=nbas), d_j=rng.normal(size=nbas),
        lam_i=rng.normal(size=(nbas, 3)), lam_j=rng.normal(size=(nbas, 3)),
        d2_ij=antisym(rng, nbas), det_sb=0.39,
        meta={"index": 1},
    )
    for channel in ("singlet", "triplet"):
        res = spectrum.integrate_state(dy, cfg, 50.0, channel, n_quad=48)
        assert np.isclose(sum(res.terms.values()), res.intensity, rtol=1e-13)
    for channel in ("jc32", "jc12"):
        res = spectrum.integrate_state(dy, cfg, 50.0, channel, n_quad=48,
                                       dyson_triplet=dy_t)
        assert np.isclose(sum(res.terms.values()), res.intensity, rtol=1e-13)


def test_negative_intensity_is_reported_not_clamped():
    """Bug [B-3]: a negative intensity must survive into the output.

    A two-AO state with orthogonal Dyson orbitals and a large
    ``d2_ij``: the triplet's ``-4*c_cross`` is a signed joint contraction
    with no positivity bound, so it overwhelms the positive-definite direct
    term.  This -- not the singlet's ``-4*X_ij``, which is bounded (see
    :func:`test_singlet_is_non_negative`) -- is the channel the guard
    against clamping protects.
    """
    dy = FakeDyson(
        d_i=np.array([1.0, 0.0]), d_j=np.array([0.0, 1.0]),
        d2_ij=np.array([[0.0, 5.0], [-5.0, 0.0]]), det_sb=1.0,
        meta={"index": 2, "label": "neg"},
    )
    terms = TermSwitches(direct=True, cross_dyson=True, indirect=False,
                         aa_bb=False, c_cross=True)
    cfg = FakeConfig(omega_ev=90.0, terms=terms,
                     sigma_ao=np.array([4.0, 1.0]),
                     pshake_ao=np.array([0.01, 1.0]))
    res = spectrum.integrate_state(dy, cfg, 40.0, "triplet", n_quad=64)
    assert res.intensity < 0.0
    assert res.negative is True
    assert res.amplitude_min < 0.0
    assert res.terms["c_cross"] < 0.0
    assert res.terms["direct"] > 0.0
    print(f"triplet intensity {res.intensity:.6e} "
          f"(direct {res.terms['direct']:.6e}, "
          f"c_cross {res.terms['c_cross']:.6e})")

    # The same state's singlet stays positive, as the bound requires.
    sing = spectrum.integrate_state(dy, cfg, 40.0, "singlet", n_quad=64)
    assert sing.intensity > 0.0 and sing.negative is False


def test_singlet_is_non_negative():
    """Cauchy-Schwarz plus AM-GM bound the singlet's -4 X_ij term.

    Sampled over random states, including the nearly-parallel regime
    REVIEW.md [B-3] identifies as dangerous.  With the physics terms on,
    A(S=0) >= 0; the bound fails only for the optional X_f term, whose
    sign is a free choice.
    """
    rng = np.random.default_rng(4242)
    nbas, nq = 10, 24
    terms = TermSwitches(direct=True, cross_dyson=True, indirect=True,
                         aa_bb=False, c_cross=False)
    worst = np.inf
    for trial in range(300):
        d_i = rng.normal(size=nbas)
        # Half the trials are nearly parallel, where the bound is tight.
        d_j = (d_i + 0.01 * rng.normal(size=nbas) if trial % 2 == 0
               else rng.normal(size=nbas))
        dy = FakeDyson(
            d_i=d_i, d_j=d_j,
            lam_i=rng.normal(size=(nbas, 3)),
            lam_j=rng.normal(size=(nbas, 3)),
        )
        blk = amp.build_blocks(
            dy,
            rng.uniform(0.0, 2.0, size=(nq, nbas)),
            rng.uniform(0.0, 1.0, size=(nq, nbas)),
            rng.uniform(0.0, 3.0, size=nq),
            terms,
        )
        a0 = amp.amplitude_ls(blk, "singlet", terms)
        worst = min(worst, float(a0.min()))
    assert worst >= 0.0, f"singlet went negative: {worst!r}"
    print(f"min A(S=0) over 300 random states: {worst:.6e}")


def test_prefactor_is_one_and_documented():
    assert spectrum.prefactor(770.0) == 1.0
    assert spectrum.prefactor(2610.0) == 1.0
    doc = spectrum.prefactor.__doc__ or ""
    assert "relative" in doc.lower() and "Mb*a.u." in doc
    with pytest.raises(ModelError):
        spectrum.prefactor(float("nan"))


def test_state_result_carries_spectroscopic_factors():
    dy, cfg, dip_ev, *_ = _flat_case()
    res = spectrum.integrate_state(dy, cfg, dip_ev, "singlet", n_quad=16)
    assert res.p_i == pytest.approx(0.88)
    assert res.p_j == pytest.approx(0.79)
    assert res.label == "T1"
    assert res.n_quad == 16


def test_voigt_integrates_to_its_area():
    area = 3.7
    x = np.linspace(-400.0, 400.0, 400001)
    for sigma_g, gamma_l in ((0.4, 0.0), (0.0, 0.3), (0.4, 0.3)):
        y = spectrum.voigt(x, 12.0, area, sigma_g, gamma_l)
        integral = np.trapezoid(y, x) if hasattr(np, "trapezoid") \
            else np.trapezoid(y, x)
        assert np.isclose(integral, area, rtol=2e-3), (
            f"sigma_g={sigma_g} gamma_l={gamma_l}: {integral}"
        )
        print(f"voigt area sigma_g={sigma_g} gamma_l={gamma_l}: "
              f"{integral:.6f} (target {area})")


def test_voigt_limits_match_analytic_profiles():
    x = np.linspace(-5.0, 5.0, 101)
    gauss = spectrum.voigt(x, 0.0, 1.0, 0.7, 0.0)
    ref_g = np.exp(-0.5 * (x / 0.7) ** 2) / (0.7 * np.sqrt(2.0 * np.pi))
    assert np.allclose(gauss, ref_g, rtol=1e-14)
    lorentz = spectrum.voigt(x, 0.0, 1.0, 0.0, 0.5)
    ref_l = 0.5 / (np.pi * (x**2 + 0.25))
    assert np.allclose(lorentz, ref_l, rtol=1e-14)
    with pytest.raises(ModelError):
        spectrum.voigt(x, 0.0, 1.0, 0.0, 0.0)
    with pytest.raises(ModelError):
        spectrum.voigt(x, 0.0, 1.0, -0.1, 0.2)


def _mkstate(index, e, intensity, channel="singlet"):
    return spectrum.StateResult(
        index=index, channel=channel, label=f"S{index}",
        e_dication_ev=e, e_shifted_ev=e, dip_ev=e, e_excess_ev=10.0,
        intensity=intensity, terms={"direct": intensity}, open=True,
    )


def test_broaden_is_linear_in_intensity():
    grid = np.linspace(0.0, 40.0, 801)
    states = [_mkstate(0, 12.0, 2.0), _mkstate(1, 18.0, -0.5)]
    y1 = spectrum.broaden(grid, states, 0.5, 0.2)
    scaled = [_mkstate(0, 12.0, 4.0), _mkstate(1, 18.0, -1.0)]
    y2 = spectrum.broaden(grid, scaled, 0.5, 0.2)
    assert np.allclose(y2, 2.0 * y1, rtol=1e-14)

    # additivity over states
    ya = spectrum.broaden(grid, states[:1], 0.5, 0.2)
    yb = spectrum.broaden(grid, states[1:], 0.5, 0.2)
    assert np.allclose(y1, ya + yb, rtol=1e-14)


def test_broaden_matches_voigt_sum():
    grid = np.linspace(5.0, 30.0, 501)
    states = [_mkstate(0, 12.0, 2.0), _mkstate(1, 18.5, 1.25)]
    ref = sum(
        spectrum.voigt(grid, s.e_shifted_ev, s.intensity, 0.6, 0.25)
        for s in states
    )
    assert np.allclose(spectrum.broaden(grid, states, 0.6, 0.25), ref,
                       rtol=1e-14)


def test_broaden_conserves_total_area():
    grid = np.linspace(-200.0, 250.0, 450001)
    states = [_mkstate(0, 12.0, 2.0), _mkstate(1, 18.5, 1.25)]
    y = spectrum.broaden(grid, states, 0.6, 0.25)
    area = np.trapezoid(y, grid) if hasattr(np, "trapezoid") \
        else np.trapezoid(y, grid)
    assert np.isclose(area, 3.25, rtol=2e-3)


def test_broaden_handles_closed_states_and_empty_list():
    grid = np.linspace(0.0, 20.0, 101)
    closed = spectrum.StateResult(
        index=0, channel="singlet", label=None, e_dication_ev=10.0,
        e_shifted_ev=10.0, dip_ev=10.0, e_excess_ev=-3.0, intensity=0.0,
        open=False,
    )
    assert np.allclose(spectrum.broaden(grid, [closed], 0.5, 0.1), 0.0)
    assert np.allclose(spectrum.broaden(grid, [], 0.5, 0.1), 0.0)


def test_quadrature_grid_rejects_bad_arguments():
    with pytest.raises(ModelError, match="positive"):
        spectrum.quadrature_grid(-1.0, n_quad=8)
    with pytest.raises(ModelError, match="n_quad"):
        spectrum.quadrature_grid(10.0, n_quad=1)


def test_integrate_state_rejects_unknown_channel_and_bad_cfg():
    dy, cfg, dip_ev, *_ = _flat_case()
    with pytest.raises(ModelError, match="unknown channel"):
        spectrum.integrate_state(dy, cfg, dip_ev, "doublet")

    class NoTerms:
        omega_ev = 100.0
    with pytest.raises(ModelError, match="terms"):
        spectrum.integrate_state(dy, NoTerms(), dip_ev, "singlet")

    class NoSigma:
        omega_ev = 100.0
        terms = TermSwitches(aa_bb=False, c_cross=False)
    with pytest.raises(ModelError, match="sigma"):
        spectrum.integrate_state(dy, NoSigma(), dip_ev, "singlet")


def test_eps1_never_negative_at_endpoint():
    g = spectrum.quadrature_grid(1e-6, n_quad=200)
    assert np.all(g.eps1_ev >= 0.0)
    assert np.all(g.k2_au >= 0.0)
    assert au_to_ev(g.e_excess_au) == pytest.approx(1e-6, rel=1e-12)

def test_jj_needs_both_multiplicity_orbital_sets():
    """A j_C peak is A(S=0)+A(S=1), from two DIFFERENT dication states.

    The two LS multiplicities have separately converged OSRHF orbitals, so
    evaluating both amplitudes from one Blocks is wrong. With real SF6
    orbital sets that error moved the branching ratio from 2.0 to ~200.
    Omitting the second set must therefore be refused, not silently wrong.
    """
    nbas = 8
    rng = np.random.default_rng(4)

    def fake(seed_scale, det):
        return FakeDyson(
            d_i=rng.normal(size=nbas), d_j=rng.normal(size=nbas),
            d2_ij=antisym(rng, nbas) * seed_scale, det_sb=det,
            meta={"index": 1})

    dy_s, dy_t = fake(1.0, 0.91), fake(0.6, 0.88)
    terms = TermSwitches(direct=True, cross_dyson=True, indirect=False,
                         aa_bb=True, c_cross=True)
    cfg = FakeConfig(omega_ev=120.0, terms=terms,
                     sigma_ao=rng.uniform(0.1, 1.0, size=nbas),
                     pshake_ao=rng.uniform(0.02, 0.3, size=nbas))

    for channel in ("jc32", "jc12"):
        try:
            spectrum.integrate_state(dy_s, cfg, 50.0, channel, n_quad=32)
        except ModelError as exc:
            assert "S_dic=0 and S_dic=1" in str(exc), str(exc)
        else:
            raise AssertionError(
                f"{channel} accepted a single orbital set")

    a32 = spectrum.integrate_state(dy_s, cfg, 50.0, "jc32", n_quad=200,
                                   dyson_triplet=dy_t).intensity
    a12 = spectrum.integrate_state(dy_s, cfg, 50.0, "jc12", n_quad=200,
                                   dyson_triplet=dy_t).intensity
    # The 2:1 branching is a pure recoupling factor and must be exact.
    assert abs(a32 / a12 - 2.0) < 1e-12, a32 / a12

    # And jc32 must equal S=0 from the singlet set plus S=1 from the triplet
    # set -- each amplitude taken from ITS OWN orbitals.
    a0 = spectrum.integrate_state(dy_s, cfg, 50.0, "singlet",
                                  n_quad=200).intensity
    a1 = spectrum.integrate_state(dy_t, cfg, 50.0, "triplet",
                                  n_quad=200).intensity
    assert np.isclose(a32, a0 + a1, rtol=1e-12), (a32, a0 + a1)

    # Sourcing both from the singlet set -- the old behaviour -- must differ,
    # or this test would not be able to detect the bug it guards.
    wrong = spectrum.integrate_state(dy_s, cfg, 50.0, "jc32", n_quad=200,
                                     dyson_triplet=dy_s).intensity
    assert not np.isclose(wrong, a32, rtol=1e-6), (
        "the fixture's two orbital sets are too similar to discriminate")

