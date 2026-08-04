"""Tests for dpi.atomic_sigma.

Covers the four evaluation regions, exactness at the tabulated knots, the
overridable high-energy exponent, and the AO -> subshell resolution of
SigmaBuilder (including that the grid evaluation equals a loop over single
energies, which is the vectorisation that replaces bug [C-4]).
"""

from __future__ import annotations

import numpy as np
import pytest

from dpi.atomic_sigma import (
    DEFAULT_SUBSHELL_MAP,
    HYDROGENIC_EXPONENT,
    REGION_BELOW_THRESHOLD,
    REGION_LINEAR_RISE,
    REGION_POWER_LAW,
    REGION_TABULATED,
    SigmaBuilder,
    YEH_LINDAU,
    provenance_note,
    sigma,
    sigma_region,
    threshold,
)
from dpi.constants import ConfigError

from .basis_stub import make_sf6_like_basis, make_spherical_basis

ALL_KEYS = sorted(YEH_LINDAU)


def test_table_is_complete_and_consistent():
    """Every S and F subshell of the old table survives, with equal lengths."""
    assert set(ALL_KEYS) == {
        "S_1s", "S_2s", "S_2p", "S_3s", "S_3p", "F_1s", "F_2s", "F_2p",
    }
    for key in ALL_KEYS:
        d = YEH_LINDAU[key]
        assert len(d.hv_ev) == len(d.sigma_mb) > 0
        assert d.threshold_ev > 0.0
        assert all(s > 0.0 for s in d.sigma_mb)
        assert list(d.hv_ev) == sorted(d.hv_ev)
        assert d.key == key


@pytest.mark.parametrize(
    "element,subshell,expected",
    [("F", "1s", 692.3), ("S", "1s", 2472.0), ("S", "2p", 170.2)],
)
def test_threshold_spot_values(element, subshell, expected):
    """Thresholds quoted in the manuscript."""
    assert threshold(element, subshell) == pytest.approx(expected)


def test_zero_at_and_below_threshold():
    """sigma is exactly zero at or below threshold, not merely small."""
    for key in ALL_KEYS:
        d = YEH_LINDAU[key]
        thr = d.threshold_ev
        hv = np.array([0.0, 0.5 * thr, thr - 1e-9, thr])
        got = sigma(d.element, d.subshell, hv)
        assert np.all(got == 0.0)
        assert np.all(
            sigma_region(d.element, d.subshell, hv)
            == REGION_BELOW_THRESHOLD
        )
    # F 1s at 500 eV is below its 692.3 eV threshold.
    assert sigma("F", "1s", 500.0) == 0.0


def test_reproduces_tabulated_values_at_knots():
    """A log-log spline passes through its knots to round-off."""
    worst = 0.0
    for key in ALL_KEYS:
        d = YEH_LINDAU[key]
        hv = np.asarray(d.hv_ev)
        sg = np.asarray(d.sigma_mb)
        keep = hv > d.threshold_ev
        got = sigma(d.element, d.subshell, hv[keep])
        worst = max(worst, float(np.max(np.abs(got / sg[keep] - 1.0))))
    assert worst < 1e-10


@pytest.mark.parametrize(
    "element,subshell,hv_ev,expected",
    [
        ("F", "1s", 770.0, 0.3224),
        ("F", "1s", 1486.6, 0.06041),
        ("S", "2p", 300.0, 1.920),
        ("S", "2s", 300.0, 0.3134),
        ("S", "3s", 100.0, 0.2397),
        ("S", "3p", 100.0, 0.5370),
        ("F", "2s", 100.0, 0.5686),
    ],
)
def test_spot_values_against_table(element, subshell, hv_ev, expected):
    """Values read off the tabulation itself."""
    assert sigma(element, subshell, hv_ev) == pytest.approx(
        expected, rel=1e-10
    )


def test_continuity_across_first_tabulated_point():
    """The linear rise meets the spline at hv_lo without a jump."""
    for key in ALL_KEYS:
        d = YEH_LINDAU[key]
        hv = np.asarray(d.hv_ev)
        keep = hv > d.threshold_ev
        hv_lo = float(hv[keep][0])
        at = sigma(d.element, d.subshell, hv_lo)
        # Both one-sided limits converge on the knot value linearly in the
        # probe offset, so halving the offset must halve the discrepancy;
        # a jump discontinuity would leave a floor instead.
        gaps = []
        for eps in (1e-6, 5e-7):
            below = sigma(d.element, d.subshell, hv_lo - eps)
            above = sigma(d.element, d.subshell, hv_lo + eps)
            assert below == pytest.approx(at, rel=1e-4)
            assert above == pytest.approx(at, rel=1e-4)
            gaps.append(abs(below - at))
        assert gaps[1] < 0.6 * gaps[0]


def test_linear_rise_is_linear_and_flagged():
    """Region 1 rises linearly from zero at threshold."""
    d = YEH_LINDAU["F_1s"]
    thr = d.threshold_ev
    hv_lo = 695.0
    mid = 0.5 * (thr + hv_lo)
    s_lo = sigma("F", "1s", hv_lo)
    s_mid = sigma("F", "1s", mid)
    assert s_mid == pytest.approx(0.5 * s_lo, rel=1e-12)
    assert sigma_region("F", "1s", mid) == REGION_LINEAR_RISE
    assert sigma_region("F", "1s", hv_lo) == REGION_TABULATED


def test_power_law_region_uses_requested_exponent():
    """The tail follows the exponent asked for, and is continuous."""
    hv = np.array([2000.0, 4000.0])
    for exponent in (-3.0, HYDROGENIC_EXPONENT, -4.0):
        got = sigma("F", "1s", hv, high_energy_exponent=exponent)
        slope = np.log(got[1] / got[0]) / np.log(hv[1] / hv[0])
        assert slope == pytest.approx(exponent, rel=1e-12)
    # continuity at the last tabulated point, whatever the exponent
    hv_hi = 1500.0
    s_hi = sigma("F", "1s", hv_hi)
    s_eps = sigma("F", "1s", hv_hi + 1e-9,
                  high_energy_exponent=HYDROGENIC_EXPONENT)
    assert s_eps == pytest.approx(s_hi, rel=1e-11)
    assert sigma_region("F", "1s", 2610.0) == REGION_POWER_LAW


def test_default_exponent_is_last_two_point_slope():
    """With no override the tail slope comes from the last two points."""
    d = YEH_LINDAU["S_2p"]
    hv = np.asarray(d.hv_ev)
    sg = np.asarray(d.sigma_mb)
    want = (np.log(sg[-1] / sg[-2])) / (np.log(hv[-1] / hv[-2]))
    got_hv = np.array([1600.0, 3200.0])
    got = sigma("S", "2p", got_hv)
    slope = np.log(got[1] / got[0]) / np.log(got_hv[1] / got_hv[0])
    assert slope == pytest.approx(want, rel=1e-12)


def test_positive_exponent_rejected():
    """A rising high-energy cross section is unphysical."""
    with pytest.raises(ConfigError, match="must be negative"):
        sigma("F", "1s", 3000.0, high_energy_exponent=1.5)


def test_unknown_subshell_raises_config_error():
    with pytest.raises(ConfigError, match="no atomic cross section"):
        sigma("Xe", "4d", 100.0)
    with pytest.raises(ConfigError):
        threshold("S", "4f")


def test_vectorisation_matches_scalar_calls():
    """Array evaluation equals a loop of scalar evaluations."""
    hv = np.array([100.0, 692.4, 700.0, 1234.5, 1500.0, 2610.0, 9000.0])
    for key in ALL_KEYS:
        d = YEH_LINDAU[key]
        vec = sigma(d.element, d.subshell, hv)
        loop = np.array(
            [sigma(d.element, d.subshell, float(x)) for x in hv]
        )
        assert np.allclose(vec, loop, rtol=0.0, atol=0.0)


def test_shape_preserved_for_2d_input():
    hv = np.array([[700.0, 800.0], [900.0, 1000.0]])
    assert sigma("F", "1s", hv).shape == (2, 2)


def test_s1s_entry_is_flagged_as_an_estimate():
    """S 1s is a hydrogenic estimate, and says so in its provenance."""
    assert YEH_LINDAU["S_1s"].is_tabulated is False
    assert YEH_LINDAU["F_1s"].is_tabulated is True
    note = provenance_note(["S_1s"])
    assert "hydrogenic" in note
    assert "NOT a Yeh-Lindau tabulation" in note


class TestSigmaBuilder:
    """The per-AO builder that resolves the subshell map once."""

    def test_sf6_like_basis_assignment(self):
        """cc-pVDZ S + 6 F gives nbas = 102 with 7 polarisation shells."""
        basis = make_sf6_like_basis()
        assert basis.nbas == 102
        builder = SigmaBuilder(basis)
        # Every AO of the default map is assigned; the map covers d
        # shells too, so nothing is unassigned in this layout.
        assert builder.n_unassigned + builder.report()["n_assigned"] == 102
        assert set(builder.subshells_used) <= set(YEH_LINDAU)
        thr = builder.threshold_ev
        assigned = np.array([k is not None for k in builder.ao_keys])
        assert np.all(thr[assigned] > 0.0)
        assert np.all(thr[~assigned] == 0.0)

    def test_unassigned_aos_are_counted_and_zero(self):
        """AOs with no tabulated subshell get sigma = 0 and are reported."""
        basis = make_sf6_like_basis()
        # Drop the d entry so the 5 d AOs per centre become unassigned.
        trimmed = {
            el: {k: v for k, v in m.items() if k[0] != 2}
            for el, m in DEFAULT_SUBSHELL_MAP.items()
        }
        builder = SigmaBuilder(basis, subshell_map=trimmed)
        n_d = int(np.sum(np.asarray(basis.l) == 2))
        assert builder.n_unassigned == n_d
        assert builder.report()["n_unassigned"] == n_d
        vals = builder.at_eps1(300.0)
        assert np.all(vals[builder.unassigned_indices] == 0.0)
        assert np.any(vals > 0.0)

    def test_at_eps1_grid_equals_loop_of_at_eps1(self):
        """The vectorised grid pass reproduces the per-point evaluation."""
        basis = make_sf6_like_basis()
        builder = SigmaBuilder(basis)
        grid = np.array([5.0, 50.0, 137.0, 500.0, 2000.0])
        block = builder.at_eps1_grid(grid)
        loop = np.array([builder.at_eps1(float(e)) for e in grid])
        assert block.shape == (grid.size, basis.nbas)
        assert np.array_equal(block, loop)

    def test_argument_is_eps1_plus_atomic_threshold(self):
        """Remark 1: sigma_mu is read at eps1 + I_mu, per AO."""
        basis = make_sf6_like_basis()
        builder = SigmaBuilder(basis)
        eps1 = 240.0
        vals = builder.at_eps1(eps1)
        keys = builder.ao_keys
        for mu, key in enumerate(keys):
            if key is None:
                continue
            d = YEH_LINDAU[key]
            want = sigma(d.element, d.subshell, eps1 + d.threshold_ev)
            assert vals[mu] == pytest.approx(want, rel=0.0, abs=0.0)
        # An AO of a deep subshell is therefore *not* evaluated at eps1.
        mu_1s = keys.index("S_1s")
        assert vals[mu_1s] != sigma("S", "1s", eps1)

    def test_naive_argument_would_suppress_valence_sigma(self):
        """Why Remark 1 matters: the naive omega - eps2 form is far off."""
        # A valence subshell read at a core-level DIP sits deep in the
        # power-law tail and is suppressed by orders of magnitude.
        eps1, dip = 100.0, 700.0
        proper = sigma("F", "2p", eps1 + threshold("F", "2p"))
        naive = sigma("F", "2p", eps1 + dip)
        assert proper > 50.0 * naive

    def test_linear_rise_flag_is_set_when_used(self):
        """used_linear_rise reports reliance on the near-threshold stopgap."""
        basis = make_sf6_like_basis()
        builder = SigmaBuilder(basis)
        assert builder.used_linear_rise is False
        builder.at_eps1_grid(np.array([500.0]))
        assert builder.used_linear_rise is False
        # eps1 = 1 eV puts every subshell just above its own threshold,
        # inside the linear-rise region.
        builder.at_eps1_grid(np.array([1.0]))
        assert builder.used_linear_rise is True
        assert builder.report()["used_linear_rise"] is True

    def test_high_energy_exponent_override_per_subshell(self):
        """A per-subshell exponent reaches the per-AO evaluation."""
        basis = make_sf6_like_basis()
        plain = SigmaBuilder(basis)
        steep = SigmaBuilder(
            basis, high_energy_exponents={"F_1s": HYDROGENIC_EXPONENT}
        )
        eps1 = 2000.0                       # F 1s tail, above 1500 eV
        mu = plain.ao_keys.index("F_1s")
        a = plain.at_eps1(eps1)[mu]
        b = steep.at_eps1(eps1)[mu]
        assert a != pytest.approx(b, rel=1e-6)
        want = sigma("F", "1s", eps1 + threshold("F", "1s"),
                     high_energy_exponent=HYDROGENIC_EXPONENT)
        assert b == pytest.approx(want, rel=0.0, abs=0.0)
        # An unrelated subshell is untouched by the override.
        mu_2p = plain.ao_keys.index("F_2p")
        assert plain.at_eps1(eps1)[mu_2p] == steep.at_eps1(eps1)[mu_2p]

    def test_bad_override_key_rejected(self):
        basis = make_sf6_like_basis()
        with pytest.raises(ConfigError, match="unknown subshell"):
            SigmaBuilder(basis, high_energy_exponents={"S_9z": -3.0})

    def test_map_to_untabulated_subshell_rejected(self):
        """A map naming a subshell with no data fails loudly at build time."""
        basis = make_sf6_like_basis()
        bad = {"S": {(0, 0): "7s"}, "F": {}}
        with pytest.raises(ConfigError, match="no tabulated cross section"):
            SigmaBuilder(basis, subshell_map=bad)

    def test_missing_basis_field_rejected(self):
        class Incomplete:
            nbas = 3
            elements = ["S", "S", "S"]
            l = np.zeros(3, dtype=int)

        with pytest.raises(ConfigError, match="shell_index"):
            SigmaBuilder(Incomplete())

    def test_no_hdf5_reopening(self, monkeypatch):
        """The builder never touches h5py: the map is resolved once."""
        import builtins

        real_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name == "h5py":
                raise AssertionError(
                    "SigmaBuilder must not import h5py at evaluation time"
                )
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guard)
        builder = SigmaBuilder(make_sf6_like_basis())
        builder.at_eps1_grid(np.linspace(1.0, 500.0, 200))

    def test_synthetic_spherical_basis_has_no_f_assignment(self):
        """An l = 3 shell has no tabulated atomic subshell: counted, zero."""
        basis = make_spherical_basis()
        builder = SigmaBuilder(basis)
        n_f = int(np.sum(np.asarray(basis.l) == 3))
        assert builder.n_unassigned >= n_f
        vals = builder.at_eps1(200.0)
        assert np.all(vals[np.asarray(basis.l) == 3] == 0.0)
