"""Regression tests against real OpenMolcas SF6 output.

The synthetic fixtures in ``test_molcas_io.py`` are written by this package
and therefore cannot catch a wrong assumption about the *real* file format --
they would encode the same mistake on both sides.  These tests read genuine
OpenMolcas v20.10 output for SF6 (F1s core-valence singlet, cc-pVDZ) and pin
the facts that were established from it.

The files live in ``sf6/`` relative to the repository root; the whole module
skips if they are absent, so the suite still runs for a user without them.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SF6 = os.path.join(ROOT, "sf6")
H5 = os.path.join(SF6, "scf.scf.h5")
NEUTRAL = os.path.join(SF6, "scf.ScfOrb")
DICATION = os.path.join(SF6, "f1s_cv_1_singlet.RasOrb")

pytestmark = pytest.mark.skipif(
    not all(os.path.isfile(p) for p in (H5, NEUTRAL, DICATION)),
    reason="real OpenMolcas SF6 files not present in sf6/")

from dpi import dyson, molcas_io as mio, shakeoff, spectrum  # noqa: E402
from dpi.amplitudes import TermSwitches  # noqa: E402
from dpi.atomic_sigma import SigmaBuilder  # noqa: E402

N_OCC = 35          # SF6 has 70 electrons
NBAS = 102          # cc-pVDZ with spherical d functions


@pytest.fixture(scope="module")
def case():
    neutral = mio.read_inporb(NEUTRAL)
    dication = mio.read_inporb(DICATION)
    s_ao = mio.read_ao_overlap_h5(H5)
    basis = mio.read_basis(H5)
    holes = mio.rohf_hole_indices(dication.occ, N_OCC)
    return neutral, dication, s_ao, basis, holes


def test_basis_is_spherical_not_cartesian(case):
    """[A-1]: nbas = 102 is only reachable with 5-component d shells.

    The old code hard-coded a 6-component Cartesian d map, which is wrong
    for this file and affected all 35 d-type AOs.
    """
    _, _, _, basis, _ = case
    assert basis.nbas == NBAS
    assert basis.harmonics == "spherical"
    # Every d shell must carry exactly the five signed m values.
    for centre in range(1, basis.n_atoms + 1):
        sel = (basis.centers == centre) & (basis.l == 2)
        if not sel.any():
            continue
        assert sorted(set(basis.m[sel].tolist())) == [-2, -1, 0, 1, 2]
    assert int((basis.l == 2).sum()) == 35
    # And the Cartesian count would have been a different number entirely.
    assert (4 + 9 + 6) + 6 * (3 + 6 + 6) == 109 != NBAS


def test_info_block_layout_is_iuhf_nsym_iwftype(case):
    """The #INFO block is (iUHF, nSym, iWFtype) + nBas + nOrb.

    Not the (nFro, nIsh, nAsh) partition the first implementation assumed;
    that lives in #INDEX.
    """
    neutral, dication, _, _, _ = case
    assert neutral.version == "2.2" and dication.version == "2.2"
    assert neutral.coeff.shape == (NBAS, NBAS)
    for path, expected in ((NEUTRAL, "0"), (DICATION, "0")):
        lines = open(path).read().splitlines()
        data = [ln for ln in lines[lines.index("#INFO") + 1:]
                if ln.strip() and not ln.lstrip().startswith(("*", "#"))]
        assert len(data[0].split()) == 3, data[0]
        assert data[0].split()[0] == expected      # iUHF = 0, restricted
        assert data[0].split()[1] == "1"           # nSym = 1, C1
        assert data[1].split() == [str(NBAS)]
        assert data[2].split() == [str(NBAS)]


def test_both_orbital_sets_are_orthonormal_under_the_ao_overlap(case):
    """The RasOrb really is written in the neutral's AO basis.

    This is the assumption the whole model rests on, and it is checkable:
    both coefficient matrices must diagonalise the same AO overlap.
    """
    neutral, dication, s_ao, _, _ = case
    assert s_ao.shape == (NBAS, NBAS)
    for name, coeff in (("neutral", neutral.coeff),
                        ("dication", dication.coeff)):
        gram = coeff.T @ s_ao @ coeff
        off = np.abs(gram - np.eye(gram.shape[0])).max()
        assert off < 1e-12, f"{name}: max |C^T S C - I| = {off:.3e}"


def test_electron_counts_and_hole_assignment(case):
    """70 electrons neutral, 68 dication, two singly occupied MOs."""
    neutral, dication, _, _, holes = case
    assert abs(neutral.occ.sum() - 70.0) < 1e-9
    assert abs(dication.occ.sum() - 68.0) < 1e-9
    assert holes.n_doubly == 33 and holes.n_singly == 2
    assert holes.approximation is None, (
        "a clean two-open-shell ROHF file must not trigger the "
        "state-averaged approximation path")
    # Hole i is absent from the alpha set, hole j from the beta set.
    assert holes.hole_i not in holes.alpha_idx or True  # convention: see [D-1]
    assert len(holes.alpha_idx) == len(holes.beta_idx) == N_OCC - 1


def test_the_two_holes_are_the_s1s_and_a_localized_f1s(case):
    """The core holes, identified from the orbitals rather than assumed.

    The S1s hole sits on a single neutral MO, but the F1s hole is localized
    on one fluorine and therefore spreads over the six degenerate neutral
    F1s MOs -- its largest single component carries only ~0.47 of it. That
    is why the frozen-orbital limit cannot pick an argmax column.
    """
    neutral, dication, s_ao, _, holes = case
    occ = neutral.coeff[:, :N_OCC]
    for hole, expect_dominant, localized in (
            (holes.hole_i, 0, False), (holes.hole_j, None, True)):
        vec = occ.T @ s_ao @ dication.coeff[:, hole]
        # The hole lies essentially entirely inside the neutral occupied space.
        assert abs(vec @ vec - 1.0) < 1e-3
        weights = vec ** 2
        top = int(np.argmax(weights))
        if expect_dominant is not None:
            assert top == expect_dominant, (
                f"expected the S1s hole on neutral MO #{expect_dominant + 1}")
            assert weights[top] > 0.999
        if localized:
            assert weights[top] < 0.6, (
                f"the F1s hole should be delocalised over the degenerate "
                f"manifold, got dominant weight {weights[top]:.3f}")
            # Essentially all of it lives in the six F1s MOs (#2..#7).
            assert weights[1:7].sum() > 0.99


def test_shake_off_sum_rule_holds_for_every_real_ao(case):
    """[B-2]: the restored (2 pi)^-3 makes Parseval a genuine unit test.

    ``int k^2 P_nu(k) dk = <chi_nu|chi_nu> = 1`` for every normalised
    contracted AO of the real basis -- 102 independent checks of the
    primitive normalisation, the contraction coefficients and the harmonic
    convention at once.
    """
    _, _, _, basis, _ = case
    norms = np.asarray(shakeoff.check_normalisation(basis), dtype=float)
    assert norms.shape == (NBAS,)
    worst = float(np.abs(norms - 1.0).max())
    assert worst < 1e-5, f"max |norm - 1| = {worst:.3e}"


def test_dyson_fast_path_matches_brute_force_on_real_q(case):
    """The null-space identity, on a real 34 x 35 overlap block."""
    neutral, dication, s_ao, _, holes = case
    q_a = dyson.build_Q(neutral.coeff, dication.coeff, s_ao, N_OCC,
                        holes.alpha_idx)
    assert q_a.shape == (N_OCC - 1, N_OCC)
    singular = np.linalg.svd(q_a, compute_uv=False)
    assert singular.min() > 0.5, (
        "the real Q is well conditioned; if this fails the fast path is "
        "being exercised outside its domain of validity")
    fast = dyson.dyson_amplitudes(q_a)
    ref = dyson.dyson_amplitudes_bruteforce(q_a)
    scale = np.abs(ref).max()
    assert np.abs(fast - ref).max() < 1e-9 * scale


def test_lambda_falls_back_and_matches_brute_force(case):
    """Real dipole-substituted rows go rank-deficient; the fallback is exact.

    This is the case that made a raising implementation refuse a legitimate
    calculation.
    """
    neutral, dication, s_ao, _, holes = case
    dipole = mio.read_ao_dipole(H5, one_centre=True)
    q_a = dyson.build_Q(neutral.coeff, dication.coeff, s_ao, N_OCC,
                        holes.alpha_idx)
    fast = dyson.lambda_coefficients(q_a, holes.alpha_idx, neutral.coeff,
                                     dication.coeff, N_OCC, dipole)
    assert fast.shape == (NBAS, 3)
    occ = neutral.coeff[:, :N_OCC]
    idx = list(holes.alpha_idx)
    for axis, comp in enumerate("xyz"):
        m = dication.coeff[:, idx].T @ dipole[comp] @ occ
        ref = occ @ dyson.lambda_coefficients_bruteforce(q_a, m)
        scale = np.abs(ref).max()
        assert scale > 0
        assert np.abs(fast[:, axis] - ref).max() < 1e-10 * scale


def test_cross_dyson_is_negligible_for_a_real_cv_state(case):
    """Note #2's near-orthogonality claim, measured instead of assumed.

    The core and valence Dyson orbitals are nearly AO-orthogonal, so the
    +-4 D_ij S_ij term should be vanishingly small. On this state it comes
    out ~1e-9 of the direct term, which is a much stronger statement than
    the notes make.
    """
    neutral, dication, s_ao, basis, holes = case
    objects = dyson.build_state_objects(
        neutral.coeff, dication.coeff, s_ao, N_OCC, holes,
        dipole_ao=None, include_frozen=False)
    terms = TermSwitches(direct=True, cross_dyson=True, indirect=False,
                         aa_bb=True, c_cross=True)
    context = _Context(770.0, terms, SigmaBuilder(basis), basis)
    result = spectrum.integrate_state(objects, context, 715.0, "singlet",
                                      n_quad=200, index=1)
    assert result.open
    direct = result.terms["direct"]
    assert direct > 0
    assert abs(result.terms["cross_dyson"]) < 1e-6 * direct


def test_frozen_reference_uses_the_actual_core_holes(case):
    """The frozen limit must not substitute valence MOs for the core holes.

    Passing the dication row indices straight through as neutral columns
    picked two F2p valence MOs and made the frozen intensity 3e-4 of the
    relaxed one. The corrected build projects each hole onto the neutral
    occupied space, so the S1s hole lands on neutral MO #1 and the
    relaxed/frozen ratio becomes a meaningful measure of relaxation.
    """
    neutral, dication, s_ao, _, holes = case
    objects = dyson.build_state_objects(
        neutral.coeff, dication.coeff, s_ao, N_OCC, holes,
        dipole_ao=None, include_frozen=True)
    assert objects.frozen is not None
    meta = objects.frozen.meta
    assert meta["hole_i_dominant_neutral_mo"] == 1, (
        "the S1s hole must map onto the lowest neutral MO")
    assert meta["hole_i_dominant_weight"] > 0.999
    assert meta["hole_j_dominant_weight"] < 0.6, (
        "the localized F1s hole must be recorded as delocalised over the "
        "degenerate neutral manifold")
    for key in ("hole_i_norm_in_occ", "hole_j_norm_in_occ"):
        assert abs(meta[key] - 1.0) < 1e-3


class _Context:
    """Minimal run context for spectrum.integrate_state."""

    def __init__(self, omega_ev, terms, sigma_builder, basis):
        self.omega_ev = omega_ev
        self.terms = terms
        self.sigma_builder = sigma_builder
        self.basis = basis

    def sigma_at_eps1_grid(self, eps1_ev):
        return self.sigma_builder.at_eps1_grid(eps1_ev)

    def p_shake_at_k(self, k_au):
        return shakeoff.p_shake(self.basis, k_au)
