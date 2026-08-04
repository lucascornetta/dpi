"""Tests of the OpenMolcas readers in :mod:`dpi.molcas_io`.

Runnable with ``python -m pytest tests/ -q`` or directly as
``python tests/test_molcas_io.py``.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpi import molcas_io as mio
from dpi.constants import (
    MolcasFormatError,
    ModelError,
    OCC_DOUBLY_MIN,
)


# ── INPORB round trip ───────────────────────────────────────────────────────

def test_inporb_round_trip_is_exact():
    """read_inporb recovers written coefficients and occupations exactly.

    The fixture writer quantises its arrays to the decimals the file will
    carry, so this is an equality test rather than a tolerance test: any
    non-zero difference is a parsing defect, not a formatting artefact.
    """
    with tempfile.TemporaryDirectory() as tmp:
        for vn, vd in (("2.0", "2.2"), ("2.2", "2.0")):
            case = mio.write_synthetic_case(
                os.path.join(tmp, f"c{vn}{vd}"),
                version_neutral=vn, version_dication=vd,
            )
            neu = mio.read_inporb(case.inporb_neutral)
            dic = mio.read_inporb(case.inporb_dication)
            assert neu.version == vn and dic.version == vd
            assert np.array_equal(neu.coeff, case.c_neu)
            assert np.array_equal(dic.coeff, case.c_dic)
            assert np.array_equal(neu.occ, case.occ_neutral)
            assert np.array_equal(dic.occ, case.occ_dication)
            assert neu.coeff.shape == (case.nbas, case.nmo)
            assert neu.nbas == case.nbas and neu.nmo == case.nmo
            assert neu.nbas_per_sym == (case.nbas,)
            assert neu.nmo_per_sym == (case.nmo,)
            assert neu.path == case.inporb_neutral


def test_both_format_versions_are_detected_not_guessed():
    """The version comes from the header and both #INFO layouts parse.

    Version 2.0 writes nSym on the second data line, 2.2 writes the
    (nFro, nIsh, nAsh) partition.  A single-symmetry 2.2 file and a
    three-symmetry 2.0 file can present the same token count, which is why
    the layout is selected from the declared version.
    """
    with tempfile.TemporaryDirectory() as tmp:
        case = mio.write_synthetic_case(os.path.join(tmp, "c"))
        text = open(case.inporb_neutral).read()
        assert text.startswith("#INPORB 2.0")
        for version in ("2.0", "2.2"):
            path = os.path.join(tmp, f"orb{version}")
            mio._write_inporb(
                path, case.c_neu, case.occ_neutral, version, "t"
            )
            got = mio.read_inporb(path)
            assert got.version == version
            assert np.array_equal(got.coeff, case.c_neu)
        # The #INFO block has exactly three data lines, matching real
        # OpenMolcas v20.10 output:
        #     iUHF  nSym  iWFtype
        #     nBas(1..nSym)
        #     nOrb(1..nSym)
        # Verified against sf6/scf.ScfOrb ("0 1 2") and a RasOrb ("0 1 0").
        lines = open(os.path.join(tmp, "orb2.2")).read().splitlines()
        info_at = lines.index("#INFO")
        data = [
            k for k in range(info_at + 1, len(lines))
            if lines[k].strip()
            and not lines[k].lstrip().startswith(("*", "#"))
        ]
        assert len(lines[data[0]].split()) == 3, lines[data[0]]
        assert len(lines[data[1]].split()) == 1, lines[data[1]]
        assert len(lines[data[2]].split()) == 1, lines[data[2]]

        # A spin-unrestricted file (iUHF = 1) must be refused: the model
        # needs RHF/ROHF spatial orbitals.
        flags = lines[data[0]].split()
        lines[data[0]] = f"       1       {flags[1]}       {flags[2]}"
        bad = os.path.join(tmp, "bad_uhf")
        open(bad, "w").write("\n".join(lines) + "\n")
        try:
            mio.read_inporb(bad)
        except MolcasFormatError as exc:
            assert "iUHF" in str(exc) and "unrestricted" in str(exc)
        else:
            raise AssertionError("a spin-unrestricted INPORB was accepted")


def test_orbitals_are_orthonormal_with_respect_to_the_overlap_read_back():
    """C^T S C = 1 using only data that made the round trip through disk.

    This is the fixture's central invariant: it validates the orbital
    writer, the overlap writer and both readers at once, and everything
    downstream (the Dyson gauge, the spectroscopic factors) depends on it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        for harmonics in ("spherical", "cartesian"):
            case = mio.write_synthetic_case(
                os.path.join(tmp, harmonics), harmonics=harmonics
            )
            neu = mio.read_inporb(case.inporb_neutral)
            dic = mio.read_inporb(case.inporb_dication)
            s_ao = mio.read_overlap(case.overlap, case.nbas)
            eye = np.eye(case.nmo)
            for c in (neu.coeff, dic.coeff):
                err = np.abs(c.T @ s_ao @ c - eye).max()
                assert err < 1e-12, (harmonics, err)


def test_overlap_square_and_triangular_layouts_agree():
    """The layout is inferred from the value count, both giving one matrix."""
    with tempfile.TemporaryDirectory() as tmp:
        case = mio.write_synthetic_case(os.path.join(tmp, "c"))
        tri = mio.read_overlap(case.overlap, case.nbas)
        assert np.array_equal(tri, case.s_ao)
        assert tri.shape == (case.nbas, case.nbas)

        sq_path = os.path.join(tmp, "sq.txt")
        with open(sq_path, "w") as fh:
            fh.write("# full square\n")
            for row in case.s_ao:
                fh.write(" ".join(mio._OVERLAP_FMT % v for v in row) + "\n")
        sq = mio.read_overlap(sq_path, case.nbas)
        assert np.array_equal(sq, case.s_ao)
        assert np.abs(sq - tri).max() == 0.0


def test_overlap_rejects_a_wrong_value_count():
    """A count matching neither layout names both candidate counts."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ovl.txt")
        open(path, "w").write("1.0 0.1 1.0\n")
        try:
            mio.read_overlap(path, 5)
        except MolcasFormatError as exc:
            msg = str(exc)
            assert "25" in msg and "15" in msg, msg
        else:
            raise AssertionError("a wrong overlap value count was accepted")


def test_overlap_rejects_an_asymmetric_square_matrix():
    """A square file that is not symmetric is not an AO overlap."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ovl.txt")
        a = np.array([[1.0, 0.3], [0.9, 1.0]])
        with open(path, "w") as fh:
            for row in a:
                fh.write(f"{float(row[0]):.16e} {float(row[1]):.16e}\n")
        try:
            mio.read_overlap(path, 2)
        except MolcasFormatError as exc:
            assert "symmetric" in str(exc)
        else:
            raise AssertionError("an asymmetric overlap was accepted")


# ── malformed files ─────────────────────────────────────────────────────────

def test_malformed_inporb_raises_mentioning_the_line():
    """Every parse failure names the file and the offending line number."""
    with tempfile.TemporaryDirectory() as tmp:
        case = mio.write_synthetic_case(os.path.join(tmp, "c"))
        good = open(case.inporb_neutral).read().splitlines()

        # (a) a coefficient that is not a number.
        lines = list(good)
        target = next(
            k for k, ln in enumerate(lines)
            if ln.strip() and ln.strip()[0].isdigit() is False
            and "E" in ln and "ORBITAL" not in ln and not ln.startswith("#")
        )
        lines[target] = lines[target].replace("E", "Q", 1)
        p = os.path.join(tmp, "bad_coeff")
        open(p, "w").write("\n".join(lines) + "\n")
        try:
            mio.read_inporb(p)
        except MolcasFormatError as exc:
            msg = str(exc)
            assert f":{target + 1}:" in msg, (msg, target + 1)
            assert "bad_coeff" in msg and "Fortran real" in msg
        else:
            raise AssertionError("a non-numeric coefficient was accepted")

        # (b) a truncated orbital block.
        lines = list(good)
        del lines[target]
        p = os.path.join(tmp, "short_orb")
        open(p, "w").write("\n".join(lines) + "\n")
        try:
            mio.read_inporb(p)
        except MolcasFormatError as exc:
            msg = str(exc)
            assert "coefficients, expected" in msg and "short_orb" in msg
            assert any(ch.isdigit() for ch in msg)
        else:
            raise AssertionError("a short orbital block was accepted")

        # (c) no #INPORB header at all.
        p = os.path.join(tmp, "no_header")
        open(p, "w").write("just some text\n1.0 2.0\n")
        try:
            mio.read_inporb(p)
        except MolcasFormatError as exc:
            assert "not an INPORB file" in str(exc) and ":1:" in str(exc)
        else:
            raise AssertionError("a headerless file was accepted")

        # (d) an #OCC block of the wrong length.
        lines = list(good)
        occ_at = lines.index("#OCC")
        lines.insert(occ_at + 2, mio._INPORB_FMT % 1.0)
        p = os.path.join(tmp, "bad_occ")
        open(p, "w").write("\n".join(lines) + "\n")
        try:
            mio.read_inporb(p)
        except MolcasFormatError as exc:
            assert "occupation numbers" in str(exc) and "bad_occ" in str(exc)
        else:
            raise AssertionError("a wrong-length #OCC was accepted")

        # (e) a missing file, and an unsupported version.
        try:
            mio.read_inporb(os.path.join(tmp, "nope"))
        except MolcasFormatError as exc:
            assert "no such file" in str(exc)
        else:
            raise AssertionError("a missing file was accepted")

        lines = list(good)
        lines[0] = "#INPORB 1.1"
        p = os.path.join(tmp, "old_version")
        open(p, "w").write("\n".join(lines) + "\n")
        try:
            mio.read_inporb(p)
        except MolcasFormatError as exc:
            assert "unsupported INPORB version" in str(exc)
        else:
            raise AssertionError("an unsupported version was accepted")


def test_fortran_d_exponent_is_parsed():
    """OpenMolcas writes D exponents; they must read as E exponents."""
    assert mio._to_float("1.25D-03", "f", 1) == 1.25e-3
    assert mio._to_float("-4.0d+02", "f", 1) == -400.0
    assert mio._to_float("3.5E-2", "f", 1) == 3.5e-2
    # A dropped exponent letter in a tight Fortran field.
    assert mio._to_float("1.234567-101", "f", 1) == float("1.234567E-101")
    try:
        mio._to_float("not_a_number", "some/file", 42)
    except MolcasFormatError as exc:
        assert "some/file:42" in str(exc)
    else:
        raise AssertionError("a non-numeric token was accepted")


# ── occupation classification ───────────────────────────────────────────────

def test_rohf_hole_indices_strict_two_open_shell():
    """Two MOs at occ ~ 1 give the holes directly, with no approximation.

    The convention is checked explicitly: hole i is absent from the alpha
    set and present in the beta set, and vice versa for hole j
    (REVIEW.md [D-1]).
    """
    n = 6
    occ = np.zeros(10)
    occ[:n] = 2.0
    occ[1] = 1.0
    occ[4] = 1.0
    ha = mio.rohf_hole_indices(occ, n)
    assert (ha.hole_i, ha.hole_j) == (1, 4)
    assert ha.approximation is None
    assert ha.n_doubly == n - 2 and ha.n_singly == 2
    assert ha.alpha_idx == (0, 2, 3, 4, 5)
    assert ha.beta_idx == (0, 1, 2, 3, 5)
    assert ha.hole_i not in ha.alpha_idx and ha.hole_i in ha.beta_idx
    assert ha.hole_j not in ha.beta_idx and ha.hole_j in ha.alpha_idx
    assert len(ha.alpha_idx) == len(ha.beta_idx) == n - 1
    assert list(ha.alpha_idx) == sorted(ha.alpha_idx)
    assert list(ha.beta_idx) == sorted(ha.beta_idx)


def test_rohf_hole_indices_state_averaged_natural_orbitals():
    """Three MOs at 5/3 plus one at 1 trigger the documented localisation.

    This is the S 2p pattern: the core hole is delocalised over three
    symmetry partners, which no single determinant can represent.  The
    approximation localises it on the lowest-index partner and says so.
    """
    n = 6
    occ = np.zeros(10)
    occ[:n] = 2.0
    occ[0] = occ[1] = occ[2] = 5.0 / 3.0
    occ[5] = 1.0
    ha = mio.rohf_hole_indices(occ, n)
    assert (ha.hole_i, ha.hole_j) == (0, 5)
    assert ha.approximation is not None
    assert "localised" in ha.approximation
    assert "single determinant" in ha.approximation
    assert ha.n_doubly == n - 2 and ha.n_singly == 2
    # The two promoted partners are now doubly occupied, so they appear in
    # both spin sectors.
    for p in (1, 2):
        assert p in ha.alpha_idx and p in ha.beta_idx
    assert ha.hole_i not in ha.alpha_idx and ha.hole_j not in ha.beta_idx
    assert abs(occ.sum() - (2 * n - 2)) < 1e-9


def test_rohf_hole_indices_rejects_unrecognised_patterns():
    """An unsupported occupation pattern raises rather than guessing."""
    n = 6
    base = np.zeros(10)
    base[:n] = 2.0

    # Four singly occupied MOs: a tetra-radical, not a dication CV state.
    occ = base.copy()
    for k in (0, 1, 2, 3):
        occ[k] = 1.0
    try:
        mio.rohf_hole_indices(occ, n)
    except ModelError as exc:
        assert "unrecognised" in str(exc) or "electrons" in str(exc)
    else:
        raise AssertionError("four open shells were accepted")

    # Wrong electron count: a monocation, not a dication.
    occ = base.copy()
    occ[0] = 1.0
    try:
        mio.rohf_hole_indices(occ, n)
    except ModelError as exc:
        assert "electrons" in str(exc), str(exc)
    else:
        raise AssertionError("a monocation occupation was accepted")

    # An occupation in none of the windows.  OCC_SINGLY and
    # OCC_FRACTIONAL abut at 1.50, so the genuine gap is between
    # OCC_VIRTUAL_MAX = 0.05 and the bottom of OCC_SINGLY at 0.50: 0.30
    # lands there.  The partner at 1.70 keeps the electron count correct
    # so that the window check, not the count check, is what fires.
    occ = base.copy()
    occ[0] = 0.30
    occ[1] = 1.70
    assert abs(occ.sum() - (2 * n - 2)) < 1e-9
    try:
        mio.rohf_hole_indices(occ, n)
    except ModelError as exc:
        assert "none of the windows" in str(exc), str(exc)
    else:
        raise AssertionError("an out-of-window occupation was accepted")

    # An empty #OCC block.
    try:
        mio.rohf_hole_indices(np.zeros(0), n)
    except ModelError as exc:
        assert "#OCC" in str(exc)
    else:
        raise AssertionError("an empty occupation array was accepted")


def test_hole_indices_from_the_synthetic_dication_file():
    """The occupations written by the fixture classify back to its holes."""
    with tempfile.TemporaryDirectory() as tmp:
        for style in ("rohf", "natural"):
            case = mio.write_synthetic_case(
                os.path.join(tmp, style), occ_style=style
            )
            dic = mio.read_inporb(case.inporb_dication)
            ha = mio.rohf_hole_indices(dic.occ, case.n_neu_occ)
            assert (ha.hole_i, ha.hole_j) == (case.hole_i, case.hole_j)
            assert (ha.approximation is not None) == (style == "natural")


# ── basis and harmonic detection ────────────────────────────────────────────

def test_harmonic_convention_detection():
    """5 m values per d shell means spherical, 6 means Cartesian.

    REVIEW.md [A-1]: the old code hard-coded six Cartesian d components,
    but SF6/cc-pVDZ has nbas = 102, which is only reachable with five
    real spherical d functions per shell.  Detection replaces the
    assumption.
    """
    with tempfile.TemporaryDirectory() as tmp:
        for harmonics, per_shell, nbas in (
            ("spherical", 5, 14), ("cartesian", 6, 15),
        ):
            case = mio.write_synthetic_case(
                os.path.join(tmp, harmonics), harmonics=harmonics
            )
            basis = mio.read_basis(case.h5)
            assert basis.harmonics == harmonics, (harmonics, basis.harmonics)
            assert basis.nbas == nbas == case.nbas
            d_sel = basis.l == 2
            assert d_sel.sum() == per_shell
            assert len(set(basis.m[d_sel].tolist())) == per_shell
            # p shells are 3-fold either way, so they cannot discriminate.
            assert (basis.l == 1).sum() == 6


def test_basis_per_ao_fields():
    """The per-AO arrays are aligned, correctly typed and correctly valued."""
    with tempfile.TemporaryDirectory() as tmp:
        case = mio.write_synthetic_case(os.path.join(tmp, "c"))
        basis = mio.read_basis(case.h5)
        n = basis.nbas
        for name in ("centers", "elements", "l", "m", "shell_index",
                     "n_prim"):
            assert len(getattr(basis, name)) == n, name
        assert len(basis.prim_exp) == n and len(basis.prim_coef) == n

        # Element symbols have the trailing atom index stripped.
        # S and F, so the synthetic case exercises the real Yeh-Lindau
        # table rather than falling through to sigma = 0.
        assert basis.atom_labels == ("S1", "F1")
        assert tuple(basis.atom_elements) == ("S", "F")
        assert set(basis.elements.tolist()) == {"S", "F"}
        assert basis.elements[0] == "S" and basis.elements[-1] == "F"
        assert basis.n_atoms == 2
        assert basis.center_coords.shape == (2, 3)
        assert np.allclose(basis.center_coords, case.center_coords)

        # The layout that was resolved, and the l values it implies.
        assert basis.id_layout == ("center", "l", "m", "shell")
        assert basis.l.min() == 0 and basis.l.max() == 2

        # Centre 0 carries two s shells and one p shell, so shell_index
        # counts 0, 1 within (centre 0, l=0) and 0 within (centre 0, l=1).
        s0 = (basis.centers == 0) & (basis.l == 0)
        assert sorted(basis.shell_index[s0].tolist()) == [0, 1]
        p0 = (basis.centers == 0) & (basis.l == 1)
        assert set(basis.shell_index[p0].tolist()) == {0}

        # All AOs of one shell share the radial part, by identity of value.
        for c in range(basis.n_atoms):
            for ll in np.unique(basis.l[basis.centers == c]):
                for sh in np.unique(
                    basis.shell_index[(basis.centers == c) & (basis.l == ll)]
                ):
                    sel = np.flatnonzero(
                        (basis.centers == c) & (basis.l == ll)
                        & (basis.shell_index == sh)
                    )
                    first = basis.prim_exp[sel[0]]
                    for k in sel[1:]:
                        assert np.array_equal(basis.prim_exp[k], first)
                        assert np.array_equal(
                            basis.prim_coef[k], basis.prim_coef[sel[0]]
                        )
        assert all(e.min() > 0.0 for e in basis.prim_exp)
        assert set(basis.n_prim.tolist()) == {3}


def test_basis_refuses_f_functions():
    """l > 2 raises ModelError instead of silently returning zeros.

    The failure mode being prevented is [A-1]'s: an unhandled component
    that quietly contributed an s-function normalisation with an l = 2
    angular factor.
    """
    import h5py

    with tempfile.TemporaryDirectory() as tmp:
        case = mio.write_synthetic_case(os.path.join(tmp, "c"))
        bad = os.path.join(tmp, "f.h5")
        with h5py.File(case.h5, "r") as src, h5py.File(bad, "w") as dst:
            for key in src:
                src.copy(key, dst)
            ids = np.asarray(dst["BASIS_FUNCTION_IDS"])
            prim = np.asarray(dst["PRIMITIVE_IDS"])
            del dst["BASIS_FUNCTION_IDS"], dst["PRIMITIVE_IDS"]
            # Promote the d shell to an f shell with its 7 components.
            keep = ids[ids[:, 1] != 2]
            fshell = np.array(
                [[2, 3, m, 1] for m in range(-3, 4)], dtype=np.int64
            )
            dst.create_dataset(
                "BASIS_FUNCTION_IDS", data=np.vstack([keep, fshell])
            )
            prim = prim.copy()
            prim[prim[:, 1] == 2, 1] = 3
            dst.create_dataset("PRIMITIVE_IDS", data=prim)
        try:
            mio.read_basis(bad)
        except ModelError as exc:
            msg = str(exc)
            assert "l=3" in msg and "d functions" in msg, msg
        else:
            raise AssertionError("an f shell was accepted")


def test_basis_reports_a_missing_dataset():
    """A missing HDF5 dataset names it and lists what is present."""
    import h5py

    with tempfile.TemporaryDirectory() as tmp:
        case = mio.write_synthetic_case(os.path.join(tmp, "c"))
        bad = os.path.join(tmp, "nodata.h5")
        with h5py.File(case.h5, "r") as src, h5py.File(bad, "w") as dst:
            for key in src:
                if key != "PRIMITIVES":
                    src.copy(key, dst)
        try:
            mio.read_basis(bad)
        except MolcasFormatError as exc:
            msg = str(exc)
            assert "PRIMITIVES" in msg and "nodata.h5" in msg
            assert "CENTER_LABELS" in msg  # the "present" listing
        else:
            raise AssertionError("a missing dataset was accepted")


# ── dipole integrals ────────────────────────────────────────────────────────

def test_one_centre_dipole_is_origin_shifted_and_block_diagonal():
    """D -> (D - R_A S) * delta(atom, atom), manuscript Eq. 117.

    Two-centre blocks vanish and each same-atom block is re-referenced to
    its own nucleus, which is what removes the large R_A <chi|chi> term
    that has no counterpart in the atomic cross sections.
    """
    with tempfile.TemporaryDirectory() as tmp:
        case = mio.write_synthetic_case(os.path.join(tmp, "c"))
        basis = mio.read_basis(case.h5)
        one = mio.read_ao_dipole(case.h5, one_centre=True)
        full = mio.read_ao_dipole(case.h5, one_centre=False)
        s_ao = mio.read_ao_overlap_h5(case.h5)
        same = basis.centers[:, None] == basis.centers[None, :]

        for comp in "xyz":
            assert one[comp].shape == (case.nbas, case.nbas)
            assert np.array_equal(full[comp], case.dipole_full[comp])
            assert np.array_equal(one[comp], case.dipole_one_centre[comp])
            # Two-centre blocks are exactly zero.
            assert np.abs(one[comp][~same]).max() == 0.0
            # Same-atom blocks equal the shifted full matrix.
            axis = "xyz".index(comp)
            r_a = basis.center_coords[basis.centers][:, axis]
            want = full[comp] - r_a[:, None] * s_ao
            assert np.abs(one[comp][same] - want[same]).max() < 1e-12

        # The point of the reduction: the magnitude drops substantially,
        # which is the M_rms diagnostic the old code printed.
        rms_full = np.sqrt(np.mean(full["z"] ** 2))
        rms_one = np.sqrt(np.mean(one["z"] ** 2))
        assert rms_one < rms_full, (rms_one, rms_full)


def test_h5_and_text_overlap_agree():
    """The same overlap comes back from the text file and the HDF5 file."""
    with tempfile.TemporaryDirectory() as tmp:
        case = mio.write_synthetic_case(os.path.join(tmp, "c"))
        from_text = mio.read_overlap(case.overlap, case.nbas)
        from_h5 = mio.read_ao_overlap_h5(case.h5)
        assert np.array_equal(from_text, from_h5)
        assert np.array_equal(from_h5, case.s_ao)
        w = np.linalg.eigvalsh(from_h5)
        assert w.min() > 0.0, f"overlap is not positive definite: {w.min()}"


# ── the fixture writer itself ───────────────────────────────────────────────

def test_synthetic_case_is_self_consistent_and_documented():
    """Every advertised field is populated and internally consistent."""
    with tempfile.TemporaryDirectory() as tmp:
        case = mio.write_synthetic_case(os.path.join(tmp, "c"))
        for name in (
            "dirpath", "inporb_neutral", "inporb_dication", "overlap", "h5",
            "nbas", "nmo", "n_neu_occ", "harmonics", "s_ao", "c_neu",
            "c_dic", "occ_neutral", "occ_dication", "occ_style", "hole_i",
            "hole_j", "dipole_full", "dipole_one_centre", "center_coords",
            "atom_labels",
        ):
            assert hasattr(case, name), name
        for path in (case.inporb_neutral, case.inporb_dication,
                     case.overlap, case.h5):
            assert os.path.isfile(path), path
        assert case.occ_neutral.sum() == 2 * case.n_neu_occ
        assert abs(case.occ_dication.sum() - (2 * case.n_neu_occ - 2)) < 1e-9
        assert case.hole_i < case.hole_j
        assert (case.occ_neutral >= OCC_DOUBLY_MIN).sum() == case.n_neu_occ

        # Q is near identity but not identity: rotation=0 would put the
        # code in the frozen limit and make the Dyson amplitudes trivial.
        q = case.c_dic.T @ case.s_ao @ case.c_neu
        off = np.abs(q - np.eye(case.nmo)).max()
        assert 1e-4 < off < 0.9, off
        flat = mio.write_synthetic_case(
            os.path.join(tmp, "flat"), rotation=0.0
        )
        q0 = flat.c_dic.T @ flat.s_ao @ flat.c_neu
        assert np.abs(q0 - np.eye(flat.nmo)).max() < 1e-9

        # The AO basis is genuinely non-orthogonal, which is what makes
        # the [D-3] distinction between the two spectroscopic-factor
        # routes observable.
        assert np.abs(case.s_ao - np.eye(case.nbas)).max() > 0.01
        assert np.allclose(np.diag(case.s_ao), 1.0, atol=1e-9)


def test_synthetic_case_rejects_inconsistent_requests():
    """Bad fixture parameters fail loudly at generation time."""
    with tempfile.TemporaryDirectory() as tmp:
        for kw, needle in (
            ({"hole_i": 3, "hole_j": 1}, "hole_i < hole_j"),
            ({"harmonics": "cubic"}, "spherical"),
            ({"occ_style": "rohf2"}, "occ_style"),
            ({"occ_style": "natural", "n_neu_occ": 4}, "n_neu_occ >= 5"),
        ):
            try:
                mio.write_synthetic_case(os.path.join(tmp, "x"), **kw)
            except ValueError as exc:
                assert needle in str(exc), (kw, str(exc))
            else:
                raise AssertionError(f"accepted bad parameters {kw}")


def _main() -> int:
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
