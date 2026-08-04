"""Validation behaviour of the TOML input layer."""

import os
import tempfile

from dpi.config import load_config, TermSwitches, OutputConfig, PhysicsConfig
from dpi.constants import ConfigError

MINIMAL = """
[physics]
photon_energy_ev = 2610.0
true_dip_ev = 2511.0
n_neu_occ = 35
{physics_extra}

[paths]
neutral_orbitals = "{neu}"
overlap = "{ovl}"
h5file = "{h5}"
dication_orbitals = {{ singlet = "{d1}", triplet = "{d2}" }}
energies = {{ singlet = "{e1}", triplet = "{e2}" }}

[output]
display_onset_ev = 15.82
{output_extra}

[terms]
{terms_extra}
"""


def _case(tmp, physics_extra="", output_extra="", terms_extra="", **kw):
    """Materialise a runnable input file plus the files it references."""
    paths = {}
    for name in ("neu", "ovl", "h5", "e1", "e2"):
        p = os.path.join(tmp, name + ".txt")
        with open(p, "w") as fh:
            fh.write("0.0\n")
        paths[name] = p
    for name in ("d1", "d2"):
        p = os.path.join(tmp, name)
        os.makedirs(p, exist_ok=True)
        paths[name] = p
    text = MINIMAL.format(physics_extra=physics_extra,
                          output_extra=output_extra,
                          terms_extra=terms_extra, **paths)
    for k, v in kw.items():
        text = text.replace(k, v)
    cfg_path = os.path.join(tmp, "in.toml")
    with open(cfg_path, "w") as fh:
        fh.write(text)
    return cfg_path


def _expect_error(cfg_path, *fragments):
    try:
        load_config(cfg_path)
    except ConfigError as exc:
        msg = str(exc)
        for frag in fragments:
            assert frag in msg, f"error message lacks {frag!r}:\n{msg}"
        return msg
    raise AssertionError("expected ConfigError, none raised")


def _relabel_line(cfg_path, key):
    """Rewrite one per-channel table from singlet/triplet to jc32/jc12."""
    text = open(cfg_path).read()
    marker = f"{key} = {{ singlet ="
    head, sep, tail = text.partition(marker)
    assert sep, f"{key!r} not found in the generated input file"
    tail = tail.replace("triplet =", "jc12 =", 1)
    open(cfg_path, "w").write(head + f"{key} = {{ jc32 =" + tail)


def _relabel_energies_as_jj(cfg_path):
    """A valid jj input: jj-labelled energies, multiplicity-labelled orbitals."""
    _relabel_line(cfg_path, "energies")



def test_minimal_loads_with_expected_defaults():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_config(_case(tmp))
        assert cfg.physics.coupling == "ls"
        assert cfg.channels == ("singlet", "triplet")
        assert cfg.physics.n_quad == 200
        assert cfg.physics.spin_degeneracy_factor == 1.0
        assert cfg.terms.active() == ("direct", "cross_dyson", "aa_bb",
                                      "c_cross")
        assert cfg.output.display_onset_ev == 15.82
        assert any("true DIP" in line for line in cfg.summary_lines())


def test_jj_coupling_switches_energy_keys_but_not_orbital_keys():
    """Only the ENERGIES are jj-resolved; orbital sets stay multiplicities.

    MOLCAS never optimises orbitals for a j_C state: the S 2p CV states are
    a post-hoc recombination of the separately converged S_dic = 0 and
    S_dic = 1 OSRHF solutions.  So a jj input keeps
    ``dication_orbitals = { singlet = ..., triplet = ... }`` while its
    ``energies`` are keyed jc32/jc12.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = _case(tmp, physics_extra='coupling = "jj"')
        _relabel_energies_as_jj(path)
        cfg = load_config(path)
        assert cfg.channels == ("jc32", "jc12")
        assert cfg.orbital_channels == ("singlet", "triplet")
        assert set(cfg.paths.dication_orbitals) == {"singlet", "triplet"}
        assert set(cfg.paths.energies) == {"jc32", "jc12"}


def test_energy_keys_must_match_coupling():
    with tempfile.TemporaryDirectory() as tmp:
        # ls energy keys left in place while asking for jj coupling
        path = _case(tmp, physics_extra='coupling = "jj"')
        _expect_error(path, "energies", "jc32")


def test_jj_orbital_keys_must_be_multiplicities():
    """Writing jc32/jc12 for the ORBITALS must be refused, with a pointer."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _case(tmp, physics_extra='coupling = "jj"')
        _relabel_energies_as_jj(path)
        _relabel_line(path, "dication_orbitals")
        _expect_error(path, "dication_orbitals", "singlet")


def test_photon_energy_below_dip_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        path = _case(tmp, **{"photon_energy_ev = 2610.0":
                             "photon_energy_ev = 2000.0"})
        _expect_error(path, "must exceed true_dip_ev", "E_excess")


def test_unknown_key_is_rejected_with_suggestion():
    with tempfile.TemporaryDirectory() as tmp:
        path = _case(tmp, physics_extra="photon_energy_eV = 1.0")
        msg = _expect_error(path, "unrecognised key")
        assert "photon_energy_ev" in msg


def test_missing_input_file_is_reported_by_key():
    with tempfile.TemporaryDirectory() as tmp:
        path = _case(tmp, **{'overlap = "': 'overlap = "/nonexistent/'})
        _expect_error(path, "not found", "overlap")


def test_interference_requires_both_parents():
    with tempfile.TemporaryDirectory() as tmp:
        path = _case(tmp, terms_extra="dir_ind_interference = true")
        _expect_error(path, "requires both direct and indirect")


def test_c_cross_without_aa_bb_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        path = _case(tmp, terms_extra="aa_bb = false")
        _expect_error(path, "c_cross")


def test_frozen_requires_frozen_energies():
    with tempfile.TemporaryDirectory() as tmp:
        path = _case(tmp, physics_extra="include_frozen = true")
        _expect_error(path, "frozen_energies")


def test_zero_voigt_widths_are_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        path = _case(tmp, output_extra="voigt_sigma_ev = 0.0\n"
                                      "voigt_gamma_ev = 0.0")
        _expect_error(path, "delta functions")


def test_overrides_apply_before_validation():
    with tempfile.TemporaryDirectory() as tmp:
        path = _case(tmp)
        cfg = load_config(path, overrides={"physics.photon_energy_ev": 3000.0,
                                           "output.e_step_ev": 0.05})
        assert cfg.physics.photon_energy_ev == 3000.0
        assert cfg.output.e_step_ev == 0.05


def test_grid_bounds_default_to_state_range_plus_margin():
    out = OutputConfig(voigt_sigma_ev=0.5, voigt_gamma_ev=0.2)
    lo, hi = out.grid_bounds([20.0, 30.0])
    assert lo < 20.0 and hi > 30.0
    assert abs((20.0 - lo) - 3.0) < 1e-12


def test_spin_degeneracy_override_rejected_in_jj():
    try:
        PhysicsConfig(photon_energy_ev=100.0, true_dip_ev=50.0, n_neu_occ=5,
                      coupling="jj", spin_degeneracy_factor=3.0)
    except ConfigError:
        raise AssertionError("PhysicsConfig should not reject this alone")
    # the cross-section check lives in load_config
    with tempfile.TemporaryDirectory() as tmp:
        path = _case(tmp,
                     physics_extra='coupling = "jj"\n'
                                   'spin_degeneracy_factor = 3.0')
        _relabel_energies_as_jj(path)
        _expect_error(path, "spin_degeneracy_factor", "jj")


def test_root_prefixes_relative_paths():
    """[paths] root stands in for the concatenation TOML cannot express."""
    with tempfile.TemporaryDirectory() as tmp:
        run = os.path.join(tmp, "sf6")
        os.makedirs(os.path.join(run, "cv_singlet"))
        os.makedirs(os.path.join(run, "cv_triplet"))
        for name in ("scf.ScfOrb", "scf.scf.h5", "e_s.dat", "e_t.dat"):
            open(os.path.join(run, name), "w").write("x\n")
        path = os.path.join(tmp, "in.toml")
        open(path, "w").write(f"""
[physics]
photon_energy_ev = 770.0
true_dip_ev = 715.0
n_neu_occ = 35

[paths]
root = "{run}"
neutral_orbitals = "scf.ScfOrb"
overlap = "scf.scf.h5"
h5file = "scf.scf.h5"
dication_orbitals = {{ singlet = "cv_singlet", triplet = "cv_triplet" }}
energies = {{ singlet = "e_s.dat", triplet = "e_t.dat" }}
""")
        cfg = load_config(path)
        # Consumers see fully joined paths; nothing downstream knows about root.
        assert cfg.paths.neutral_orbitals == os.path.join(run, "scf.ScfOrb")
        assert cfg.paths.h5file == os.path.join(run, "scf.scf.h5")
        assert cfg.paths.energies["singlet"] == os.path.join(run, "e_s.dat")
        assert cfg.paths.dication_orbitals["triplet"] == os.path.join(
            run, "cv_triplet")


def test_root_leaves_absolute_paths_alone():
    """An absolute entry can still point outside the run directory."""
    with tempfile.TemporaryDirectory() as tmp:
        run = os.path.join(tmp, "sf6")
        other = os.path.join(tmp, "shared")
        os.makedirs(os.path.join(run, "cv_singlet"))
        os.makedirs(os.path.join(run, "cv_triplet"))
        os.makedirs(other)
        for name in ("scf.ScfOrb", "e_s.dat", "e_t.dat"):
            open(os.path.join(run, name), "w").write("x\n")
        shared_h5 = os.path.join(other, "scf.scf.h5")
        open(shared_h5, "w").write("x\n")
        path = os.path.join(tmp, "in.toml")
        open(path, "w").write(f"""
[physics]
photon_energy_ev = 770.0
true_dip_ev = 715.0
n_neu_occ = 35

[paths]
root = "{run}"
neutral_orbitals = "scf.ScfOrb"
overlap = "{shared_h5}"
h5file = "{shared_h5}"
dication_orbitals = {{ singlet = "cv_singlet", triplet = "cv_triplet" }}
energies = {{ singlet = "e_s.dat", triplet = "e_t.dat" }}
""")
        cfg = load_config(path)
        assert cfg.paths.h5file == shared_h5
        assert cfg.paths.neutral_orbitals == os.path.join(run, "scf.ScfOrb")


def test_missing_file_error_reports_the_joined_path():
    """The error must name the path actually looked for, not the bare name."""
    with tempfile.TemporaryDirectory() as tmp:
        run = os.path.join(tmp, "sf6")
        os.makedirs(os.path.join(run, "cv_singlet"))
        os.makedirs(os.path.join(run, "cv_triplet"))
        for name in ("scf.scf.h5", "e_s.dat", "e_t.dat"):
            open(os.path.join(run, name), "w").write("x\n")
        path = os.path.join(tmp, "in.toml")
        open(path, "w").write(f"""
[physics]
photon_energy_ev = 770.0
true_dip_ev = 715.0
n_neu_occ = 35

[paths]
root = "{run}"
neutral_orbitals = "typo.ScfOrb"
overlap = "scf.scf.h5"
h5file = "scf.scf.h5"
dication_orbitals = {{ singlet = "cv_singlet", triplet = "cv_triplet" }}
energies = {{ singlet = "e_s.dat", triplet = "e_t.dat" }}
""")
        try:
            load_config(path)
        except ConfigError as exc:
            assert os.path.join(run, "typo.ScfOrb") in str(exc), str(exc)
        else:
            raise AssertionError("a missing neutral_orbitals was accepted")


def test_accepted_path_keys_track_the_dataclass():
    """The key whitelist is derived, so it cannot drift from InputPaths."""
    from dpi.config import InputPaths, _PATH_KEYS
    assert _PATH_KEYS == set(InputPaths.__dataclass_fields__)
    assert "root" in _PATH_KEYS


if __name__ == "__main__":
    import sys
    mod = sys.modules[__name__]
    fns = [(n, f) for n, f in vars(mod).items()
           if n.startswith("test_") and callable(f)]
    for name, fn in fns:
        fn()
        print(f"  ok  {name}")
    print(f"{len(fns)} config tests passed")
