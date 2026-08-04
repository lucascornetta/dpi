"""Tests of the output layer: round trips, headers and column order."""

from __future__ import annotations

import numpy as np
import pytest

from dpi import report
from dpi.amplitudes import TERM_ORDER, TermSwitches
from dpi.constants import ConfigError
from dpi.spectrum import StateResult


def _states():
    return [
        StateResult(
            index=0, channel="singlet", label="F1s T22",
            e_dication_ev=41.2, e_shifted_ev=15.82, dip_ev=41.2,
            e_excess_ev=58.8, intensity=1.234568e-03,
            terms={"direct": 1.5e-3, "cross_dyson": -2.65432e-04},
            open=True, negative=False, amplitude_min=-1.0e-9,
            p_i=0.912345, p_j=0.874321, n_quad=200,
        ),
        StateResult(
            index=1, channel="triplet", label=None,
            e_dication_ev=43.9, e_shifted_ev=18.51, dip_ev=43.9,
            e_excess_ev=56.1, intensity=-4.20000e-05,
            terms={"direct": 3.0e-5, "cross_dyson": 1.0e-5,
                   "aa_bb": 2.0e-6, "c_cross": -1.02e-4},
            open=True, negative=True, amplitude_min=-3.3e-6,
            p_i=0.5, p_j=0.25, n_quad=200,
        ),
        StateResult(
            index=2, channel="jc32", label="S2p V4",
            e_dication_ev=60.0, e_shifted_ev=34.5, dip_ev=60.0,
            e_excess_ev=-2.0, intensity=0.0, terms={}, open=False,
            negative=False, amplitude_min=0.0, p_i=0.7, p_j=0.6,
        ),
    ]


def _settings():
    return {
        "photon_energy_ev": 770.0,
        "n_quad": 200,
        "sigma_g_ev": 0.6,
        "gamma_l_ev": 0.25,
        "terms": TermSwitches(),
    }


def test_spectrum_column_order(tmp_path):
    path = tmp_path / "spectrum.dat"
    grid = np.linspace(10.0, 20.0, 11)
    channels = {
        "singlet": np.linspace(0.0, 1.0, 11),
        "triplet": np.linspace(1.0, 0.0, 11),
    }
    report.write_spectrum(str(path), grid, channels, settings=_settings())
    spec = report.read_spectrum(str(path))
    assert spec.columns == ["E_shifted_eV", "singlet", "triplet", "total"]
    print("spectrum.dat columns: " + "  ".join(spec.columns))


def test_spectrum_round_trip(tmp_path):
    path = tmp_path / "spectrum.dat"
    rng = np.random.default_rng(5)
    grid = np.linspace(12.0, 32.0, 201)
    channels = {
        "singlet": rng.uniform(0.0, 1e-3, size=201),
        "triplet": rng.uniform(-1e-4, 1e-3, size=201),
    }
    report.write_spectrum(str(path), grid, channels, settings=_settings())
    spec = report.read_spectrum(str(path))

    assert np.allclose(spec.e_grid_ev, grid, atol=5e-5)
    for name, arr in channels.items():
        assert np.allclose(spec.channels[name], arr, rtol=1e-6, atol=0.0)
    assert np.allclose(spec.total, sum(channels.values()), rtol=1e-6)


def test_spectrum_total_defaults_to_channel_sum(tmp_path):
    path = tmp_path / "spectrum.dat"
    grid = np.array([1.0, 2.0, 3.0])
    channels = {"a": np.array([1.0, 2.0, 3.0]),
                "b": np.array([0.5, 0.5, 0.5])}
    report.write_spectrum(str(path), grid, channels)
    spec = report.read_spectrum(str(path))
    assert np.allclose(spec.total, [1.5, 2.5, 3.5], rtol=1e-6)


def test_spectrum_explicit_total_is_honoured(tmp_path):
    path = tmp_path / "spectrum.dat"
    grid = np.array([1.0, 2.0])
    channels = {"a": np.array([1.0, 1.0])}
    report.write_spectrum(str(path), grid, channels,
                          total=np.array([9.0, 9.0]))
    assert np.allclose(report.read_spectrum(str(path)).total, 9.0, rtol=1e-6)


def test_sticks_column_order(tmp_path):
    path = tmp_path / "sticks.dat"
    report.write_sticks(str(path), _states(), settings=_settings())
    sticks = report.read_sticks(str(path))
    expected = (
        ["index", "channel", "label", "E_shifted_eV", "DIP_eV",
         "E_excess_eV", "intensity"]
        + [f"I_{t}" for t in TERM_ORDER]
        + ["p_i", "p_j", "amplitude_min", "open", "negative"]
    )
    assert sticks.columns == expected
    assert list(report.STICKS_COLUMNS) == expected
    print("sticks.dat columns: " + "  ".join(sticks.columns))


def test_sticks_round_trip(tmp_path):
    path = tmp_path / "sticks.dat"
    original = _states()
    report.write_sticks(str(path), original, settings=_settings())
    read = report.read_sticks(str(path)).states
    assert len(read) == len(original)
    for a, b in zip(original, read):
        assert b.index == a.index
        assert b.channel == a.channel
        # Internal whitespace in a label is written as '_' to keep the
        # column split unambiguous; that substitution is the format.
        assert b.label == (a.label.replace(" ", "_") if a.label else None)
        assert b.e_shifted_ev == pytest.approx(a.e_shifted_ev, rel=1e-8)
        assert b.dip_ev == pytest.approx(a.dip_ev, rel=1e-8)
        assert b.e_excess_ev == pytest.approx(a.e_excess_ev, rel=1e-8)
        assert b.intensity == pytest.approx(a.intensity, rel=1e-8, abs=1e-30)
        assert b.p_i == pytest.approx(a.p_i, rel=1e-8)
        assert b.p_j == pytest.approx(a.p_j, rel=1e-8)
        assert b.amplitude_min == pytest.approx(a.amplitude_min, rel=1e-8,
                                                abs=1e-30)
        assert b.open == a.open
        assert b.negative == a.negative
        for term in TERM_ORDER:
            assert b.terms[term] == pytest.approx(
                a.terms.get(term, 0.0), rel=1e-8, abs=1e-30
            )


def test_sticks_writes_all_term_columns_regardless_of_model(tmp_path):
    """The column count must not depend on which terms are switched on."""
    path = tmp_path / "sticks.dat"
    minimal = [StateResult(
        index=0, channel="singlet", label="only-direct",
        e_dication_ev=1.0, e_shifted_ev=1.0, dip_ev=1.0, e_excess_ev=5.0,
        intensity=2.0, terms={"direct": 2.0}, open=True,
    )]
    report.write_sticks(str(path), minimal)
    read = report.read_sticks(str(path))
    assert read.columns == list(report.STICKS_COLUMNS)
    assert read.states[0].terms["aa_bb"] == 0.0


def test_header_is_self_documenting(tmp_path):
    path = tmp_path / "spectrum.dat"
    grid = np.array([1.0, 2.0])
    report.write_spectrum(str(path), grid, {"singlet": np.zeros(2)},
                          settings=_settings(),
                          note="sub-threshold linear rise used for S_1s")
    spec = report.read_spectrum(str(path))
    settings = spec.settings()
    assert settings["photon_energy_ev"] == "770.0"
    assert settings["n_quad"] == "200"
    # Nested TermSwitches flattened to terms.<field>.
    assert settings["terms.direct"] == "True"
    assert settings["terms.indirect"] == "False"
    assert settings["terms.spin_degeneracy_factor"] == "1.0"
    joined = " ".join(spec.header)
    assert "dpi_run.py" in joined
    assert "Mb*a.u." in joined and "RELATIVE" in joined
    assert "sub-threshold linear rise" in joined
    # ISO-8601 date on the provenance line.
    assert spec.header[0].count("-") >= 2 and "T" in spec.header[0]


def test_readers_reject_files_without_column_line(tmp_path):
    path = tmp_path / "bad.dat"
    path.write_text("# no columns line here\n1.0 2.0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="columns"):
        report.read_spectrum(str(path))
    with pytest.raises(ConfigError, match="columns"):
        report.read_sticks(str(path))


def test_readers_reject_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        report.read_spectrum(str(tmp_path / "absent.dat"))


def test_writer_rejects_mismatched_lengths(tmp_path):
    path = tmp_path / "spectrum.dat"
    with pytest.raises(ConfigError, match="energy grid"):
        report.write_spectrum(str(path), np.zeros(5),
                              {"singlet": np.zeros(4)})


def test_writer_rejects_unknown_channel(tmp_path):
    path = tmp_path / "sticks.dat"
    bad = [StateResult(
        index=0, channel="nonet", label=None, e_dication_ev=0.0,
        e_shifted_ev=0.0, dip_ev=0.0, e_excess_ev=1.0, intensity=0.0,
    )]
    with pytest.raises(ConfigError, match="unknown channel"):
        report.write_sticks(str(path), bad)


def test_writer_rejects_missing_directory(tmp_path):
    with pytest.raises(ConfigError, match="does not exist"):
        report.write_spectrum(str(tmp_path / "nope" / "spectrum.dat"),
                              np.zeros(2), {"singlet": np.zeros(2)})


def test_latex_table_writes_and_escapes(tmp_path):
    path = tmp_path / "states.tex"
    report.write_latex_table(str(path), _states(), settings=_settings())
    text = path.read_text(encoding="utf-8")
    assert text.startswith("% dpi_run.py")
    assert "\\begin{tabular}" in text and "\\end{table}" in text
    # The label is used verbatim here (no _token() pass), so the space
    # survives and only genuine underscores are escaped.
    assert "F1s T22" in text
    assert "dagger" in text            # the closed channel is footnoted
    assert "cross\\_dyson" in text     # a term column with an underscore


def test_latex_table_column_selection(tmp_path):
    path = tmp_path / "states.tex"
    report.write_latex_table(str(path), _states(), terms=["direct"])
    text = path.read_text(encoding="utf-8")
    assert "direct" in text
    assert "aa\\_bb" not in text


def test_empty_state_list_round_trips(tmp_path):
    path = tmp_path / "sticks.dat"
    report.write_sticks(str(path), [])
    assert report.read_sticks(str(path)).states == []
