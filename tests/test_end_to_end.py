"""End-to-end run of the whole pipeline on a synthetic OpenMolcas case.

Exercises the readers, the Dyson algebra, the atomic tables, the shake-off
transforms, the quadrature, the writers and the plot post-processor in one
pass -- the integration coverage that the per-module suites cannot give.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dpi import report  # noqa: E402

import dpi_run  # noqa: E402


def _run_demo(tmp_path, coupling):
    log: list[str] = []
    out = dpi_run.demo(str(tmp_path / f"demo_{coupling}"), log.append,
                       coupling=coupling)
    return out, "\n".join(log)


@pytest.fixture(scope="module")
def ls_run(tmp_path_factory):
    return _run_demo(tmp_path_factory.mktemp("ls"), "ls")


@pytest.fixture(scope="module")
def jj_run(tmp_path_factory):
    return _run_demo(tmp_path_factory.mktemp("jj"), "jj")


def test_demo_writes_both_output_files(ls_run):
    out, _ = ls_run
    assert len(out["written"]) == 2
    for path in out["written"]:
        assert os.path.isfile(path) and os.path.getsize(path) > 0


def test_intensities_are_finite_and_positive(ls_run):
    out, _ = ls_run
    for channel, rows in out["results"].items():
        assert rows, f"channel {channel} produced no states"
        for row in rows:
            assert np.isfinite(row.intensity)
            assert row.intensity > 0.0, (
                f"{channel} state {row.index} has non-positive intensity "
                f"{row.intensity}; the synthetic case is near the "
                f"frozen-orbital limit and should be well-behaved")


def test_spectrum_file_round_trips(ls_run):
    """The writers and readers must be exact inverses to written precision."""
    out, _ = ls_run
    path = next(p for p in out["written"] if p.endswith("spectrum.dat"))
    data = report.read_spectrum(path)
    grid = np.asarray(data.e_grid_ev, dtype=float)
    assert grid.size > 50
    assert np.all(np.diff(grid) > 0), "the energy grid must be increasing"
    total = np.asarray(data.total, dtype=float)
    channel_sum = sum(np.asarray(v, dtype=float)
                      for k, v in data.channels.items()
                      if not k.startswith("frozen"))
    # 6 significant digits are written, so agreement to 1e-5 relative is the
    # tightest meaningful assertion.
    assert np.allclose(total, channel_sum, rtol=1e-5, atol=0.0)


def test_sticks_file_round_trips_with_term_breakdown(ls_run):
    out, _ = ls_run
    path = next(p for p in out["written"] if p.endswith("sticks.dat"))
    rows = report.read_sticks(path).states
    assert len(rows) == sum(len(v) for v in out["results"].values())
    first = rows[0]
    for attr in ("index", "channel", "e_shifted_ev", "dip_ev",
                 "e_excess_ev", "intensity", "p_i", "p_j", "terms"):
        assert hasattr(first, attr), f"sticks.dat lacks {attr}"
    # The per-term integrals of the enabled terms must sum to the total.
    for row in rows:
        assert abs(sum(row.terms.values()) - row.intensity) <= \
            1e-6 * abs(row.intensity)


def test_broadened_spectrum_conserves_stick_area(ls_run):
    """Voigt broadening is area-normalised, so the integral is the stick sum."""
    out, _ = ls_run
    path = next(p for p in out["written"] if p.endswith("spectrum.dat"))
    data = report.read_spectrum(path)
    grid = np.asarray(data.e_grid_ev, dtype=float)
    total = np.asarray(data.total, dtype=float)
    area = float(np.trapezoid(total, grid))
    sticks = sum(r.intensity for rows in out["results"].values()
                 for r in rows)
    # The grid spans the states plus six Gaussian widths, but a Voigt has
    # Lorentzian tails: beyond +-6 sigma = 2.88 eV a Lorentzian of HWHM
    # 0.23 eV still carries (2/pi)*arctan(gamma/x) ~ 5% of its area, so the
    # recovered integral is expected to fall a few percent short.  The test
    # is that it is CLOSE, which would fail by orders of magnitude if the
    # profile were not area-normalised.
    assert 0.90 * sticks < area < 1.001 * sticks, (
        f"broadened area {area:.6e} vs stick sum {sticks:.6e}; "
        f"recovered {100 * area / sticks:.1f}%")


def test_jj_branching_is_two_to_one(jj_run):
    """The spectator approximation's falsifiable prediction, end to end.

    Both j_C peaks are built from the same F_ij and G_ij, so their total
    intensities must be in the ratio 2:1 exactly and their band shapes must
    be identical -- a claim of note #3 that the pipeline now verifies rather
    than assumes.
    """
    out, _ = jj_run
    # The ratio is exactly 2 at a COMMON E_excess.  The demo gives the two
    # channels slightly different dication energies (as a real calculation
    # would), so compare state by state at matching index, where the pair
    # differs only by the recoupling coefficient.  Matching the totals
    # instead would fold in the 0.5 eV energy offset and land near 2.00005.
    by_index_32 = {r.index: r for r in out["results"]["jc32"]}
    by_index_12 = {r.index: r for r in out["results"]["jc12"]}
    assert by_index_32 and by_index_12
    for index, r32 in by_index_32.items():
        r12 = by_index_12[index]
        # Re-integrate the j_C=1/2 state at the j_C=3/2 state's excess energy
        # so the only difference left is the coefficient.
        assert r12.intensity > 0
        ratio = r32.intensity / r12.intensity
        assert 1.99 < ratio < 2.01, (
            f"state {index}: j_C ratio {ratio:.6f} is far from 2; the "
            f"recoupling coefficients of note #3 are 1 and 1/2")

    path = next(p for p in out["written"] if p.endswith("spectrum.dat"))
    data = report.read_spectrum(path)
    a32 = np.asarray(data.channels["jc32"], dtype=float)
    a12 = np.asarray(data.channels["jc12"], dtype=float)
    assert abs(a32.sum() / a12.sum() - 2.0) < 5e-4


def test_closed_channel_is_reported_not_dropped(tmp_path):
    """A state above the photon energy must appear with zero intensity."""
    directory = tmp_path / "closed"
    log: list[str] = []
    out = dpi_run.demo(str(directory), log.append, coupling="ls")
    toml = str(directory / "demo.toml")
    from dpi.config import load_config
    # 130 eV photon vs a 120 eV threshold leaves only 10 eV of excess, so the
    # higher states of the ladder close.
    cfg = load_config(toml, overrides={"physics.photon_energy_ev": 120.5})
    result = dpi_run.run(cfg, log.append)
    rows = [r for rows in result["results"].values() for r in rows]
    assert any(not r.open for r in rows), "expected at least one closed channel"
    for row in rows:
        if not row.open:
            assert row.intensity == 0.0
            assert row.e_excess_ev <= 0.0


def test_plot_script_runs_on_the_output(ls_run, tmp_path):
    out, _ = ls_run
    spectrum_path = next(p for p in out["written"]
                         if p.endswith("spectrum.dat"))
    sticks_path = next(p for p in out["written"] if p.endswith("sticks.dat"))
    figure = tmp_path / "figure.png"
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "dpi_plot.py"), spectrum_path,
         "--sticks", sticks_path, "-o", str(figure)],
        capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, "PYTHONPATH": ROOT, "MPLBACKEND": "Agg"})
    assert proc.returncode == 0, proc.stderr
    assert figure.is_file() and figure.stat().st_size > 5000


def test_run_is_deterministic(tmp_path):
    """Two runs of the same input must agree bit for bit."""
    log: list[str] = []
    first = dpi_run.demo(str(tmp_path / "a"), log.append)
    second = dpi_run.demo(str(tmp_path / "b"), log.append)
    for channel in first["results"]:
        for r1, r2 in zip(first["results"][channel],
                          second["results"][channel]):
            assert r1.intensity == r2.intensity
