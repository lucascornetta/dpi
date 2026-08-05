#!/usr/bin/env python3
"""Compute a double-photoionization CV spectrum from OpenMolcas output.

    python dpi_run.py dpi_input.toml
    python dpi_run.py dpi_input.toml --set physics.photon_energy_ev=770
    python dpi_run.py --demo out_demo        # synthetic self-test, no MOLCAS

Everything the physics depends on lives in the TOML input file; see
``dpi_input.toml`` for the annotated example.  The only command-line
options are an output-directory override, ``--set`` for scratch parameter
sweeps, and the reporting verbosity, so that a run is always reproducible
from its input file alone -- and the input file is echoed into the header
of every output file it produces.

Outputs
-------
``spectrum.dat``  broadened spectrum vs shifted double-ionization energy,
                  one column per channel plus the total
``sticks.dat``    one row per state: energies, DIP, E_excess, intensity,
                  every per-term integral and the spectroscopic factors
``states.tex``    optional LaTeX table

Plot them with ``dpi_plot.py``, which reads nothing else.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

# Make `import dpi` work regardless of how the script was invoked. Python
# normally puts the script's own directory on sys.path, but not under
# `python -I`/`-P`/PYTHONSAFEPATH=1, nor when the file is executed through a
# symlink or a wrapper from another directory. Prepending it explicitly costs
# nothing and removes the most common failure a new user hits:
#     ModuleNotFoundError: No module named 'dpi'
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from dpi import __version__, dyson, molcas_io, report, spectrum
except ModuleNotFoundError as _exc:                      # pragma: no cover
    if _exc.name != "dpi":
        raise
    raise SystemExit(
        f"dpi_run.py: cannot find the 'dpi' package.\n"
        f"  looked in: {_HERE}\n"
        f"  expected:  {os.path.join(_HERE, 'dpi', '__init__.py')}\n"
        f"\n"
        f"The modules must sit in a 'dpi/' subdirectory next to this script,\n"
        f"not beside it:\n"
        f"    dpi_run.py\n"
        f"    dpi_plot.py\n"
        f"    dpi/__init__.py  constants.py  config.py  molcas_io.py\n"
        f"        dyson.py  atomic_sigma.py  shakeoff.py  amplitudes.py\n"
        f"        spectrum.py  report.py\n"
    ) from None
from dpi.atomic_sigma import SigmaBuilder
from dpi.config import Config, load_config
from dpi.constants import HARTREE_EV, DPIError


# ── run context: the object the spectrum layer duck-types against ───────────

@dataclass
class RunContext:
    """What :func:`dpi.spectrum.integrate_state` needs to see.

    The spectrum layer resolves its atomic inputs through this object once
    per state rather than per quadrature point, which is what makes the
    per-AO cross-section lookup affordable.
    """

    omega_ev: float
    terms: Any
    sigma_builder: SigmaBuilder
    basis: Any
    omega_over_omega_eff: bool = False
    """Whether to apply the per-AO ``omega/(eps1 + I_mu)`` weight, [A-8]."""

    def sigma_at_eps1_grid(self, eps1_ev: np.ndarray) -> np.ndarray:
        # omega is forwarded only when the [physics] omega_over_omega_eff
        # switch is on; the builder then applies the per-AO
        # omega/(eps1 + I_mu) weight of REVIEW.md [A-8].  Passing None
        # reproduces the historical convention exactly.
        omega = self.omega_ev if self.omega_over_omega_eff else None
        return self.sigma_builder.at_eps1_grid(eps1_ev, omega_ev=omega)

    def p_shake_at_k(self, k_au: np.ndarray) -> np.ndarray:
        from dpi.shakeoff import p_shake
        return p_shake(self.basis, k_au)


# ── state bookkeeping ───────────────────────────────────────────────────────

_INDEX_RE = re.compile(r"(\d+)")


def state_index(path: str) -> int | None:
    """Extract the state index embedded in a RasOrb filename.

    Files are matched to energy-file lines by this index rather than by
    sort order, so that a missing state cannot silently shift every
    subsequent state's energy against its orbitals.
    """
    matches = _INDEX_RE.findall(os.path.basename(path))
    return int(matches[-1]) if matches else None


def collect_orbital_files(spec: str) -> dict[int, str]:
    """Map state index -> RasOrb path for a directory or glob."""
    if os.path.isdir(spec):
        paths = sorted(glob.glob(os.path.join(spec, "*")))
        paths = [p for p in paths if os.path.isfile(p)
                 and not p.endswith((".dat", ".txt", ".h5", ".npz"))]
    else:
        paths = sorted(glob.glob(spec))
    indexed: dict[int, str] = {}
    for path in paths:
        idx = state_index(path)
        if idx is None:
            continue
        if idx in indexed:
            raise DPIError(
                f"two orbital files claim state index {idx}:\n"
                f"  {indexed[idx]}\n  {path}\n"
                f"the index embedded in the filename must be unique.")
        indexed[idx] = path
    if not paths:
        raise DPIError(f"no orbital files found matching {spec!r}")
    return indexed


def read_energies(path: str, unit: str) -> np.ndarray:
    """Read one dication total energy per line, returned in eV."""
    values: list[float] = []
    with open(path) as handle:
        for lineno, line in enumerate(handle, 1):
            text = line.strip()
            if not text or text[0] in "#*%":
                continue
            try:
                values.append(float(text.split()[0].replace("D", "E")))
            except ValueError as exc:
                raise DPIError(
                    f"{path}:{lineno}: cannot read an energy from "
                    f"{text!r}") from exc
    if not values:
        raise DPIError(f"{path} contains no energies.")
    array = np.asarray(values, dtype=float)
    return array * HARTREE_EV if unit == "au" else array


@dataclass
class StateSpec:
    """One dication state of one channel, before the physics is done."""

    index: int
    channel: str
    e_dication_ev: float
    e_shifted_ev: float
    dip_ev: float
    orbital_path: str | None
    """Dication orbitals for the S_dic = 0 set."""

    triplet_path: str | None = None
    """Dication orbitals for the S_dic = 1 set.

    Only populated in jj coupling, where each j_C peak is A(S=0) + A(S=1)
    and the two LS pieces come from different OSRHF solutions.  In LS
    coupling each channel is a single multiplicity and this stays None.
    """


def build_state_specs(cfg: Config, log: Callable[[str], None]
                      ) -> dict[str, list[StateSpec]]:
    """Pair energies with orbital files and assign each state its own DIP.

    The lowest state across all channels defines both the physical
    threshold (``true_dip_ev``) and the cosmetic x-axis origin
    (``display_onset_ev``); every other state keeps its computed spacing
    relative to it.  Keeping these two shifts separate is the point: the
    first enters ``E_excess`` and therefore every cross section, the second
    only moves the plot.
    """
    raw = {ch: read_energies(cfg.paths.energies[ch], cfg.output.energies_unit)
           for ch in cfg.channels}
    e_ref = min(float(values.min()) for values in raw.values())

    # Orbital sets are keyed by multiplicity in BOTH coupling modes: MOLCAS
    # converges S_dic = 0 and S_dic = 1 separately and the j_C states are a
    # post-hoc recombination of the two, so no `jc32` orbital set exists.
    orbitals = {key: collect_orbital_files(cfg.paths.dication_orbitals[key])
                for key in cfg.orbital_channels}
    jj = cfg.physics.coupling == "jj"

    specs: dict[str, list[StateSpec]] = {}
    for channel in cfg.channels:
        # In LS mode the channel names ARE the orbital keys; in jj mode both
        # peaks draw on both sets.
        indexed = orbitals[channel] if not jj else orbitals["singlet"]
        indexed_t = orbitals["triplet"] if jj else None
        energies = raw[channel]
        missing: list[int] = []
        rows: list[StateSpec] = []
        for line, energy in enumerate(energies, start=1):
            offset = float(energy) - e_ref
            path = indexed.get(line)
            path_t = indexed_t.get(line) if indexed_t is not None else None
            # A jj state needs BOTH sets; one alone cannot give A(S=0)+A(S=1).
            if path is None or (jj and path_t is None):
                missing.append(line)
            rows.append(StateSpec(
                index=line,
                channel=channel,
                e_dication_ev=float(energy),
                e_shifted_ev=cfg.output.display_onset_ev + offset,
                dip_ev=cfg.physics.true_dip_ev + offset,
                orbital_path=path,
                triplet_path=path_t))
        specs[channel] = rows
        extra = sorted(set(indexed) - set(range(1, len(energies) + 1)))
        log(f"  {channel:8s} {len(energies):3d} states, "
            f"{len(energies) - len(missing):3d} with orbitals, "
            f"E_shifted {rows[0].e_shifted_ev:.2f} .. "
            f"{rows[-1].e_shifted_ev:.2f} eV")
        if missing:
            log(f"    no orbital file for state(s) {missing}; "
                f"these contribute zero intensity")
        if extra:
            log(f"    orbital files with index {extra} exceed the "
                f"{len(energies)} energies listed; ignored")
    return specs


# ── the calculation ─────────────────────────────────────────────────────────

def dyson_for_orbitals(orbital_path: str, index: int, cfg: Config,
                       shared: dict[str, Any],
                       log: Callable[[str], None]) -> dyson.DysonObjects:
    """Load or build the Dyson objects for one dication orbital set.

    Every object of a state is built in a single call so that they share
    one sign gauge; their relative signs are physical and enter the
    one-/two-electron interference term.

    The cache is keyed by the orbital FILENAME rather than by channel: in jj
    coupling both j_C peaks are assembled from the same pair of multiplicity
    orbital sets, so keying by channel would either collide two different
    objects on one file or rebuild the same objects twice.
    """
    stem = os.path.splitext(os.path.basename(orbital_path))[0]
    cache = os.path.join(cfg.paths.cache_dir, f"{stem}_{index:04d}.npz")
    # The cache is keyed by filename, so a file written by an earlier run
    # with different SWITCHES is a name match but a content mismatch.  The
    # signature records which optional blocks that run built; without it a
    # stale cache either raises a confusing error deep in build_blocks
    # (indirect terms, no lam_i) or -- worse -- silently omits the frozen
    # block from the output.  See REVIEW.md [C-7].
    want = "|".join((
        f"lam={cfg.terms.indirect or cfg.terms.dir_ind_interference}",
        f"frozen={cfg.physics.include_frozen}",
    ))
    if os.path.isfile(cache):
        try:
            return dyson.DysonObjects.load(cache, require=want)
        except Exception as exc:
            log(f"    cache {cache} stale or unreadable ({exc}); rebuilding")

    dication = molcas_io.read_inporb(orbital_path)
    holes = molcas_io.rohf_hole_indices(dication.occ, cfg.physics.n_neu_occ)
    if holes.approximation:
        log(f"    state {index}: {holes.approximation}")

    objects = dyson.build_state_objects(
        C_neu=shared["C_neu"],
        C_dic=dication.coeff,
        S_ao=shared["S_ao"],
        n_neu_occ=cfg.physics.n_neu_occ,
        holes=holes,
        dipole_ao=shared["dipole"],
        include_frozen=cfg.physics.include_frozen,
        meta={"index": index,
              "orbitals": os.path.basename(orbital_path)})

    os.makedirs(cfg.paths.cache_dir, exist_ok=True)
    objects.save(cache)
    return objects


def dyson_for_state(spec: StateSpec, cfg: Config, shared: dict[str, Any],
                    log: Callable[[str], None]
                    ) -> tuple[dyson.DysonObjects, dyson.DysonObjects | None] \
                         | None:
    """Dyson objects for one state: the S=0 set, and the S=1 set in jj mode."""
    if spec.orbital_path is None:
        return None
    primary = dyson_for_orbitals(spec.orbital_path, spec.index, cfg, shared,
                                 log)
    secondary = None
    if spec.triplet_path is not None:
        secondary = dyson_for_orbitals(spec.triplet_path, spec.index, cfg,
                                       shared, log)
    return primary, secondary


def run(cfg: Config, log: Callable[[str], None]) -> dict[str, Any]:
    """Execute the calculation and write the output files."""
    started = time.time()
    log("")
    for line in cfg.summary_lines():
        log(f"  {line}")
    log("")

    log("reading shared inputs")
    neutral = molcas_io.read_inporb(cfg.paths.neutral_orbitals)
    nbas = neutral.coeff.shape[0]
    # SEWARD already writes AO_OVERLAP_MATRIX into the HDF5, so a separate
    # text file is optional: point `overlap` at the .h5 (or omit it) and the
    # matrix is taken from there. One less file to keep in sync, and it
    # cannot disagree with the basis the same file describes.
    if cfg.paths.overlap.lower().endswith((".h5", ".hdf5")):
        S_ao = molcas_io.read_ao_overlap_h5(cfg.paths.overlap)
        if S_ao.shape != (nbas, nbas):
            raise DPIError(
                f"{cfg.paths.overlap} holds a {S_ao.shape} overlap but the "
                f"orbital file has {nbas} basis functions.")
    else:
        S_ao = molcas_io.read_overlap(cfg.paths.overlap, nbas)
    basis = molcas_io.read_basis(cfg.paths.h5file)
    if basis.nbas != nbas:
        raise DPIError(
            f"the orbital file has {nbas} AO basis functions but "
            f"{cfg.paths.h5file} describes {basis.nbas}; they must come "
            f"from the same SEWARD run.")
    log(f"  {nbas} AOs, {basis.harmonics} harmonics, "
        f"{cfg.physics.n_neu_occ} occupied neutral MOs")

    dipole = None
    if cfg.terms.indirect or cfg.terms.dir_ind_interference:
        dipole = molcas_io.read_ao_dipole(
            cfg.paths.h5file, one_centre=cfg.physics.one_centre_dipole)
        kind = ("one-centre origin-shifted" if cfg.physics.one_centre_dipole
                else "full molecular")
        log(f"  AO dipole integrals read ({kind})")

    sigma_builder = SigmaBuilder(
        basis,
        high_energy_exponents=cfg.physics.high_energy_exponent,
        threshold_overrides=cfg.physics.threshold_override,
        threshold_model=cfg.physics.threshold_model,
        anchor_delta_ev=cfg.physics.anchor_delta_ev)
    assigned = getattr(sigma_builder, "n_assigned", None)
    if assigned is not None:
        log(f"  {assigned}/{nbas} AOs carry a tabulated atomic subshell; "
            f"the rest contribute zero cross section")
    unsafe = getattr(sigma_builder, "unsafe_extrapolations", lambda: ())()
    if unsafe:
        # Two independent things make a near-threshold estimate untrustworthy,
        # and they need different wording because the fix differs.  Name the
        # reason per group rather than refuse: the affected subshells usually
        # carry little intensity, so the run is still useful.
        wide, disagree = [], []
        for k in unsafe:
            arm = sigma_builder.lever_arm(k)
            if arm > 0.10:
                wide.append(f"{k} (gap is {100 * arm:.0f}% of I_mu)")
            else:
                disagree.append(f"{k} (gap only {100 * arm:.1f}% of I_mu)")
        model = cfg.physics.threshold_model
        if wide:
            log(f"  WARNING: threshold_model='{model}' extrapolates over a "
                f"wide gap for {', '.join(wide)}; if the table starts past "
                f"the subshell maximum this manufactures a rise the real "
                f"curve does not have (REVIEW.md [A-14])")
        if disagree:
            log(f"  WARNING: 'coulomb' and 'extrapolate' disagree by >15% at "
                f"threshold for {', '.join(disagree)} despite a short gap -- "
                f"the tabulated curve is not locally power-law there (a "
                f"delayed maximum at the edge), so neither estimate is "
                f"reliable (REVIEW.md [A-16])")

    context = RunContext(
        omega_ev=cfg.physics.photon_energy_ev,
        terms=cfg.terms, sigma_builder=sigma_builder, basis=basis,
        omega_over_omega_eff=cfg.physics.omega_over_omega_eff)

    log("")
    log("pairing states with orbital files")
    specs = build_state_specs(cfg, log)

    shared = {"C_neu": neutral.coeff, "S_ao": S_ao, "dipole": dipole}
    results: dict[str, list[spectrum.StateResult]] = {}
    frozen_results: dict[str, list[spectrum.StateResult]] = {}

    log("")
    for channel in cfg.channels:
        log(f"integrating {channel}")
        rows: list[spectrum.StateResult] = []
        frozen_rows: list[spectrum.StateResult] = []
        closed = negative = 0
        for spec in specs[channel]:
            built = dyson_for_state(spec, cfg, shared, log)
            if built is None:
                continue
            objects, objects_t = built
            result = spectrum.integrate_state(
                objects, context, spec.dip_ev, channel,
                n_quad=cfg.physics.n_quad, dyson_triplet=objects_t,
                index=spec.index,
                e_dication_ev=spec.e_dication_ev,
                e_shifted_ev=spec.e_shifted_ev)
            rows.append(result)
            closed += not result.open
            negative += bool(getattr(result, "negative", False))

            if cfg.physics.include_frozen and objects.frozen is not None:
                frozen_rows.append(spectrum.integrate_state(
                    objects.frozen, context, spec.dip_ev, channel,
                    n_quad=cfg.physics.n_quad,
                    dyson_triplet=(objects_t.frozen
                                   if objects_t is not None else None),
                    index=spec.index,
                    e_dication_ev=spec.e_dication_ev,
                    e_shifted_ev=spec.e_shifted_ev))

        results[channel] = rows
        if frozen_rows:
            frozen_results[channel] = frozen_rows
        total = sum(r.intensity for r in rows)
        log(f"  {len(rows)} states, sum of intensities {total:.6e}")
        if closed:
            log(f"  {closed} channel(s) closed (E_excess <= 0); reported "
                f"with zero intensity, not dropped")
        if negative:
            log(f"  [!] {negative} state(s) have a negative integrated "
                f"amplitude.  The singlet's -4 D_ij S_ij term can dominate "
                f"when the two Dyson orbitals are not AO-orthogonal; this "
                f"is a model breakdown, reported rather than clamped.")

    # Quadrature convergence, on the strongest state of the first channel.
    first = cfg.channels[0]
    if results[first]:
        strongest = max(results[first], key=lambda r: abs(r.intensity))
        spec = next(s for s in specs[first] if s.index == strongest.index)
        built = dyson_for_state(spec, cfg, shared, log)
        if built is not None and hasattr(spectrum, "convergence"):
            objects, objects_t = built
            try:
                conv = spectrum.convergence(objects, context, spec.dip_ev,
                                            first, n_quad=cfg.physics.n_quad,
                                            dyson_triplet=objects_t)
                log("")
                log(f"quadrature check on the strongest {first} state "
                    f"(index {strongest.index}): "
                    f"relative change on doubling n_quad = "
                    f"{_conv_value(conv):.2e}")
            except Exception as exc:
                log(f"  quadrature convergence check skipped: {exc}")

    written = write_outputs(cfg, results, frozen_results, log)
    log("")
    log(f"done in {time.time() - started:.1f} s")
    return {"results": results, "frozen": frozen_results, "written": written}


def _conv_value(conv: Any) -> float:
    """Relative quadrature residual between n_quad and 2*n_quad."""
    if isinstance(conv, dict):
        return float(conv.get("rel_residual", float("nan")))
    if isinstance(conv, (int, float)):
        return float(conv)
    return float("nan")


def write_outputs(cfg: Config,
                  results: dict[str, list[spectrum.StateResult]],
                  frozen: dict[str, list[spectrum.StateResult]],
                  log: Callable[[str], None]) -> list[str]:
    """Broaden, then write spectrum.dat, sticks.dat and the LaTeX table."""
    os.makedirs(cfg.output.directory, exist_ok=True)

    positions = [r.e_shifted_ev for rows in results.values() for r in rows]
    lo, hi = cfg.output.grid_bounds(positions)
    n_point = max(int(round((hi - lo) / cfg.output.e_step_ev)) + 1, 2)
    grid = np.linspace(lo, hi, n_point)

    columns = {
        channel: spectrum.broaden(grid, rows, cfg.output.voigt_sigma_ev,
                                  cfg.output.voigt_gamma_ev)
        for channel, rows in results.items()}

    if frozen:
        # The frozen limit is a shape reference, so it is emitted as a single
        # summed column rather than per channel.
        frozen_total = np.zeros_like(grid)
        for rows in frozen.values():
            frozen_total += spectrum.broaden(grid, rows,
                                             cfg.output.voigt_sigma_ev,
                                             cfg.output.voigt_gamma_ev)
        columns["frozen_total"] = frozen_total

    # The provenance block goes in as `note` rather than as a settings value:
    # settings entries are flattened to one `key = value` line each, which
    # would collapse the whole summary onto a single unreadable line.
    settings = {"terms": cfg.terms.as_dict(), "version": __version__}
    note = list(cfg.summary_lines()) + [
        "",
        "intensities are RELATIVE (Mb * a.u.); see REVIEW.md [P-6]",
        "E_shifted_eV = display_onset_ev + (E_f - E_lowest)",
        "the physics uses DIP_f = true_dip_ev + (E_f - E_lowest)"]

    written: list[str] = []
    path = report.write_spectrum(cfg.out_path(cfg.output.spectrum), grid,
                                columns, settings=settings, note=note)
    written.append(path)
    log(f"wrote {path}  ({n_point} points, "
        f"{len(columns)} channel column(s) + total)")

    flat = [r for channel in cfg.channels for r in results[channel]]
    path = report.write_sticks(cfg.out_path(cfg.output.sticks), flat,
                               settings=settings, note=note)
    written.append(path)
    log(f"wrote {path}  ({len(flat)} states)")

    if cfg.output.latex_table:
        path = report.write_latex_table(
            cfg.out_path(cfg.output.latex_table), flat)
        written.append(path)
        log(f"wrote {path}")
    return written


# ── synthetic demo ──────────────────────────────────────────────────────────

def demo(directory: str, log: Callable[[str], None],
         coupling: str = "ls",
         overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the whole pipeline on a synthetic case, with no MOLCAS output.

    Exercises every module end to end: the readers, the Dyson algebra, the
    atomic tables, the shake-off transforms, the quadrature and the
    writers.  Useful as an installation check and as the fixture for the
    end-to-end test.

    ``overrides`` are the parsed ``--set`` assignments, applied to the
    generated input file exactly as they would be to a user's own.  Without
    this, ``--demo`` silently ignored every ``--set`` on the command line,
    which made the synthetic case useless for trying a switch out before
    committing a real run to it.
    """
    os.makedirs(directory, exist_ok=True)
    case_dir = os.path.join(directory, "synthetic")
    case = molcas_io.write_synthetic_case(case_dir, n_neu_occ=5,
                                          hole_i=0, hole_j=3)
    log(f"synthetic case written to {case_dir}: "
        f"{case.nbas} AOs, {case.n_neu_occ} occupied MOs, "
        f"{case.harmonics} harmonics")

    channels = ("singlet", "triplet") if coupling == "ls" else ("jc32", "jc12")
    orbital_keys = ("singlet", "triplet")
    n_state = 4

    # Orbital sets are keyed by MULTIPLICITY in both coupling modes: MOLCAS
    # never optimises orbitals for a j_C state, so a jj input still supplies
    # singlet and triplet RasOrb directories and only its energies carry the
    # jc32/jc12 labels.
    for key in orbital_keys:
        orb_dir = os.path.join(directory, f"orb_{key}")
        os.makedirs(orb_dir, exist_ok=True)
        for k in range(1, n_state + 1):
            # Distinct dication files per state would need distinct orbital
            # sets; one set per multiplicity suffices to exercise the
            # pipeline, with the state energies supplying the structure.
            target = os.path.join(orb_dir, f"state_{k}.RasOrb")
            with open(case.inporb_dication) as src, open(target, "w") as dst:
                dst.write(src.read())

    for channel in channels:
        energies = [-100.0 + 0.05 * (k - 1)
                    + (0.02 if channel in ("triplet", "jc12") else 0.0)
                    for k in range(1, n_state + 1)]
        with open(os.path.join(directory, f"e_{channel}.dat"), "w") as fh:
            fh.write("\n".join(f"{e:.10f}" for e in energies) + "\n")

    toml = [
        "[physics]",
        "photon_energy_ev = 300.0",
        "true_dip_ev = 120.0",
        "n_neu_occ = 5",
        f'coupling = "{coupling}"',
        "n_quad = 120",
        "",
        "[paths]",
        f'neutral_orbitals = "{case.inporb_neutral}"',
        f'overlap = "{case.overlap}"',
        f'h5file = "{case.h5}"',
        "dication_orbitals = { "
        + ", ".join(f'{c} = "{os.path.join(directory, "orb_" + c)}"'
                    for c in orbital_keys) + " }",
        "energies = { "
        + ", ".join(f'{c} = "{os.path.join(directory, "e_" + c + ".dat")}"'
                    for c in channels) + " }",
        f'cache_dir = "{os.path.join(directory, "cache")}"',
        "",
        "[output]",
        f'directory = "{os.path.join(directory, "out")}"',
        'energies_unit = "au"',
        "display_onset_ev = 20.0",
        "e_step_ev = 0.05",
        "",
        "[terms]",
        "",
    ]
    toml_path = os.path.join(directory, "demo.toml")
    with open(toml_path, "w") as fh:
        fh.write("\n".join(toml))
    log(f"demo input file: {toml_path}")

    return run(load_config(toml_path, overrides=overrides or {}), log)


# ── command line ────────────────────────────────────────────────────────────

def parse_set(assignments: Sequence[str]) -> dict[str, Any]:
    """Turn ``section.key=value`` strings into typed overrides."""
    overrides: dict[str, Any] = {}
    for item in assignments:
        if "=" not in item:
            raise SystemExit(
                f"dpi_run: --set expects section.key=value, got {item!r}")
        key, _, text = item.partition("=")
        key, text = key.strip(), text.strip()
        lowered = text.lower()
        if lowered in ("true", "false"):
            value: Any = lowered == "true"
        else:
            try:
                value = int(text)
            except ValueError:
                try:
                    value = float(text)
                except ValueError:
                    value = text
        overrides[key] = value
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute a DPI core-valence spectrum from an input file.")
    parser.add_argument("input", nargs="?",
                        help="TOML input file; see dpi_input.toml")
    parser.add_argument("--output-dir", metavar="DIR",
                        help="override [output] directory")
    parser.add_argument("--set", action="append", default=[],
                        metavar="SECTION.KEY=VALUE",
                        help="override one input-file setting; repeatable")
    parser.add_argument("--demo", metavar="DIR",
                        help="run a synthetic case in DIR, no MOLCAS needed")
    parser.add_argument("--demo-coupling", choices=("ls", "jj"), default="ls",
                        help="coupling mode for --demo")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="print only warnings and the files written")
    parser.add_argument("--version", action="version",
                        version=f"dpi {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def log(message: str) -> None:
        if not args.quiet:
            print(message)

    try:
        if args.demo:
            demo_overrides = parse_set(args.set)
            if args.output_dir:
                demo_overrides["output.directory"] = args.output_dir
            demo(args.demo, log, coupling=args.demo_coupling,
                 overrides=demo_overrides)
            return 0
        if not args.input:
            build_parser().print_help()
            return 2
        overrides = parse_set(args.set)
        if args.output_dir:
            overrides["output.directory"] = args.output_dir
        run(load_config(args.input, overrides=overrides), log)
    except DPIError as exc:
        print(f"\ndpi_run: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

