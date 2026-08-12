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


def build_state_specs(cfg: Config, log: Callable[[str], None],
                      frozen: bool = False) -> dict[str, list[StateSpec]]:
    """Pair energies with orbital files and assign each state its own DIP.

    The lowest state across all channels defines both the physical
    threshold (``true_dip_ev``) and the cosmetic x-axis origin
    (``display_onset_ev``); every other state keeps its computed spacing
    relative to it.  Keeping these two shifts separate is the point: the
    first enters ``E_excess`` and therefore every cross section, the second
    only moves the plot.

    With ``frozen=True`` the same pairing is done for the frozen-orbital
    calculation, from ``paths.frozen_energies`` and
    ``paths.frozen_orbitals``.  **The two sets are paired independently.**
    A frozen-orbital calculation converges its states in its own order, and
    nothing guarantees that its state 3 is the relaxed state 3 -- the
    orbital files carry their own indices and are matched to their own
    energy list, so a reordering between the two calculations cannot
    silently pair a relaxed energy with a frozen orbital set.

    The frozen ladder is referenced to the frozen set's **own** lowest
    state, so both curves begin at ``true_dip_ev`` and their shapes are
    directly comparable.  That discards the relaxation energy, which is
    physically real, so the offset between the two references is computed
    and reported rather than thrown away -- see
    :func:`frozen_reference_offset`.
    """
    energy_paths = (cfg.paths.frozen_energies if frozen
                    else cfg.paths.energies)
    orbital_paths = (cfg.paths.frozen_orbitals if frozen
                     else cfg.paths.dication_orbitals)
    raw = {ch: read_energies(energy_paths[ch], cfg.output.energies_unit)
           for ch in cfg.channels}
    e_ref = min(float(values.min()) for values in raw.values())

    # Orbital sets are keyed by multiplicity in BOTH coupling modes: MOLCAS
    # converges S_dic = 0 and S_dic = 1 separately and the j_C states are a
    # post-hoc recombination of the two, so no `jc32` orbital set exists.
    orbitals = {key: collect_orbital_files(orbital_paths[key])
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
        log(f"  {'[frozen] ' if frozen else ''}{channel:8s} "
            f"{len(energies):3d} states, "
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


def frozen_reference_offset(cfg: Config) -> float | None:
    """``E_lowest^frozen - E_lowest^relaxed`` in eV, or ``None``.

    Both spectra are placed on their own internal ladder so their band
    *shapes* are comparable (each starts at ``true_dip_ev``).  That is the
    right choice for a diagnostic, but it discards a physically real
    quantity: an unrelaxed dication lies **above** the relaxed one, because
    relaxation is variational and can only lower the energy.  Rather than
    lose it, the offset is computed here and reported in the run header and
    the per-state table.

    Returns ``None`` when the two energy lists are not on a common absolute
    scale, which is detected rather than assumed: if either list is already
    referenced to its own minimum (its smallest entry is 0 within
    ``1e-9``), differencing the two minima is meaningless.  A negative
    offset is returned but flagged by the caller, since relaxation lowering
    the energy means the frozen minimum should be the higher of the two.
    """
    if not (cfg.paths.frozen_energies and cfg.paths.energies):
        return None
    unit = cfg.output.energies_unit
    try:
        relaxed = [read_energies(cfg.paths.energies[ch], unit)
                   for ch in cfg.channels]
        frozen = [read_energies(cfg.paths.frozen_energies[ch], unit)
                  for ch in cfg.channels]
    except Exception:
        return None
    lo_r = min(float(v.min()) for v in relaxed)
    lo_f = min(float(v.min()) for v in frozen)
    if abs(lo_r) < 1e-9 or abs(lo_f) < 1e-9:
        # One of the lists is already relative to its own minimum, so the
        # two minima do not share a zero and their difference is not a
        # relaxation energy.
        return None
    return lo_f - lo_r


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
        # The relaxed objects never carry a frozen sub-block now, so this
        # component is constant; it stays in the signature so that caches
        # written by the older code -- which DID attach one under
        # include_frozen -- read as mismatched and are rebuilt.
        "frozen=False",
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
        # NOT cfg.physics.include_frozen: the frozen limit is now a separate
        # calculation from its own orbital files (frozen_dyson_for_state), so
        # attaching a frozen sub-block to every relaxed state would be dead
        # work -- and would derive the holes from the RELAXED file, which is
        # the mis-assignment the separate path exists to avoid.
        include_frozen=False,
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


def frozen_dyson_for_state(spec: StateSpec, cfg: Config,
                           shared: dict[str, Any],
                           log: Callable[[str], None]
                           ) -> tuple[dyson.DysonObjects,
                                      dyson.DysonObjects | None] | None:
    """Frozen-limit Dyson objects, from the frozen calculation's own orbitals.

    The frozen ``.RasOrb`` is read for one reason: it is the only record of
    **which orbitals carry the holes** in that calculation.  Its
    coefficients are (verifiably) the neutral ones up to a signed
    permutation, so they add no orbital information -- but its ``#INDEX``
    identifies the hole pair, and its state ordering need not match the
    relaxed one.  Deriving the holes from the relaxed file instead would
    silently mis-assign every state whenever the two calculations converge
    in different orders, which is exactly the case this function exists for.

    Objects are then built by :func:`dpi.dyson.frozen_objects` from the
    **neutral** coefficients, which is what the frozen limit means: the
    overlap matrix is a Kronecker delta, ``det(S_beta) = 1``, and the
    two-electron Dyson minor is the frozen-orbital expression of note #4.
    Passing the frozen file's own coefficients would give the same numbers
    up to the permutation, so the neutral set is used and the frozen file is
    the *check* on that assumption -- :func:`molcas_io.frozen_hole_indices`
    refuses a file that is not a permutation.
    """
    if spec.orbital_path is None:
        return None

    expected = cfg.physics.frozen_expect_holes.get(spec.channel, {}).get(
        spec.index)

    def build(path: str) -> dyson.DysonObjects:
        orb = molcas_io.read_inporb(path)
        holes = molcas_io.frozen_hole_indices(
            orb, shared["neutral"], shared["S_ao"], cfg.physics.n_neu_occ,
            active_mos=cfg.physics.frozen_active_mos or None,
            expect_holes=expected)
        return dyson.frozen_objects(
            shared["C_neu"], shared["S_ao"], cfg.physics.n_neu_occ,
            holes.hole_i, holes.hole_j,
            dipole_ao=shared["dipole"],
            meta={"index": spec.index, "limit": "frozen",
                  "orbitals": os.path.basename(path),
                  "hole_i": holes.hole_i, "hole_j": holes.hole_j})

    primary = build(spec.orbital_path)
    secondary = build(spec.triplet_path) if spec.triplet_path else None
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
        anchor_delta_ev=cfg.physics.anchor_delta_ev,
        # Energy conservation against the REAL photon.  Without this the
        # builder reads every subshell at hv = eps1 + I_mu >= I_mu, so an AO
        # whose threshold exceeds omega still gets the cross section it would
        # have at its own onset -- see REVIEW.md [A-18].
        photon_energy_ev=cfg.physics.photon_energy_ev)
    assigned = getattr(sigma_builder, "n_assigned", None)
    if assigned is not None:
        log(f"  {assigned}/{nbas} AOs carry a tabulated atomic subshell; "
            f"the rest contribute zero cross section")
    # `closed_subshells` is filled lazily, on the first at_eps1_grid call, so
    # it is empty here.  Ask the thresholds directly instead: the criterion is
    # just I_mu > omega and needs no evaluation.
    closed = sorted(k for k in sigma_builder.subshells_used
                    if sigma_builder.threshold_of(k)
                    > cfg.physics.photon_energy_ev)
    if closed:
        # Reported, not fatal: at a soft edge some deep subshell legitimately
        # sits above the photon energy and contributes nothing.  But if an AO
        # carrying real dipole strength is closed, the photon energy and the
        # state list probably belong to different edges.
        pretty = ", ".join(
            f"{k} (I_mu = {sigma_builder.threshold_of(k):.1f} eV)"
            for k in closed)
        log(f"  {len(closed)} subshell(s) are energetically CLOSED at "
            f"omega = {cfg.physics.photon_energy_ev:.1f} eV and contribute "
            f"zero: {pretty}  [REVIEW.md A-18]")
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

    shared = {"C_neu": neutral.coeff, "S_ao": S_ao, "dipole": dipole,
              "neutral": neutral}
    results: dict[str, list[spectrum.StateResult]] = {}
    frozen_results: dict[str, list[spectrum.StateResult]] = {}

    log("")
    for channel in cfg.channels:
        log(f"integrating {channel}")
        rows: list[spectrum.StateResult] = []
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

        results[channel] = rows
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

    # ── the frozen-orbital limit, as an INDEPENDENT calculation ────────────
    #
    # Not a by-product of the relaxed states: its own energy list, its own
    # orbital files, its own state ordering, its own DIP ladder.  The two
    # results are reported side by side and never summed -- they are the same
    # spectrum computed under two different orbital assumptions, so adding
    # them would double-count the process.
    frozen_offset = None
    if cfg.physics.include_frozen:
        log("")
        log("pairing frozen-orbital states with their own orbital files")
        frozen_specs = build_state_specs(cfg, log, frozen=True)
        frozen_offset = frozen_reference_offset(cfg)
        if frozen_offset is None:
            log("  the two energy lists are not on a common absolute scale, "
                "so the relaxation energy cannot be reported")
        else:
            log(f"  E_lowest(frozen) - E_lowest(relaxed) = "
                f"{frozen_offset:+.4f} eV")
            if frozen_offset < 0.0:
                log("  [!] the frozen minimum lies BELOW the relaxed one. "
                    "Relaxation is variational and can only lower the "
                    "energy, so either the lists are not comparable or the "
                    "'relaxed' calculation is not converged.")
        log("")
        hole_log: list[tuple[str, int, int, int]] = []
        for channel in cfg.channels:
            log(f"integrating {channel} [frozen limit]")
            frozen_rows: list[spectrum.StateResult] = []
            for spec in frozen_specs[channel]:
                built = frozen_dyson_for_state(spec, cfg, shared, log)
                if built is None:
                    continue
                fobj, fobj_t = built
                # Report which orbitals each state was holed in, with their
                # neutral orbital energies.  No check on the amplitudes can
                # catch a wrong hole -- in the frozen limit det(S_beta) = 1
                # and p_i = p_j = 1 whatever orbitals are named, so every
                # mis-assignment yields a well-formed, plausible number.
                # Printing the depths makes it visible instead: a core hole
                # at -26.4 Ha where -92.5 was intended stands out on sight.
                hi = fobj.meta.get("hole_i")
                hj = fobj.meta.get("hole_j")
                if hi is not None:
                    hole_log.append((channel, spec.index, hi, hj))
                frozen_rows.append(spectrum.integrate_state(
                    fobj, context, spec.dip_ev, channel,
                    n_quad=cfg.physics.n_quad, dyson_triplet=fobj_t,
                    index=spec.index,
                    e_dication_ev=spec.e_dication_ev,
                    e_shifted_ev=spec.e_shifted_ev))
            if frozen_rows:
                frozen_results[channel] = frozen_rows
                log(f"  {len(frozen_rows)} states, sum of intensities "
                    f"{sum(r.intensity for r in frozen_rows):.6e}")

        if hole_log:
            e_orb = getattr(neutral, "energies", None)
            checked = cfg.physics.frozen_expect_holes
            log("")
            log("  frozen hole assignment (neutral MO, 1-based):")
            for channel, index, hi, hj in hole_log:
                mark = " [checked]" if index in checked.get(channel, {}) else ""
                if e_orb is not None and e_orb.size > max(hi, hj):
                    log(f"    {channel:8s} state {index:3d}: "
                        f"core MO {hi + 1:3d} ({e_orb[hi]:9.3f} Ha)  "
                        f"valence MO {hj + 1:3d} ({e_orb[hj]:8.3f} Ha)"
                        f"{mark}")
                else:
                    log(f"    {channel:8s} state {index:3d}: "
                        f"holes MO {hi + 1:3d}, {hj + 1:3d}{mark}")
            n_checked = sum(len(r) for r in checked.values())
            if n_checked:
                log(f"    {n_checked} state(s) cross-checked against "
                    f"frozen_expect_holes; the rest are unverified")

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

    written = write_outputs(cfg, results, frozen_results, log,
                            frozen_offset=frozen_offset)
    log("")
    log(f"done in {time.time() - started:.1f} s")
    return {"results": results, "frozen": frozen_results,
            "frozen_offset": frozen_offset, "written": written}


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
                  log: Callable[[str], None],
                  frozen_offset: float | None = None) -> list[str]:
    """Broaden, then write spectrum.dat, sticks.dat and the LaTeX table."""
    os.makedirs(cfg.output.directory, exist_ok=True)

    # The frozen states have their own positions, and they need not span the
    # same range as the relaxed ones -- a frozen calculation can order its
    # states differently and spread them differently.  Both sets must fit on
    # one grid or the frozen curve would be silently truncated.
    positions = [r.e_shifted_ev for rows in results.values() for r in rows]
    positions += [r.e_shifted_ev for rows in frozen.values() for r in rows]
    lo, hi = cfg.output.grid_bounds(positions)
    n_point = max(int(round((hi - lo) / cfg.output.e_step_ev)) + 1, 2)
    grid = np.linspace(lo, hi, n_point)

    columns = {
        channel: spectrum.broaden(grid, rows, cfg.output.voigt_sigma_ev,
                                  cfg.output.voigt_gamma_ev)
        for channel, rows in results.items()}

    if frozen:
        # The frozen limit is emitted as ONE total column, not per channel.
        # Its singlet and triplet parts are essential to computing it -- the
        # LS amplitudes differ and both are integrated -- but the frozen
        # limit is used as a reference shape, and splitting it invites the
        # reader to compare a frozen channel against a relaxed channel whose
        # state ordering is unrelated.  Summing over channels HERE is the
        # sum over the two spin couplings of one spectrum, which is
        # physical; it is not a sum with the relaxed result, which would
        # double-count the process.
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
    if frozen:
        note += [
            "",
            "frozen_total is the FROZEN-ORBITAL limit: an independent "
            "calculation with",
            "its own energies, orbitals and state ordering.  It is a "
            "reference, NOT a",
            "contribution -- do not add it to the relaxed channels.",
            "Each ladder is referenced to its OWN lowest state, so both "
            "start at",
            "true_dip_ev and their shapes are comparable."]
        note.append(
            "E_lowest(frozen) - E_lowest(relaxed) = "
            f"{frozen_offset:+.4f} eV  [the relaxation energy, removed from "
            f"both ladders]" if frozen_offset is not None else
            "the relaxation energy could not be reported: the two energy "
            "lists are not on a common absolute scale")

    # `total` defaults to the sum over every column, which would fold the
    # frozen-orbital limit into the relaxed result.  The two are the SAME
    # spectrum under two orbital assumptions, so adding them double-counts
    # the process: the total must be the sum over the RELAXED channels only.
    relaxed_total = np.zeros_like(grid)
    for channel in results:
        relaxed_total += columns[channel]

    written: list[str] = []
    path = report.write_spectrum(cfg.out_path(cfg.output.spectrum), grid,
                                columns, settings=settings, note=note,
                                total=relaxed_total)
    written.append(path)
    log(f"wrote {path}  ({n_point} points, "
        f"{len(columns)} channel column(s) + total)")

    flat = [r for channel in cfg.channels for r in results[channel]]
    path = report.write_sticks(cfg.out_path(cfg.output.sticks), flat,
                               settings=settings, note=note)
    written.append(path)
    log(f"wrote {path}  ({len(flat)} states)")

    if frozen:
        # A SEPARATE stick file, never merged into the relaxed one.  The
        # frozen singlet and triplet rows are what the total is built from
        # and are needed to check it, but their state indices refer to the
        # frozen energy list -- index 3 there is not index 3 in the relaxed
        # table -- so interleaving the two would invite exactly the
        # cross-calculation comparison the separate ladders are meant to
        # prevent.
        flat_frozen = [r for channel in cfg.channels
                       for r in frozen.get(channel, ())]
        base, ext = os.path.splitext(cfg.output.sticks)
        path = report.write_sticks(
            cfg.out_path(f"{base}_frozen{ext}"), flat_frozen,
            settings=settings,
            note=list(note) + [
                "",
                "FROZEN-ORBITAL limit, per state.  `index` refers to the "
                "frozen energy",
                "list, which need not order its states as the relaxed "
                "calculation does."])
        written.append(path)
        log(f"wrote {path}  ({len(flat_frozen)} frozen states)")

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

    # ── frozen-orbital inputs ──────────────────────────────────────────────
    #
    # Deliberately in a DIFFERENT state order from the relaxed list, and
    # shifted up in energy.  Both are what a real frozen calculation gives:
    # it converges its states independently, so its state 3 need not be the
    # relaxed state 3, and being unrelaxed it lies above the relaxed set.
    # A demo whose two orderings agreed would pass even if the code paired
    # frozen orbitals against relaxed energies.
    for key in orbital_keys:
        orb_dir = os.path.join(directory, f"frozen_orb_{key}")
        os.makedirs(orb_dir, exist_ok=True)
        for k in range(1, n_state + 1):
            target = os.path.join(orb_dir, f"state_{k}.RasOrb")
            with open(case.inporb_frozen) as src, open(target, "w") as dst:
                dst.write(src.read())
    for channel in channels:
        base = [-99.2 + 0.05 * (k - 1)
                + (0.02 if channel in ("triplet", "jc12") else 0.0)
                for k in range(1, n_state + 1)]
        reordered = [base[i] for i in (1, 0, 3, 2)][:n_state]
        with open(os.path.join(directory,
                               f"e_frozen_{channel}.dat"), "w") as fh:
            fh.write("\n".join(f"{e:.10f}" for e in reordered) + "\n")

    toml = [
        "[physics]",
        "photon_energy_ev = 300.0",
        "true_dip_ev = 120.0",
        "n_neu_occ = 5",
        f'coupling = "{coupling}"',
        "n_quad = 120",
        # The synthetic frozen file puts its two actives first, so the
        # convention for it is [1, 2].  Stated explicitly rather than left to
        # #INDEX so the demo exercises the cross-check that a real run
        # depends on: with both present they must agree.
        "frozen_active_mos = [1, 2]",
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
        "frozen_orbitals = { "
        + ", ".join(f'{c} = "{os.path.join(directory, "frozen_orb_" + c)}"'
                    for c in orbital_keys) + " }",
        "frozen_energies = { "
        + ", ".join(
            f'{c} = "{os.path.join(directory, "e_frozen_" + c + ".dat")}"'
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

