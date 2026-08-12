"""Input-file configuration layer.

The calculation is driven by a single annotated TOML file rather than by
command-line flags; see ``dpi_input.toml`` for a commented example.  This
module maps that file onto validated dataclasses and is the only place
where user-supplied strings are turned into typed values.

Validation is eager and total: every consistency requirement between keys
is checked here, before any orbital file is opened, so that a misconfigured
run fails in under a second with a message naming the offending key rather
than three minutes later inside a quadrature loop.
"""

from __future__ import annotations

import difflib
import math
import os
import tomllib
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from .amplitudes import TermSwitches
from .atomic_sigma import THRESHOLD_MODELS
from .constants import ConfigError

Coupling = Literal["ls", "jj"]
EnergyUnit = Literal["au", "ev"]


def build_term_switches(table: dict[str, Any]) -> TermSwitches:
    """Turn the ``[terms]`` table into the amplitude layer's TermSwitches.

    ``TermSwitches`` itself is defined once, in :mod:`dpi.amplitudes`, next
    to the expressions whose terms it selects; this function is only the
    input-file adapter.  It accepts ``interference_sign`` as a friendlier
    spelling of ``dir_ind_sign`` and enforces the two cross-key
    requirements that the amplitude layer cannot see on its own.
    """
    table = dict(table)
    if "interference_sign" in table:
        if "dir_ind_sign" in table:
            raise ConfigError(
                "[terms] give either interference_sign or dir_ind_sign, "
                "not both; they name the same quantity.")
        table["dir_ind_sign"] = float(table.pop("interference_sign"))
    if table.get("dir_ind_sign") not in (None, 1.0, -1.0, 1, -1):
        raise ConfigError(
            f"[terms] interference_sign must be -1 or +1, "
            f"got {table['dir_ind_sign']!r}")
    if "dir_ind_sign" in table:
        table["dir_ind_sign"] = float(table["dir_ind_sign"])

    switches = TermSwitches(**table)

    if switches.dir_ind_interference and not (switches.direct
                                              and switches.indirect):
        raise ConfigError(
            "[terms] dir_ind_interference requires both direct and indirect "
            "to be enabled: it is their cross term.")
    if switches.c_cross and not switches.aa_bb:
        raise ConfigError(
            "[terms] c_cross is the interference between the one- and "
            "two-electron Dyson amplitudes and is meaningless with aa_bb "
            "disabled; set aa_bb = true or c_cross = false.")
    return switches


@dataclass(frozen=True)
class PhysicsConfig:
    """Photon energy, the double-ionization threshold, and model options."""

    photon_energy_ev: float
    true_dip_ev: float
    """Double ionization potential of the LOWEST dication state, in eV.

    This sets the physics: E_excess(f) = photon_energy - DIP_f with
    DIP_f = true_dip_ev + (E_f - E_lowest).  It is emphatically not the
    cosmetic x-axis offset ``display_onset_ev``; conflating the two was a
    live trap in the previous implementation.
    """

    n_neu_occ: int
    coupling: Coupling = "ls"
    n_quad: int = 200
    spin_degeneracy_factor: float = 1.0
    """Diagnostic only; leave at 1.0.

    The triplet amplitude already sums the three M_S^dic substates
    coherently with their Clebsch-Gordan weights, so an external (2S+1)
    would double-count.  Exposed because the notes' near-unity
    singlet:triplet branching claim appears to require it; see
    REVIEW.md [P-2].
    """

    one_centre_dipole: bool = True
    """Restrict the bound-to-bound dipole to one-centre origin-shifted blocks.

    Required for consistency with the Gelius reduction (notes Eq. 146);
    the full molecular dipole injects a large R_A <chi|chi> positional
    term with no counterpart in the atomic cross sections.
    """

    omega_over_omega_eff: bool = False
    """Apply the residual ``omega / (eps1 + I_mu)`` weight to each AO.

    Substituting a tabulated ``sigma^AO`` for the computed dipole matrix
    element imports sigma's own normalisation.  Inverting its definition
    (notes Eq. 93, via Eq. (99)) cancels the sDCS
    prefactor's ``k_1`` and its ``1/3`` polarisation average exactly -- the
    ``1/3`` of Eq. (94) appears once in each and no more
    (Remark 3) -- and the ``k_2`` of the shake-off block is
    electron 2's phase-space factor -- so ``4 pi^2 omega / 3c * k_1 k_2``
    cancels *except* for one ratio: the prefactor carries the molecular
    ``omega`` while sigma carries the per-AO ``omega_eff = eps1 + I_mu``.

    Off by default because every spectrum computed before this option
    existed omitted it, and the published SF6 band intensities were
    validated in that convention.  Switching it on is not a rescaling: the
    weight varies across the energy-sharing integral (1.01-1.11 for an F1s
    AO but 8.8-42.8 for F2p at omega = 770 eV), so it changes the sharing
    profile and the core-vs-valence balance.  See REVIEW.md [A-8] and
    NORMALISATION.md.
    """

    include_frozen: bool = False
    #: 1-based positions of the active (C and V) orbitals in a
    #: frozen-orbital ``.RasOrb``, i.e. the calculation's own convention:
    #: ``[34, 35]`` for the SF6 S1s and F1s edges, ``[32, 33, 34, 35]`` for
    #: S2p, where all three degenerate cartesian S 2p projections are
    #: active.  A frozen file carries the NEUTRAL occupancy in ``#OCC`` and
    #: so cannot be read by occupation; when this is given it is the
    #: authority and ``#INDEX`` is used as a cross-check, with a
    #: disagreement raising rather than silently trusting either.  Leave
    #: empty to rely on ``#INDEX`` alone.
    frozen_active_mos: tuple[int, ...] = ()
    #: Optional per-state cross-check on the frozen hole assignment, keyed by
    #: channel then by 1-based state index:
    #:
    #:     [physics.frozen_expect_holes.singlet]
    #:     1 = [1, 33]
    #:
    #: The values are **neutral** MO numbers (1-based) -- not positions in
    #: the frozen file, which differ: on the shipped SF6 S1s file the holes
    #: are neutral MOs (1, 33) while the file positions are (34, 35).
    #:
    #: This never feeds the assignment.  The holes are always resolved from
    #: the orbital file, and a listed state whose resolved holes disagree
    #: raises.  Supplying holes directly instead would skip reading the
    #: coefficients, and with them the signed-permutation check that stops a
    #: relaxed file being computed as frozen; a wrong hole is otherwise
    #: silent, since det(S_beta) = 1 in the frozen limit whatever holes are
    #: named.  States not listed are simply not checked.
    frozen_expect_holes: dict[str, dict[int, tuple[int, ...]]] = field(
        default_factory=dict)

    high_energy_exponent: dict[str, float] = field(default_factory=dict)
    """Per-subshell power-law exponent above the tabulated range.

    Keys look like ``"S_1s"``.  Yeh-Lindau stops at 1500 eV while the S1s
    edge is probed near 2610 eV, so that subshell rests on an
    extrapolation; -3.5 is the hydrogenic value.
    """

    anchor_delta_ev: float = 0.1
    """Offset of the second artificial knot in the ``"anchored"`` model, eV.

    Only used when ``threshold_model = "anchored"``.  It sets the width over
    which ``sigma`` climbs from the imposed zero at ``I_mu`` to the value the
    data extrapolates to, so it is physical content rather than a numerical
    tolerance: as it goes to zero the model becomes a step, and as it
    approaches the gap width it becomes a smooth rise across the whole gap.
    The dependence is monotone and mild -- at the F1s edge the integrated
    intensity moves from 1.02763 at 0.001 eV to 1.02649 at 0.1 (0.11% across
    two decades) and 1.01688 at 1.0 eV -- so 0.1 is where a further tenfold
    reduction changes the answer by about 0.1%.  REVIEW.md [A-17].
    """

    threshold_model: str = "linear"
    """Shape of ``sigma^AO`` between ``I_mu`` and the first tabulated point.

    ``"linear"`` (default), ``"flat"``, ``"coulomb"``, ``"wigner"``,
    ``"extrapolate"`` or ``"anchored"``.  This
    region is not tabulated data, and it is entered unavoidably whenever
    ``omega_over_omega_eff`` is on, because ``omega_eff -> I_mu`` as the
    sharing ``eps1 -> 0``: at the F1s edge it supplies 18% of the weighted
    singlet intensity against 14% unweighted (REVIEW.md [A-12]).

    The four are not equally defensible and the default is the *least* so:

    * ``"linear"`` -- ``sigma_lo * (hv - I)/(hv_lo - I)``.  Vanishes at
      threshold with slope 1, which no final-state potential produces.  Kept
      as the default only because every spectrum computed before this option
      existed used it.
    * ``"coulomb"`` -- the exact hydrogenic (Stobbe) energy dependence,
      rescaled to the tabulated magnitude at ``hv_lo``.  **This is the
      physically correct family for a tabulated atomic cross section**: the
      residual ion is charged, the Coulomb penetration factor cancels the
      phase-space suppression, and ``sigma`` approaches a finite constant at
      threshold rather than vanishing.  Consistent with the tabulated data,
      whose first-two-point log-log slopes are flat (-0.02 for F1s, +0.00 for
      S2s) rather than rising.
    * ``"flat"`` -- ``sigma_lo`` held constant.  Crude, but it is the correct
      *limit* of ``"coulomb"``, so the pair brackets the answer.
    * ``"wigner"`` -- ``(hv - I)^(l' + 1/2)``, the threshold law for a
      **short-range** final state.  Wrong for a tabulated atomic ``sigma``,
      but right for the plane-wave continuum this model uses everywhere else,
      so the mismatch is a real internal tension rather than an error to
      pick a side on.  Offered to be quantified, not to be trusted.
    * ``"extrapolate"`` -- no physics model at all: continue the log-log
      spline itself backwards into the gap.  Assumes only the data's own
      local shape, which makes it an *independent* check on ``"coulomb"``
      rather than a rival: where they agree the near-threshold value is
      corroborated by two unrelated arguments, and where they diverge the
      group is flagged by ``unsafe_extrapolations``.  On the shipped table
      they agree to 0.06% over the subshells carrying 94.6% of an F1s-edge
      spectrum.  Unbounded when abused, so never use it alone over a long
      lever arm.

    * ``"anchored"`` -- the two-artificial-point construction.  Drop every
      point at or below ``I_mu``; insert a knot at ``I_mu`` with
      ``sigma = 0`` exactly, and a second at ``I_mu + anchor_delta_ev``
      whose value comes from extrapolating the cleaned data; then
      interpolate the union with a shape-preserving PCHIP in *linear*
      ``sigma`` (``log 0`` is undefined, so this model cannot use the
      log-log variable the others share).  It is the only model that
      imposes ``sigma(I_mu) = 0`` while inheriting the *slope* leaving
      threshold from the data rather than assuming one.

    Sweep it (``--set physics.threshold_model=coulomb``) and quote the spread
    as a systematic on the band intensities.  With the flagged groups left on
    ``"linear"``, the three defensible models ``flat``/``coulomb``/
    ``extrapolate`` span only 0.087% (F1s) and 0.047% (S1s); the full
    five-model spread is 3.3% / 1.8%, but that is dominated by ``linear`` and
    ``wigner``, both of which are known to be wrong for a Coulombic threshold.
    The systematic to quote is the +2.7% / +1.4% shift of the defensible
    cluster away from ``linear``.
    """

    threshold_override: dict[str, float] = field(default_factory=dict)
    """Per-subshell ionization threshold ``I_mu`` in eV, overriding the table.

    Keys look like ``"F_2p"``.  By default ``I_mu`` is the *free-atom*
    threshold shipped with the Yeh-Lindau entry -- the choice consistent
    with substituting a free-atom ``sigma`` -- and it is not derived from
    the user's MOLCAS files at all.  That is a modelling assumption, not a
    result, and it is worth being able to test: ``I_mu`` sets both where the
    tabulated curve is read (``sigma_mu(eps1 + I_mu)``) and the denominator
    of the ``omega_over_omega_eff`` weight, so an error in it does not
    cancel between the two.  Measured sensitivity at the F1s edge is about
    2% per eV near zero.

    The natural use is to substitute molecular binding energies for the
    valence subshells, whose free-atom values shift most on bonding, while
    leaving the core alone.  Overriding does *not* move the tabulated
    curve's own onset: ``sigma^AO`` stays a free-atom property.  See
    REVIEW.md [A-13].
    """

    def __post_init__(self) -> None:
        if self.photon_energy_ev <= 0:
            raise ConfigError("[physics] photon_energy_ev must be positive.")
        if self.true_dip_ev <= 0:
            raise ConfigError(
                "[physics] true_dip_ev must be positive: it is the double "
                "ionization potential of the lowest dication state in eV "
                "(e.g. 2511 for the SF6 S1s edge), not an offset.")
        if self.photon_energy_ev <= self.true_dip_ev:
            raise ConfigError(
                f"[physics] photon_energy_ev ({self.photon_energy_ev} eV) "
                f"must exceed true_dip_ev ({self.true_dip_ev} eV): otherwise "
                f"E_excess <= 0 and no channel is open.")
        if self.frozen_active_mos:
            mos = self.frozen_active_mos
            if len(set(mos)) != len(mos):
                raise ConfigError(
                    f"[physics] frozen_active_mos {mos} repeats an orbital; "
                    f"each active MO must appear once")
            if any(m < 1 for m in mos):
                raise ConfigError(
                    f"[physics] frozen_active_mos {mos} contains a "
                    f"non-positive index; positions are 1-based as written "
                    f"in the RasOrb file")
            if len(mos) < 2:
                raise ConfigError(
                    f"[physics] frozen_active_mos {mos} lists {len(mos)} "
                    f"orbital(s); a core-valence dication needs at least 2 "
                    f"(one core, one valence)")
        for channel, rows in self.frozen_expect_holes.items():
            for index, holes in rows.items():
                if index < 1:
                    raise ConfigError(
                        f"[physics.frozen_expect_holes.{channel}] state "
                        f"index {index} is not positive; state indices are "
                        f"1-based, matching the energy-list line numbers")
                if len(holes) != 2:
                    raise ConfigError(
                        f"[physics.frozen_expect_holes.{channel}] state "
                        f"{index} lists {len(holes)} orbital(s) {holes}; a "
                        f"core-valence hole pair is exactly 2 neutral MO "
                        f"numbers")
                if len(set(holes)) != 2:
                    raise ConfigError(
                        f"[physics.frozen_expect_holes.{channel}] state "
                        f"{index} repeats orbital {holes[0]}; the two holes "
                        f"must be different orbitals")
                if any(h < 1 for h in holes):
                    raise ConfigError(
                        f"[physics.frozen_expect_holes.{channel}] state "
                        f"{index} contains a non-positive orbital in "
                        f"{holes}; these are 1-based NEUTRAL MO numbers")
                if any(h > self.n_neu_occ for h in holes):
                    raise ConfigError(
                        f"[physics.frozen_expect_holes.{channel}] state "
                        f"{index} names orbital(s) above n_neu_occ="
                        f"{self.n_neu_occ} in {holes}; a hole must be in an "
                        f"occupied neutral orbital.  Note these are NEUTRAL "
                        f"MO numbers, not positions in the frozen file")
        if self.frozen_expect_holes and not self.include_frozen:
            raise ConfigError(
                "[physics] frozen_expect_holes is set but include_frozen is "
                "false, so nothing would be checked.  Enable the frozen "
                "limit or remove the expectation table.")
        if self.coupling not in ("ls", "jj"):
            raise ConfigError(
                f"[physics] coupling must be 'ls' (F1s, S1s edges) or 'jj' "
                f"(S2p edge, spin-orbit split core hole), "
                f"got {self.coupling!r}")
        if not math.isfinite(self.anchor_delta_ev) or self.anchor_delta_ev <= 0.0:
            raise ConfigError(
                f"[physics] anchor_delta_ev must be positive and finite, got "
                f"{self.anchor_delta_ev}. It is the offset of the second "
                f"artificial knot above the threshold in the 'anchored' model."
            )
        if self.anchor_delta_ev > 1.0 and self.threshold_model == "anchored":
            raise ConfigError(
                f"[physics] anchor_delta_ev = {self.anchor_delta_ev} eV "
                f"exceeds 1 eV, a sizeable fraction of the narrower gaps "
                f"(S_2s's is 0.8 eV), so the second artificial knot would "
                f"land past real data. Keep it <= 1.0; the integral is flat "
                f"below ~0.1 eV (REVIEW.md [A-17])."
            )
        if self.threshold_model not in THRESHOLD_MODELS:
            raise ConfigError(
                f"[physics] threshold_model must be one of "
                f"{list(THRESHOLD_MODELS)}, got {self.threshold_model!r}. "
                f"'coulomb' is the physically correct family for a tabulated "
                f"atomic cross section; 'linear' is the historical default.")
        if self.n_quad < 20:
            raise ConfigError(
                f"[physics] n_quad = {self.n_quad} is too small for the "
                f"energy-sharing integral; ~200 is the converged value.")
        if self.n_neu_occ <= 0:
            raise ConfigError(
                "[physics] n_neu_occ must be the number of doubly occupied "
                "spatial MOs of the neutral molecule (35 for SF6).")


@dataclass(frozen=True)
class InputPaths:
    """Where the OpenMolcas output lives."""

    neutral_orbitals: str
    overlap: str
    h5file: str

    root: str = ""
    """Directory prefixed onto every relative path in this section.

    TOML has no expressions, so ``a = base + "x"`` is a syntax error and
    there is no way to factor out a common directory inside the file itself.
    This key does that job: set ``root`` once and write the rest of the
    section as bare filenames.  Absolute paths are left untouched, so a
    single entry can still point outside the run directory.

    The values stored on this dataclass are already joined -- ``root`` is
    applied when the input file is parsed, so every consumer sees a complete
    path and no module needs to know this key exists.
    """

    dication_orbitals: dict[str, str] = field(default_factory=dict)
    """Channel name -> directory or glob of dication RasOrb files."""

    energies: dict[str, str] = field(default_factory=dict)
    """Channel name -> file of dication total energies, one per line."""

    frozen_energies: dict[str, str] = field(default_factory=dict)
    frozen_orbitals: dict[str, str] = field(default_factory=dict)
    experiment: str | None = None
    cache_dir: str = "dyson_cache"

    def check_exists(self) -> None:
        """Verify every referenced path is present before the run starts."""
        missing: list[str] = []
        for key in ("neutral_orbitals", "overlap", "h5file"):
            if not os.path.isfile(getattr(self, key)):
                missing.append(f"[paths] {key} = {getattr(self, key)!r}")
        for label, mapping in (("energies", self.energies),
                               ("frozen_energies", self.frozen_energies)):
            for channel, path in mapping.items():
                if not os.path.isfile(path):
                    missing.append(f"[paths.{label}] {channel} = {path!r}")
        for label, mapping in (("dication_orbitals", self.dication_orbitals),
                               ("frozen_orbitals", self.frozen_orbitals)):
            for channel, path in mapping.items():
                if not (os.path.isdir(path) or "*" in path or "?" in path):
                    missing.append(f"[paths.{label}] {channel} = {path!r}")
        if self.experiment is not None and not os.path.isfile(self.experiment):
            missing.append(f"[paths] experiment = {self.experiment!r}")
        if missing:
            raise ConfigError(
                "input files not found:\n  " + "\n  ".join(missing))


@dataclass(frozen=True)
class OutputConfig:
    """Filenames and presentation settings for the emitted text files."""

    directory: str = "."
    spectrum: str = "spectrum.dat"
    sticks: str = "sticks.dat"
    latex_table: str | None = None
    energies_unit: EnergyUnit = "au"

    display_onset_ev: float = 0.0
    """Cosmetic x-axis offset: the lowest state is placed here.

    Affects the ``E_shifted_eV`` column and nothing else.  The physics uses
    ``true_dip_ev``.
    """

    e_min_ev: float | None = None
    e_max_ev: float | None = None
    e_step_ev: float = 0.02
    voigt_sigma_ev: float = 0.48
    voigt_gamma_ev: float = 0.23
    label: str = "CV states"
    max_table_ev: float = 40.0

    def __post_init__(self) -> None:
        if self.energies_unit not in ("au", "ev"):
            raise ConfigError(
                f"[output] energies_unit must be 'au' or 'ev', "
                f"got {self.energies_unit!r}")
        if self.e_step_ev <= 0:
            raise ConfigError("[output] e_step_ev must be positive.")
        if self.voigt_sigma_ev < 0 or self.voigt_gamma_ev < 0:
            raise ConfigError("[output] Voigt widths must be non-negative.")
        if self.voigt_sigma_ev == 0 and self.voigt_gamma_ev == 0:
            raise ConfigError(
                "[output] both Voigt widths are zero; the broadened spectrum "
                "would be a sum of delta functions.  Set voigt_sigma_ev "
                "(resolution and vibrational envelope) and/or voigt_gamma_ev "
                "(core-hole lifetime).")
        if (self.e_min_ev is not None and self.e_max_ev is not None
                and self.e_min_ev >= self.e_max_ev):
            raise ConfigError(
                f"[output] e_min_ev ({self.e_min_ev}) must be below "
                f"e_max_ev ({self.e_max_ev}).")

    def grid_bounds(self, e_states_ev: Sequence[float]) -> tuple[float, float]:
        """Spectrum grid limits, defaulting to the state range plus margin."""
        if len(e_states_ev) == 0:
            raise ConfigError("no states available to define a spectrum grid.")
        margin = 6.0 * max(self.voigt_sigma_ev, self.voigt_gamma_ev, 0.1)
        lo = (self.e_min_ev if self.e_min_ev is not None
              else min(e_states_ev) - margin)
        hi = (self.e_max_ev if self.e_max_ev is not None
              else max(e_states_ev) + margin)
        return float(lo), float(hi)


@dataclass(frozen=True)
class Config:
    """A complete, validated specification of one calculation."""

    physics: PhysicsConfig
    paths: InputPaths
    output: OutputConfig
    terms: TermSwitches
    source: str = "<none>"

    @property
    def channels(self) -> tuple[str, ...]:
        """The two spin or spin-orbit channels, in output-column order."""
        return (("singlet", "triplet") if self.physics.coupling == "ls"
                else ("jc32", "jc12"))

    @property
    def orbital_channels(self) -> tuple[str, ...]:
        """Keys of the dication ORBITAL sets, which are always multiplicities.

        The distinction matters only in jj coupling.  A ``j_C`` state is not
        something MOLCAS optimises orbitals for: the CV states at the S 2p
        edge are obtained by recombining the separately converged
        ``S_dic = 0`` and ``S_dic = 1`` OSRHF solutions.  So the RasOrb files
        are always singlet and triplet sets, while the *energies* are
        genuinely resolved into ``j_C = 3/2`` and ``1/2``.  Keeping the two
        keyed differently reflects that asymmetry instead of pretending a
        ``jc32`` orbital set exists.
        """
        return ("singlet", "triplet")

    def out_path(self, name: str) -> str:
        return os.path.join(self.output.directory, name)

    def summary_lines(self) -> list[str]:
        """One-line-per-setting provenance block for output file headers."""
        phys, out = self.physics, self.output
        lines = [
            f"input file        : {self.source}",
            f"coupling          : {phys.coupling}"
            f"  (channels: {', '.join(self.channels)})",
            f"photon energy     : {phys.photon_energy_ev:.4f} eV",
            f"true DIP (lowest) : {phys.true_dip_ev:.4f} eV",
            f"E_excess (lowest) : "
            f"{phys.photon_energy_ev - phys.true_dip_ev:.4f} eV",
            f"n_neu_occ         : {phys.n_neu_occ}",
            f"quadrature points : {phys.n_quad}",
            f"terms included    : {', '.join(self.terms.active())}",
            f"one-centre dipole : {phys.one_centre_dipole}",
            f"omega/omega_eff   : {phys.omega_over_omega_eff}",
            f"frozen reference  : {phys.include_frozen}"
            + (f"  [active MOs {phys.frozen_active_mos}]"
               if phys.include_frozen and phys.frozen_active_mos else "")
            + (f"  [{sum(len(r) for r in phys.frozen_expect_holes.values())}"
               f" state(s) hole-checked]"
               if phys.include_frozen and phys.frozen_expect_holes else "")
            + ("  [own energies + own orbitals; reported separately, "
               "never summed with the relaxed result]"
               if phys.include_frozen else ""),
            f"display onset     : {out.display_onset_ev:.4f} eV "
            f"(cosmetic x-axis shift only)",
            f"Voigt sigma/gamma : {out.voigt_sigma_ev:.4f} / "
            f"{out.voigt_gamma_ev:.4f} eV",
        ]
        if phys.spin_degeneracy_factor != 1.0:
            lines.append(
                f"spin degeneracy   : {phys.spin_degeneracy_factor}"
                f"  [DIAGNOSTIC OVERRIDE -- see REVIEW.md P-2]")
        if phys.high_energy_exponent:
            pretty = ", ".join(
                f"{k}:{v:+.2f}"
                for k, v in sorted(phys.high_energy_exponent.items()))
            lines.append(f"sigma extrapolation: {pretty}")
        if phys.threshold_model != "linear":
            lines.append(
                f"near-threshold sigma model: {phys.threshold_model}"
                + (f" (delta = {phys.anchor_delta_ev} eV)"
                   if phys.threshold_model == "anchored" else "")
                + "  [non-default -- REVIEW.md A-14]")
        if phys.threshold_override:
            pretty = ", ".join(
                f"{k}:{v:.2f}eV"
                for k, v in sorted(phys.threshold_override.items()))
            lines.append(
                f"I_mu override: {pretty}"
                f"  [free-atom thresholds replaced -- REVIEW.md A-13]")
        return lines


_SECTIONS = {"physics", "paths", "output", "terms"}

# Derived from the dataclass rather than hand-listed, so adding a field to
# InputPaths cannot leave the accepted-key set behind and reject a key the
# code itself supports.
_PATH_KEYS = set(InputPaths.__dataclass_fields__)


def _require(table: dict[str, Any], key: str, section: str) -> Any:
    if key not in table:
        raise ConfigError(f"[{section}] is missing the required key {key!r}.")
    return table[key]


def _expect_holes_table(raw: Any) -> dict[str, dict[int, tuple[int, ...]]]:
    """Parse ``[physics.frozen_expect_holes]`` with legible errors.

    The shape is channel -> state index -> two neutral MO numbers::

        [physics.frozen_expect_holes.singlet]
        1 = [1, 33]

    Written by hand and easy to get slightly wrong, so every level reports
    what it found rather than letting a bare ``dict()`` raise
    ``dictionary update sequence element #0 has length 1``, which says
    nothing about which channel or state is at fault.
    """
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"[physics] frozen_expect_holes must be a table of "
            f"channel -> state -> [core_mo, valence_mo], got "
            f"{type(raw).__name__}")
    out: dict[str, dict[int, tuple[int, ...]]] = {}
    for channel, rows in raw.items():
        if not isinstance(rows, dict):
            raise ConfigError(
                f"[physics.frozen_expect_holes] channel {channel!r} must map "
                f"state indices to hole pairs, e.g. "
                f"`[physics.frozen_expect_holes.{channel}]` with `1 = "
                f"[1, 33]`; got {type(rows).__name__}")
        parsed: dict[int, tuple[int, ...]] = {}
        for key, vals in rows.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                raise ConfigError(
                    f"[physics.frozen_expect_holes.{channel}] key {key!r} is "
                    f"not a state index; keys are the 1-based line numbers "
                    f"of the frozen energy list") from None
            if isinstance(vals, (str, bytes)) or not isinstance(
                    vals, (list, tuple)):
                raise ConfigError(
                    f"[physics.frozen_expect_holes.{channel}] state {index} "
                    f"must be a list of two NEUTRAL MO numbers, e.g. "
                    f"[1, 33]; got {vals!r}")
            try:
                parsed[index] = tuple(int(v) for v in vals)
            except (TypeError, ValueError):
                raise ConfigError(
                    f"[physics.frozen_expect_holes.{channel}] state {index} "
                    f"contains a non-integer orbital in {list(vals)!r}") \
                    from None
        out[str(channel)] = parsed
    return out


def _reject_unknown(table: dict[str, Any], allowed: set[str],
                    section: str) -> None:
    """Fail on unrecognised keys rather than ignoring a typo silently."""
    unknown = set(table) - allowed
    if not unknown:
        return
    hints = []
    for key in sorted(unknown):
        close = difflib.get_close_matches(key, sorted(allowed), n=1)
        hints.append(f"{key!r}"
                     + (f" (did you mean {close[0]!r}?)" if close else ""))
    raise ConfigError(
        f"[{section}] unrecognised key(s): {', '.join(hints)}.\n"
        f"  valid keys: {', '.join(sorted(allowed))}")


def _channel_map(table: dict[str, Any], key: str, channels: Sequence[str],
                 section: str, required: bool) -> dict[str, str]:
    """Read a per-channel sub-table, checking it covers exactly the channels."""
    raw = table.get(key)
    if raw is None:
        if required:
            raise ConfigError(
                f"[{section}] is missing {key!r}, which must be a table with "
                f"one entry per channel ({', '.join(channels)}), e.g.\n"
                f"  {key} = {{ {channels[0]} = \"...\", "
                f"{channels[1]} = \"...\" }}")
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"[{section}] {key!r} must be a table keyed by channel "
            f"({', '.join(channels)}), not a bare {type(raw).__name__}.")
    missing = set(channels) - set(raw)
    extra = set(raw) - set(channels)
    if missing:
        raise ConfigError(
            f"[{section}] {key!r} has no entry for "
            f"{', '.join(sorted(missing))}; this coupling mode requires "
            f"{', '.join(channels)}.")
    if extra:
        raise ConfigError(
            f"[{section}] {key!r} has entries for {', '.join(sorted(extra))}, "
            f"which are not channels of this coupling mode "
            f"({', '.join(channels)}).")
    return {c: str(raw[c]) for c in channels}


def _field_names(cls: type) -> set[str]:
    return set(cls.__dataclass_fields__)


def load_config(path: str,
                overrides: dict[str, Any] | None = None) -> Config:
    """Read and validate a TOML input file.

    Parameters
    ----------
    path
        Path to the input file.
    overrides
        Optional flat mapping of ``"section.key"`` to value, applied after
        parsing and before validation, so a driver can expose a handful of
        command-line overrides without duplicating the schema.

    Returns
    -------
    Config
        Fully validated; every referenced input file is confirmed present.
    """
    if not os.path.isfile(path):
        raise ConfigError(f"input file not found: {path}")
    try:
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc

    _reject_unknown(raw, _SECTIONS, "top level")
    for section in ("physics", "paths"):
        if section not in raw:
            raise ConfigError(f"input file {path} has no [{section}] section.")

    raw = _apply_overrides(raw, overrides or {})

    phys_t = dict(raw.get("physics", {}))
    paths_t = dict(raw.get("paths", {}))
    out_t = dict(raw.get("output", {}))
    terms_t = dict(raw.get("terms", {}))

    _reject_unknown(phys_t, _field_names(PhysicsConfig), "physics")
    _reject_unknown(paths_t, _PATH_KEYS, "paths")
    _reject_unknown(out_t, _field_names(OutputConfig), "output")
    _reject_unknown(terms_t,
                    _field_names(TermSwitches) | {"interference_sign"},
                    "terms")

    coupling = phys_t.get("coupling", "ls")
    if coupling not in ("ls", "jj"):
        raise ConfigError(
            f"[physics] coupling must be 'ls' or 'jj', got {coupling!r}")
    channels = ("singlet", "triplet") if coupling == "ls" else ("jc32", "jc12")
    # Orbital files are keyed by multiplicity whatever the coupling; only the
    # energies carry the jj labels (Config.orbital_channels explains why).
    orbital_keys = ("singlet", "triplet")

    physics = PhysicsConfig(
        photon_energy_ev=float(_require(phys_t, "photon_energy_ev", "physics")),
        true_dip_ev=float(_require(phys_t, "true_dip_ev", "physics")),
        n_neu_occ=int(_require(phys_t, "n_neu_occ", "physics")),
        coupling=coupling,
        n_quad=int(phys_t.get("n_quad", 200)),
        spin_degeneracy_factor=float(
            phys_t.get("spin_degeneracy_factor", 1.0)),
        one_centre_dipole=bool(phys_t.get("one_centre_dipole", True)),
        omega_over_omega_eff=bool(
            phys_t.get("omega_over_omega_eff", False)),
        include_frozen=bool(phys_t.get("include_frozen", False)),
        frozen_active_mos=tuple(
            int(v) for v in phys_t.get("frozen_active_mos", ())),
        frozen_expect_holes=_expect_holes_table(
            phys_t.get("frozen_expect_holes", {})),
        high_energy_exponent={
            str(k): float(v)
            for k, v in dict(phys_t.get("high_energy_exponent", {})).items()},
        threshold_model=str(phys_t.get("threshold_model", "linear")),
        anchor_delta_ev=float(phys_t.get("anchor_delta_ev", 0.1)),
        threshold_override={
            str(k): float(v)
            for k, v in dict(phys_t.get("threshold_override", {})).items()},
    )

    # `root` stands in for the string concatenation TOML cannot express.
    # Joining happens here, once, so every downstream module receives a
    # complete path and none of them needs to know the key exists.
    root = str(paths_t.get("root", ""))

    def _join(value: str) -> str:
        """Prefix `root` onto a relative path; leave absolute ones alone."""
        text = str(value)
        if not root or os.path.isabs(text):
            return text
        return os.path.join(root, text)

    def _join_map(mapping: dict[str, str]) -> dict[str, str]:
        return {channel: _join(path) for channel, path in mapping.items()}

    paths = InputPaths(
        root=root,
        neutral_orbitals=_join(_require(paths_t, "neutral_orbitals", "paths")),
        overlap=_join(_require(paths_t, "overlap", "paths")),
        h5file=_join(_require(paths_t, "h5file", "paths")),
        # Orbital sets are keyed by MULTIPLICITY in both coupling modes; see
        # Config.orbital_channels.  A jj input therefore writes
        #   dication_orbitals = { singlet = "...", triplet = "..." }
        # while its `energies` stay keyed by jc32/jc12.
        dication_orbitals=_join_map(
            _channel_map(paths_t, "dication_orbitals", orbital_keys, "paths",
                         required=True)),
        energies=_join_map(
            _channel_map(paths_t, "energies", channels, "paths",
                         required=True)),
        frozen_energies=_join_map(
            _channel_map(paths_t, "frozen_energies", channels, "paths",
                         required=physics.include_frozen)),
        # Required whenever the frozen limit is on: the frozen .RasOrb files
        # are the ONLY record of which orbitals carry the holes in that
        # calculation (its #OCC is the neutral occupancy) and of its own
        # state ordering, which need not match the relaxed one.
        frozen_orbitals=_join_map(
            _channel_map(paths_t, "frozen_orbitals", orbital_keys, "paths",
                         required=physics.include_frozen)),
        experiment=(_join(paths_t["experiment"])
                    if paths_t.get("experiment") else None),
        # cache_dir is an OUTPUT directory, so it is deliberately not
        # prefixed by root: root points at where the MOLCAS files were read
        # from, which is often read-only or shared between runs.
        cache_dir=str(paths_t.get("cache_dir", "dyson_cache")),
    )

    output = OutputConfig(**out_t)
    terms = build_term_switches(terms_t)

    if physics.omega_over_omega_eff and terms.c_cross:
        raise ConfigError(
            "[physics] omega_over_omega_eff = true is not consistent with "
            "[terms] c_cross = true. The C_ij term contracts sqrt(sigma_mu) "
            "against the antisymmetric two-electron Dyson amplitude, so it "
            "carries the omega/omega_eff weight only to the half power while "
            "every other term carries it to the first; on the real SF6 files "
            "this drives the triplet amplitude NEGATIVE (A_f < 0, which is "
            "impossible for a squared amplitude). Set c_cross = false to use "
            "the weight, or omega_over_omega_eff = false to keep C_ij. "
            "See REVIEW.md [A-8] and [A-5].")
    if physics.spin_degeneracy_factor != 1.0 and physics.coupling == "jj":
        raise ConfigError(
            "[physics] spin_degeneracy_factor is an LS-coupling diagnostic; "
            "in jj coupling the 2:1 branching follows from the substate "
            "enumeration and must not be rescaled.")

    paths.check_exists()
    return Config(physics=physics, paths=paths, output=output, terms=terms,
                  source=os.path.abspath(path))


# Keys whose value is itself a table, so that a three-level override
# "section.key.subkey" addresses an entry inside it rather than creating a
# literal dotted key (which would then be rejected as unrecognised).
_NESTED_KEYS = {
    "physics": {"high_energy_exponent", "threshold_override",
                "frozen_expect_holes"},
    "paths": {"dication_orbitals", "energies", "frozen_energies",
              "frozen_orbitals"},
}


def _apply_overrides(raw: dict[str, Any],
                     overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply ``"section.key"`` or ``"section.key.subkey"`` overrides.

    The two-level form replaces a scalar setting.  The three-level form
    addresses one entry of a table-valued key -- ``high_energy_exponent`` and
    the per-channel path maps -- so that

        --set physics.high_energy_exponent.S_1s=-3.0

    sweeps a single subshell exponent while leaving the others in place.
    Without this, that key would be stored literally as
    ``"high_energy_exponent.S_1s"`` and then rejected by the unknown-key
    check, which is what happened before this branch existed.
    """
    out: dict[str, Any] = {k: (dict(v) if isinstance(v, dict) else v)
                           for k, v in raw.items()}
    for dotted, value in overrides.items():
        if value is None:
            continue
        parts = dotted.split(".")
        if len(parts) < 2:
            raise ConfigError(
                f"override {dotted!r} must have the form 'section.key' or "
                f"'section.key.subkey'.")
        section, key = parts[0], parts[1]
        if section not in _SECTIONS:
            raise ConfigError(
                f"override {dotted!r} names unknown section {section!r}.")
        table = out.setdefault(section, {})
        if len(parts) == 2:
            table[key] = value
            continue
        if len(parts) > 3:
            raise ConfigError(
                f"override {dotted!r} is nested too deeply; at most "
                f"'section.key.subkey' is supported.")
        if key not in _NESTED_KEYS.get(section, frozenset()):
            raise ConfigError(
                f"override {dotted!r} addresses a sub-key of "
                f"[{section}] {key}, but that key is not table-valued. "
                f"Table-valued keys in [{section}]: "
                f"{sorted(_NESTED_KEYS.get(section, ())) or 'none'}.")
        inner = table.get(key)
        table[key] = ({**inner} if isinstance(inner, dict) else {})
        table[key][parts[2]] = value
    return out
