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
import os
import tomllib
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from .amplitudes import TermSwitches
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

    Required for consistency with the Gelius reduction (manuscript
    Eq. 117); the full molecular dipole injects a large R_A <chi|chi>
    positional term with no counterpart in the atomic cross sections.
    """

    include_frozen: bool = False

    high_energy_exponent: dict[str, float] = field(default_factory=dict)
    """Per-subshell power-law exponent above the tabulated range.

    Keys look like ``"S_1s"``.  Yeh-Lindau stops at 1500 eV while the S1s
    edge is probed near 2610 eV, so that subshell rests on an
    extrapolation; -3.5 is the hydrogenic value.
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
        if self.coupling not in ("ls", "jj"):
            raise ConfigError(
                f"[physics] coupling must be 'ls' (F1s, S1s edges) or 'jj' "
                f"(S2p edge, spin-orbit split core hole), "
                f"got {self.coupling!r}")
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
            f"frozen reference  : {phys.include_frozen}",
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
        include_frozen=bool(phys_t.get("include_frozen", False)),
        high_energy_exponent={
            str(k): float(v)
            for k, v in dict(phys_t.get("high_energy_exponent", {})).items()},
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
        frozen_orbitals=_join_map(
            _channel_map(paths_t, "frozen_orbitals", orbital_keys, "paths",
                         required=False)),
        experiment=(_join(paths_t["experiment"])
                    if paths_t.get("experiment") else None),
        # cache_dir is an OUTPUT directory, so it is deliberately not
        # prefixed by root: root points at where the MOLCAS files were read
        # from, which is often read-only or shared between runs.
        cache_dir=str(paths_t.get("cache_dir", "dyson_cache")),
    )

    output = OutputConfig(**out_t)
    terms = build_term_switches(terms_t)

    if physics.spin_degeneracy_factor != 1.0 and physics.coupling == "jj":
        raise ConfigError(
            "[physics] spin_degeneracy_factor is an LS-coupling diagnostic; "
            "in jj coupling the 2:1 branching follows from the substate "
            "enumeration and must not be rescaled.")

    paths.check_exists()
    return Config(physics=physics, paths=paths, output=output, terms=terms,
                  source=os.path.abspath(path))


def _apply_overrides(raw: dict[str, Any],
                     overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply flat ``"section.key" -> value`` overrides to a parsed table."""
    out: dict[str, Any] = {k: (dict(v) if isinstance(v, dict) else v)
                           for k, v in raw.items()}
    for dotted, value in overrides.items():
        if value is None:
            continue
        if "." not in dotted:
            raise ConfigError(
                f"override {dotted!r} must have the form 'section.key'.")
        section, key = dotted.split(".", 1)
        if section not in _SECTIONS:
            raise ConfigError(
                f"override {dotted!r} names unknown section {section!r}.")
        out.setdefault(section, {})[key] = value
    return out
