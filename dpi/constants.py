"""Physical constants, unit conversions and shared exception types.

All internal arithmetic is in atomic units.  Quantities that cross a module
or file boundary in eV carry an ``_ev`` suffix; atomic-unit quantities carry
``_au``.  Atomic subshell cross sections are in Mb everywhere.
"""

from __future__ import annotations

# ── unit conversions ────────────────────────────────────────────────────────
HARTREE_EV = 27.211386245988      # CODATA 2018
EV_AU = 1.0 / HARTREE_EV
# 1 Mb = 1e-18 cm^2 and a0 = 0.529177210903e-8 cm, so 1 Mb = 0.0357106 a0^2.
# Unused while the reported intensity is relative (see spectrum.prefactor),
# but it must be right the day an absolute cross section is wanted:
# dpi_cross_section.py carried MBARN_TO_AU = 1/28002.57, too small by a
# factor of 1000 (the denominator is a0^2/Mb = 28.00285, not 28002.85).
MB_TO_BOHR2 = 1.0 / 28.00285
C_AU = 137.035999084             # speed of light, atomic units


def ev_to_au(e_ev):
    """eV -> Hartree."""
    return e_ev * EV_AU


def au_to_ev(e_au):
    """Hartree -> eV."""
    return e_au * HARTREE_EV


def momentum_au(eps_au):
    """Electron momentum k = sqrt(2*eps) from kinetic energy, atomic units."""
    return (2.0 * eps_au) ** 0.5


# ── occupation-number windows for classifying RasOrb #OCC entries ───────────
# A strict 2-open-shell ROHF dication has two MOs at occ ~ 1.  A
# state-averaged natural-orbital RasOrb (e.g. the S 2p edge, where the core
# hole is delocalised over three symmetry partners) has three MOs at
# occ ~ 5/3 plus one at occ ~ 1; those fractional entries land in
# OCC_FRACTIONAL and trigger the documented single-determinant approximation.
OCC_DOUBLY_MIN = 1.95
OCC_FRACTIONAL = (1.50, 1.95)
OCC_SINGLY = (0.50, 1.50)
OCC_VIRTUAL_MAX = 0.05

# ── channel identifiers ─────────────────────────────────────────────────────
CHANNELS_LS = ("singlet", "triplet")
CHANNELS_JJ = ("jc32", "jc12")
CHANNEL_LABELS = {
    "singlet": "S_dic = 0",
    "triplet": "S_dic = 1",
    "jc32": "j_C = 3/2",
    "jc12": "j_C = 1/2",
}


class DPIError(Exception):
    """Base class for all errors raised by the dpi package."""


class MolcasFormatError(DPIError, ValueError):
    """An OpenMolcas file could not be parsed or failed validation."""


class ConfigError(DPIError, ValueError):
    """The input file is missing a key, or a key has an invalid value."""


class ModelError(DPIError, ValueError):
    """The requested combination of model options is not implementable."""
