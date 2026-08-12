"""Tabulated atomic subshell photoionization cross sections, in Mb.

Equation numbers refer to **dpi_notes_revised.tex** (the 47-page
revision; the earlier manuscript numbering differs).
The Gelius one-centre intensity model weights each AO :math:`\\chi_\\mu` by
the photoionization cross section of the free-atom subshell that AO
represents.  This module holds that table for the elements of SF6 and
evaluates it, with an explicit treatment of the three energy regions where
the tabulation does not apply.

Data source
-----------
J. J. Yeh and I. Lindau, *Atomic Data and Nuclear Data Tables* **32**,
1--155 (1985), doi:10.1016/0092-640X(85)90016-6, as redistributed in
continuous-curve form by the Elettra synchrotron "WebElements" service
(https://vuo.elettra.eu/services/elements/WebElements.html).  Values are
carried over verbatim from the previous implementation's ``_RAW`` table;
they have not been re-rounded or re-interpolated.

The S 1s entry is *not* a Yeh--Lindau tabulation.  Yeh--Lindau stops at
1500 eV, well below the S 1s threshold at 2472 eV, so that entry is a
hydrogenic :math:`\\sigma \\propto (h\\nu)^{-7/2}` estimate anchored to the
published value at 8047.8 eV.  Its ``source`` field says so, and
:func:`provenance_note` collects such caveats for an output header.

Remark 1 (why the argument is ``eps1 + I_mu``)
---------------------------------------------
:math:`\\sigma^{AO}_\\mu` is tabulated against *photon* energy.  For a free
atom carrying AO :math:`\\mu`, a photon producing a photoelectron of
kinetic energy :math:`\\varepsilon_1` must supply both that kinetic energy
and the atomic binding energy :math:`I_\\mu`, so the tabulation is to be
read at :math:`\\varepsilon_1 + I_\\mu` -- per AO, since :math:`I_\\mu`
differs between subshells.  It is *not* to be read at
:math:`\\omega - \\varepsilon_2`, which would replace :math:`I_\\mu` by the
molecular double ionization potential.  For valence subshells
:math:`I_\\mu \\ll \\mathrm{DIP}`, so the naive form evaluates the
tabulation far above threshold, where the cross section is strongly
suppressed; the two forms differ by a factor of order
:math:`(\\mathrm{DIP}/I_\\mu)^{3.5}` for the outermost shells.

Units
-----
All photon and kinetic energies in eV (``_ev`` suffix); all cross sections
in Mb.  No conversion to :math:`a_0^2` is applied anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from .constants import ConfigError, ModelError

__all__ = [
    "SubshellData",
    "YEH_LINDAU",
    "DEFAULT_SUBSHELL_MAP",
    "HYDROGENIC_EXPONENT",
    "REGION_BELOW_THRESHOLD",
    "REGION_LINEAR_RISE",
    "REGION_NEAR_THRESHOLD",
    "THRESHOLD_MODELS",
    "REGION_TABULATED",
    "REGION_POWER_LAW",
    "sigma",
    "sigma_region",
    "threshold",
    "provenance_note",
    "SigmaBuilder",
]


_ELETTRA_NOTE = (
    "Yeh & Lindau (1985) via the Elettra WebElements continuous curves"
)
_HYDROGENIC_NOTE = (
    "hydrogenic sigma ~ (hv)^-7/2 estimate anchored to Yeh & Lindau "
    "(1985) at 8047.8 eV; NOT a Yeh-Lindau tabulation"
)

# Region codes returned by sigma_region().  The driver reports whether any
# evaluation fell in the near-threshold region, because it is a model rather
# than tabulated data.  ``REGION_LINEAR_RISE`` keeps the original name (the
# linear rise is still the default model) and ``REGION_NEAR_THRESHOLD`` is the
# model-neutral alias.
REGION_BELOW_THRESHOLD = 0
REGION_LINEAR_RISE = 1
REGION_NEAR_THRESHOLD = 1
REGION_TABULATED = 2
REGION_POWER_LAW = 3

# Interpolation models for the gap between the ionization threshold and the
# first usable tabulated point.  This region is entered unavoidably whenever
# the omega/omega_eff weight is on, because omega_eff(mu) -> I_mu as the
# energy sharing eps1 -> 0, so the tabulation gets read at its own onset
# (REVIEW.md [A-12]).
#
# The choice is not cosmetic and it is not arbitrary.  What happens to an
# atomic subshell cross section at threshold depends on the final-state
# potential:
#
#   * COULOMB final state (a real atom: the residual ion is charged).  The
#     Coulomb penetration factor |C_0|^2 = 2 pi nu / (1 - exp(-2 pi nu))
#     with nu = Z/k diverges as 2 pi/k for k -> 0, cancelling the phase-space
#     suppression.  The exact hydrogenic (Stobbe) result therefore approaches
#     a FINITE constant: sigma_0 = 2^9 pi^2/(3 e^4) alpha a0^2 = 6.3043 Mb for
#     H 1s, matching the tabulated 6.30 Mb.  This is what Yeh-Lindau data is,
#     and the measured log-log slopes of the first two tabulated points bear
#     it out: -0.02 (F1s), +0.00 (S2s), -0.05 (S1s) -- flat, not rising.
#
#   * SHORT-RANGE or PLANE-WAVE final state.  No penetration factor, so the
#     Wigner law sigma ~ (hv - I)^(l'+1/2) survives; for an l'=1 outgoing
#     electron that is eps^1.5, verified against the plane-wave hydrogenic
#     cross section (slope +1.500 at eps = 1e-3 Ha).
#
# So WIGNER is the wrong law for a tabulated atomic sigma and the right one
# for a model whose continuum is a plane wave -- which is what this code uses
# everywhere else.  Both are offered because that tension is real and
# unresolved: see the ``threshold_model`` docstring on SigmaBuilder.
THRESHOLD_MODELS = ("linear", "flat", "coulomb", "wigner", "extrapolate",
                    "anchored")

#: Offset of the second artificial knot above the threshold, eV.  See
#: :func:`_anchored_knots`.
ANCHOR_DELTA_EV = 0.1

# ``extrapolate`` is the "why model it at all?" option: continue the log-log
# spline itself backwards into the gap, treating the near-threshold value as
# an interpolation question rather than a physics question.  Its merit is
# that it assumes nothing beyond the data's own local shape, and on the
# shipped table it agrees with ``coulomb`` to 0.1% for every subshell whose
# lever arm is short -- an independent confirmation, from an agnostic
# interpolant, of the Coulomb argument above.
#
# It is NOT safe over a long lever arm, and fails far more violently than a
# shape model does when abused: extrapolating a cubic in log-log space away
# from its knots is unbounded.  For S_3p (arm 0.61) the not-a-knot spline
# returns 3.9x sigma_lo, a natural-boundary spline 28x, and an Akima
# interpolant 25x -- against 3.6x for ``coulomb``, which at least cannot
# exceed the Stobbe shape.  ``unsafe_extrapolations`` therefore flags this
# model exactly as it flags ``coulomb``.

# Wigner exponent l' + 1/2 for the dominant outgoing partial wave, keyed by
# the angular momentum l of the bound orbital.  sigma ~ k^(2l'+1) and
# eps = k^2/2, hence sigma ~ eps^(l' + 1/2).  A dipole transition from l
# reaches l' = l +- 1, and the LOWER channel dominates at threshold because
# its centrifugal barrier is smaller -- so l' = l - 1 where that exists, and
# l' = 1 from an s orbital, which has no lower channel:
#
#   l = 0 (s) -> l' = 1 only     -> 1.5
#   l = 1 (p) -> l' = 0 dominates -> 0.5
#   l = 2 (d) -> l' = 1 dominates -> 1.5
#   l = 3 (f) -> l' = 2 dominates -> 2.5
_WIGNER_EXPONENT = {0: 1.5, 1: 0.5, 2: 1.5, 3: 2.5}

# Angular momentum of each tabulated subshell, for the Wigner exponent.
_SUBSHELL_L = {"s": 0, "p": 1, "d": 2, "f": 3}

# Exponent of a hydrogenic cross section well above threshold.  Supplied as
# a named constant because it is the physically motivated override for
# subshells used far beyond the end of the Yeh-Lindau table -- notably at
# the S 1s edge (omega = 2610 eV), where every other subshell's sigma is an
# extrapolation over a factor ~1.7 in energy.  REVIEW.md recommendation 3
# asks for the sensitivity of band intensities to this value to be quoted,
# so it must stay a parameter and never become a literal.
HYDROGENIC_EXPONENT = -3.5


@dataclass(frozen=True)
class SubshellData:
    """One subshell's tabulated cross section.

    Attributes
    ----------
    element : str
        Element symbol, e.g. ``'S'``.
    subshell : str
        Subshell label, e.g. ``'2p'``.
    threshold_ev : float
        Ionization threshold of the free atom, eV.  ``sigma`` is zero at or
        below this energy.
    hv_ev : tuple of float
        Tabulated photon energies, eV, ascending.
    sigma_mb : tuple of float
        Tabulated cross sections, Mb, parallel to ``hv_ev``.
    source : str
        Provenance of this entry, reproduced in output headers.
    """

    element: str
    subshell: str
    threshold_ev: float
    hv_ev: tuple[float, ...]
    sigma_mb: tuple[float, ...]
    source: str

    @property
    def key(self) -> str:
        """Table key, ``'<element>_<subshell>'``."""
        return f"{self.element}_{self.subshell}"

    @property
    def is_tabulated(self) -> bool:
        """False for entries that are estimates rather than measurements."""
        return self.source is not _HYDROGENIC_NOTE


# Tabulated data, carried over verbatim from the previous
# implementation's _RAW dict by AST literal-eval (no hand
# transcription).  hv_ev in eV, sigma_mb in Mb.
YEH_LINDAU: dict[str, SubshellData] = {
    "S_1s": SubshellData(
        element="S",
        subshell="1s",
        threshold_ev=2472.0,
        hv_ev=(
            2480., 2500., 2600., 2700., 3000., 4000., 5000., 8048.,
        ),
        sigma_mb=(
            0.092, 0.086, 0.073, 0.063, 0.044, 0.023,
            0.014, 0.005,
        ),
        source=_HYDROGENIC_NOTE,
    ),
    "S_2s": SubshellData(
        element="S",
        subshell="2s",
        threshold_ev=229.2,
        hv_ev=(
            225., 230., 235., 240., 245., 250., 255., 260.,
            265., 270., 275., 280., 285., 290., 295., 300.,
            305., 310., 315., 320., 325., 330., 335., 340.,
            345., 350., 355., 360., 365., 370., 375., 380.,
            385., 390., 395., 400., 405., 410., 415., 420.,
            425., 430., 435., 440., 445., 450., 455., 460.,
            465., 470., 475., 480., 485., 490., 495., 500.,
            505., 510., 515., 520., 540., 560., 580., 600.,
            620., 640., 660., 680., 700., 720., 740., 760.,
            780., 800., 850., 900., 950., 1000., 1041., 1050.,
            1100., 1150., 1200., 1250., 1253.6, 1300., 1350., 1400.,
            1450., 1486.6, 1500.,
        ),
        sigma_mb=(
            0.3675, 0.3728, 0.376, 0.3782, 0.3789, 0.3776,
            0.3743, 0.3694, 0.3633, 0.3563, 0.3488, 0.3413,
            0.3339, 0.3267, 0.3199, 0.3134, 0.3073, 0.3015,
            0.2959, 0.2906, 0.2855, 0.2805, 0.2756, 0.2708,
            0.2661, 0.2614, 0.2567, 0.2521, 0.2476, 0.243,
            0.2386, 0.2342, 0.2299, 0.2257, 0.2216, 0.2176,
            0.2136, 0.2098, 0.206, 0.2024, 0.1989, 0.1954,
            0.192, 0.1888, 0.1856, 0.1824, 0.1794, 0.1764,
            0.1735, 0.1706, 0.1678, 0.1651, 0.1624, 0.1598,
            0.1572, 0.1547, 0.1522, 0.1498, 0.1474, 0.1451,
            0.1363, 0.1282, 0.1209, 0.1141, 0.1079, 0.1021,
            0.09676, 0.09174, 0.08704, 0.08266, 0.07856, 0.07474,
            0.07118, 0.06786, 0.06048, 0.05419, 0.04882, 0.04417,
            0.04081, 0.04012, 0.03654, 0.03335, 0.03051, 0.02799,
            0.02782, 0.02576, 0.02378, 0.02203, 0.02046, 0.01941,
            0.01904,
        ),
        source=_ELETTRA_NOTE,
    ),
    "S_2p": SubshellData(
        element="S",
        subshell="2p",
        threshold_ev=170.2,
        hv_ev=(
            175., 180., 185., 190., 195., 200., 205., 210.,
            215., 220., 225., 230., 235., 240., 245., 250.,
            255., 260., 265., 270., 275., 280., 285., 290.,
            295., 300., 305., 310., 315., 320., 325., 330.,
            335., 340., 345., 350., 355., 360., 365., 370.,
            375., 380., 385., 390., 395., 400., 405., 410.,
            415., 420., 425., 430., 435., 440., 445., 450.,
            455., 460., 465., 470., 480., 500., 520., 540.,
            560., 580., 600., 620., 640., 660., 680., 700.,
            720., 740., 760., 780., 800., 850., 900., 950.,
            1000., 1041., 1050., 1100., 1150., 1200., 1250., 1253.6,
            1300., 1350., 1400., 1450., 1486.6, 1500.,
        ),
        sigma_mb=(
            3.198, 5.184, 4.569, 4.131, 3.914, 3.799,
            3.721, 3.647, 3.563, 3.465, 3.356, 3.239,
            3.119, 3., 2.884, 2.774, 2.669, 2.569,
            2.475, 2.385, 2.3, 2.218, 2.14, 2.064,
            1.991, 1.92, 1.852, 1.785, 1.721, 1.66,
            1.601, 1.544, 1.489, 1.437, 1.387, 1.339,
            1.294, 1.25, 1.209, 1.169, 1.131, 1.094,
            1.06, 1.026, 0.9942, 0.9635, 0.934, 0.9056,
            0.8784, 0.8521, 0.8268, 0.8025, 0.7791, 0.7565,
            0.7347, 0.7138, 0.6936, 0.6741, 0.6553, 0.6373,
            0.603, 0.5416, 0.4882, 0.4415, 0.4004, 0.3639,
            0.3315, 0.3027, 0.2769, 0.254, 0.2335, 0.2152,
            0.1988, 0.184, 0.1706, 0.1585, 0.1475, 0.1239,
            0.1049, 0.08931, 0.07654, 0.06776, 0.06601, 0.05732,
            0.05013, 0.04411, 0.039, 0.03866, 0.0346, 0.03078,
            0.02745, 0.02454, 0.02265, 0.02201,
        ),
        source=_ELETTRA_NOTE,
    ),
    "S_3s": SubshellData(
        element="S",
        subshell="3s",
        threshold_ev=20.7,
        hv_ev=(
            25., 26.86, 30., 35., 40., 40.81, 45., 50.,
            55., 60., 65., 70., 75., 80., 85., 90.,
            95., 100., 105., 110., 115., 120., 125., 130.,
            132.3, 135., 140., 145., 150., 151.4, 155., 160.,
            165., 170., 175., 180., 185., 190., 195., 200.,
            205., 210., 215., 220., 225., 230., 235., 240.,
            245., 250., 255., 260., 265., 270., 275., 280.,
            285., 290., 295., 300., 305., 310., 315., 320.,
            340., 360., 380., 400., 420., 440., 460., 480.,
            500., 520., 540., 560., 580., 600., 620., 640.,
            660., 680., 700., 720., 740., 760., 780., 800.,
            850., 900., 950., 1000., 1041., 1050., 1100., 1150.,
            1200., 1250., 1253.6, 1300., 1350., 1400., 1450., 1486.6,
            1500.,
        ),
        sigma_mb=(
            0.1921, 0.2482, 0.3253, 0.4072, 0.4463, 0.4493,
            0.4545, 0.4453, 0.4284, 0.4084, 0.3865, 0.3631,
            0.3389, 0.315, 0.2927, 0.2726, 0.2551, 0.2397,
            0.226, 0.2132, 0.2011, 0.1895, 0.1785, 0.1682,
            0.1637, 0.1587, 0.1502, 0.1427, 0.1359, 0.1341,
            0.1298, 0.1241, 0.1186, 0.1134, 0.1083, 0.1033,
            0.09855, 0.09405, 0.08988, 0.08607, 0.08265, 0.07957,
            0.07679, 0.07425, 0.07187, 0.06959, 0.06735, 0.06514,
            0.06293, 0.06075, 0.05861, 0.05655, 0.05459, 0.05276,
            0.05108, 0.04954, 0.04814, 0.04686, 0.04567, 0.04455,
            0.04347, 0.0424, 0.04133, 0.04025, 0.03585, 0.03199,
            0.02918, 0.02702, 0.02481, 0.02249, 0.02051, 0.01908,
            0.01796, 0.0168, 0.01549, 0.01425, 0.0133, 0.01265,
            0.01207, 0.0114, 0.01062, 0.009875, 0.009303, 0.008914,
            0.008598, 0.008229, 0.007762, 0.007257, 0.006389, 0.005881,
            0.005089, 0.004588, 0.004367, 0.004298, 0.003775, 0.003405,
            0.003246, 0.002919, 0.00289, 0.002586, 0.002473, 0.002337,
            0.002072, 0.001935, 0.001909,
        ),
        source=_ELETTRA_NOTE,
    ),
    "S_3p": SubshellData(
        element="S",
        subshell="3p",
        threshold_ev=10.4,
        hv_ev=(
            16.7, 20., 21.2, 25., 26.8, 30., 35., 40.,
            40.8, 45., 50., 55., 60., 65., 70., 75.,
            80., 85., 90., 95., 100., 105., 110., 115.,
            120., 125., 130., 132.3, 135., 140., 145., 150.,
            151.4, 155., 160., 165., 170., 175., 180., 185.,
            190., 195., 200., 205., 210., 215., 220., 225.,
            230., 235., 240., 245., 250., 255., 260., 265.,
            270., 275., 280., 285., 290., 295., 300., 305.,
            310., 320., 340., 360., 380., 400., 420., 440.,
            460., 480., 500., 520., 540., 560., 580., 600.,
            620., 640., 660., 680., 700., 720., 740., 760.,
            780., 800., 850., 900., 950., 1000., 1041., 1050.,
            1100., 1150., 1200., 1250., 1253.6, 1300., 1350., 1400.,
            1450., 1486.6, 1500.,
        ),
        sigma_mb=(
            18.25, 6.353, 4.371, 1.495, 0.9906, 0.6099,
            0.5207, 0.5911, 0.6051, 0.6741, 0.7345, 0.7674,
            0.7761, 0.7664, 0.7439, 0.7135, 0.6789, 0.6425,
            0.6062, 0.5708, 0.537, 0.505, 0.4748, 0.4465,
            0.42, 0.3952, 0.372, 0.3618, 0.3503, 0.3301,
            0.3112, 0.2936, 0.2889, 0.2772, 0.2619, 0.2477,
            0.2345, 0.2221, 0.2107, 0.2, 0.19, 0.1807,
            0.172, 0.1639, 0.1563, 0.1491, 0.1425, 0.1362,
            0.1303, 0.1248, 0.1195, 0.1146, 0.1099, 0.1055,
            0.1012, 0.09724, 0.09343, 0.0898, 0.08635, 0.08306,
            0.07993, 0.07695, 0.07412, 0.07142, 0.06886, 0.06409,
            0.05582, 0.04888, 0.04299, 0.038, 0.03381, 0.03023,
            0.02711, 0.02435, 0.02195, 0.01991, 0.01818, 0.01662,
            0.01517, 0.01384, 0.0127, 0.01175, 0.01089, 0.01006,
            0.009252, 0.008536, 0.007936, 0.007416, 0.00693, 0.006455,
            0.005405, 0.004624, 0.003967, 0.003412, 0.003047, 0.002975,
            0.002603, 0.002277, 0.002006, 0.001779, 0.001764, 0.001579,
            0.001407, 0.001262, 0.001137, 0.001053, 0.001025,
        ),
        source=_ELETTRA_NOTE,
    ),
    "F_1s": SubshellData(
        element="F",
        subshell="1s",
        threshold_ev=692.3,
        hv_ev=(
            690., 695., 700., 705., 710., 715., 720., 725.,
            730., 735., 740., 745., 750., 755., 760., 765.,
            770., 775., 780., 785., 790., 795., 800., 805.,
            810., 815., 820., 825., 830., 835., 840., 845.,
            850., 855., 860., 865., 870., 875., 880., 885.,
            890., 895., 900., 905., 910., 915., 920., 925.,
            930., 935., 940., 945., 950., 955., 960., 965.,
            970., 975., 980., 985., 1000., 1041., 1050., 1100.,
            1150., 1200., 1250., 1253.6, 1300., 1350., 1400., 1450.,
            1486.6, 1500.,
        ),
        sigma_mb=(
            0.4103, 0.3869, 0.3778, 0.3727, 0.3691, 0.3661,
            0.3633, 0.3603, 0.3571, 0.3537, 0.3499, 0.3459,
            0.3416, 0.337, 0.3323, 0.3274, 0.3224, 0.3173,
            0.3123, 0.3072, 0.3022, 0.2972, 0.2923, 0.2875,
            0.2829, 0.2783, 0.2739, 0.2696, 0.2654, 0.2613,
            0.2574, 0.2536, 0.2498, 0.2462, 0.2427, 0.2393,
            0.236, 0.2327, 0.2296, 0.2265, 0.2235, 0.2205,
            0.2176, 0.2148, 0.212, 0.2093, 0.2066, 0.204,
            0.2014, 0.1988, 0.1963, 0.1938, 0.1914, 0.189,
            0.1866, 0.1842, 0.1819, 0.1796, 0.1774, 0.1752,
            0.1687, 0.1525, 0.1492, 0.1325, 0.1184, 0.1062,
            0.09558, 0.09487, 0.08629, 0.07811, 0.07091, 0.06457,
            0.06041, 0.05898,
        ),
        source=_ELETTRA_NOTE,
    ),
    "F_2s": SubshellData(
        element="F",
        subshell="2s",
        threshold_ev=37.9,
        hv_ev=(
            40., 40.81, 45., 50., 55., 60., 65., 70.,
            75., 80., 85., 90., 95., 100., 105., 110.,
            115., 120., 125., 130., 132.3, 135., 140., 145.,
            150., 151.4, 155., 160., 165., 170., 175., 180.,
            185., 190., 195., 200., 205., 210., 215., 220.,
            225., 230., 235., 240., 245., 250., 255., 260.,
            265., 270., 275., 280., 285., 290., 295., 300.,
            305., 310., 315., 320., 325., 330., 335., 340.,
            360., 380., 400., 420., 440., 460., 480., 500.,
            520., 540., 560., 580., 600., 620., 640., 660.,
            680., 700., 720., 740., 760., 780., 800., 850.,
            900., 950., 1000., 1041., 1050., 1100., 1150., 1200.,
            1250., 1253.6, 1300., 1350., 1400., 1450., 1486.6, 1500.,
        ),
        sigma_mb=(
            0.4967, 0.5185, 0.607, 0.6717, 0.7072, 0.723,
            0.7242, 0.7145, 0.6969, 0.6742, 0.6486, 0.6219,
            0.595, 0.5686, 0.543, 0.5183, 0.4943, 0.471,
            0.4484, 0.4265, 0.4167, 0.4055, 0.3854, 0.3663,
            0.3483, 0.3435, 0.3315, 0.3158, 0.3013, 0.2878,
            0.2752, 0.2635, 0.2525, 0.2422, 0.2324, 0.2231,
            0.2142, 0.2056, 0.1975, 0.1897, 0.1822, 0.1751,
            0.1684, 0.1621, 0.1561, 0.1505, 0.1452, 0.1403,
            0.1356, 0.1312, 0.127, 0.1231, 0.1193, 0.1156,
            0.1121, 0.1087, 0.1054, 0.1022, 0.09909, 0.09605,
            0.0931, 0.09025, 0.08748, 0.08482, 0.07528, 0.06747,
            0.06105, 0.05552, 0.05046, 0.04577, 0.04155, 0.03792,
            0.03489, 0.03235, 0.03009, 0.02797, 0.02591, 0.02394,
            0.02213, 0.02056, 0.01922, 0.01809, 0.01707, 0.01611,
            0.01515, 0.0142, 0.01327, 0.01133, 0.01002, 0.008913,
            0.007748, 0.006911, 0.006757, 0.006096, 0.005574, 0.004988,
            0.004396, 0.004358, 0.003957, 0.003674, 0.003412, 0.00309,
            0.002846, 0.002763,
        ),
        source=_ELETTRA_NOTE,
    ),
    "F_2p": SubshellData(
        element="F",
        subshell="2p",
        threshold_ev=17.4,
        hv_ev=(
            20., 21.22, 25., 26.86, 30., 35., 40., 40.81,
            45., 50., 55., 60., 65., 70., 75., 80.,
            85., 90., 95., 100., 105., 110., 115., 120.,
            125., 130., 132.3, 135., 140., 145., 150., 151.4,
            155., 160., 165., 170., 175., 180., 185., 190.,
            195., 200., 205., 210., 215., 220., 225., 230.,
            235., 240., 245., 250., 255., 260., 265., 270.,
            275., 280., 285., 290., 295., 300., 305., 310.,
            315., 320., 340., 360., 380., 400., 420., 440.,
            460., 480., 500., 520., 540., 560., 580., 600.,
            620., 640., 660., 680., 700., 720., 740., 760.,
            780., 800., 850., 900., 950., 1000., 1041., 1050.,
            1100., 1150., 1200., 1250., 1253.6, 1300., 1350., 1400.,
            1450., 1486.6, 1500.,
        ),
        sigma_mb=(
            8.917, 9.305, 9.839, 9.876, 9.742, 9.223,
            8.535, 8.417, 7.803, 7.075, 6.363, 5.681,
            5.045, 4.469, 3.958, 3.508, 3.112, 2.764,
            2.457, 2.187, 1.953, 1.75, 1.574, 1.421,
            1.288, 1.171, 1.121, 1.066, 0.9722, 0.8883,
            0.8132, 0.7936, 0.7462, 0.6867, 0.6337, 0.5866,
            0.5442, 0.5058, 0.4707, 0.4384, 0.4085, 0.3808,
            0.3553, 0.3318, 0.3103, 0.2907, 0.2729, 0.2567,
            0.2419, 0.2283, 0.2157, 0.2038, 0.1927, 0.1821,
            0.1721, 0.1627, 0.1538, 0.1455, 0.1379, 0.1308,
            0.1243, 0.1183, 0.1128, 0.1077, 0.1029, 0.09838,
            0.08225, 0.06861, 0.05801, 0.05008, 0.04349, 0.03747,
            0.03231, 0.02833, 0.02524, 0.02245, 0.01975, 0.01736,
            0.01549, 0.01406, 0.0128, 0.01152, 0.01026, 0.009192,
            0.008385, 0.00776, 0.007169, 0.006535, 0.005893, 0.005334,
            0.004472, 0.003687, 0.002968, 0.002603, 0.002247, 0.00216,
            0.001799, 0.001633, 0.001378, 0.001148, 0.001139, 0.001059,
            0.0009311, 0.0007757, 0.0007144, 0.0006817, 0.0006601,
        ),
        source=_ELETTRA_NOTE,
    ),
}


# Mapping (l, shell_index) -> subshell label, per element, for S and F in
# cc-pVDZ.  ``shell_index`` is the sequential index of the contracted shell
# within its (centre, l) group, as defined in SPEC.md section 2.  This is a
# *default*: it is basis-set specific and is a plain argument to
# SigmaBuilder, not a hidden global.  Shells with no tabulated atomic
# counterpart -- the cc-pVDZ polarisation functions, and the second
# contracted shell of an l for which only one atomic subshell exists -- are
# deliberately absent, and SigmaBuilder counts them.
DEFAULT_SUBSHELL_MAP: dict[str, dict[tuple[int, int], str]] = {
    "S": {
        (0, 0): "1s",
        (0, 1): "2s",
        (0, 2): "3s",
        (0, 3): "3s",
        (1, 0): "2p",
        (1, 1): "3p",
        (1, 2): "3p",
        (2, 0): "3p",
    },
    "F": {
        (0, 0): "1s",
        (0, 1): "2s",
        (0, 2): "2s",
        (1, 0): "2p",
        (1, 1): "2p",
        (2, 0): "2p",
    },
}


@dataclass(frozen=True)
class _Evaluator:
    """Precomputed spline and extrapolation data for one subshell."""

    data: SubshellData
    spline: CubicSpline
    log_hv: np.ndarray
    log_sigma: np.ndarray
    hv_lo: float
    hv_hi: float
    sigma_lo: float
    default_exponent: float
    n_dropped: int
    anchored: PchipInterpolator | None = None
    anchor_hv: float = 0.0
    anchor_sigma: float = 0.0


_EVALUATORS: dict[tuple[str, float], _Evaluator] = {}


def _entry(element: str, subshell: str) -> SubshellData:
    key = f"{element}_{subshell}"
    try:
        return YEH_LINDAU[key]
    except KeyError:
        raise ConfigError(
            f"no atomic cross section tabulated for {element} {subshell} "
            f"(key {key!r}); available keys: {sorted(YEH_LINDAU)}"
        ) from None


def _first_real_point(data: SubshellData) -> float | None:
    """Lowest tabulated energy strictly above the declared threshold.

    This is the boundary the ``"anchored"`` model bridges to, and it differs
    from ``_Evaluator.hv_lo`` for the two entries whose grid starts below
    their own threshold.
    """
    real = [h for h, sg in zip(data.hv_ev, data.sigma_mb)
            if h > data.threshold_ev and sg > 0.0]
    return float(min(real)) if real else None


def _anchored_knots(
    data: SubshellData, delta_ev: float | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build the knot set for the ``"anchored"`` near-threshold model.

    The recipe, in five steps:

    1. Discard every tabulated point at or below ``threshold_ev``.  For the
       two straddling entries (F 1s, S 2s) this *removes* real data, which
       is the deliberate trade: the model needs a clean statement of where
       the onset is, and mixing a published point below the declared
       threshold with an artificial zero *at* it is contradictory.
    2. Insert an artificial knot at ``I_mu`` and a second at
       ``I_mu + delta_ev``.
    3. The first carries ``sigma = 0`` exactly -- the physical boundary
       condition that no model here previously imposed.
    4. The second carries the value obtained by extrapolating the *cleaned
       original* data (log-log cubic) to that energy.  This is what keeps
       the rise physical rather than linear: the slope leaving the
       threshold is inherited from the data instead of assumed.
    5. Interpolate the union.

    Step 3 forces the interpolation variable.  Every other model works in
    ``log sigma``, and ``log 0`` is ``-inf``, so the assembled curve must be
    interpolated in **linear** ``sigma`` against **linear** ``hv``.  That is
    a real difference in character, not a detail: the interpolant is no
    longer exact on a power law.  Between the artificial knots and the first
    real one the curve is a cubic in ``hv``, and above ``hv_lo`` the
    ``anchored`` model hands back to the ordinary log-log spline, so the
    change is confined to the gap.

    PCHIP rather than a natural cubic for the union: it is shape-preserving,
    so ``sigma`` cannot dip negative between the zero at ``I_mu`` and the
    first positive knot.  A cubic through the same points does undershoot,
    which would be unphysical.

    Returns
    -------
    hv_knots, sigma_knots, sigma_at_delta
    """
    # Read the module global at CALL time, not as a default argument -- a
    # default binds once at definition and would silently ignore any later
    # change to ANCHOR_DELTA_EV, making the offset look insensitive when it
    # is in fact never varying.
    if delta_ev is None:
        delta_ev = ANCHOR_DELTA_EV
    thr = float(data.threshold_ev)
    hv = np.asarray(data.hv_ev, dtype=float)
    sig = np.asarray(data.sigma_mb, dtype=float)

    keep = (hv > thr) & (sig > 0.0)                       # step 1
    hv, sig = hv[keep], sig[keep]
    if hv.size < 4:
        raise ModelError(
            f"{data.element} {data.subshell}: only {hv.size} tabulated "
            f"points survive above the {thr} eV threshold; the anchored "
            f"model needs 4 to extrapolate from"
        )
    order = np.argsort(hv)
    hv, sig = hv[order], sig[order]

    anchor_hv = thr + float(delta_ev)
    if anchor_hv >= hv[0]:
        # The second artificial point would land on or past the first real
        # one.  Nothing to bridge: return the data with only the zero.
        return (np.concatenate([[thr], hv]),
                np.concatenate([[0.0], sig]),
                float(sig[0]))

    ext = CubicSpline(np.log(hv), np.log(sig), extrapolate=True)  # step 4
    with np.errstate(over="ignore"):
        s_anchor = float(np.exp(ext(np.log(anchor_hv))))
    if not np.isfinite(s_anchor) or s_anchor <= 0.0:
        s_anchor = float(sig[0])

    hv_k = np.concatenate([[thr, anchor_hv], hv])          # steps 2, 3
    sg_k = np.concatenate([[0.0, s_anchor], sig])
    return hv_k, sg_k, s_anchor


def _evaluator(element: str, subshell: str,
               delta_ev: float | None = None) -> _Evaluator:
    """Build (once) the log-log spline and power-law tail for a subshell.

    Every point with ``sigma > 0`` is kept, **including any that lie below
    the declared** ``threshold_ev``.  Those points are not artefacts: they
    are the published curve's own data, and their existence means the
    tabulation's implicit onset is *lower* than the threshold this table
    declares.  F 1s has a point at 690.0 eV against a declared 692.3 eV,
    and S 2s one at 225.0 against 229.2 -- the two values come from
    different sources (round-number photon-energy grids from the
    digitisation, experimental binding energies for the threshold), and
    they are not required to agree to better than a grid spacing.

    Discarding them used to have a consequence out of all proportion to
    two data points: it moved ``hv_lo`` up to the first point *above* the
    threshold, manufacturing a 2.7 eV "gap" for F 1s where the tabulation
    in fact has data on both sides.  Since F 1s carries ~70% of the
    intensity of an F1s-edge CV spectrum, most of what looked like
    near-threshold model dependence was this bookkeeping.  Keeping the
    points makes evaluation there ordinary spline interpolation.

    ``sigma`` is still exactly zero at and below ``threshold_ev`` -- that
    is physics, and :func:`sigma_region` enforces it independently of the
    spline's domain.
    """
    key = f"{element}_{subshell}"
    # The anchored knot set depends on ``delta_ev``, so it must be part of
    # the cache key.  Keying on the subshell alone would return the first
    # caller's offset to every later one -- which is exactly the class of
    # bug that made the delta sweep look flat while it was being developed.
    if delta_ev is None:
        delta_ev = ANCHOR_DELTA_EV
    delta_ev = float(delta_ev)
    cache_key = (key, delta_ev)
    cached = _EVALUATORS.get(cache_key)
    if cached is not None:
        return cached

    data = _entry(element, subshell)
    hv = np.asarray(data.hv_ev, dtype=float)
    sig = np.asarray(data.sigma_mb, dtype=float)
    if hv.shape != sig.shape:
        raise ModelError(
            f"{key}: hv_ev has {hv.size} points but sigma_mb has "
            f"{sig.size}; the table is corrupt"
        )

    keep = sig > 0.0
    n_dropped = int(hv.size - keep.sum())
    hv, sig = hv[keep], sig[keep]
    if hv.size < 4:
        raise ModelError(
            f"{key}: only {hv.size} tabulated points with a positive cross "
            f"section; a cubic spline needs 4"
        )
    order = np.argsort(hv)
    hv, sig = hv[order], sig[order]
    if np.any(np.diff(hv) <= 0.0):
        dup = hv[:-1][np.diff(hv) <= 0.0]
        raise ModelError(
            f"{key}: tabulated photon energies are not strictly "
            f"increasing near {dup[0]} eV"
        )

    log_hv = np.log(hv)
    log_sigma = np.log(sig)
    # A cubic spline in log-log space passes exactly through every knot, so
    # tabulated energies are reproduced to round-off, and a power law is a
    # straight line -- the right interpolation variable for a quantity that
    # falls off as a power of the photon energy over three decades.
    spline = CubicSpline(log_hv, log_sigma, extrapolate=False)
    default_exponent = float(
        (log_sigma[-1] - log_sigma[-2]) / (log_hv[-1] - log_hv[-2])
    )
    # The "anchored" model needs its own knot set and interpolant: it drops
    # the sub-threshold points the main spline keeps, and it works in linear
    # sigma because one of its knots is exactly zero.  Built here so the cost
    # is paid once per subshell, like the main spline.
    try:
        a_hv, a_sg, a_val = _anchored_knots(data, delta_ev)
        anchored = PchipInterpolator(a_hv, a_sg, extrapolate=False)
        anchor_hv = float(a_hv[1]) if a_hv.size > 1 else float(a_hv[0])
        anchor_sigma = float(a_val)
    except ModelError:
        anchored, anchor_hv, anchor_sigma = None, 0.0, 0.0

    ev = _Evaluator(
        data=data,
        spline=spline,
        log_hv=log_hv,
        log_sigma=log_sigma,
        hv_lo=float(hv[0]),
        hv_hi=float(hv[-1]),
        sigma_lo=float(sig[0]),
        default_exponent=default_exponent,
        n_dropped=n_dropped,
        anchored=anchored,
        anchor_hv=anchor_hv,
        anchor_sigma=anchor_sigma,
    )
    _EVALUATORS[key] = ev
    return ev


def threshold(element: str, subshell: str) -> float:
    """Free-atom ionization threshold of a subshell.

    Parameters
    ----------
    element, subshell : str
        e.g. ``'S'``, ``'2p'``.

    Returns
    -------
    float
        Threshold energy, eV.
    """
    return float(_entry(element, subshell).threshold_ev)


def sigma_region(
    element: str, subshell: str, hv_ev: float | np.ndarray,
    threshold_model: str = "linear",
) -> np.ndarray:
    """Classify each photon energy into one of the four evaluation regions.

    Returns an integer array of ``REGION_*`` codes, shaped like ``hv_ev``
    (0-d for scalar input).  This lets a caller establish *before* or
    *after* evaluation whether any point relied on the linear-rise
    stopgap, without a module-level mutable flag.

    Parameters
    ----------
    element, subshell : str
    hv_ev : float or ndarray
        Photon energy, eV.

    Returns
    -------
    ndarray of int
        Region codes.
    """
    ev = _evaluator(element, subshell)
    hv = np.asarray(hv_ev, dtype=float)
    codes = np.full(hv.shape, REGION_TABULATED, dtype=int)
    codes = np.where(hv <= ev.data.threshold_ev, REGION_BELOW_THRESHOLD,
                     codes)
    codes = np.where(
        (hv > ev.data.threshold_ev) & (hv < ev.hv_lo),
        REGION_LINEAR_RISE, codes,
    )
    codes = np.where(hv > ev.hv_hi, REGION_POWER_LAW, codes)

    if threshold_model == "anchored":
        # The anchored model owns a wider interval than the others: its step
        # 1 discards the points at or below ``threshold_ev``, so for the two
        # straddling entries (F 1s, S 2s) ``ev.hv_lo`` sits BELOW the
        # threshold and the tests above find no modelled region at all.  The
        # reclaim used to live only inside :func:`sigma`, which left this
        # function -- and therefore ``SigmaBuilder.used_linear_rise`` and any
        # diagnostic plot -- reporting no model use where the model was in
        # fact active.  Classifying it here keeps the one definition of the
        # region in one place.
        first_real = _first_real_point(ev.data)
        if first_real is not None:
            reclaim = (hv > ev.data.threshold_ev) & (hv < first_real)
            codes = np.where(reclaim, REGION_NEAR_THRESHOLD, codes)

    return codes


def _near_threshold(ev, hv: np.ndarray, model: str) -> np.ndarray:
    """Cross section between the threshold and the first tabulated point.

    Every model is normalised to pass through ``(hv_lo, sigma_lo)``, so the
    result is continuous with the spline and the *only* thing that changes
    is the shape of the approach to threshold.  That makes the four
    directly comparable: any difference in an integrated intensity is
    attributable to the shape alone, not to a discontinuity.

    Parameters
    ----------
    ev : _Evaluator
        Carries ``data.threshold_ev``, ``hv_lo`` and ``sigma_lo``.
    hv : ndarray
        Photon energies strictly inside ``(threshold, hv_lo)``.
    model : str
        One of :data:`THRESHOLD_MODELS`.

    Returns
    -------
    ndarray
        Cross sections in Mb, same shape as ``hv``.
    """
    if model not in THRESHOLD_MODELS:
        raise ConfigError(
            f"threshold_model must be one of {THRESHOLD_MODELS}, "
            f"got {model!r}"
        )
    thr = float(ev.data.threshold_ev)

    if model == "anchored":
        # Handled before the ``span`` guard below, because this model does
        # not measure against ``ev.hv_lo``: it discards the sub-threshold
        # points that ``hv_lo`` may sit on and bridges to the first
        # *surviving* real point instead.  For F 1s and S 2s, ``span`` is
        # negative and the guard would return ``sigma_lo`` -- the value of a
        # point this model has deliberately dropped.
        if ev.anchored is None:
            raise ModelError(
                f"{ev.data.element} {ev.data.subshell}: the anchored model "
                f"could not be built for this subshell"
            )
        out = np.asarray(ev.anchored(hv), dtype=float)
        # PCHIP is shape-preserving, so it cannot undershoot below the zero
        # knot; the clamp guards only against a NaN from an hv outside the
        # knot range, which the region logic should already prevent.
        return np.where(np.isfinite(out), np.maximum(out, 0.0), 0.0)

    span = float(ev.hv_lo) - thr
    s_lo = float(ev.sigma_lo)
    if span <= 0.0:                     # first tabulated point at threshold
        return np.full(hv.shape, s_lo)
    x = (hv - thr) / span               # 0 at threshold, 1 at hv_lo

    if model == "linear":
        # Historical default: sigma_lo * x.  Vanishes at threshold with slope
        # 1, which no final-state potential produces -- kept because every
        # spectrum computed before this option existed used it.
        return s_lo * x

    if model == "flat":
        # sigma_lo held constant down to threshold.  Crude, but it is the
        # correct *limit* for a Coulomb final state (see below), so it
        # brackets 'coulomb' from above and costs nothing to evaluate.
        return np.full(hv.shape, s_lo)

    if model == "wigner":
        # sigma ~ (hv - I)^(l' + 1/2), the threshold law for a SHORT-RANGE
        # final state.  Correct for the plane-wave continuum this model uses
        # elsewhere; NOT correct for a tabulated atomic sigma, whose residual
        # ion is charged.  The exponent follows the bound orbital's l.
        letter = ev.data.subshell[-1].lower()
        if letter not in _SUBSHELL_L:
            raise ConfigError(
                f"cannot infer the angular momentum of subshell "
                f"{ev.data.subshell!r} for the Wigner exponent; the label "
                f"must end in one of {sorted(_SUBSHELL_L)}"
            )
        p = _WIGNER_EXPONENT[_SUBSHELL_L[letter]]
        return s_lo * x ** p

    if model == "extrapolate":
        # Continue the log-log spline backwards.  ``CubicSpline`` is built
        # with ``extrapolate=False`` so that an accidental out-of-range
        # evaluation returns NaN rather than nonsense; here the extension is
        # deliberate, so build a one-off extrapolating view of the same
        # knots.  Anything non-finite (possible if the extension dives) is
        # clamped to the flat value, which is the nearest defensible answer.
        ext = CubicSpline(ev.log_hv, ev.log_sigma, extrapolate=True)
        with np.errstate(over="ignore"):
            out = np.exp(ext(np.log(hv)))
        return np.where(np.isfinite(out) & (out > 0.0), out, s_lo)

    # model == "coulomb": the exact hydrogenic (Stobbe) energy dependence,
    # scaled to match sigma_lo at hv_lo.  Only the SHAPE is hydrogenic; the
    # magnitude stays the tabulated one, so this does not import a hydrogenic
    # cross section into a many-electron subshell.
    #
    # The extrapolation is only trustworthy over a SHORT lever arm.  The
    # Stobbe shape is monotone falling, so if the first tabulated point
    # already sits past the subshell's maximum -- which for S_3p it does,
    # 61% above threshold with a local log-log slope of -2.5 -- then
    # extrapolating that shape backwards manufactures a rise (3.6x sigma_lo)
    # that the real curve does not have.  ``lever_arm`` records
    # (hv_lo - I)/I so a caller can see which groups are safe: it is 0.3-0.4%
    # for the core subshells that carry the intensity and 61% for S_3p.
    #
    #   sigma(x) ~ x^-4 exp(4 - 4 arctan(eta)/eta) / (1 - exp(-2 pi/eta)),
    #   x = hv/I,  eta = sqrt(x - 1)
    #
    # As eta -> 0 both exponential factors -> 1 and sigma -> sigma(I), finite.
    # This is why a tabulated atomic sigma is flat at threshold rather than
    # vanishing: the Coulomb penetration factor 2 pi nu/(1 - exp(-2 pi nu))
    # ~ 2 pi/k cancels the phase-space suppression.
    shape = _stobbe_shape(hv / thr)
    shape_lo = float(_stobbe_shape(np.asarray([ev.hv_lo / thr]))[0])
    if not np.isfinite(shape_lo) or shape_lo <= 0.0:
        return np.full(hv.shape, s_lo)
    return s_lo * shape / shape_lo


def _stobbe_shape(x: np.ndarray) -> np.ndarray:
    """Hydrogenic 1s photoionization energy dependence, up to a constant.

    ``x = hv / I >= 1``.  Returns
    ``x^-4 exp(4 - 4 arctan(eta)/eta) / (1 - exp(-2 pi/eta))`` with
    ``eta = sqrt(x - 1)``, continuous at ``x = 1`` where both exponential
    factors tend to 1 and the value is finite (this is the whole point --
    see :func:`_near_threshold`).  Verified against the literature H 1s
    threshold cross section, 6.30 Mb.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    eta = np.sqrt(np.maximum(x - 1.0, 0.0))
    out = np.empty(x.shape, dtype=float)
    tiny = eta < 1e-7
    out[tiny] = x[tiny] ** -4.0
    e = eta[~tiny]
    out[~tiny] = (x[~tiny] ** -4.0
                  * np.exp(4.0 - 4.0 * np.arctan(e) / e)
                  / (1.0 - np.exp(-2.0 * np.pi / e)))
    return out


def sigma(
    element: str,
    subshell: str,
    hv_ev: float | np.ndarray,
    high_energy_exponent: float | None = None,
    threshold_model: str = "linear",
    anchor_delta_ev: float | None = None,
) -> float | np.ndarray:
    """Atomic subshell photoionization cross section, Mb.

    Implements the piecewise evaluation of SPEC.md section 3 with four
    explicit regions:

    ``hv <= threshold``
        zero, exactly.
    ``threshold < hv < hv_lo`` (below the first usable tabulated point)
        one of four models selected by ``threshold_model``; all are
        continuous with the spline at ``hv_lo``.  This region is a
        **model**, not tabulated data, and none of the four captures
        autoionizing resonances.  Use :func:`sigma_region` (or
        :attr:`SigmaBuilder.used_linear_rise`) to detect reliance on it and
        say so in an output header.  See :data:`THRESHOLD_MODELS`.
    ``hv_lo <= hv <= hv_hi``
        cubic spline in ``(log hv, log sigma)``; exact at the knots.
    ``hv > hv_hi``
        power law ``sigma_hi * (hv / hv_hi) ** p`` continuous at
        ``hv_hi``, with ``p`` the slope of the last two tabulated points
        unless overridden.

    Parameters
    ----------
    element, subshell : str
        e.g. ``'F'``, ``'1s'``.
    hv_ev : float or ndarray
        Photon energy, eV.  Vectorised; the result has the input shape.
    threshold_model : str, optional
        ``'linear'`` (default, the historical behaviour), ``'flat'``,
        ``'coulomb'`` or ``'wigner'``; see :data:`THRESHOLD_MODELS`.  For a
        tabulated *atomic* cross section ``'coulomb'`` is the physically
        correct family and ``'wigner'`` is not, because the residual ion is
        charged; ``'wigner'`` is offered because the model's own continuum
        is a plane wave, for which it *is* correct.  Reported as a
        sensitivity, not resolved.
    high_energy_exponent : float, optional
        Replaces the fitted slope in the power-law region.  Yeh--Lindau
        stops at 1500 eV while the S 1s edge is used at 2610 eV, so the
        tail exponent is a modelling choice, not data;
        :data:`HYDROGENIC_EXPONENT` (-3.5) is the hydrogenic value.  A
        positive value is rejected, since a photoionization cross section
        cannot grow with photon energy far above threshold.

    Returns
    -------
    float or ndarray
        Cross section in Mb; float for scalar input.
    """
    ev = _evaluator(element, subshell, anchor_delta_ev)
    if high_energy_exponent is None:
        exponent = ev.default_exponent
    else:
        exponent = float(high_energy_exponent)
        if exponent >= 0.0:
            raise ConfigError(
                f"high_energy_exponent for {element} {subshell} must be "
                f"negative (got {exponent}); a subshell cross section "
                "cannot rise with photon energy far above threshold"
            )

    hv_in = np.asarray(hv_ev, dtype=float)
    scalar = hv_in.ndim == 0
    hv = np.atleast_1d(hv_in)
    out = np.zeros(hv.shape, dtype=float)

    codes = sigma_region(element, subshell, hv, threshold_model)

    rise = codes == REGION_NEAR_THRESHOLD
    if np.any(rise):
        out[rise] = _near_threshold(ev, hv[rise], threshold_model)

    inside = codes == REGION_TABULATED
    if np.any(inside):
        out[inside] = np.exp(ev.spline(np.log(hv[inside])))

    tail = codes == REGION_POWER_LAW
    if np.any(tail):
        out[tail] = np.exp(
            ev.log_sigma[-1]
            + exponent * (np.log(hv[tail]) - ev.log_hv[-1])
        )

    return float(out[0]) if scalar else out.reshape(hv_in.shape)


def provenance_note(keys: Sequence[str] | None = None) -> str:
    """One-line-per-subshell provenance string for an output header.

    Parameters
    ----------
    keys : sequence of str, optional
        Table keys (``'S_1s'``, ...).  Default: every entry.

    Returns
    -------
    str
        Newline-separated ``'<key>: <source>'`` lines, so a run that used
        the hydrogenic S 1s estimate says so in its own output.
    """
    chosen = sorted(YEH_LINDAU) if keys is None else list(keys)
    lines = []
    for key in chosen:
        if key not in YEH_LINDAU:
            raise ConfigError(f"unknown subshell key {key!r}")
        lines.append(f"{key}: {YEH_LINDAU[key].source}")
    return "\n".join(lines)


@dataclass
class _AoAssignment:
    """Bookkeeping for the AO -> (element, subshell) resolution."""

    keys: tuple[str | None, ...]
    threshold_ev: np.ndarray
    groups: dict[str, np.ndarray]
    unassigned: np.ndarray


class SigmaBuilder:
    """Per-AO atomic cross sections on an energy-sharing grid.

    Resolves the AO -> ``(element, subshell)`` assignment and the per-AO
    threshold :math:`I_\\mu` **once**, at construction, from the
    :class:`BasisSet` fields ``nbas``, ``elements``, ``l`` and
    ``shell_index``.  Evaluation then costs one spline call per distinct
    subshell per grid, not one per AO per quadrature point: the previous
    implementation reopened the HDF5 file and rebuilt this map at every
    quadrature point of every state (bug [C-4] in REVIEW.md, ~14400 file
    opens per run, all returning the same map).

    Parameters
    ----------
    basis : BasisSet
        Needs ``nbas``, ``elements`` (element symbol per AO), ``l``
        (angular momentum per AO) and ``shell_index`` (sequential index of
        the contracted shell within its ``(centre, l)`` group).
    subshell_map : mapping, optional
        ``{element: {(l, shell_index): subshell}}``.  Defaults to
        :data:`DEFAULT_SUBSHELL_MAP` (S and F in cc-pVDZ).
    high_energy_exponents : mapping, optional
        ``{'<element>_<subshell>': exponent}`` overriding the fitted
        power-law tail for individual subshells, e.g.
        ``{'F_1s': HYDROGENIC_EXPONENT}``.
    threshold_overrides : mapping, optional
        ``{'<element>_<subshell>': I_mu_ev}`` replacing the tabulated
        free-atom threshold.  The default :math:`I_\\mu` is the free-atom
        value that comes with the Yeh--Lindau entry, which is the choice
        consistent with substituting a free-atom :math:`\\sigma`; this
        override exists because that choice is *not* obviously right for
        valence subshells, whose binding energies shift on bonding, and
        because :math:`I_\\mu` enters twice -- as the :math:`\\sigma`
        argument and as the denominator of the
        ``omega_over_omega_eff`` weight -- so an error in it does not
        cancel.  Sensitivity is roughly 2% per eV at the F1s edge; see
        REVIEW.md [A-13].  Overriding shifts *both* uses consistently.

    Attributes
    ----------
    used_linear_rise : bool
        True once any evaluation has fallen in the near-threshold
        linear-rise stopgap region.  The driver reads this after
        integrating and notes it in the output header.
    n_unassigned : int
        Number of AOs with no tabulated atomic subshell (cc-pVDZ
        polarisation functions).  These get ``sigma = 0``; they are
        counted, not silently ignored.
    """

    def __init__(
        self,
        basis,
        subshell_map: Mapping[str, Mapping[tuple[int, int], str]] | None
        = None,
        high_energy_exponents: Mapping[str, float] | None = None,
        threshold_overrides: Mapping[str, float] | None = None,
        threshold_model: str = "linear",
        anchor_delta_ev: float = ANCHOR_DELTA_EV,
        photon_energy_ev: float | None = None,
    ) -> None:
        self.basis = basis
        self.anchor_delta_ev = float(anchor_delta_ev)
        #: Table keys zeroed because ``I_mu`` exceeded the photon energy on
        #: some call to :meth:`at_eps1_grid`.  A closed subshell is usually a
        #: sign that the photon energy and the state list belong to different
        #: edges.
        self.closed_subshells: set[str] = set()
        #: Photon energy in eV, when known.  Kept SEPARATE from the
        #: ``omega_ev`` argument of :meth:`at_eps1_grid`, which is the switch
        #: for the ``omega/omega_eff`` weight: energy conservation is not
        #: optional and must hold whether or not that weight is applied.
        #: ``None`` disables the check, for callers with no photon energy.
        self.omega_ev: float | None = (
            None if photon_energy_ev is None else float(photon_energy_ev))
        self.subshell_map = (
            DEFAULT_SUBSHELL_MAP if subshell_map is None
            else {k: dict(v) for k, v in subshell_map.items()}
        )
        self.high_energy_exponents = dict(high_energy_exponents or {})
        for key in self.high_energy_exponents:
            if key not in YEH_LINDAU:
                raise ConfigError(
                    f"high_energy_exponents names unknown subshell "
                    f"{key!r}; available: {sorted(YEH_LINDAU)}"
                )
        self.threshold_overrides = {
            k: float(v) for k, v in dict(threshold_overrides or {}).items()
        }
        for key, val in self.threshold_overrides.items():
            if key not in YEH_LINDAU:
                raise ConfigError(
                    f"threshold_overrides names unknown subshell "
                    f"{key!r}; available: {sorted(YEH_LINDAU)}"
                )
            if not np.isfinite(val) or val <= 0.0:
                raise ConfigError(
                    f"threshold_overrides[{key!r}] = {val!r} must be a "
                    f"positive, finite ionization threshold in eV"
                )
        if threshold_model not in THRESHOLD_MODELS:
            raise ConfigError(
                f"threshold_model must be one of {THRESHOLD_MODELS}, "
                f"got {threshold_model!r}"
            )
        self.threshold_model = str(threshold_model)
        self._assign = self._resolve(basis)
        self.used_linear_rise = False

    def lever_arm(self, key: str) -> float:
        """``(hv_lo - I_mu) / I_mu``: how far the table starts above threshold.

        The fraction of the threshold energy over which a near-threshold
        model has to extrapolate.  Small means the gap is a sliver of the
        curve and any smooth model is safe there; large means the first
        tabulated point may already be past the subshell maximum, in which
        case the monotone-falling ``coulomb`` shape extrapolates backwards
        through a peak it cannot see.

        Measured on the shipped table: 0.003 (S_1s), 0.004 (F_1s, S_2s),
        0.03 (S_2p), 0.06 (F_2s), 0.15 (F_2p), 0.21 (S_3s), 0.61 (S_3p).
        """
        entry = YEH_LINDAU[key]
        thr = self.threshold_of(key)
        if thr <= 0.0:
            return 0.0
        # Measure against the lower bound of the knot set the ACTIVE model
        # actually uses.  The main spline keeps points below ``threshold_ev``
        # (F 1s at 690.0 eV, S 2s at 225.0), so for those two the grid
        # straddles the threshold and the arm is zero -- but the ``anchored``
        # model discards them by construction (its step 1), which restores a
        # gap it must bridge.  Reporting zero there would hide exactly the
        # extrapolation ``anchored`` performs in its step 4.
        ev = _evaluator(entry.element, entry.subshell, self.anchor_delta_ev)
        lo = ev.hv_lo
        if self.threshold_model == "anchored":
            real = [h for h, sg in zip(entry.hv_ev, entry.sigma_mb)
                    if h > entry.threshold_ev and sg > 0.0]
            if real:
                lo = min(real)
        return max(0.0, (lo - thr) / thr)

    def unsafe_extrapolations(self, tol: float = 0.10,
                              disagreement: float = 0.15) -> tuple[str, ...]:
        """Groups whose near-threshold model should not be trusted.

        A group is flagged when **either** test fails:

        1. :meth:`lever_arm` exceeds ``tol`` -- the model must reach across
           more than ``tol`` of the threshold energy.  ``S_3p`` (arm 0.61)
           is the shipped table's example: it begins past its own maximum,
           so a monotone shape extrapolated backwards manufactures a rise.
        2. ``coulomb`` and ``extrapolate`` disagree at the threshold by more
           than ``disagreement`` in relative terms.  These two are
           independent -- one is the hydrogenic energy dependence, the other
           the tabulated curve's own local shape -- so when they agree the
           answer is corroborated and when they diverge neither can be
           trusted, whatever the lever arm says.

        The second test is not redundant.  ``S_2p`` has a lever arm of only
        0.028 and still fails it badly (ratio 0.14): its first two tabulated
        points *rise* 62%, a delayed maximum at the edge, so the local
        log-log slope at the first knot is ``+17`` and a backwards extension
        plunges.  A distance criterion alone cannot see that; comparing two
        independent estimates can.

        Empty for ``linear`` and ``flat``, which anchor only on ``sigma_lo``
        and so cannot manufacture structure, and for ``wigner``, whose
        exponent is fixed by the outgoing partial wave rather than fitted to
        the tabulated curve.

        On the shipped table this flags ``F_2p``, ``F_2s``, ``S_2p``,
        ``S_3p`` and ``S_3s``, leaving ``F_1s``, ``S_1s`` and ``S_2s`` --
        which carry 94.6% of an F1s-edge CV spectrum's intensity.
        """
        if self.threshold_model not in ("coulomb", "extrapolate", "anchored"):
            return ()
        flagged = []
        for key in self.subshells_used:
            if self.lever_arm(key) > tol:
                flagged.append(key)
                continue
            entry = YEH_LINDAU[key]
            thr = self.threshold_of(key)
            if self.lever_arm(key) == 0.0:
                continue                      # no gap: nothing is modelled
            # Compare two independent estimates *of the same quantity*.  For
            # ``anchored`` that quantity is its step-4 knot value, which is
            # the log-log extrapolation to ``I + delta``; ``sigma`` at the
            # threshold itself is zero by construction and so carries no
            # information about whether the extrapolation is sound.
            if self.threshold_model == "anchored":
                ev = _evaluator(entry.element, entry.subshell,
                                self.anchor_delta_ev)
                probe = ev.anchor_hv if ev.anchor_hv > thr else thr + 1e-9
            else:
                probe = thr + 1e-9
            a = float(sigma(entry.element, entry.subshell, probe, None,
                            threshold_model="coulomb",
                            anchor_delta_ev=self.anchor_delta_ev))
            b = float(sigma(entry.element, entry.subshell, probe, None,
                            threshold_model="extrapolate",
                            anchor_delta_ev=self.anchor_delta_ev))
            scale = max(abs(a), abs(b), 1e-30)
            if abs(a - b) / scale > disagreement:
                flagged.append(key)
        return tuple(sorted(flagged))

    def threshold_of(self, key: str) -> float:
        """Threshold :math:`I_\\mu` in use for a table key, eV.

        The override if one was supplied, else the tabulated free-atom
        value.  Every place that needs :math:`I_\\mu` goes through here, so
        an override cannot be applied to one of its two uses and not the
        other.
        """
        if key in self.threshold_overrides:
            return self.threshold_overrides[key]
        return float(YEH_LINDAU[key].threshold_ev)

    def _resolve(self, basis) -> _AoAssignment:
        for attr in ("nbas", "elements", "l", "shell_index"):
            if not hasattr(basis, attr):
                raise ConfigError(
                    f"BasisSet is missing the {attr!r} field required to "
                    "assign atomic subshells to AOs (SPEC.md section 2)"
                )
        nbas = int(basis.nbas)
        elements = list(basis.elements)
        l_arr = np.asarray(basis.l, dtype=int)
        shell_arr = np.asarray(basis.shell_index, dtype=int)
        for name, arr in (("elements", elements), ("l", l_arr),
                          ("shell_index", shell_arr)):
            if len(arr) != nbas:
                raise ConfigError(
                    f"BasisSet.{name} has length {len(arr)} but nbas is "
                    f"{nbas}"
                )

        keys: list[str | None] = []
        thresholds = np.zeros(nbas, dtype=float)
        groups: dict[str, list[int]] = {}
        unassigned: list[int] = []
        for mu in range(nbas):
            elem = str(elements[mu])
            per_elem = self.subshell_map.get(elem, {})
            sub = per_elem.get((int(l_arr[mu]), int(shell_arr[mu])))
            if sub is None:
                keys.append(None)
                unassigned.append(mu)
                continue
            key = f"{elem}_{sub}"
            if key not in YEH_LINDAU:
                raise ConfigError(
                    f"subshell_map assigns AO {mu} (element {elem}, "
                    f"l={int(l_arr[mu])}, shell_index="
                    f"{int(shell_arr[mu])}) to {key!r}, which has no "
                    f"tabulated cross section; available: "
                    f"{sorted(YEH_LINDAU)}"
                )
            keys.append(key)
            thresholds[mu] = self.threshold_of(key)
            groups.setdefault(key, []).append(mu)

        return _AoAssignment(
            keys=tuple(keys),
            threshold_ev=thresholds,
            groups={k: np.asarray(v, dtype=int) for k, v in groups.items()},
            unassigned=np.asarray(unassigned, dtype=int),
        )

    @property
    def nbas(self) -> int:
        """Number of contracted AOs."""
        return int(self.basis.nbas)

    @property
    def threshold_ev(self) -> np.ndarray:
        """Per-AO atomic threshold :math:`I_\\mu`, eV; 0 where unassigned."""
        return self._assign.threshold_ev.copy()

    @property
    def ao_keys(self) -> tuple[str | None, ...]:
        """Per-AO table key, ``None`` for unassigned AOs."""
        return self._assign.keys

    @property
    def n_unassigned(self) -> int:
        """Number of AOs with no tabulated atomic subshell."""
        return int(self._assign.unassigned.size)

    @property
    def unassigned_indices(self) -> np.ndarray:
        """Indices of the AOs that get ``sigma = 0``."""
        return self._assign.unassigned.copy()

    @property
    def subshells_used(self) -> tuple[str, ...]:
        """Table keys actually reached by this basis, sorted."""
        return tuple(sorted(self._assign.groups))

    def at_eps1_grid(self, eps1_ev: np.ndarray,
                     omega_ev: float | None = None) -> np.ndarray:
        """Per-AO cross sections on a whole energy-sharing grid.

        Evaluates :math:`\\sigma_\\mu(\\varepsilon_1 + I_\\mu)` -- see
        Remark 1 in the module docstring for why the argument is the
        *atomic* threshold and not :math:`\\omega - \\varepsilon_2`.

        Parameters
        ----------
        eps1_ev : ndarray, shape (nq,)
            Kinetic energy of the fast photoelectron at each quadrature
            point, eV.
        omega_ev : float, optional
            Photon energy, eV.  When given, each AO column is multiplied by
            ``omega_ev / (eps1 + I_mu)``, the residual normalisation factor
            left over when the tabulated ``sigma^AO`` replaces the computed
            dipole matrix element (REVIEW.md [A-8]).  When ``None`` the
            weight is omitted, reproducing the behaviour of every result
            obtained before this option existed.

        Returns
        -------
        ndarray, shape (nq, nbas)
            Cross sections in Mb.  Columns for unassigned AOs are zero.
        """
        eps1 = np.atleast_1d(np.asarray(eps1_ev, dtype=float))
        if eps1.ndim != 1:
            raise ConfigError(
                f"eps1_ev must be 1-D, got shape {eps1.shape}"
            )
        if omega_ev is not None and not np.isfinite(omega_ev):
            raise ConfigError(
                f"omega_ev must be finite when given, got {omega_ev!r}"
            )
        out = np.zeros((eps1.size, self.nbas), dtype=float)
        for key, cols in self._assign.groups.items():
            entry = YEH_LINDAU[key]
            # I_mu via threshold_of, so a [physics] threshold_override moves
            # the sigma argument AND the omega/omega_eff denominator below by
            # the same amount.  Note that it deliberately does *not* move the
            # tabulated curve's own threshold: sigma^AO is a free-atom
            # property and keeps its free-atom onset, while I_mu is the
            # offset between eps1 and the photon energy at which that curve
            # is read.  Overriding one without the other would be incoherent.
            hv = eps1 + self.threshold_of(key)
            vals = sigma(
                entry.element, entry.subshell, hv,
                self.high_energy_exponents.get(key),
                threshold_model=self.threshold_model,
                anchor_delta_ev=self.anchor_delta_ev,
            )
            if np.any(
                sigma_region(entry.element, entry.subshell, hv,
                             self.threshold_model)
                == REGION_LINEAR_RISE
            ):
                self.used_linear_rise = True
            if self.omega_ev is not None and float(
                    self.omega_ev) < entry.threshold_ev:
                # ENERGY CONSERVATION against the real photon.  sigma is read
                # at hv = eps1 + I_mu, which is >= I_mu by construction, so
                # without this test every subshell looks open however small
                # omega is: an AO whose threshold exceeds the photon energy
                # still receives the cross section it would have at its OWN
                # onset.  The physical value is exactly zero -- the photon
                # cannot ionise that subshell at all.
                #
                # The symptom is visible in the weight: omega/omega_eff(mu)
                # < 1 requires omega < eps1 + I_mu, and since eps1 >= 0 the
                # only way that can hold across the whole grid is omega <
                # I_mu.  So a sub-unity weight IS the signature of a closed
                # channel, which is how this was found.  See REVIEW.md [A-18].
                out[:, cols] = 0.0
                self.closed_subshells.add(key)
                continue
            if omega_ev is not None:
                # Substituting a *tabulated* sigma^AO for the computed dipole
                # matrix element imports sigma's own normalisation: inverting
                # its definition (notes Eq. 93, with
                # omega_eff of Definition 1 = Eq.
                # (92)) via Eq. (99) yields
                #
                #   INT dOmega_k1 sum_alpha |<psi_k1|r_alpha|chi_mu>|^2
                #       = sigma_mu(w_eff) * 3c / (4 pi^2 w_eff k_1)
                #
                # The 1/k_1 cancels the sDCS prefactor's k_1 and the 1/3
                # cancels its polarisation average -- Eq. (94),
                # applied once only (Remark 3) -- but the
                # prefactor carries
                # the MOLECULAR omega while sigma carries the PER-AO
                # w_eff = eps1 + I_mu.  Their ratio does not cancel and is a
                # function of eps1, so it cannot be hoisted into a scalar
                # prefactor -- it belongs here, inside the per-AO loop, where
                # w_eff is known.  See NORMALISATION.md and REVIEW.md [A-8].
                out[:, cols] = (vals * (float(omega_ev) / hv))[:, None]
            else:
                # One spline evaluation for the whole grid, broadcast to every
                # AO of this subshell: sigma depends on the AO only through
                # (element, subshell), so AOs in a group share a column value.
                out[:, cols] = vals[:, None]
        return out

    def at_eps1(self, eps1_ev: float,
                omega_ev: float | None = None) -> np.ndarray:
        """Per-AO cross sections at one energy sharing.

        Parameters
        ----------
        eps1_ev : float
            Kinetic energy of the fast photoelectron, eV.

        Returns
        -------
        ndarray, shape (nbas,)
            Cross sections in Mb.
        """
        return self.at_eps1_grid(np.asarray([float(eps1_ev)]),
                                 omega_ev=omega_ev)[0]

    def report(self) -> dict[str, object]:
        """Summary for an output header.

        Returns
        -------
        dict
            ``nbas``, ``n_assigned``, ``n_unassigned``,
            ``unassigned_indices``, ``subshells_used``,
            ``used_linear_rise``, ``thresholds_ev``,
            ``threshold_overrides`` and ``provenance``.  The thresholds are
            reported because :math:`I_\\mu` is a modelling choice with a
            ~2%/eV effect (REVIEW.md [A-13]), so a spectrum is not
            reproducible without them.
        """
        return {
            "nbas": self.nbas,
            "n_assigned": self.nbas - self.n_unassigned,
            "n_unassigned": self.n_unassigned,
            "unassigned_indices": self.unassigned_indices.tolist(),
            "subshells_used": self.subshells_used,
            "used_linear_rise": self.used_linear_rise,
            "threshold_model": self.threshold_model,
            "anchor_delta_ev": self.anchor_delta_ev,
            "lever_arm": {k: self.lever_arm(k) for k in self.subshells_used},
            "unsafe_extrapolations": self.unsafe_extrapolations(),
            "thresholds_ev": {k: self.threshold_of(k)
                              for k in self.subshells_used},
            "threshold_overrides": dict(self.threshold_overrides),
            "provenance": provenance_note(self.subshells_used),
        }
