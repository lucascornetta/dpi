"""Tabulated atomic subshell photoionization cross sections, in Mb.

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

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy.interpolate import CubicSpline

from .constants import ConfigError, ModelError

__all__ = [
    "SubshellData",
    "YEH_LINDAU",
    "DEFAULT_SUBSHELL_MAP",
    "HYDROGENIC_EXPONENT",
    "REGION_BELOW_THRESHOLD",
    "REGION_LINEAR_RISE",
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
# evaluation fell in REGION_LINEAR_RISE, because that region is a stopgap
# (see _rise_region below) rather than tabulated data.
REGION_BELOW_THRESHOLD = 0
REGION_LINEAR_RISE = 1
REGION_TABULATED = 2
REGION_POWER_LAW = 3

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


_EVALUATORS: dict[str, _Evaluator] = {}


def _entry(element: str, subshell: str) -> SubshellData:
    key = f"{element}_{subshell}"
    try:
        return YEH_LINDAU[key]
    except KeyError:
        raise ConfigError(
            f"no atomic cross section tabulated for {element} {subshell} "
            f"(key {key!r}); available keys: {sorted(YEH_LINDAU)}"
        ) from None


def _evaluator(element: str, subshell: str) -> _Evaluator:
    """Build (once) the log-log spline and power-law tail for a subshell.

    Tabulated points at or below the threshold are discarded: the cross
    section is identically zero there, so such points are digitisation
    artefacts of the published curves (F 1s and S 2s each have one).
    Keeping them would let the spline return a finite value below
    threshold.
    """
    key = f"{element}_{subshell}"
    cached = _EVALUATORS.get(key)
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

    keep = (hv > data.threshold_ev) & (sig > 0.0)
    n_dropped = int(hv.size - keep.sum())
    hv, sig = hv[keep], sig[keep]
    if hv.size < 4:
        raise ModelError(
            f"{key}: only {hv.size} usable tabulated points above the "
            f"{data.threshold_ev} eV threshold; a cubic spline needs 4"
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
    element: str, subshell: str, hv_ev: float | np.ndarray
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
    return codes


def sigma(
    element: str,
    subshell: str,
    hv_ev: float | np.ndarray,
    high_energy_exponent: float | None = None,
) -> float | np.ndarray:
    """Atomic subshell photoionization cross section, Mb.

    Implements the piecewise evaluation of SPEC.md section 3 with four
    explicit regions:

    ``hv <= threshold``
        zero, exactly.
    ``threshold < hv < hv_lo`` (below the first usable tabulated point)
        linear rise, ``sigma_lo * (hv - threshold) / (hv_lo - threshold)``.
        This is a **stopgap**, not physics: the true near-threshold shape
        depends on the centrifugal barrier and on autoionizing resonances,
        neither of which is in the tabulation.  Use
        :func:`sigma_region` (or :attr:`SigmaBuilder.used_linear_rise`) to
        detect reliance on it and say so in an output header.
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
    ev = _evaluator(element, subshell)
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

    codes = sigma_region(element, subshell, hv)

    rise = codes == REGION_LINEAR_RISE
    if np.any(rise):
        thr = ev.data.threshold_ev
        out[rise] = ev.sigma_lo * (hv[rise] - thr) / (ev.hv_lo - thr)

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
    ) -> None:
        self.basis = basis
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
        self._assign = self._resolve(basis)
        self.used_linear_rise = False

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
            thresholds[mu] = YEH_LINDAU[key].threshold_ev
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

    def at_eps1_grid(self, eps1_ev: np.ndarray) -> np.ndarray:
        """Per-AO cross sections on a whole energy-sharing grid.

        Evaluates :math:`\\sigma_\\mu(\\varepsilon_1 + I_\\mu)` -- see
        Remark 1 in the module docstring for why the argument is the
        *atomic* threshold and not :math:`\\omega - \\varepsilon_2`.

        Parameters
        ----------
        eps1_ev : ndarray, shape (nq,)
            Kinetic energy of the fast photoelectron at each quadrature
            point, eV.

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
        out = np.zeros((eps1.size, self.nbas), dtype=float)
        for key, cols in self._assign.groups.items():
            entry = YEH_LINDAU[key]
            hv = eps1 + entry.threshold_ev
            vals = sigma(
                entry.element, entry.subshell, hv,
                self.high_energy_exponents.get(key),
            )
            if np.any(
                sigma_region(entry.element, entry.subshell, hv)
                == REGION_LINEAR_RISE
            ):
                self.used_linear_rise = True
            # One spline evaluation for the whole grid, broadcast to every
            # AO of this subshell: sigma depends on the AO only through
            # (element, subshell), so AOs in a group share a column value.
            out[:, cols] = vals[:, None] # type: ignore
        return out

    def at_eps1(self, eps1_ev: float) -> np.ndarray:
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
        return self.at_eps1_grid(np.asarray([float(eps1_ev)]))[0]

    def report(self) -> dict[str, object]:
        """Summary for an output header.

        Returns
        -------
        dict
            ``nbas``, ``n_assigned``, ``n_unassigned``,
            ``unassigned_indices``, ``subshells_used``,
            ``used_linear_rise`` and ``provenance``.
        """
        return {
            "nbas": self.nbas,
            "n_assigned": self.nbas - self.n_unassigned,
            "n_unassigned": self.n_unassigned,
            "unassigned_indices": self.unassigned_indices.tolist(),
            "subshells_used": self.subshells_used,
            "used_linear_rise": self.used_linear_rise,
            "provenance": provenance_note(self.subshells_used),
        }
