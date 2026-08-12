#!/usr/bin/env python3
"""Post-process a dpi calculation into a figure.

Reads only the text files written by ``dpi_run.py`` plus an optional
two-column experimental spectrum.  It never opens an HDF5 file, imports
``dpi.dyson`` or recomputes any physics, so replotting is instant and the
figure can be regenerated from an archived pair of .dat files long after
the calculation itself.

    python dpi_plot.py out/spectrum.dat
    python dpi_plot.py out/spectrum.dat --sticks out/sticks.dat \\
        --experiment exp_s1s.dat --xlim 14 46 --output s1s.png

Panels
------
top     One equal-height stick per state, coloured by channel, with the
        state-density envelope drawn above it.  Neither carries the
        intensity model: the panel answers *where are the levels*, and
        the lower panel answers *how strong are they*, so a band that is
        strong below but sparse here is carried by a few intense states
        rather than by a dense manifold.
bottom  The broadened spectrum: one curve per channel, their sum, the
        optional frozen-orbital reference and the optional measurement.

Omit ``--sticks`` for a single-panel figure of the broadened spectrum.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Figure geometry.  Near-unity aspect: the two-panel form is read as a
# single square block in a one-column journal layout rather than as a wide
# strip, which is what the extra width used to make of it.
FIGSIZE_TWO_PANEL = (7.6, 7.2)
FIGSIZE_ONE_PANEL = (7.6, 5.4)

# Font sizes, collected here so the whole figure can be rescaled for a
# different column width by editing five numbers.  The x tick numbers are
# deliberately the largest text: they are the only quantitative axis and
# are what a reader checks first.
FONT_TICK_X = 15
FONT_TICK_Y = 11
FONT_AXIS_LABEL = 12
FONT_LEGEND = 10
FONT_ANNOTATION = 15

# Channel colours.  A deep-teal / MAGENTA pair on the blue-yellow axis,
# which is the axis that survives red-green colour blindness; the default
# blue/orange is avoided.  Relative luminances are 0.138 and 0.500, a
# factor 3.6 apart, so the two curves also separate in greyscale print.
CHANNEL_TEAL = "#16A8C2"     # deep teal, low luminance, reads as the dark
CHANNEL_MAGENTA = "#E8298C"  # MAGENTA, high luminance, reads as the light

# Extra hues for channel names the table below does not know about,
# ordered so a two-channel fallback pair is still luminance-separated.
ACCENT_SLATE = "#3B6E8F"   # muted slate blue, distinct from CHANNEL_TEAL
ACCENT_ROSE = "#C2456E"    # desaturated rose, mid luminance
ACCENT_OLIVE = "#8A8B2C"   # olive, high luminance without reading yellow
ACCENT_VIOLET = "#6A3D9A"  # violet, darkest of the four
FALLBACK_COLOURS = (ACCENT_SLATE, ACCENT_ROSE, ACCENT_OLIVE, ACCENT_VIOLET)

# The three non-channel traces each keep a distinct treatment (colour plus
# line style or marker) so they never compete with the channel hues.
TOTAL_COLOUR = "#22282D"       # graphite; a dashed near-black for the sum
FROZEN_COLOUR = "#95A0A8"      # cool grey; a dotted reference, not a datum
# Sienna for the experimental overlay.  Two constraints: it must not duplicate
# any channel or fallback colour (it previously carried ACCENT_VIOLET's exact
# hex, so a 4th/unrecognised channel would have collided with it), and it must
# separate from TOTAL_COLOUR in greyscale -- near-black failed that at a 1.17x
# luminance ratio, while this gives 2.21x.  The overlay is additionally drawn
# as markers and never as a line, so shape carries the distinction too.
EXPERIMENT_COLOUR = "#140EB4"  # sienna; drawn as markers, never a line
DENSITY_COLOUR = "#41576A"     # slate grey; dashed level-counting curve

# Stick geometry.  Every state gets the SAME height: the panel is a level
# map, so drawing physical intensities here would duplicate the lower
# panel and hide the sparse-but-intense bands.  The density curve is
# normalised to the same height and offset above the sticks.
STICK_HEIGHT = 0.5
STICK_WIDTH = 2.0
DENSITY_AMPLITUDE = 1.9 * STICK_HEIGHT
DENSITY_BASELINE = 1.2 * STICK_HEIGHT
STICK_PANEL_TOP = 3.85 * STICK_HEIGHT

CHANNEL_STYLE = {
    "singlet": (CHANNEL_TEAL, "Singlets"),
    "triplet": (CHANNEL_MAGENTA, "Triplets"),
    "jc32": (CHANNEL_TEAL, r"$j_C = 3/2$"),
    "jc12": (CHANNEL_MAGENTA, r"$j_C = 1/2$"),
}


def read_table(path: str) -> tuple[list[str], np.ndarray, list[str]]:
    """Read a dpi .dat file.

    The format is a comment block of ``#``-prefixed lines, one of which is
    ``# columns: <name> <name> ...``, followed by whitespace-separated
    numeric rows.  Non-numeric fields (the channel and term labels in
    ``sticks.dat``) are returned as NaN in the array; use
    :func:`read_sticks` for that file instead.

    Returns
    -------
    columns
        Column names taken from the ``# columns:`` line, or positional
        names if that line is absent.
    data
        Array of shape ``(nrow, ncol)``.
    header
        The comment lines verbatim, without the leading ``#``.
    """
    if not os.path.isfile(path):
        raise SystemExit(f"dpi_plot: file not found: {path}")

    header: list[str] = []
    columns: list[str] = []
    rows: list[list[float]] = []
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                body = stripped.lstrip("#").strip()
                header.append(body)
                if body.lower().startswith("columns:"):
                    columns = body.split(":", 1)[1].split()
                continue
            fields = stripped.replace(",", " ").split()
            rows.append([_as_float(f) for f in fields])

    if not rows:
        raise SystemExit(f"dpi_plot: {path} contains no data rows.")
    width = max(len(r) for r in rows)
    data = np.full((len(rows), width), np.nan)
    for i, row in enumerate(rows):
        data[i, :len(row)] = row
    if not columns:
        columns = [f"col{i}" for i in range(width)]
    return columns, data, header


def _as_float(field: str) -> float:
    try:
        return float(field.replace("D", "E").replace("d", "e"))
    except ValueError:
        return np.nan


def read_sticks(path: str) -> tuple[list[str], list[dict[str, object]]]:
    """Read ``sticks.dat``, keeping text columns as strings.

    Returns the column names and one dict per state, values converted to
    float where possible and left as strings otherwise.
    """
    columns, _, _ = read_table(path)
    records: list[dict[str, object]] = []
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            record: dict[str, object] = {}
            for name, field in zip(columns, fields):
                try:
                    record[name] = float(field.replace("D", "E"))
                except ValueError:
                    record[name] = field
            records.append(record)
    return columns, records


def read_experiment(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a two-column (energy_eV, intensity) measured spectrum."""
    energy, signal = [], []
    with open(path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped[0] in "#%":
                continue
            fields = stripped.replace(",", " ").split()
            if len(fields) < 2:
                continue
            try:
                energy.append(float(fields[0]))
                signal.append(float(fields[1]))
            except ValueError:
                continue
    if not energy:
        raise SystemExit(f"dpi_plot: no usable rows in {path}")
    return np.asarray(energy), np.asarray(signal)


def channel_columns(columns: list[str]) -> list[str]:
    """Spectrum columns that are physical channels, in file order."""
    skip = {"e_shifted_ev", "total", "frozen_total"}
    return [c for c in columns[1:]
            if c.lower() not in skip and not c.lower().startswith("frozen")]


def style_for(channel: str, index: int) -> tuple[str, str]:
    """Colour and legend label for a channel name."""
    if channel in CHANNEL_STYLE:
        return CHANNEL_STYLE[channel]
    colour = FALLBACK_COLOURS[index % len(FALLBACK_COLOURS)]
    return colour, channel.replace("_", " ")


def frame_axes(ax, label_bottom: bool = True) -> None:
    """Close the panel with inward ticks on all four edges.

    The convention in this field is a boxed panel with the ticks pointing
    into the data, so a reader can read a peak position off any edge.
    Numbers stay on the bottom and are suppressed on the other three: the
    y scale is arbitrary units, so its ticks are drawn unlabelled rather
    than deleted, which keeps the frame complete without implying a
    meaningful magnitude.
    """
    from matplotlib.ticker import AutoMinorLocator

    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.1)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", direction="in",
                   top=True, bottom=True, left=True, right=True,
                   labeltop=False, labelright=False, labelleft=False)
    ax.tick_params(which="major", length=7.0, width=1.1,
                   labelsize=FONT_TICK_Y)
    ax.tick_params(which="minor", length=3.5, width=0.9)
    ax.tick_params(axis="x", which="major", labelsize=FONT_TICK_X,
                   labelbottom=label_bottom)


def scale_experiment(exp_energy: np.ndarray, exp_signal: np.ndarray,
                     grid: np.ndarray, model: np.ndarray,
                     offset_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Match the measured trace to the model's dynamic range and offset it.

    The calculation yields relative intensities, so the comparison is one of
    band positions and relative heights, not absolute magnitude.  The trace
    is affinely mapped onto the model's range over the overlapping energy
    window and then displaced vertically by ``offset_fraction`` of the model
    maximum so the two curves can be read separately.  A vertical offset of
    zero overlays them instead.
    """
    inside = (exp_energy >= grid.min()) & (exp_energy <= grid.max())
    exp_energy, exp_signal = exp_energy[inside], exp_signal[inside]
    if exp_energy.size == 0:
        return exp_energy, exp_signal

    model_at_exp = np.interp(exp_energy, grid, model)
    exp_span = np.ptp(exp_signal)
    model_span = np.ptp(model_at_exp)
    if exp_span > 0 and model_span > 0:
        scaled = ((exp_signal - exp_signal.min()) / exp_span * model_span
                  + model_at_exp.min())
    else:
        scaled = exp_signal.astype(float)
    scaled = scaled + offset_fraction * float(np.max(model))
    return exp_energy, scaled


def build_figure(args: argparse.Namespace):
    """Assemble the figure; returns the matplotlib Figure."""
    import matplotlib
    if args.output:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns, data, header = read_table(args.spectrum)
    grid = data[:, 0]
    by_name = {name.lower(): data[:, i] for i, name in enumerate(columns)}
    channels = channel_columns(columns)

    total = by_name.get("total")
    if total is None:
        total = sum(by_name[c.lower()] for c in channels)

    sticks = None
    if args.sticks:
        _, sticks = read_sticks(args.sticks)

    xlo, xhi = (args.xlim if args.xlim
                else (float(grid.min()), float(grid.max())))
    if args.xlim:
        # Matplotlib happily inverts an axis given a reversed pair, and a
        # window outside the data produces a blank figure -- both look like
        # the script silently failed, so say what happened instead.
        if xlo >= xhi:
            raise SystemExit(
                f"dpi_plot: --xlim {xlo:g} {xhi:g} is empty or reversed; "
                f"give it as LOW HIGH.")
        if xhi < grid.min() or xlo > grid.max():
            raise SystemExit(
                f"dpi_plot: --xlim {xlo:g} {xhi:g} lies outside the "
                f"spectrum, which spans {grid.min():.2f} to "
                f"{grid.max():.2f} eV.\n"
                f"  (that range is display_onset_ev plus each state's "
                f"spacing, not a raw dication energy)")

    if sticks:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=FIGSIZE_TWO_PANEL, sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.45], "hspace": 0.08})
        draw_sticks(ax_top, sticks, grid, channels, args, xlo, xhi)
    else:
        fig, ax_bot = plt.subplots(figsize=FIGSIZE_ONE_PANEL)
        ax_top = None

    draw_spectrum(ax_bot, grid, by_name, channels, total, args, xlo, xhi)

    label = args.label or header_value(header, "label") or ""
    target = ax_top if ax_top is not None else ax_bot
    if label:
        target.text(0.025, 0.955, label, transform=target.transAxes,
                    fontsize=FONT_ANNOTATION, va="top")

    ax_bot.set_xlabel("Shifted double ionization energy (eV)",
                      fontsize=FONT_AXIS_LABEL)
    ax_bot.set_xlim(xlo, xhi)
    fig.align_ylabels()
    return fig


def draw_sticks(ax, sticks, grid, channels, args, xlo, xhi) -> None:
    """Top panel: equal-height state sticks plus the state-density curve."""
    energy_key = _first_key(sticks[0], ("e_shifted_ev", "energy_ev"))

    for index, channel in enumerate(channels):
        colour, label = style_for(channel, index)
        # Height carries no information, so a single vlines call per
        # channel is enough and guarantees the sticks cannot drift apart.
        energies = [float(s[energy_key]) for s in sticks
                    if str(s.get("channel", "")) == channel
                    and _visible(s, energy_key, xlo, xhi)]
        if energies:
            ax.vlines(energies, 0.0, STICK_HEIGHT, color=colour,
                      lw=STICK_WIDTH, label=label, zorder=3)

    # Each state contributes a Voigt profile of unit AREA, so the envelope
    # counts levels at equal weight exactly as the sticks do -- the two
    # halves of the panel now say the same thing at different resolution.
    density = np.zeros_like(grid)
    for row in sticks:
        density += voigt(grid, float(row[energy_key]), 1.0,
                         args.sigma, args.gamma)
    # Normalise over the VISIBLE window only. Every state contributes to the
    # curve (a state just outside the window still has tails inside it), but
    # dividing by a global maximum that lies outside a zoomed view would
    # flatten the curve to a fraction of the panel.
    in_view = (grid >= xlo) & (grid <= xhi)
    scale = density[in_view].max() if in_view.any() else density.max()
    if scale > 0:
        density = density / scale * DENSITY_AMPLITUDE
    ax.plot(grid, density + DENSITY_BASELINE, color=DENSITY_COLOUR, lw=1.6,
            ls="--", label="conv. DOS", zorder=2)

    ax.set_ylim(0.0, STICK_PANEL_TOP)
    ax.set_ylabel("States (equal weight)", fontsize=FONT_AXIS_LABEL)
    ax.legend(loc="upper right", frameon=False, fontsize=FONT_LEGEND,
              handlelength=1.6, borderaxespad=0.5)
    frame_axes(ax, label_bottom=False)


def draw_spectrum(ax, grid, by_name, channels, total, args, xlo, xhi) -> None:
    """Bottom panel: broadened per-channel curves, their sum and overlays."""
    for index, channel in enumerate(channels):
        colour, label = style_for(channel, index)
        ax.plot(grid, by_name[channel.lower()], color=colour, lw=2.0,
                label=label, zorder=3)

    ax.plot(grid, total, color=TOTAL_COLOUR, lw=1.8, ls="--", label="Sum",
            zorder=4)

    # Overlay curves, collected as they are drawn so the y-scaling below can
    # take their visible extent into account.
    overlays: list[np.ndarray] = []

    frozen = by_name.get("frozen_total")
    if frozen is not None and np.nanmax(frozen) > 0:
        # Normalised to the relaxed sum: the frozen limit is a shape
        # reference for how much structure orbital relaxation supplies, not
        # an independent intensity prediction.
        scaled = frozen * (np.nanmax(total) / np.nanmax(frozen))
        ax.plot(grid, scaled, color=FROZEN_COLOUR, lw=1.9, ls=":",
                label="Frozen-orbital limit", zorder=2)
        overlays.append(scaled)

    exp_energy = np.empty(0)
    exp_scaled = np.empty(0)
    if args.experiment:
        exp_energy, exp_signal = read_experiment(args.experiment)
        exp_energy, exp_scaled = scale_experiment(
            exp_energy, exp_signal, grid, total, args.exp_offset)
        if exp_energy.size:
            ax.scatter(exp_energy, exp_scaled, s=11,
                       color=EXPERIMENT_COLOUR, alpha=0.85, zorder=5,
                       label="Experiment")

    # Scale the y-axis to what is VISIBLE, not to the whole spectrum.
    # Matplotlib's autoscale uses every plotted point, so zooming into a weak
    # region with --xlim would otherwise leave the curves flat against the
    # axis while a strong band outside the window sets the scale. Headroom
    # above the visible maximum leaves room for the legend.
    visible = (grid >= xlo) & (grid <= xhi)
    if visible.any():
        curves = [total] + [by_name[c.lower()] for c in channels] + overlays
        top = max(float(np.nanmax(np.asarray(c)[visible])) for c in curves)
        if exp_energy.size:
            in_view = (exp_energy >= xlo) & (exp_energy <= xhi)
            if in_view.any():
                top = max(top, float(np.nanmax(exp_scaled[in_view])))
        if top > 0:
            ax.set_ylim(-0.04 * top, 1.45 * top)

    ax.set_ylabel("Intensity (arb. units)", fontsize=FONT_AXIS_LABEL)
    ax.set_xlim(xlo, xhi)
    ax.legend(loc="upper left", frameon=False, fontsize=FONT_LEGEND,
              handlelength=1.6, borderaxespad=0.5)
    frame_axes(ax, label_bottom=True)


def voigt(x: np.ndarray, centre: float, area: float,
          sigma: float, gamma: float) -> np.ndarray:
    """Area-normalised Voigt profile.

    Duplicated from ``dpi.spectrum`` on purpose: this script must run
    against nothing but the .dat files and numpy/scipy, so that an archived
    output pair remains plottable without the package.
    """
    from scipy.special import wofz
    if sigma <= 0:
        sigma = 1e-6
    z = ((x - centre) + 1j * gamma) / (sigma * np.sqrt(2.0))
    return area * np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))


def header_value(header: list[str], key: str) -> str | None:
    """Look up a ``key : value`` line in the .dat comment block."""
    for line in header:
        if ":" in line:
            name, _, value = line.partition(":")
            if name.strip().lower().startswith(key.lower()):
                return value.strip()
    return None


def _first_key(record: dict, candidates: tuple[str, ...]) -> str:
    lowered = {k.lower(): k for k in record}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    raise SystemExit(
        f"dpi_plot: sticks file has none of the expected columns "
        f"{candidates}; found {sorted(record)}")


def _visible(record: dict, energy_key: str, xlo: float, xhi: float) -> bool:
    try:
        return xlo <= float(record[energy_key]) <= xhi
    except (KeyError, TypeError, ValueError):
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a dpi spectrum from its .dat output files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("spectrum",
                        help="spectrum.dat written by dpi_run.py")
    parser.add_argument("--sticks", metavar="FILE",
                        help="sticks.dat; adds the per-state stick panel")
    parser.add_argument("--experiment", metavar="FILE",
                        help="two-column (energy_eV, intensity) measurement")
    parser.add_argument("--output", "-o", metavar="FILE",
                        help="save here instead of opening a window")
    parser.add_argument("--xlim", nargs=2, type=float,
                        metavar=("LO", "HI"),
                        help="energy window, eV (default: the whole grid)")
    parser.add_argument("--label", help="annotation for the upper panel")
    parser.add_argument("--sigma", type=float, default=0.48,
                        help="Gaussian width for the state-density curve, eV")
    parser.add_argument("--gamma", type=float, default=0.23,
                        help="Lorentzian HWHM for the state-density curve, eV")
    parser.add_argument("--exp-offset", type=float, default=0.30,
                        metavar="FRAC",
                        help="vertical displacement of the measured trace, "
                             "as a fraction of the model maximum; 0 overlays")
    parser.add_argument("--dpi", type=int, default=300,
                        help="raster resolution when saving")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fig = build_figure(args)
    if args.output:
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"figure written: {args.output}")
    else:
        import matplotlib.pyplot as plt
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
