"""Writers and readers for the ``dpi`` output files.

Formats are fixed by SPEC.md section 8.  Two rules govern everything here:

* a ``.dat`` file must be reproducible from its own header, so every active
  setting, the code version and an ISO-8601 timestamp are written as
  comment lines;
* every writer has a matching reader, so the formats are round-trippable.
  ``dpi_plot.py`` and the test suite both consume the readers, and the
  round trip is itself a test.

Files are whitespace-delimited text.  Free-text fields (the channel and the
term label of ``sticks.dat``) are written as single tokens with internal
whitespace replaced by ``_`` and an absent label written as ``-``, so that
``numpy.loadtxt``-style column splitting is unambiguous.

Units.  Energies are in eV; intensities are the relative Mb*a.u. of
:func:`dpi.spectrum.prefactor`; the broadened spectrum is Mb*a.u./eV.
"""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import __version__
from .amplitudes import TERM_ORDER
from .constants import CHANNEL_LABELS, ConfigError
from .spectrum import StateResult

__all__ = [
    "SPECTRUM_FIXED_COLUMNS",
    "STICKS_COLUMNS",
    "write_spectrum",
    "read_spectrum",
    "write_sticks",
    "read_sticks",
    "write_latex_table",
    "header_lines",
]


# spectrum.dat: this many leading columns, then one per channel, then total.
SPECTRUM_FIXED_COLUMNS: tuple[str, ...] = ("E_shifted_eV",)

# sticks.dat: fixed leading columns, then the per-term integrals in
# amplitudes.TERM_ORDER, then the trailing diagnostic columns.
STICKS_LEADING: tuple[str, ...] = (
    "index",
    "channel",
    "label",
    "E_shifted_eV",
    "DIP_eV",
    "E_excess_eV",
    "intensity",
)
STICKS_TRAILING: tuple[str, ...] = (
    "p_i",
    "p_j",
    "amplitude_min",
    "open",
    "negative",
)
STICKS_COLUMNS: tuple[str, ...] = (
    STICKS_LEADING + tuple(f"I_{t}" for t in TERM_ORDER) + STICKS_TRAILING
)

_MISSING = "-"


def _token(value: Any) -> str:
    """Whitespace-free token for a free-text column."""
    if value is None:
        return _MISSING
    text = str(value).strip()
    if not text:
        return _MISSING
    return "_".join(text.split())


def _untoken(text: str) -> str | None:
    return None if text == _MISSING else text


def _timestamp() -> str:
    """ISO-8601 UTC timestamp to whole seconds."""
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def _flatten(prefix: str, obj: Any) -> list[tuple[str, Any]]:
    """Flatten a settings mapping/dataclass into ``key = value`` pairs."""
    if obj is None:
        return []
    if hasattr(obj, "as_dict") and callable(obj.as_dict):
        obj = obj.as_dict()
    elif hasattr(obj, "__dataclass_fields__"):
        obj = {f: getattr(obj, f) for f in obj.__dataclass_fields__}
    if isinstance(obj, Mapping):
        out: list[tuple[str, Any]] = []
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, Mapping) or hasattr(
                value, "__dataclass_fields__"
            ):
                out.extend(_flatten(name, value))
            else:
                out.append((name, value))
        return out
    return [(prefix, obj)]


def header_lines(
    settings: Mapping[str, Any] | Any | None = None,
    *,
    producer: str = "dpi_run.py",
    note: str | Iterable[str] | None = None,
) -> list[str]:
    """Self-documenting comment header.

    Parameters
    ----------
    settings:
        Mapping, dataclass, or object exposing ``as_dict()``.  Nested
        mappings and dataclasses (e.g. a
        :class:`dpi.amplitudes.TermSwitches` under the key ``terms``) are
        flattened to ``parent.child`` keys, one comment line each.
    producer:
        Name of the program writing the file.
    note:
        Extra free-text line(s), e.g. the stopgap warning of
        :mod:`dpi.atomic_sigma` when a sub-threshold linear rise was used.

    Returns
    -------
    list[str]
        Lines *without* the leading ``# `` or a trailing newline; the
        writers add both.  The first line is
        ``<producer>  <version>  <ISO-8601 timestamp>``.
    """
    lines = [f"{producer}  {__version__}  {_timestamp()}"]
    lines.append(
        "intensity units: Mb*a.u. (RELATIVE, spectrum.prefactor() == 1.0)"
    )
    for key, value in _flatten("", settings):
        lines.append(f"{key} = {value}")
    if note is None:
        notes: list[str] = []
    elif isinstance(note, str):
        notes = [note]
    else:
        notes = list(note)
    lines.extend(str(n) for n in notes)
    return lines


def _write_header(fh, lines: Sequence[str]) -> None:
    for line in lines:
        for piece in str(line).splitlines() or [""]:
            fh.write(f"# {piece}\n")


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        raise ConfigError(
            f"report: output directory {parent!r} does not exist "
            f"(writing {path!r})."
        )


def write_spectrum(
    path: str,
    e_grid_ev: np.ndarray,
    channels: Mapping[str, np.ndarray],
    *,
    settings: Mapping[str, Any] | Any | None = None,
    producer: str = "dpi_run.py",
    note: str | Iterable[str] | None = None,
    total: np.ndarray | None = None,
    precision: int = 6,
) -> str:
    """Write the broadened spectrum, the primary deliverable.

    Parameters
    ----------
    path:
        Output path, conventionally ``spectrum.dat``.
    e_grid_ev:
        ``(ne,)`` uniform grid of shifted double-ionization energy, eV.
    channels:
        Ordered mapping channel name -> ``(ne,)`` broadened intensity in
        Mb*a.u./eV.  Insertion order fixes the column order, so pass the
        channels in the order the driver reports them (``dict`` preserves
        it).
    settings:
        Active run settings for the header; see :func:`header_lines`.
    producer, note:
        Passed to :func:`header_lines`.
    total:
        Optional total column.  Defaults to the sum over ``channels``,
        which is what the driver wants; supply it explicitly only when the
        total is not that sum.
    precision:
        Significant digits of the intensity columns.

    Returns
    -------
    str
        ``path``, for chaining.

    Notes
    -----
    Column layout, which :func:`read_spectrum` and ``dpi_plot.py`` code
    against::

        E_shifted_eV  <channel_1> ... <channel_n>  total

    The energy column is written with four decimals in fixed-point form
    (the grid is uniform and in eV, so this is exact to well below any
    physical broadening); the intensity columns in exponential form.
    """
    e_grid_ev = np.asarray(e_grid_ev, dtype=float).ravel()
    ne = e_grid_ev.size
    names = list(channels)
    cols = []
    for name in names:
        arr = np.asarray(channels[name], dtype=float).ravel()
        if arr.size != ne:
            raise ConfigError(
                f"report.write_spectrum: channel {name!r} has {arr.size} "
                f"points but the energy grid has {ne} (writing {path!r})."
            )
        cols.append(arr)
    if total is None:
        tot = np.sum(cols, axis=0) if cols else np.zeros(ne)
    else:
        tot = np.asarray(total, dtype=float).ravel()
        if tot.size != ne:
            raise ConfigError(
                f"report.write_spectrum: total has {tot.size} points but "
                f"the energy grid has {ne} (writing {path!r})."
            )

    _ensure_parent(path)
    header = header_lines(settings, producer=producer, note=note)
    col_names = list(SPECTRUM_FIXED_COLUMNS) + [_token(n) for n in names]
    col_names.append("total")
    data = np.column_stack([e_grid_ev] + cols + [tot])
    fmt = "%14.4f" + "".join(f"  %{precision + 8}.{precision}e"
                             for _ in range(data.shape[1] - 1))

    with open(path, "w", encoding="utf-8") as fh:
        _write_header(fh, header)
        fh.write("# columns: " + "  ".join(col_names) + "\n")
        for row in data:
            fh.write(fmt % tuple(row) + "\n")
    return path


@dataclass(frozen=True)
class SpectrumFile:
    """Contents of a ``spectrum.dat``.

    Attributes
    ----------
    e_grid_ev:
        ``(ne,)`` shifted double-ionization energy, eV.
    channels:
        Channel name -> ``(ne,)`` intensity, Mb*a.u./eV, in file order.
    total:
        ``(ne,)`` total column, Mb*a.u./eV.
    header:
        Comment lines with the leading ``# `` stripped.
    columns:
        Column names as declared by the ``# columns:`` line.
    """

    e_grid_ev: np.ndarray
    channels: dict[str, np.ndarray]
    total: np.ndarray
    header: list[str]
    columns: list[str]

    def settings(self) -> dict[str, str]:
        """``key = value`` header lines parsed back into a mapping."""
        return _parse_settings(self.header)


def _parse_settings(header: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in header:
        if "=" in line:
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    return out


def _read_commented(path: str) -> tuple[list[str], list[str], list[str]]:
    """Split a ``.dat`` into header lines, column names and data lines."""
    if not os.path.isfile(path):
        raise ConfigError(f"report: file not found: {path!r}.")
    header: list[str] = []
    columns: list[str] = []
    data: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                body = line.lstrip()[1:].strip()
                if body.lower().startswith("columns:"):
                    columns = body.split(":", 1)[1].split()
                else:
                    header.append(body)
            else:
                data.append(line)
    if not columns:
        raise ConfigError(
            f"report: {path!r} has no '# columns:' line; the file was not "
            "written by dpi.report and its column order is unknown."
        )
    del lineno
    return header, columns, data


def read_spectrum(path: str) -> SpectrumFile:
    """Read a ``spectrum.dat`` written by :func:`write_spectrum`.

    Parameters
    ----------
    path:
        Input path.

    Returns
    -------
    SpectrumFile
        Energies in eV, channel and total columns in Mb*a.u./eV, plus the
        header lines so that a plot can quote the settings the spectrum was
        produced with.
    """
    header, columns, data = _read_commented(path)
    if not columns or columns[0] != SPECTRUM_FIXED_COLUMNS[0]:
        raise ConfigError(
            f"report.read_spectrum: {path!r} declares first column "
            f"{columns[:1]!r}, expected {SPECTRUM_FIXED_COLUMNS[0]!r}."
        )
    if columns[-1] != "total":
        raise ConfigError(
            f"report.read_spectrum: {path!r} declares last column "
            f"{columns[-1]!r}, expected 'total'."
        )
    table = np.array(
        [[float(v) for v in line.split()] for line in data], dtype=float
    ) if data else np.zeros((0, len(columns)))
    if table.shape[1] != len(columns):
        raise ConfigError(
            f"report.read_spectrum: {path!r} declares {len(columns)} "
            f"columns but the data rows have {table.shape[1]}."
        )
    chan_names = columns[1:-1]
    return SpectrumFile(
        e_grid_ev=table[:, 0].copy(),
        channels={n: table[:, i + 1].copy() for i, n in enumerate(chan_names)},
        total=table[:, -1].copy(),
        header=header,
        columns=columns,
    )


def write_sticks(
    path: str,
    states: Iterable[StateResult],
    *,
    settings: Mapping[str, Any] | Any | None = None,
    producer: str = "dpi_run.py",
    note: str | Iterable[str] | None = None,
    precision: int = 8,
) -> str:
    """Write the per-state stick and term table.

    Parameters
    ----------
    path:
        Output path, conventionally ``sticks.dat``.
    states:
        :class:`dpi.spectrum.StateResult` instances, in the order the
        driver produced them.  Closed states are written too, with zero
        intensity and ``open = 0``, so that the file is a complete record.
    settings, producer, note:
        Header content; see :func:`header_lines`.
    precision:
        Significant digits of the floating-point columns.

    Returns
    -------
    str
        ``path``.

    Notes
    -----
    Column layout, which :func:`read_sticks` and ``dpi_plot.py`` code
    against -- ``index channel label E_shifted_eV DIP_eV E_excess_eV
    intensity``, then one ``I_<term>`` column for every term in
    :data:`dpi.amplitudes.TERM_ORDER` (``direct``, ``cross_dyson``,
    ``indirect``, ``indirect_cross``, ``aa_bb``, ``c_cross``,
    ``dir_ind_interference``), then ``p_i p_j amplitude_min open
    negative``.

    Every term column is present whether or not the term is active, so the
    number of columns does not depend on the model; a term switched off is
    written as ``0.0`` and the header records that it was off.  ``p_i`` and
    ``p_j`` are the spectroscopic factors, the model's own measure of the
    distance from the frozen-orbital limit, and belong in the output
    (REVIEW.md section 6.2).  ``open`` and ``negative`` are written as
    ``0``/``1``.
    """
    st = list(states)
    _ensure_parent(path)
    header = header_lines(settings, producer=producer, note=note)
    fw = precision + 8
    with open(path, "w", encoding="utf-8") as fh:
        _write_header(fh, header)
        fh.write("# columns: " + "  ".join(STICKS_COLUMNS) + "\n")
        for s in st:
            if s.channel not in CHANNEL_LABELS:
                raise ConfigError(
                    f"report.write_sticks: state {s.index} has unknown "
                    f"channel {s.channel!r} (writing {path!r})."
                )
            fields = [
                f"{int(s.index):6d}",
                f"{_token(s.channel):>10s}",
                f"{_token(s.label):>14s}",
                f"{s.e_shifted_ev:{fw}.{precision}e}",
                f"{s.dip_ev:{fw}.{precision}e}",
                f"{s.e_excess_ev:{fw}.{precision}e}",
                f"{s.intensity:{fw}.{precision}e}",
            ]
            fields += [
                f"{float(s.terms.get(term, 0.0)):{fw}.{precision}e}"
                for term in TERM_ORDER
            ]
            fields += [
                f"{s.p_i:{fw}.{precision}e}",
                f"{s.p_j:{fw}.{precision}e}",
                f"{s.amplitude_min:{fw}.{precision}e}",
                f"{int(bool(s.open)):5d}",
                f"{int(bool(s.negative)):9d}",
            ]
            fh.write("  ".join(fields) + "\n")
    return path


@dataclass(frozen=True)
class SticksFile:
    """Contents of a ``sticks.dat``.

    Attributes
    ----------
    states:
        :class:`dpi.spectrum.StateResult` instances rebuilt from the file.
        ``e_dication_ev`` is ``nan``: it is not part of the format because
        it plays no role in the physics or the plot.
    header, columns:
        As in :class:`SpectrumFile`.
    """

    states: list[StateResult]
    header: list[str]
    columns: list[str]

    def settings(self) -> dict[str, str]:
        """``key = value`` header lines parsed back into a mapping."""
        return _parse_settings(self.header)


def read_sticks(path: str) -> SticksFile:
    """Read a ``sticks.dat`` written by :func:`write_sticks`.

    Parameters
    ----------
    path:
        Input path.

    Returns
    -------
    SticksFile
        States with energies in eV and intensities in Mb*a.u.  Term
        integrals are keyed by the ``I_``-stripped column names, i.e. by
        :data:`dpi.amplitudes.TERM_ORDER`.

    Notes
    -----
    Columns are located by name from the ``# columns:`` line rather than by
    position, so a file written by a future version that appends a column
    still reads.
    """
    header, columns, data = _read_commented(path)
    missing = [c for c in STICKS_LEADING if c not in columns]
    if missing:
        raise ConfigError(
            f"report.read_sticks: {path!r} is missing required column(s) "
            f"{missing}; declared columns are {columns}."
        )
    idx = {name: i for i, name in enumerate(columns)}
    term_cols = {
        name[2:]: i for name, i in idx.items() if name.startswith("I_")
    }

    states: list[StateResult] = []
    for lineno, line in enumerate(data, start=1):
        fields = line.split()
        if len(fields) != len(columns):
            raise ConfigError(
                f"report.read_sticks: {path!r} line {lineno} has "
                f"{len(fields)} fields, expected {len(columns)}."
            )

        def num(name: str, default: float = float("nan")) -> float:
            i = idx.get(name)
            return default if i is None else float(fields[i])

        def flag(name: str, default: bool) -> bool:
            i = idx.get(name)
            return default if i is None else bool(int(float(fields[i])))

        states.append(
            StateResult(
                index=int(fields[idx["index"]]),
                channel=fields[idx["channel"]],
                label=_untoken(fields[idx["label"]]),
                e_dication_ev=float("nan"),
                e_shifted_ev=num("E_shifted_eV"),
                dip_ev=num("DIP_eV"),
                e_excess_ev=num("E_excess_eV"),
                intensity=num("intensity"),
                terms={t: float(fields[i]) for t, i in term_cols.items()},
                open=flag("open", True),
                negative=flag("negative", False),
                amplitude_min=num("amplitude_min", 0.0),
                p_i=num("p_i"),
                p_j=num("p_j"),
            )
        )
    return SticksFile(states=states, header=header, columns=columns)


def write_latex_table(
    path: str,
    states: Iterable[StateResult],
    *,
    caption: str = "Core-valence DPI intensities.",
    label: str = "tab:dpi",
    terms: Sequence[str] | None = None,
    settings: Mapping[str, Any] | Any | None = None,
    producer: str = "dpi_run.py",
    precision: int = 3,
) -> str:
    """Write an optional LaTeX table of the per-state results.

    Parameters
    ----------
    path:
        Output path, conventionally ``states.tex``.
    states:
        States to tabulate, in order.
    caption, label:
        LaTeX caption and ``\\label`` key.
    terms:
        Term columns to include; defaults to the terms that are non-zero
        for at least one state, which keeps a table of a reduced model
        narrow.  Pass an explicit sequence to pin the columns.
    settings, producer:
        Written as LaTeX comments above the table, so the ``.tex`` carries
        the same provenance as the ``.dat`` files.
    precision:
        Significant digits in the numeric cells.

    Returns
    -------
    str
        ``path``.

    Notes
    -----
    ``sticks.dat`` is the machine-readable record; this writer exists for
    the manuscript only, and deliberately rounds.  Underscores in labels
    are escaped.
    """
    st = list(states)
    if terms is None:
        terms = [
            t for t in TERM_ORDER
            if any(abs(float(s.terms.get(t, 0.0))) > 0.0 for s in st)
        ]
    terms = list(terms)
    _ensure_parent(path)

    def esc(text: str) -> str:
        return text.replace("_", r"\_")

    ncol = 5 + len(terms)
    head = (
        ["State", "Channel", r"$E_{\mathrm{shift}}$ (eV)",
         r"$E_{\mathrm{exc}}$ (eV)", r"$I_f$"]
        + [rf"$I_{{\mathrm{{{esc(t)}}}}}$" for t in terms]
    )
    with open(path, "w", encoding="utf-8") as fh:
        for line in header_lines(settings, producer=producer):
            fh.write(f"% {line}\n")
        fh.write("\\begin{table}\n\\centering\n")
        fh.write(f"\\caption{{{caption}}}\n\\label{{{label}}}\n")
        fh.write("\\begin{tabular}{" + "l" * 2 + "r" * (ncol - 2) + "}\n")
        fh.write("\\hline\n" + " & ".join(head) + " \\\\\n\\hline\n")
        for s in st:
            name = esc(s.label) if s.label else str(s.index)
            cells = [
                name,
                esc(CHANNEL_LABELS.get(s.channel, s.channel)),
                f"{s.e_shifted_ev:.2f}",
                f"{s.e_excess_ev:.2f}",
                f"{s.intensity:.{precision}e}",
            ]
            cells += [
                f"{float(s.terms.get(t, 0.0)):.{precision}e}" for t in terms
            ]
            if not s.open:
                cells[-1] = cells[-1] + r"$^{\dagger}$"
            fh.write(" & ".join(cells) + " \\\\\n")
        fh.write("\\hline\n\\end{tabular}\n")
        if any(not s.open for s in st):
            fh.write(
                r"\\ $^{\dagger}$ channel closed, $E_{\mathrm{exc}} \le 0$."
                "\n"
            )
        fh.write("\\end{table}\n")
    return path
