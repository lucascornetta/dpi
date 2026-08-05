"""Readers for the OpenMolcas files the DPI model consumes.

Four kinds of input are handled:

* ``INPORB`` / ``RasOrb`` orbital files, format versions 2.0 and 2.2, via
  :func:`read_inporb`;
* a plain-text AO overlap matrix via :func:`read_overlap`, or the same
  matrix out of the HDF5 file via :func:`read_ao_overlap_h5`;
* the contracted-GTO basis description out of the HDF5 file via
  :func:`read_basis`, including *detection* of the harmonic convention;
* AO dipole integrals via :func:`read_ao_dipole`, by default reduced to
  the one-centre origin-shifted form the Gelius model requires.

:func:`write_synthetic_case` generates a small but fully self-consistent
fake calculation, which is what the test suite and the other tracks build
their fixtures on.

Every reader raises :class:`dpi.constants.MolcasFormatError` naming the
offending path and, where a line can be blamed, its 1-based line number.
Angular momenta above ``l = 2`` raise :class:`ModelError`: the shake-off
angular factors are only tabulated through d functions.

All quantities here are in atomic units, which is what OpenMolcas writes:
coefficients are dimensionless, overlaps dimensionless, dipole integrals
in bohr (e = 1), coordinates in bohr, primitive exponents in bohr^-2.
"""

from __future__ import annotations

import itertools
import os
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .constants import (
    MolcasFormatError,
    ModelError,
    OCC_DOUBLY_MIN,
    OCC_FRACTIONAL,
    OCC_SINGLY,
    OCC_VIRTUAL_MAX,
)

__all__ = [
    "OrbitalSet",
    "HoleAssignment",
    "BasisSet",
    "SyntheticCase",
    "read_inporb",
    "read_overlap",
    "rohf_hole_indices",
    "read_basis",
    "read_ao_dipole",
    "read_ao_overlap_h5",
    "write_synthetic_case",
]


# ── dataclasses ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OrbitalSet:
    """One set of molecular orbitals as written by OpenMolcas.

    Attributes
    ----------
    coeff : ndarray, shape (nbas, nmo)
        MO coefficients in the AO basis; ``coeff[:, k]`` is MO ``k``.
        Dimensionless.  Symmetry-blocked files are expanded to the full
        rectangular matrix with zeros off the symmetry blocks.
    occ : ndarray, shape (nmo,)
        Occupation numbers, electrons per spatial MO (0 to 2).  Empty
        array of length 0 when the file carries no ``#OCC`` block.
    nbas_per_sym, nmo_per_sym : tuple of int
        Per-irrep basis-function and orbital counts, as declared in
        ``#INFO``.
    path : str
        The file the data came from, kept for error messages.
    version : str
        The declared format version, ``'2.0'`` or ``'2.2'``.
    """

    coeff: np.ndarray
    occ: np.ndarray
    nbas_per_sym: tuple[int, ...]
    nmo_per_sym: tuple[int, ...]
    path: str
    version: str = "2.0"

    @property
    def nbas(self) -> int:
        return int(self.coeff.shape[0])

    @property
    def nmo(self) -> int:
        return int(self.coeff.shape[1])


@dataclass(frozen=True)
class HoleAssignment:
    """Which neutral MOs occupy the alpha and beta sectors of a dication.

    The convention, which REVIEW.md [D-1] found reversed in the old
    docstrings, is fixed here once and for all:

    * hole ``i`` is the MO **missing from the alpha** occupied set, so
      ``hole_i`` appears in :attr:`beta_idx` and not in :attr:`alpha_idx`;
    * hole ``j`` is the MO **missing from the beta** occupied set.

    ``i < j`` always.  With OpenMolcas' usual energy ordering that makes
    ``i`` the core hole and ``j`` the valence hole for a core-valence
    state, but nothing downstream relies on that reading: the working
    amplitude expressions are symmetric under ``i <-> j`` and only the
    internal consistency of the two index tuples matters.

    Attributes
    ----------
    alpha_idx, beta_idx : tuple of int
        Length ``n_neu_occ - 1`` tuples of 0-based neutral MO columns,
        ascending.  These index the *neutral* orbital set; they are the
        column selections that build ``Q_a`` and ``Q_b``.
    n_doubly : int
        Number of MOs doubly occupied in the dication, ``n_neu_occ - 2``.
    n_singly : int
        Number of singly occupied MOs, always 2 after any approximation.
    hole_i, hole_j : int
        0-based neutral MO columns of the two holes, ``hole_i < hole_j``.
    approximation : str or None
        ``None`` for a strict two-open-shell ROHF file.  For
        state-averaged natural orbitals, a sentence naming the
        single-determinant approximation that was applied.
    """

    alpha_idx: tuple[int, ...]
    beta_idx: tuple[int, ...]
    n_doubly: int
    n_singly: int
    hole_i: int
    hole_j: int
    approximation: str | None = None


@dataclass(frozen=True)
class BasisSet:
    """Per-AO contracted-GTO description read from an OpenMolcas HDF5 file.

    The per-AO arrays all have length :attr:`nbas` and are aligned with
    the AO index used by every other array in the package (overlap,
    dipole, Dyson orbitals).

    Attributes
    ----------
    nbas : int
        Number of contracted AO basis functions.
    n_atoms : int
        Number of centres.
    centers : ndarray of int, shape (nbas,)
        0-based centre index of each AO.
    elements : ndarray of str, shape (nbas,)
        Element symbol of each AO's centre, with the trailing atom index
        stripped: a ``CENTER_LABELS`` entry ``'S1'`` becomes ``'S'``.
    l, m : ndarray of int, shape (nbas,)
        Angular momentum and the file's magnetic/component label.  For
        ``harmonics == 'spherical'`` there are ``2l+1`` distinct ``m`` per
        shell; for ``'cartesian'``, ``(l+1)(l+2)/2``.  The *meaning* of an
        individual ``m`` value is the file's, not reinterpreted here.
    shell_index : ndarray of int, shape (nbas,)
        0-based sequential index of the contracted shell within its
        ``(center, l)`` group, ordered by the file's shell label.
    harmonics : str
        ``'spherical'`` or ``'cartesian'``, **detected** from the number
        of distinct ``m`` per ``(center, l)`` shell.  Never assumed:
        REVIEW.md [A-1] traces a whole class of shake-off errors to the
        old code hard-coding six Cartesian d components for a basis that
        stores five real spherical ones.
    prim_exp, prim_coef : tuple of ndarray
        Length-``nbas`` tuples; entry ``mu`` holds the primitive Gaussian
        exponents (bohr^-2) and contraction coefficients (dimensionless,
        as stored by OpenMolcas) of AO ``mu``.  All AOs of one shell share
        the radial part, so these arrays are shared between the ``2l+1``
        (or ``(l+1)(l+2)/2``) components of a shell.
    n_prim : ndarray of int, shape (nbas,)
        Contraction length of each AO, ``len(prim_exp[mu])``.
    center_coords : ndarray, shape (n_atoms, 3)
        Nuclear coordinates in bohr.
    atom_labels, atom_elements : tuple of str
        Per-centre raw ``CENTER_LABELS`` entries and their stripped
        element symbols.
    id_layout : tuple of str
        Which column of ``BASIS_FUNCTION_IDS`` was resolved to which
        quantum label, e.g. ``('center', 'l', 'm', 'shell')``.  Recorded
        because the column order is not stable across OpenMolcas
        versions and this is the first thing to check when a basis looks
        wrong.
    path : str
        The HDF5 file the data came from.
    """

    nbas: int
    n_atoms: int
    centers: np.ndarray
    elements: np.ndarray
    l: np.ndarray
    m: np.ndarray
    shell_index: np.ndarray
    harmonics: str
    prim_exp: tuple[np.ndarray, ...]
    prim_coef: tuple[np.ndarray, ...]
    n_prim: np.ndarray
    center_coords: np.ndarray
    atom_labels: tuple[str, ...]
    atom_elements: tuple[str, ...]
    id_layout: tuple[str, ...]
    path: str

    # ── adapter for dpi.shakeoff ──────────────────────────────────────────
    # The shake-off transforms consume the primitive data under the names
    # alpha / coeff / norm, with `norm` the per-primitive normalisation of
    # the angular convention actually in the file.  Exposing them as
    # properties keeps one source of truth for the exponents while letting
    # the two modules keep their natural vocabulary.

    @property
    def alpha(self) -> tuple[np.ndarray, ...]:
        """Primitive exponents per AO, atomic units."""
        return self.prim_exp

    @property
    def coeff(self) -> tuple[np.ndarray, ...]:
        """Contraction coefficients per AO."""
        return self.prim_coef

    @property
    def norm(self) -> tuple[np.ndarray, ...]:
        """Per-primitive normalisation constants for each AO.

        For the spherical convention the radial factor is normalised as
        ``int |N r^l exp(-alpha r^2)|^2 r^2 dr = 1``, giving
        ``N = sqrt(2 (2 alpha)^(l+3/2) / Gamma(l+3/2))``, with the angular
        factor carried by a unit-normalised real ``Y_lm``.  Cartesian
        shells instead normalise the monomial ``x^a y^b z^c``, whose
        double-factorial denominator depends on the individual powers; the
        Cartesian branch of :mod:`dpi.shakeoff` applies that factor itself
        from ``m``, so the same radial constant is returned here.
        """
        from math import gamma as _gamma
        out = []
        for exps, l_mu in zip(self.prim_exp, np.asarray(self.l, dtype=int)):
            a = np.atleast_1d(np.asarray(exps, dtype=float))
            out.append(np.sqrt(2.0 * (2.0 * a) ** (l_mu + 1.5)
                               / _gamma(l_mu + 1.5)))
        return tuple(out)


@dataclass(frozen=True)
class SyntheticCase:
    """Paths and reference data of a generated self-consistent fake case.

    Everything a test or a downstream track needs to exercise the
    pipeline without an OpenMolcas run.  The orbital sets are genuinely
    orthonormal with respect to the written AO overlap, and the dication
    set is a small rotation of the neutral one, so ``Q`` is near-identity
    but not identity and the Dyson amplitudes are non-trivial.

    Attributes
    ----------
    dirpath : str
        Directory holding the generated files.
    inporb_neutral, inporb_dication, overlap, h5 : str
        Paths of the neutral ``INPORB``, the dication ``RasOrb``, the
        text overlap file and the HDF5 file.
    nbas, nmo, n_neu_occ : int
        Basis size, orbital count and number of doubly occupied neutral
        spatial MOs.
    harmonics : str
        Harmonic convention written into the HDF5 file.
    s_ao : ndarray, shape (nbas, nbas)
        The AO overlap actually written, after decimal round-tripping,
        so tests may compare exactly.
    c_neu, c_dic : ndarray, shape (nbas, nmo)
        The MO coefficients actually written, after round-tripping.
    occ_neutral, occ_dication : ndarray, shape (nmo,)
        The occupation numbers actually written.
    occ_style : str
        ``'rohf'`` (two MOs at occ 1) or ``'natural'`` (three at 5/3 plus
        one at 1, triggering the single-determinant approximation).
    hole_i, hole_j : int
        The hole MOs the dication occupations encode, ``hole_i < hole_j``.
    dipole_full : dict of str to ndarray
        The full molecular AO dipole matrices written to the HDF5 file,
        keyed ``'x'``, ``'y'``, ``'z'``, in bohr.
    dipole_one_centre : dict of str to ndarray
        The one-centre origin-shifted reduction of the same, i.e. what
        ``read_ao_dipole(h5, one_centre=True)`` must return.
    center_coords : ndarray, shape (n_atoms, 3)
        Nuclear coordinates in bohr.
    atom_labels : tuple of str
        Raw centre labels written to the HDF5 file.
    """

    dirpath: str
    inporb_neutral: str
    inporb_dication: str
    overlap: str
    h5: str
    nbas: int
    nmo: int
    n_neu_occ: int
    harmonics: str
    s_ao: np.ndarray
    c_neu: np.ndarray
    c_dic: np.ndarray
    occ_neutral: np.ndarray
    occ_dication: np.ndarray
    occ_style: str
    hole_i: int
    hole_j: int
    dipole_full: dict[str, np.ndarray]
    dipole_one_centre: dict[str, np.ndarray]
    center_coords: np.ndarray
    atom_labels: tuple[str, ...]


# ── low-level text helpers ──────────────────────────────────────────────────

# OpenMolcas writes Fortran double-precision exponents, which may use D,
# and -- when the exponent needs three digits and the field is tight --
# may drop the exponent letter altogether ('1.234567-101').
_D_EXP = re.compile(r"(?<=[0-9.])[Dd](?=[-+]?\d)")
_BARE_EXP = re.compile(r"(?<=[0-9.])([-+]\d{2,3})$")


def _to_float(token: str, path: str, lineno: int) -> float:
    """Parse one Fortran-formatted real, or raise naming path and line."""
    t = _D_EXP.sub("E", token.strip())
    try:
        return float(t)
    except ValueError:
        pass
    # Retry assuming a dropped exponent letter.
    t2 = _BARE_EXP.sub(r"E\1", t)
    try:
        return float(t2)
    except ValueError as exc:
        raise MolcasFormatError(
            f"{path}:{lineno}: cannot parse {token!r} as a Fortran real"
        ) from exc


def _to_int(token: str, path: str, lineno: int) -> int:
    try:
        return int(token)
    except ValueError as exc:
        raise MolcasFormatError(
            f"{path}:{lineno}: cannot parse {token!r} as an integer"
        ) from exc


def _read_text(path: str | os.PathLike) -> list[str]:
    p = os.fspath(path)
    if not os.path.isfile(p):
        raise MolcasFormatError(f"{p}: no such file")
    try:
        with open(p, "r", errors="replace") as fh:
            return fh.read().splitlines()
    except OSError as exc:  # pragma: no cover - filesystem dependent
        raise MolcasFormatError(f"{p}: cannot be read ({exc})") from exc


def _strip_element(label: str) -> str:
    """'S1' -> 'S', 'Fe12' -> 'Fe', 'H' -> 'H'.

    OpenMolcas centre labels carry a trailing symmetry-unique atom index;
    the Yeh-Lindau table and the shake-off angular factors are keyed by
    element, so the index must go.  Only trailing digits are stripped, so
    an element symbol is never truncated.
    """
    return re.sub(r"\d+$", "", label.strip()).strip().capitalize()


# ── INPORB / RasOrb ─────────────────────────────────────────────────────────

_ORBITAL_SEP = re.compile(r"^\*?\s*ORBITAL\s+(\d+)\s+(\d+)\s*$", re.IGNORECASE)


def _split_blocks(
    lines: Sequence[str], path: str
) -> tuple[str, dict[str, list[tuple[int, str]]]]:
    """Return the declared version and the ``#TAG -> [(lineno, text)]`` map.

    A block is opened by a line whose first non-blank character is ``#``
    and runs to the next such line.  Line numbers are 1-based.
    """
    version = None
    blocks: dict[str, list[tuple[int, str]]] = {}
    current: list[tuple[int, str]] | None = None
    for lineno, raw in enumerate(lines, start=1):
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#"):
            tag = s.split()[0].upper()
            if tag == "#INPORB":
                parts = s.split()
                if len(parts) < 2:
                    raise MolcasFormatError(
                        f"{path}:{lineno}: '#INPORB' header carries no "
                        f"version number"
                    )
                version = parts[1]
                current = None
                continue
            if tag in blocks:
                raise MolcasFormatError(
                    f"{path}:{lineno}: duplicate {tag} block"
                )
            current = []
            blocks[tag] = current
            continue
        if current is not None:
            current.append((lineno, raw))
    if version is None:
        raise MolcasFormatError(
            f"{path}:1: not an INPORB file -- no '#INPORB <version>' "
            f"header found"
        )
    return version, blocks


def _info_ints(
    body: Iterable[tuple[int, str]], path: str
) -> list[tuple[int, list[int]]]:
    """Integer data lines of ``#INFO``, as ``(lineno, values)``."""
    out = []
    for lineno, raw in body:
        s = raw.strip()
        if not s or s.startswith("*"):
            continue
        toks = s.split()
        out.append((lineno, [_to_int(t, path, lineno) for t in toks]))
    return out


def read_inporb(path: str | os.PathLike) -> OrbitalSet:
    """Parse an OpenMolcas ``INPORB``/``RasOrb`` file, version 2.0 or 2.2.

    Parameters
    ----------
    path : str or path-like
        The orbital file.

    Returns
    -------
    OrbitalSet
        Coefficients (dimensionless, shape ``(nbas, nmo)``), occupations
        (electrons per spatial MO) and the per-irrep dimensions.

    Notes
    -----
    The two format versions differ in the second data line of ``#INFO``:
    version 2.0 writes ``nSym``, version 2.2 writes the
    ``(nFro, nIsh, nAsh)`` partition of the orbital space.  The version
    is taken from the ``#INPORB`` header and the layout is chosen from
    it; it is never guessed from the line's arity, because a
    single-symmetry 2.2 file and a three-symmetry 2.0 file can present
    the same token count.

    Symmetry-blocked files are expanded into one dense ``(nbas, nmo)``
    matrix with the off-block entries left at zero, so downstream code
    never has to know about irreps.
    """
    p = os.fspath(path)
    lines = _read_text(p)
    version, blocks = _split_blocks(lines, p)
    if version not in ("2.0", "2.2"):
        raise MolcasFormatError(
            f"{p}:1: unsupported INPORB version {version!r}; this reader "
            f"handles 2.0 and 2.2"
        )
    if "#UORB" in blocks or "#UOCC" in blocks:
        raise MolcasFormatError(
            f"{p}: file carries a #UORB/#UOCC block, i.e. spin-unrestricted "
            f"orbitals; the DPI model requires RHF/ROHF spatial orbitals"
        )
    for required in ("#INFO", "#ORB"):
        if required not in blocks:
            raise MolcasFormatError(f"{p}: mandatory {required} block missing")

    info = _info_ints(blocks["#INFO"], p)
    # The #INFO block carries exactly three data lines (the `*` title line is
    # a comment and already stripped):
    #
    #     iUHF  nSym  iWFtype        e.g. "0  1  2" for a C1 RHF SCF file
    #     nBas(1..nSym)              e.g. "102"
    #     nOrb(1..nSym)              e.g. "102"
    #
    # nSym is the SECOND integer of the first line, not a line of its own,
    # and there is no (nFro, nIsh, nAsh) partition here -- that lives in
    # #INDEX. Verified against real OpenMolcas v20.10 ScfOrb and RasOrb
    # files, whose first data lines read "0 1 2" and "0 1 0".
    if len(info) < 3:
        raise MolcasFormatError(
            f"{p}: #INFO block has {len(info)} data lines, expected 3 "
            f"(iUHF/nSym/iWFtype, nBas per symmetry, nOrb per symmetry)"
        )
    flag_line, flags = info[0]
    if len(flags) < 2:
        raise MolcasFormatError(
            f"{p}:{flag_line}: the first #INFO line must carry at least "
            f"iUHF and nSym, found {len(flags)} value(s)"
        )
    if flags[0] != 0:
        raise MolcasFormatError(
            f"{p}:{flag_line}: iUHF = {flags[0]} marks spin-unrestricted "
            f"orbitals; the DPI model requires RHF/ROHF spatial orbitals"
        )
    nsym = flags[1]
    if nsym < 1:
        raise MolcasFormatError(
            f"{p}:{flag_line}: nSym = {nsym} is not a valid symmetry count"
        )

    nbas_line, nbas_per_sym = info[1]
    nmo_line, nmo_per_sym = info[2]
    if nsym is not None and len(nbas_per_sym) != nsym:
        raise MolcasFormatError(
            f"{p}:{nbas_line}: #INFO declares nSym={nsym} but the "
            f"nbas line carries {len(nbas_per_sym)} values"
        )
    if len(nmo_per_sym) != len(nbas_per_sym):
        raise MolcasFormatError(
            f"{p}:{nmo_line}: {len(nmo_per_sym)} nmo values against "
            f"{len(nbas_per_sym)} nbas values"
        )
    if any(v < 0 for v in nbas_per_sym + nmo_per_sym):
        raise MolcasFormatError(
            f"{p}:{nbas_line}: negative dimension in #INFO"
        )
    if any(nm > nb for nb, nm in zip(nbas_per_sym, nmo_per_sym)):
        raise MolcasFormatError(
            f"{p}:{nmo_line}: an irrep declares more orbitals than basis "
            f"functions ({tuple(nmo_per_sym)} vs {tuple(nbas_per_sym)})"
        )

    nbas = int(sum(nbas_per_sym))
    nmo = int(sum(nmo_per_sym))
    bas_off = np.cumsum([0] + list(nbas_per_sym))
    mo_off = np.cumsum([0] + list(nmo_per_sym))

    coeff = np.zeros((nbas, nmo))
    seen: set[tuple[int, int]] = set()
    isym = iorb = None
    buf: list[float] = []
    sep_line = 0

    def _flush() -> None:
        if isym is None:
            return
        want = nbas_per_sym[isym - 1]
        if len(buf) != want:
            raise MolcasFormatError(
                f"{p}:{sep_line}: ORBITAL {isym} {iorb} has {len(buf)} "
                f"coefficients, expected nbas({isym})={want}"
            )
        r0 = bas_off[isym - 1]
        coeff[r0:r0 + want, mo_off[isym - 1] + iorb - 1] = buf

    for lineno, raw in blocks["#ORB"]:
        s = raw.strip()
        if not s:
            continue
        mt = _ORBITAL_SEP.match(s)
        if mt is not None:
            _flush()
            isym = _to_int(mt.group(1), p, lineno)
            iorb = _to_int(mt.group(2), p, lineno)
            sep_line = lineno
            buf = []
            if not 1 <= isym <= len(nbas_per_sym):
                raise MolcasFormatError(
                    f"{p}:{lineno}: ORBITAL symmetry index {isym} outside "
                    f"1..{len(nbas_per_sym)}"
                )
            if not 1 <= iorb <= nmo_per_sym[isym - 1]:
                raise MolcasFormatError(
                    f"{p}:{lineno}: ORBITAL index {iorb} outside "
                    f"1..nmo({isym})={nmo_per_sym[isym - 1]}"
                )
            if (isym, iorb) in seen:
                raise MolcasFormatError(
                    f"{p}:{lineno}: ORBITAL {isym} {iorb} appears twice"
                )
            seen.add((isym, iorb))
            continue
        if s.startswith("*"):
            continue  # a comment inside #ORB
        if isym is None:
            raise MolcasFormatError(
                f"{p}:{lineno}: coefficient data before the first "
                f"'* ORBITAL sym idx' separator"
            )
        buf.extend(_to_float(t, p, lineno) for t in s.split())
    _flush()

    if len(seen) != nmo:
        missing = [
            (s_, o_)
            for s_ in range(1, len(nmo_per_sym) + 1)
            for o_ in range(1, nmo_per_sym[s_ - 1] + 1)
            if (s_, o_) not in seen
        ]
        raise MolcasFormatError(
            f"{p}: #ORB carries {len(seen)} orbitals but #INFO declares "
            f"{nmo}; first missing (sym, idx) = "
            f"{missing[0] if missing else 'n/a'}"
        )

    occ = np.zeros(0)
    if "#OCC" in blocks:
        vals: list[float] = []
        last = 0
        for lineno, raw in blocks["#OCC"]:
            s = raw.strip()
            if not s or s.startswith("*"):
                continue
            last = lineno
            vals.extend(_to_float(t, p, lineno) for t in s.split())
        if len(vals) != nmo:
            raise MolcasFormatError(
                f"{p}:{last}: #OCC carries {len(vals)} occupation numbers, "
                f"expected nmo={nmo}"
            )
        occ = np.asarray(vals, dtype=float)
        if occ.min() < -1e-6 or occ.max() > 2.0 + 1e-6:
            bad = int(np.argmin(np.where(
                (occ >= -1e-6) & (occ <= 2.0 + 1e-6), np.inf, occ)))
            raise MolcasFormatError(
                f"{p}:{last}: occupation {occ[bad]!r} for MO {bad} is "
                f"outside [0, 2]"
            )

    return OrbitalSet(
        coeff=coeff,
        occ=occ,
        nbas_per_sym=tuple(int(v) for v in nbas_per_sym),
        nmo_per_sym=tuple(int(v) for v in nmo_per_sym),
        path=p,
        version=version,
    )


# ── text overlap ────────────────────────────────────────────────────────────

def read_overlap(path: str | os.PathLike, nbas: int) -> np.ndarray:
    """Read a plain-text AO overlap matrix.

    Parameters
    ----------
    path : str or path-like
        Text file of whitespace-separated reals.  Lines whose first
        non-blank character is ``#`` or ``*`` are comments.
    nbas : int
        Expected basis size, used to decide the storage layout.

    Returns
    -------
    ndarray, shape (nbas, nbas)
        The symmetric overlap, dimensionless.

    Notes
    -----
    Both layouts OpenMolcas utilities emit are accepted and
    distinguished by the *value count*, never by a flag: ``nbas**2``
    values are read as a full square matrix in row-major order,
    ``nbas*(nbas+1)/2`` as the lower triangle in row-major order.  Any
    other count is an error naming both candidate counts.  A square
    matrix is symmetrised as ``(A + A.T)/2`` and the discarded
    antisymmetric part is checked against a tolerance, since a genuine
    overlap is symmetric and an asymmetric file means the wrong integral
    was dumped.
    """
    p = os.fspath(path)
    if nbas <= 0:
        raise MolcasFormatError(f"{p}: nbas must be positive, got {nbas}")
    lines = _read_text(p)
    vals: list[float] = []
    for lineno, raw in enumerate(lines, start=1):
        s = raw.strip()
        if not s or s[0] in "#*":
            continue
        vals.extend(_to_float(t, p, lineno) for t in s.split())

    ntri = nbas * (nbas + 1) // 2
    nsq = nbas * nbas
    arr = np.asarray(vals, dtype=float)
    if arr.size == nsq:
        s_ao = arr.reshape(nbas, nbas)
        asym = np.abs(s_ao - s_ao.T).max()
        scale = max(np.abs(s_ao).max(), 1.0)
        if asym > 1e-8 * scale:
            raise MolcasFormatError(
                f"{p}: square matrix is not symmetric (max |A - A.T| = "
                f"{asym:.3e}); this is not an AO overlap"
            )
        s_ao = 0.5 * (s_ao + s_ao.T)
    elif arr.size == ntri:
        s_ao = np.zeros((nbas, nbas))
        iu = np.tril_indices(nbas)
        s_ao[iu] = arr
        s_ao = s_ao + s_ao.T - np.diag(np.diag(s_ao))
    else:
        raise MolcasFormatError(
            f"{p}: found {arr.size} values, which is neither nbas**2="
            f"{nsq} (full square) nor nbas*(nbas+1)/2={ntri} (lower "
            f"triangle) for nbas={nbas}"
        )
    if np.any(np.diag(s_ao) <= 0.0):
        bad = int(np.argmin(np.diag(s_ao)))
        raise MolcasFormatError(
            f"{p}: diagonal overlap element {bad} is "
            f"{np.diag(s_ao)[bad]:.3e}; an AO self-overlap must be positive"
        )
    return s_ao


# ── ROHF hole classification ────────────────────────────────────────────────

def rohf_hole_indices(occ: np.ndarray, n_neu_occ: int) -> HoleAssignment:
    """Classify dication occupations into alpha and beta occupied sets.

    Parameters
    ----------
    occ : ndarray, shape (nmo,)
        Occupation numbers from the dication ``#OCC`` block, electrons
        per spatial MO.
    n_neu_occ : int
        Number of doubly occupied *neutral* spatial MOs, ``n``.  The
        dication carries ``2n - 2`` electrons.

    Returns
    -------
    HoleAssignment
        See that class for the ``i``/``j`` convention, which is the one
        REVIEW.md [D-1] found stated backwards in the old code.

    Notes
    -----
    Two occupation patterns are recognised, using the windows in
    :mod:`dpi.constants` rather than inline literals:

    (a) strict two-open-shell ROHF -- exactly two MOs in ``OCC_SINGLY``
        and none in ``OCC_FRACTIONAL``.  The two holes are those MOs and
        :attr:`HoleAssignment.approximation` is ``None``.

    (b) state-averaged natural orbitals -- three MOs in
        ``OCC_FRACTIONAL`` (at occ ~ 5/3, the signature of a core hole
        delocalised over three symmetry partners, e.g. the S 2p edge)
        plus one MO in ``OCC_SINGLY``.  A single determinant cannot
        represent that density, so the documented approximation
        localises the core hole on the lowest-index fractional partner
        and promotes the other two to double occupancy.  The fact is
        recorded on the returned object so that any report can carry it.

    Anything else raises :class:`ModelError` quoting the counts found,
    rather than silently picking two indices.
    """
    occ = np.asarray(occ, dtype=float).ravel()
    if occ.size == 0:
        raise ModelError(
            "dication occupation array is empty; the RasOrb file carried "
            "no #OCC block, so the holes cannot be identified"
        )
    if n_neu_occ < 2:
        raise ModelError(
            f"n_neu_occ={n_neu_occ}: a dication needs at least two "
            f"occupied neutral orbitals to remove two electrons from"
        )

    doubly = np.flatnonzero(occ >= OCC_DOUBLY_MIN)
    fractional = np.flatnonzero(
        (occ >= OCC_FRACTIONAL[0]) & (occ < OCC_FRACTIONAL[1])
    )
    singly = np.flatnonzero((occ >= OCC_SINGLY[0]) & (occ < OCC_SINGLY[1]))
    virtual = np.flatnonzero(occ <= OCC_VIRTUAL_MAX)
    classified = (
        doubly.size + fractional.size + singly.size + virtual.size
    )
    if classified != occ.size:
        assigned = np.zeros(occ.size, dtype=bool)
        for idx in (doubly, fractional, singly, virtual):
            assigned[idx] = True
        stray = int(np.flatnonzero(~assigned)[0])
        raise ModelError(
            f"MO {stray} has occupation {occ[stray]:.6f}, which falls in "
            f"none of the windows doubly>={OCC_DOUBLY_MIN}, "
            f"fractional{OCC_FRACTIONAL}, singly{OCC_SINGLY}, "
            f"virtual<={OCC_VIRTUAL_MAX}"
        )

    n_elec = float(occ.sum())
    if abs(n_elec - (2 * n_neu_occ - 2)) > 1e-3:
        raise ModelError(
            f"occupations sum to {n_elec:.6f} electrons but a dication of "
            f"a neutral with n_neu_occ={n_neu_occ} must carry "
            f"{2 * n_neu_occ - 2}"
        )

    approximation: str | None = None
    if fractional.size == 0 and singly.size == 2:
        holes = (int(singly[0]), int(singly[1]))
        doubly_set = [int(v) for v in doubly]
    elif fractional.size == 3 and singly.size == 1:
        core = int(fractional[0])
        promoted = [int(v) for v in fractional[1:]]
        holes = (core, int(singly[0]))
        doubly_set = [int(v) for v in doubly] + promoted
        approximation = (
            "state-averaged natural orbitals: the core hole, delocalised "
            f"over MOs {tuple(int(v) for v in fractional)} at occupation "
            f"~{occ[fractional[0]]:.4f}, is localised on the lowest-index "
            f"partner (MO {core}) and MOs {tuple(promoted)} are promoted "
            "to double occupancy, giving a single determinant"
        )
    elif fractional.size == 0 and singly.size == 0 \
            and doubly.size == n_neu_occ - 1:
        # Both electrons removed from ONE spatial MO: the dication is a
        # closed-shell singlet with one empty orbital, so there is no open
        # shell to detect and the two-hole indices coincide.
        #
        # This is not a parsing gap -- the model's expressions do not cover
        # it. The whole spin algebra of notes #2/#3 rests on two DISTINCT
        # holes (i != j): the two-electron Dyson amplitude D_ij is
        # antisymmetric, so D_ii = 0 and the alpha-alpha/beta-beta channel
        # G_ij closes identically; the singlet/triplet pair collapses to a
        # single S_dic = 0 state (a triplet needs two orbitals); and the
        # cross-Dyson and interference terms reduce to i = j forms that the
        # angle-averaged expressions were never derived for. Only the
        # direct 2*D_i*S_i term survives, and the Clebsch-Gordan
        # bookkeeping in front of it differs from the i != j case.
        raise ModelError(
            f"both holes lie in the same spatial MO (MO "
            f"{int(np.setdiff1d(np.arange(n_neu_occ), doubly)[0]) + 1}): the "
            f"dication is closed-shell with {doubly.size} doubly occupied "
            f"MOs and no open shell.\n"
            f"  This is a DIFFERENT physical case, not an input error. The "
            f"two-hole expressions of the model assume i != j: D_ij is "
            f"antisymmetric so D_ii = 0 (the aa/bb channel closes), and "
            f"there is no triplet partner. Implementing it needs the "
            f"single-configuration i = j amplitudes derived separately -- "
            f"see REVIEW.md [P-7]."
        )
    else:
        raise ModelError(
            f"unrecognised dication occupation pattern: "
            f"{doubly.size} doubly occupied, {fractional.size} fractional, "
            f"{singly.size} singly occupied, {virtual.size} virtual. "
            f"Supported: (fractional=0, singly=2) for two-open-shell ROHF, "
            f"or (fractional=3, singly=1) for state-averaged natural "
            f"orbitals"
        )

    hole_i, hole_j = sorted(holes)
    doubly_set = sorted(doubly_set)
    if len(doubly_set) != n_neu_occ - 2:
        raise ModelError(
            f"after classification there are {len(doubly_set)} doubly "
            f"occupied dication MOs, expected n_neu_occ - 2 = "
            f"{n_neu_occ - 2}"
        )
    if hole_i in doubly_set or hole_j in doubly_set:
        raise ModelError(
            f"hole MO {hole_i if hole_i in doubly_set else hole_j} was also "
            f"classified as doubly occupied; occupation windows overlap"
        )

    # Hole i is absent from alpha, hole j from beta.  Both tuples are
    # ascending, which is what pins the row order of det(S^beta) to the
    # canonical (phi_i, phi_j) sense -- REVIEW.md [B-4].
    alpha_idx = tuple(sorted(doubly_set + [hole_j]))
    beta_idx = tuple(sorted(doubly_set + [hole_i]))
    return HoleAssignment(
        alpha_idx=alpha_idx,
        beta_idx=beta_idx,
        n_doubly=len(doubly_set),
        n_singly=2,
        hole_i=hole_i,
        hole_j=hole_j,
        approximation=approximation,
    )


# ── HDF5 helpers ────────────────────────────────────────────────────────────

def _h5_open(path: str | os.PathLike):
    import h5py  # imported lazily so `import dpi.molcas_io` stays cheap

    p = os.fspath(path)
    if not os.path.isfile(p):
        raise MolcasFormatError(f"{p}: no such file")
    try:
        return h5py.File(p, "r")
    except OSError as exc:
        raise MolcasFormatError(
            f"{p}: cannot be opened as HDF5 ({exc})"
        ) from exc


def _h5_get(fh, name: str, path: str) -> np.ndarray:
    if name not in fh:
        raise MolcasFormatError(
            f"{path}: dataset {name!r} missing; datasets present: "
            f"{sorted(fh.keys())}"
        )
    return np.asarray(fh[name])


def _decode_labels(raw: np.ndarray, path: str) -> tuple[str, ...]:
    out = []
    for v in np.asarray(raw).ravel():
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", "replace").strip())
        else:
            out.append(str(v).strip())
    if not out:
        raise MolcasFormatError(f"{path}: CENTER_LABELS is empty")
    return tuple(out)


def _orient_ids(ids: np.ndarray, ncol: int, name: str, path: str,
                nrow_hint: int | None = None) -> np.ndarray:
    """Return ``ids`` as ``(nitem, ncol)``, transposing if it was stored
    column-major-ish.  ``nrow_hint`` disambiguates the square case."""
    a = np.asarray(ids)
    if a.ndim != 2:
        raise MolcasFormatError(
            f"{path}: {name} has shape {a.shape}, expected 2-D"
        )
    if a.shape[0] == ncol and a.shape[1] == ncol:
        if nrow_hint is not None and nrow_hint != ncol:
            raise MolcasFormatError(
                f"{path}: {name} is {a.shape} but {nrow_hint} rows were "
                f"expected"
            )
        return a  # square and ambiguous; rows-as-items is the convention
    if a.shape[1] == ncol:
        return a
    if a.shape[0] == ncol:
        return a.T
    raise MolcasFormatError(
        f"{path}: {name} has shape {a.shape}; one axis must have length "
        f"{ncol}"
    )


def _multiplicity_convention(
    center: np.ndarray, l: np.ndarray, m: np.ndarray, shell: np.ndarray,
    path: str, strict: bool,
) -> str | None:
    """Detect 'spherical' vs 'cartesian' from m-multiplicity per shell.

    Returns ``None`` when the grouping is inconsistent and ``strict`` is
    false; raises when ``strict``.  A shell is a ``(center, l, shell)``
    triple; it must carry ``2l+1`` distinct ``m`` for real spherical
    harmonics and ``(l+1)(l+2)/2`` for Cartesians.  These differ first at
    ``l = 2`` (5 vs 6), which is exactly the SF6/cc-pVDZ discrimination
    REVIEW.md [A-1] turns on.
    """
    verdict: set[str] = set()
    keys = {}
    for idx in range(center.size):
        keys.setdefault((int(center[idx]), int(l[idx]), int(shell[idx])),
                        []).append(int(m[idx]))
    for (c, ll, sh), ms in keys.items():
        if ll < 0:
            if strict:
                raise MolcasFormatError(
                    f"{path}: negative angular momentum {ll} on centre {c}"
                )
            return None
        if len(set(ms)) != len(ms):
            if strict:
                raise MolcasFormatError(
                    f"{path}: shell (centre {c}, l={ll}, shell {sh}) "
                    f"repeats a component label: {sorted(ms)}"
                )
            return None
        n_sph = 2 * ll + 1
        n_car = (ll + 1) * (ll + 2) // 2
        got = len(ms)
        if got == n_sph and got == n_car:
            continue  # l <= 1: the two conventions coincide
        if got == n_sph:
            verdict.add("spherical")
        elif got == n_car:
            verdict.add("cartesian")
        else:
            if strict:
                raise MolcasFormatError(
                    f"{path}: shell (centre {c}, l={ll}, shell {sh}) has "
                    f"{got} components, which matches neither "
                    f"{n_sph} real spherical harmonics nor {n_car} "
                    f"Cartesian components"
                )
            return None
    if len(verdict) > 1:
        if strict:
            raise MolcasFormatError(
                f"{path}: basis mixes harmonic conventions across shells "
                f"({sorted(verdict)}); this cannot be handled"
            )
        return None
    # A basis with no shell above l = 1 is indistinguishable; call it
    # spherical, which is the OpenMolcas default and, for l <= 1, is the
    # same set of functions.
    return verdict.pop() if verdict else "spherical"


def _resolve_ao_ids(
    ids: np.ndarray, n_atoms: int, path: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str,
           tuple[str, ...]]:
    """Assign the four ``BASIS_FUNCTION_IDS`` columns to quantum labels.

    OpenMolcas stores ``(centre, l, m, shell)`` per contracted AO but the
    column order has moved between versions, so it is resolved here by
    consistency rather than assumed: the centre column is the one whose
    values index the centre list, and the remaining three are assigned by
    requiring that the resulting shells have a coherent ``m``
    multiplicity.  Ties are broken towards the documented
    ``(centre, l, m, shell)`` order.
    """
    a = ids.astype(np.int64, copy=False)
    ncols = a.shape[1]
    if ncols != 4:
        raise MolcasFormatError(
            f"{path}: BASIS_FUNCTION_IDS has {ncols} columns, expected 4 "
            f"(centre, l, m, shell)"
        )
    center_cands = [
        c for c in range(4)
        if a[:, c].min() >= 1 and a[:, c].max() <= n_atoms
        and len(np.unique(a[:, c])) <= n_atoms
    ]
    if not center_cands:
        raise MolcasFormatError(
            f"{path}: no BASIS_FUNCTION_IDS column takes values in "
            f"1..n_atoms={n_atoms}, so the centre column cannot be "
            f"identified; column ranges are "
            f"{[(int(a[:, c].min()), int(a[:, c].max())) for c in range(4)]}"
        )

    solutions = []
    for ccol in center_cands:
        rest = [c for c in range(4) if c != ccol]
        for lcol, mcol, scol in itertools.permutations(rest):
            l = a[:, lcol]
            if l.min() < 0 or l.max() > 8:
                continue
            if a[:, scol].min() < 1:
                continue  # OpenMolcas shell labels are 1-based
            harm = _multiplicity_convention(
                a[:, ccol], l, a[:, mcol], a[:, scol], path, strict=False
            )
            if harm is None:
                continue
            solutions.append(((ccol, lcol, mcol, scol), harm))
    if not solutions:
        # Re-run the preferred layout in strict mode so the user gets the
        # specific reason rather than "unresolvable".
        ccol = center_cands[0]
        rest = [c for c in range(4) if c != ccol]
        _multiplicity_convention(
            a[:, ccol], a[:, rest[0]], a[:, rest[1]], a[:, rest[2]],
            path, strict=True,
        )
        raise MolcasFormatError(
            f"{path}: BASIS_FUNCTION_IDS columns could not be assigned to "
            f"(centre, l, m, shell)"
        )
    preferred = (0, 1, 2, 3)
    layout, harmonics = next(
        (s for s in solutions if s[0] == preferred), solutions[0]
    )
    if len({h for _, h in solutions}) > 1:
        # Different column assignments imply different conventions; the
        # basis is then genuinely ambiguous and guessing would reintroduce
        # bug [A-1] by another route.
        raise MolcasFormatError(
            f"{path}: BASIS_FUNCTION_IDS admits several column "
            f"assignments implying different harmonic conventions "
            f"({sorted({h for _, h in solutions})}); cannot proceed"
        )
    ccol, lcol, mcol, scol = layout
    names = [""] * 4
    names[ccol], names[lcol] = "center", "l"
    names[mcol], names[scol] = "m", "shell"
    return (
        a[:, ccol] - 1, a[:, lcol], a[:, mcol], a[:, scol],
        harmonics, tuple(names),
    )


def read_basis(h5path: str | os.PathLike) -> BasisSet:
    """Read the contracted-GTO basis description from an OpenMolcas HDF5.

    Parameters
    ----------
    h5path : str or path-like
        An OpenMolcas ``.h5`` carrying ``PRIMITIVES``, ``PRIMITIVE_IDS``,
        ``BASIS_FUNCTION_IDS``, ``CENTER_LABELS`` and
        ``CENTER_COORDINATES``.

    Returns
    -------
    BasisSet
        Per-AO centre, element, ``l``, ``m``, shell index and radial
        primitives, plus the *detected* harmonic convention.  Exponents
        are in bohr^-2, coordinates in bohr.

    Raises
    ------
    ModelError
        If any shell has ``l > 2``.  The shake-off angular factors are
        only derived through d functions, and REVIEW.md [A-1] shows what
        silently returning zero for an unhandled component costs.
    MolcasFormatError
        On a missing dataset, an unassignable ID column layout, or a
        primitive set that does not cover every contracted shell.

    Notes
    -----
    All AOs of one shell share the radial part, so the primitive arrays
    are looked up per ``(centre, l, shell)`` triple and broadcast to the
    ``2l+1`` (spherical) or ``(l+1)(l+2)/2`` (Cartesian) components.
    """
    p = os.fspath(h5path)
    with _h5_open(p) as fh:
        labels = _decode_labels(_h5_get(fh, "CENTER_LABELS", p), p)
        n_atoms = len(labels)
        coords = np.asarray(
            _h5_get(fh, "CENTER_COORDINATES", p), dtype=float
        )
        if coords.ndim != 2 or 3 not in coords.shape:
            raise MolcasFormatError(
                f"{p}: CENTER_COORDINATES has shape {coords.shape}, "
                f"expected (n_atoms, 3)"
            )
        if coords.shape[1] != 3:
            coords = coords.T
        if coords.shape[0] != n_atoms:
            raise MolcasFormatError(
                f"{p}: CENTER_COORDINATES has {coords.shape[0]} rows but "
                f"CENTER_LABELS lists {n_atoms} centres"
            )
        bf_ids = _orient_ids(
            _h5_get(fh, "BASIS_FUNCTION_IDS", p), 4, "BASIS_FUNCTION_IDS", p
        )
        prim_ids = _orient_ids(
            _h5_get(fh, "PRIMITIVE_IDS", p), 3, "PRIMITIVE_IDS", p
        )
        prims = np.asarray(_h5_get(fh, "PRIMITIVES", p), dtype=float)

    if prims.ndim != 2 or 2 not in prims.shape:
        raise MolcasFormatError(
            f"{p}: PRIMITIVES has shape {prims.shape}, expected "
            f"(n_prim, 2) of (exponent, coefficient)"
        )
    if prims.shape[1] != 2:
        prims = prims.T
    if prims.shape[0] != prim_ids.shape[0]:
        raise MolcasFormatError(
            f"{p}: PRIMITIVES has {prims.shape[0]} rows but PRIMITIVE_IDS "
            f"has {prim_ids.shape[0]}"
        )

    centers, l, m, shell_raw, harmonics, id_layout = _resolve_ao_ids(
        bf_ids, n_atoms, p
    )
    nbas = int(centers.size)
    if l.max() > 2:
        bad = int(np.argmax(l))
        raise ModelError(
            f"{p}: AO {bad} on centre {labels[int(centers[bad])]} has "
            f"l={int(l[bad])}; the one-centre shake-off model is only "
            f"derived through l=2 (d functions), so f and higher shells "
            f"must be removed from the basis or the angular factors "
            f"extended"
        )
    if prims.size and np.any(prims[:, 0] <= 0.0):
        bad = int(np.argmin(prims[:, 0]))
        raise MolcasFormatError(
            f"{p}: PRIMITIVES row {bad} has non-positive exponent "
            f"{prims[bad, 0]!r}"
        )

    # PRIMITIVE_IDS is (centre, l, shell) but, like the AO ids, with an
    # unstable column order.  Choose the (l, shell) assignment whose
    # triples cover the AO shells; there is no other free choice, since
    # the centre column is pinned by its value range.
    pa = prim_ids.astype(np.int64, copy=False)
    want = {
        (int(centers[k]), int(l[k]), int(shell_raw[k])) for k in range(nbas)
    }
    chosen = None
    for ccol in range(3):
        if pa[:, ccol].min() < 1 or pa[:, ccol].max() > n_atoms:
            continue
        rest = [c for c in range(3) if c != ccol]
        for lcol, scol in (rest, rest[::-1]):
            trip = {
                (int(pa[r, ccol]) - 1, int(pa[r, lcol]), int(pa[r, scol]))
                for r in range(pa.shape[0])
            }
            if want <= trip:
                chosen = (ccol, lcol, scol)
                break
        if chosen is not None:
            break
    if chosen is None:
        raise MolcasFormatError(
            f"{p}: PRIMITIVE_IDS does not cover every contracted shell in "
            f"BASIS_FUNCTION_IDS under any (centre, l, shell) column "
            f"assignment; {len(want)} shells requested"
        )
    pccol, plcol, pscol = chosen
    groups: dict[tuple[int, int, int], list[int]] = {}
    for r in range(pa.shape[0]):
        key = (int(pa[r, pccol]) - 1, int(pa[r, plcol]), int(pa[r, pscol]))
        groups.setdefault(key, []).append(r)

    # shell_index: rank the file's shell labels within each (centre, l).
    shell_index = np.empty(nbas, dtype=np.int64)
    for c in range(n_atoms):
        for ll in np.unique(l[centers == c]):
            sel = (centers == c) & (l == ll)
            order = {v: k for k, v in enumerate(np.unique(shell_raw[sel]))}
            shell_index[sel] = [order[v] for v in shell_raw[sel]]

    exps: list[np.ndarray] = []
    coefs: list[np.ndarray] = []
    for k in range(nbas):
        rows = groups[(int(centers[k]), int(l[k]), int(shell_raw[k]))]
        exps.append(prims[rows, 0].copy())
        coefs.append(prims[rows, 1].copy())

    elements = np.array(
        [_strip_element(labels[int(c)]) for c in centers], dtype="<U3"
    )
    return BasisSet(
        nbas=nbas,
        n_atoms=n_atoms,
        centers=centers.astype(np.int64),
        elements=elements,
        l=l.astype(np.int64),
        m=m.astype(np.int64),
        shell_index=shell_index,
        harmonics=harmonics,
        prim_exp=tuple(exps),
        prim_coef=tuple(coefs),
        n_prim=np.array([e.size for e in exps], dtype=np.int64),
        center_coords=coords,
        atom_labels=tuple(labels),
        atom_elements=tuple(_strip_element(x) for x in labels),
        id_layout=id_layout,
        path=p,
    )


def _square_from_h5(raw: np.ndarray, nbas: int, name: str,
                    path: str) -> np.ndarray:
    """Expand an HDF5 one-electron matrix to a dense symmetric square."""
    a = np.asarray(raw, dtype=float).ravel()
    ntri = nbas * (nbas + 1) // 2
    if a.size == nbas * nbas:
        out = a.reshape(nbas, nbas)
        return 0.5 * (out + out.T)
    if a.size == ntri:
        out = np.zeros((nbas, nbas))
        out[np.tril_indices(nbas)] = a
        return out + out.T - np.diag(np.diag(out))
    raise MolcasFormatError(
        f"{path}: {name} has {a.size} values, which is neither "
        f"nbas**2={nbas * nbas} nor nbas*(nbas+1)/2={ntri} for nbas={nbas}"
    )


def read_ao_overlap_h5(h5path: str | os.PathLike) -> np.ndarray:
    """Read the AO overlap matrix from an OpenMolcas HDF5 file.

    Parameters
    ----------
    h5path : str or path-like

    Returns
    -------
    ndarray, shape (nbas, nbas)
        Symmetric, dimensionless.  Triangular and square storage are both
        accepted; ``nbas`` is taken from ``BASIS_FUNCTION_IDS``.
    """
    p = os.fspath(h5path)
    with _h5_open(p) as fh:
        nbas = _orient_ids(
            _h5_get(fh, "BASIS_FUNCTION_IDS", p), 4,
            "BASIS_FUNCTION_IDS", p,
        ).shape[0]
        for name in ("AO_OVERLAP_MATRIX", "AO_OVERLAP"):
            if name in fh:
                return _square_from_h5(
                    np.asarray(fh[name]), nbas, name, p
                )
        raise MolcasFormatError(
            f"{p}: no AO overlap dataset ('AO_OVERLAP_MATRIX'); datasets "
            f"present: {sorted(fh.keys())}"
        )


def read_ao_dipole(
    h5path: str | os.PathLike, one_centre: bool = True
) -> dict[str, np.ndarray]:
    """Read AO dipole integrals, by default in one-centre reduced form.

    Parameters
    ----------
    h5path : str or path-like
        HDF5 file carrying ``AO_MLTPL_X``, ``AO_MLTPL_Y``, ``AO_MLTPL_Z``.
    one_centre : bool, default True
        When true, return the Gelius-consistent reduced matrix described
        below.  When false, return the raw molecular dipole matrices.

    Returns
    -------
    dict
        Keys ``'x'``, ``'y'``, ``'z'``; each value is ``(nbas, nbas)`` in
        bohr (electron charge set to 1, and the electronic dipole sign
        convention of the HDF5 file is preserved).

    Notes
    -----
    With ``one_centre=True`` this returns (notes Eq. 146)

    .. code::

        D[mu, nu] = (D_full[mu, nu] - R_A(mu) * S[mu, nu])
                    * delta(atom(mu), atom(nu))

    i.e. each same-atom block is re-referenced to its own nucleus and all
    two-centre blocks are dropped.  The reason is a consistency
    requirement, not an approximation for convenience.  The bound-to-bound
    dipole enters the indirect amplitude alongside atomic subshell cross
    sections ``sigma^AO_mu`` that are, by construction, one-centre
    quantities computed about their own nucleus.  The full molecular
    matrix element carries a term ``R_A <chi_mu|chi_nu>`` that grows with
    the distance from the (arbitrary) molecular origin to the atom; that
    term has no counterpart in ``sigma^AO`` and, being proportional to a
    large overlap rather than to a small transition density, dominates the
    indirect amplitude and makes it exceed the direct one unphysically --
    the pathology REVIEW.md [P-4] documents.  Subtracting ``R_A S`` and
    keeping only same-atom blocks removes exactly that term and leaves the
    genuine intra-atomic transition dipole.
    """
    p = os.fspath(h5path)
    keys = {"x": "AO_MLTPL_X", "y": "AO_MLTPL_Y", "z": "AO_MLTPL_Z"}
    out: dict[str, np.ndarray] = {}
    with _h5_open(p) as fh:
        bf = _orient_ids(
            _h5_get(fh, "BASIS_FUNCTION_IDS", p), 4,
            "BASIS_FUNCTION_IDS", p,
        )
        nbas = bf.shape[0]
        for comp, name in keys.items():
            out[comp] = _square_from_h5(
                _h5_get(fh, name, p), nbas, name, p
            )
    if not one_centre:
        return out

    basis = read_basis(p)
    s_ao = read_ao_overlap_h5(p)
    same_atom = basis.centers[:, None] == basis.centers[None, :]
    r_a = basis.center_coords[basis.centers]  # (nbas, 3), bohr
    for axis, comp in enumerate("xyz"):
        shifted = out[comp] - r_a[:, axis][:, None] * s_ao
        out[comp] = np.where(same_atom, shifted, 0.0)
    return out


# ── synthetic case generation ───────────────────────────────────────────────

_INPORB_FMT = "%22.14E"
_OVERLAP_FMT = "%25.16E"


def _round_trip(a: np.ndarray, fmt: str) -> np.ndarray:
    """Values as they will read back after formatting with ``fmt``.

    The generated fixtures must satisfy their invariants (orthonormality,
    symmetry) in the *written* decimals, not merely in the float64 values
    that produced them, or the round-trip tests would carry a tolerance
    that hides real parsing errors.
    """
    flat = np.array([float(fmt % v) for v in np.asarray(a, float).ravel()])
    return flat.reshape(np.shape(a))


def _write_block(fh, values: Sequence[float], fmt: str,
                 per_line: int) -> None:
    for start in range(0, len(values), per_line):
        chunk = values[start:start + per_line]
        fh.write("".join(fmt % v for v in chunk) + "\n")


def _write_inporb(
    path: str, coeff: np.ndarray, occ: np.ndarray | None, version: str,
    title: str,
) -> None:
    """Write a single-symmetry INPORB 2.0 or 2.2 file.

    The ``#INFO`` block matches real OpenMolcas output exactly: three data
    lines carrying ``(iUHF, nSym, iWFtype)``, then ``nBas`` and ``nOrb`` per
    symmetry.  Real v20.10 files write ``0 1 2`` for an SCF ``ScfOrb`` and
    ``0 1 0`` for a RASSCF ``RasOrb``; the fixture must not invent a
    different layout or it stops being a test of the reader.
    """
    nbas, nmo = coeff.shape
    wf_type = 0 if occ is not None else 2
    with open(path, "w") as fh:
        fh.write(f"#INPORB {version}\n")
        fh.write("#INFO\n")
        fh.write(f"* {title}\n")
        fh.write(f"       0       1       {wf_type}\n")
        fh.write(f"    {nbas:4d}\n")
        fh.write(f"    {nmo:4d}\n")
        fh.write("#ORB\n")
        for k in range(nmo):
            fh.write(f"* ORBITAL    1 {k + 1:4d}\n")
            _write_block(fh, list(coeff[:, k]), _INPORB_FMT, 5)
        if occ is not None:
            fh.write("#OCC\n")
            fh.write("* OCCUPATION NUMBERS\n")
            _write_block(fh, list(occ), _INPORB_FMT, 5)
        fh.write("#INDEX\n")
        fh.write("* 1234567890\n")
        if occ is None:
            types = "i" * nmo
        else:
            types = "".join(
                "i" if o >= OCC_DOUBLY_MIN else
                ("2" if o > OCC_VIRTUAL_MAX else "s") for o in occ
            )
        for start in range(0, nmo, 10):
            fh.write(f" {start // 10} {types[start:start + 10]}\n")


def _m_labels(l: int, harmonics: str) -> list[int]:
    """Component labels of one shell, in the order AOs are laid out."""
    if harmonics == "spherical":
        return list(range(-l, l + 1))
    # Cartesian components are labelled 1.. in the order the old code
    # assumed (xx, yy, zz, xy, xz, yz for l = 2); only the count and the
    # distinctness matter to the reader.
    return list(range(1, (l + 1) * (l + 2) // 2 + 1))


def write_synthetic_case(
    dirpath: str | os.PathLike,
    *,
    harmonics: str = "spherical",
    n_neu_occ: int = 5,
    occ_style: str = "rohf",
    hole_i: int = 0,
    hole_j: int = 3,
    version_neutral: str = "2.0",
    version_dication: str = "2.2",
    rotation: float = 0.08,
    seed: int = 20240517,
) -> SyntheticCase:
    """Generate a small, complete, self-consistent fake OpenMolcas case.

    Two centres carrying s, p and (on the second centre) d shells, so
    ``nbas`` is 14 for spherical harmonics and 15 for Cartesian.  Written
    files: an ``INPORB`` for the neutral, a ``RasOrb`` with a valid
    ``#OCC`` block for the dication, a text AO overlap, and an ``.h5``
    with every dataset the readers above need.

    Parameters
    ----------
    dirpath : str or path-like
        Directory to create and populate.
    harmonics : {'spherical', 'cartesian'}
        Which convention the generated ``.h5`` encodes, through the number
        of distinct ``m`` labels per d shell (5 or 6).  This is what
        exercises the [A-1] detection.
    n_neu_occ : int
        Number of doubly occupied neutral spatial MOs, ``n``.
    occ_style : {'rohf', 'natural'}
        ``'rohf'`` writes two MOs at occupation 1.  ``'natural'`` writes
        three at 5/3 plus one at 1, the state-averaged pattern that
        triggers the single-determinant approximation in
        :func:`rohf_hole_indices`.
    hole_i, hole_j : int
        0-based MO columns of the two holes, ``hole_i < hole_j <
        n_neu_occ``.  Ignored for ``occ_style='natural'``, which needs
        three consecutive fractional partners and places them at
        ``0, 1, 2`` with the valence hole at ``n_neu_occ - 1``.
    version_neutral, version_dication : {'2.0', '2.2'}
        Format versions to write, so one call exercises both branches of
        the ``#INFO`` layout.
    rotation : float
        Magnitude of the orbital-relaxation rotation between the neutral
        and dication sets, in radians.  ``Q`` is then near-identity but
        not identity, which is what makes the Dyson amplitudes
        non-trivial; ``0.0`` would put the code in the frozen-orbital
        limit exactly.
    seed : int
        Seed of the random generator, so fixtures are reproducible.

    Returns
    -------
    SyntheticCase
        Paths plus the reference arrays *as written*, see that class.

    Notes
    -----
    Construction order matters and is deliberate.  The overlap is built
    first from a random positive-definite matrix with unit diagonal, then
    quantised to the decimals the text file will carry.  Both orbital sets
    are then built from that *quantised* overlap by symmetric
    orthogonalisation, ``C = S^(-1/2) U`` with ``U`` random orthogonal, so
    ``C^T S C = 1`` holds in the values that are written rather than only
    in the values that generated them.  The dication set is
    ``C_dic = C_neu R`` with ``R`` a Cayley-transform orthogonal matrix
    close to the identity, which keeps ``C_dic`` orthonormal exactly while
    making ``Q = C_dic^T S C_neu = R^T`` a non-trivial near-identity.
    """
    d = os.fspath(dirpath)
    os.makedirs(d, exist_ok=True)
    if harmonics not in ("spherical", "cartesian"):
        raise ValueError(
            f"harmonics must be 'spherical' or 'cartesian', got "
            f"{harmonics!r}"
        )
    if occ_style not in ("rohf", "natural"):
        raise ValueError(
            f"occ_style must be 'rohf' or 'natural', got {occ_style!r}"
        )
    rng = np.random.default_rng(seed)

    # Shell layout: (centre, l, shell label).
    #
    # The two centres are S and F so that the generated case exercises the
    # real Yeh-Lindau table rather than falling through to sigma = 0: those
    # are the elements dpi.atomic_sigma tabulates, and a synthetic case that
    # silently produced zero cross sections would be a poor end-to-end test.
    # Shell labels are the sequential index within a (centre, l) group, so
    # (0, 0, 1) and (0, 0, 2) are the S 1s and 2s shells.
    atom_labels = ("S1", "F1")
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.9]])
    shells = [
        (0, 0, 1), (0, 0, 2), (0, 1, 1),
        (1, 0, 1), (1, 1, 1), (1, 2, 1),
    ]
    centers_l, l_l, m_l, shell_l = [], [], [], []
    for c, ll, sh in shells:
        for mm in _m_labels(ll, harmonics):
            centers_l.append(c)
            l_l.append(ll)
            m_l.append(mm)
            shell_l.append(sh)
    nbas = len(centers_l)
    nmo = nbas
    if not 0 <= hole_i < hole_j < n_neu_occ:
        raise ValueError(
            f"need 0 <= hole_i < hole_j < n_neu_occ, got "
            f"({hole_i}, {hole_j}, {n_neu_occ})"
        )
    if n_neu_occ > nmo:
        raise ValueError(
            f"n_neu_occ={n_neu_occ} exceeds nmo={nmo} for this layout"
        )

    # Overlap: unit diagonal, off-diagonals decaying with a random
    # symmetric perturbation, then made positive definite and quantised.
    a = rng.standard_normal((nbas, nbas))
    s_raw = 0.35 * (a + a.T) / (2.0 * nbas ** 0.5)
    np.fill_diagonal(s_raw, 0.0)
    s_raw = s_raw + np.eye(nbas)
    w = np.linalg.eigvalsh(s_raw)
    if w.min() < 0.05:
        s_raw = (s_raw + (0.05 - w.min()) * np.eye(nbas)) / (
            1.0 + (0.05 - w.min())
        )
    s_ao = _round_trip(s_raw, _OVERLAP_FMT)
    s_ao = 0.5 * (s_ao + s_ao.T)
    s_ao = _round_trip(s_ao, _OVERLAP_FMT)

    # C_neu = S^(-1/2) U, exactly orthonormal w.r.t. s_ao.
    w, v = np.linalg.eigh(s_ao)
    s_inv_half = (v * w ** -0.5) @ v.T
    g = rng.standard_normal((nbas, nmo))
    q_orth, r_orth = np.linalg.qr(g)
    q_orth = q_orth * np.sign(np.diag(r_orth))
    c_neu = _round_trip(s_inv_half @ q_orth, _INPORB_FMT)

    # Cayley transform of a small antisymmetric generator: orthogonal to
    # machine precision without needing a matrix exponential.
    k = rng.standard_normal((nmo, nmo))
    k = rotation * (k - k.T) / (2.0 * nmo ** 0.5)
    ident = np.eye(nmo)
    rot = np.linalg.solve(ident - 0.5 * k, ident + 0.5 * k)
    c_dic = _round_trip(c_neu @ rot, _INPORB_FMT)

    occ_neu = np.zeros(nmo)
    occ_neu[:n_neu_occ] = 2.0
    occ_neu = _round_trip(occ_neu, _INPORB_FMT)

    occ_dic = np.zeros(nmo)
    if occ_style == "rohf":
        occ_dic[:n_neu_occ] = 2.0
        occ_dic[hole_i] = 1.0
        occ_dic[hole_j] = 1.0
        h_i, h_j = hole_i, hole_j
    else:
        if n_neu_occ < 5:
            raise ValueError(
                f"occ_style='natural' needs n_neu_occ >= 5 to host three "
                f"fractional partners plus a valence hole, got {n_neu_occ}"
            )
        occ_dic[:n_neu_occ] = 2.0
        occ_dic[0:3] = 5.0 / 3.0
        occ_dic[n_neu_occ - 1] = 1.0
        h_i, h_j = 0, n_neu_occ - 1
    occ_dic = _round_trip(occ_dic, _INPORB_FMT)

    inporb_neu = os.path.join(d, "INPORB")
    inporb_dic = os.path.join(d, "RasOrb.dication")
    ovl_path = os.path.join(d, "ao_overlap.txt")
    h5_path = os.path.join(d, "case.h5")

    _write_inporb(inporb_neu, c_neu, occ_neu, version_neutral,
                  "synthetic neutral RHF orbitals")
    _write_inporb(inporb_dic, c_dic, occ_dic, version_dication,
                  "synthetic dication OSRHF orbitals")

    with open(ovl_path, "w") as fh:
        fh.write("# synthetic AO overlap, lower triangle, row-major\n")
        fh.write(f"* nbas = {nbas}\n")
        tri = list(s_ao[np.tril_indices(nbas)])
        _write_block(fh, tri, _OVERLAP_FMT, 4)

    # Dipole integrals: build the full molecular matrix so that its
    # same-atom blocks reduce exactly to a chosen intra-atomic part.
    r_a = coords[np.asarray(centers_l)]
    dip_full: dict[str, np.ndarray] = {}
    dip_one: dict[str, np.ndarray] = {}
    cen = np.asarray(centers_l)
    same_atom = cen[:, None] == cen[None, :]
    for axis, comp in enumerate("xyz"):
        b = rng.standard_normal((nbas, nbas))
        intra = 0.25 * (b + b.T)
        full = 0.5 * (
            r_a[:, axis][:, None] + r_a[:, axis][None, :]
        ) * s_ao + intra
        dip_full[comp] = full
        # Reproduce the reader's arithmetic exactly, so the reference and
        # the reader agree to the last bit rather than to a tolerance.
        dip_one[comp] = np.where(
            same_atom, full - r_a[:, axis][:, None] * s_ao, 0.0
        )

    n_prim_per_shell = 3
    prim_rows = []
    prim_id_rows = []
    for c, ll, sh in shells:
        base = 0.6 * (2.4 ** sh) * (1.0 + 0.7 * ll)
        for ip in range(n_prim_per_shell):
            prim_rows.append((base * 3.1 ** ip, 0.5 ** (ip + 1)))
            prim_id_rows.append((c + 1, ll, sh))

    import h5py

    with h5py.File(h5_path, "w") as fh:
        fh.attrs["MOLCAS_MODULE"] = np.bytes_("SYNTHETIC")
        fh.attrs["NSYM"] = 1
        fh.attrs["NBAS"] = np.array([nbas], dtype=np.int64)
        fh.create_dataset(
            "CENTER_LABELS",
            data=np.array([s.ljust(8).encode() for s in atom_labels],
                          dtype="S8"),
        )
        fh.create_dataset("CENTER_COORDINATES", data=coords)
        fh.create_dataset("CENTER_CHARGES",
                          data=np.array([6.0, 8.0]))
        fh.create_dataset(
            "BASIS_FUNCTION_IDS",
            data=np.array(
                [[centers_l[k] + 1, l_l[k], m_l[k], shell_l[k]]
                 for k in range(nbas)],
                dtype=np.int64,
            ),
        )
        fh.create_dataset(
            "PRIMITIVE_IDS", data=np.array(prim_id_rows, dtype=np.int64)
        )
        fh.create_dataset("PRIMITIVES", data=np.array(prim_rows, dtype=float))
        fh.create_dataset(
            "AO_OVERLAP_MATRIX", data=s_ao.reshape(-1).copy()
        )
        for comp in "xyz":
            fh.create_dataset(
                f"AO_MLTPL_{comp.upper()}",
                data=dip_full[comp].reshape(-1).copy(),
            )
        fh.create_dataset("MO_VECTORS", data=c_neu.T.reshape(-1).copy())
        fh.create_dataset("MO_OCCUPATIONS", data=occ_neu.copy())

    return SyntheticCase(
        dirpath=d,
        inporb_neutral=inporb_neu,
        inporb_dication=inporb_dic,
        overlap=ovl_path,
        h5=h5_path,
        nbas=nbas,
        nmo=nmo,
        n_neu_occ=n_neu_occ,
        harmonics=harmonics,
        s_ao=s_ao,
        c_neu=c_neu,
        c_dic=c_dic,
        occ_neutral=occ_neu,
        occ_dication=occ_dic,
        occ_style=occ_style,
        hole_i=h_i,
        hole_j=h_j,
        dipole_full=dip_full,
        dipole_one_centre=dip_one,
        center_coords=coords,
        atom_labels=atom_labels,
    )
