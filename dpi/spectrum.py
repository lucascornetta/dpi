"""Energy-sharing quadrature and Voigt broadening.

The singly-differential DPI intensity of a core-valence state ``f`` is
integrated over the sharing of the excess energy between the two emitted
electrons,

    I_f = prefactor(omega) * integral_0^E_excess A_f(eps2) d(eps2),

with ``eps1 + eps2 = E_excess = omega - DIP_f``.  The substitution
``eps2 = E_excess * t**2`` maps the integral onto ``t`` in ``[0, 1]`` with
Jacobian ``d(eps2) = 2*E_excess*t*dt`` and ``k2 = t*sqrt(2*E_excess)``,
which removes the integrable ``k2 -> 0`` singularity carried by the
shake-off block ``S_i`` (see REVIEW.md, and [B-1] for the trailing factor
``t`` the old integrated shake-off diagnostic dropped).

Units.  Energies crossing this module's boundary are in eV
(``*_ev``); the quadrature works in atomic units internally.  Intensities
are *relative*, in Mb*a.u. -- see :func:`prefactor`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy.special import wofz

from . import amplitudes as amp
from .amplitudes import TermSwitches
from .constants import (
    CHANNEL_LABELS,
    ModelError,
    au_to_ev,
    ev_to_au,
)

__all__ = [
    "StateResult",
    "QuadratureGrid",
    "quadrature_grid",
    "prefactor",
    "integrate_state",
    "convergence",
    "voigt",
    "broaden",
]


@dataclass
class StateResult:
    """Integrated DPI intensity of one core-valence state in one channel.

    Attributes
    ----------
    index:
        0-based state index as supplied by the driver.
    channel:
        ``'singlet'``, ``'triplet'``, ``'jc32'`` or ``'jc12'``.
    label:
        Spectroscopic term label, or ``None``.
    e_dication_ev:
        Raw dication total energy or term energy exactly as read, eV.  Not
        used by the physics.
    e_shifted_ev:
        Position of the state on the plotted energy axis, eV.  This is a
        *cosmetic* display shift; it must never re-enter the physics.
    dip_ev:
        Double-ionization potential used for the physics, eV.  Carried
        explicitly per state because ``E_excess`` depends on it, and
        deriving it from ``e_shifted_ev`` mixes a plotting offset into the
        cross section -- a real trap in the old code.
    e_excess_ev:
        ``omega - DIP_f``, eV.  Negative or zero means the channel is
        closed.
    intensity:
        Integrated ``A_f``, relative units of Mb*a.u.
    terms:
        Per-term integrals in the same units, keyed by
        :data:`dpi.amplitudes.TERM_ORDER`.  For an open channel they sum
        to ``intensity``.
    open:
        ``False`` when ``E_excess <= 0``.  A closed state is returned with
        zero intensity rather than dropped, so that state indices, energy
        lists and term tables stay in registration (bug [C-2]).
    negative:
        ``True`` when the integrated intensity is negative.  The value is
        reported, not clamped: clamping hides a model breakdown and
        silently deletes states from the spectrum (bug [B-3]).  The
        channel at risk is the *triplet*, whose ``-4*c_cross`` is a signed
        contraction with no positivity bound; the singlet is non-negative
        by Cauchy-Schwarz unless the optional ``X_f`` term is enabled.  See
        :func:`dpi.amplitudes.amplitude_ls` for the proof.
    amplitude_min:
        Most negative value of ``A_f`` on the quadrature grid, Mb*a.u.
        Non-zero even for states whose *integral* is positive, so it flags
        a partial breakdown the integral alone would hide.
    p_i, p_j:
        Spectroscopic factors of the two Dyson orbitals (AO-metric norms,
        dimensionless), copied from the Dyson objects.  These are the
        model's own measure of the distance from the frozen-orbital limit
        and are written to ``sticks.dat``.
    n_quad:
        Number of Gauss-Legendre nodes used.
    """

    index: int
    channel: str
    label: str | None
    e_dication_ev: float
    e_shifted_ev: float
    dip_ev: float
    e_excess_ev: float
    intensity: float
    terms: dict[str, float] = field(default_factory=dict)
    open: bool = True
    negative: bool = False
    amplitude_min: float = 0.0
    p_i: float = float("nan")
    p_j: float = float("nan")
    n_quad: int = 0


@dataclass(frozen=True)
class QuadratureGrid:
    """Gauss-Legendre nodes of the energy-sharing integral.

    Attributes
    ----------
    t:
        ``(nq,)`` nodes on ``[0, 1]``, dimensionless.
    weights:
        ``(nq,)`` Gauss-Legendre weights on ``[0, 1]``, dimensionless.
    eps1_ev, eps2_ev:
        ``(nq,)`` fast- and slow-electron kinetic energies, eV.
    k2_au:
        ``(nq,)`` slow-electron momentum ``t*sqrt(2*E_excess)``, a.u.
    jacobian_au:
        ``(nq,)`` ``d(eps2)/dt = 2*E_excess*t``, a.u.  Multiplying by
        ``weights`` gives the measure of the ``eps2`` integral.
    e_excess_au:
        Excess energy, a.u.
    """

    t: np.ndarray
    weights: np.ndarray
    eps1_ev: np.ndarray
    eps2_ev: np.ndarray
    k2_au: np.ndarray
    jacobian_au: np.ndarray
    e_excess_au: float

    @property
    def measure_au(self) -> np.ndarray:
        """``(nq,)`` combined weight ``w_q * d(eps2)/dt``, a.u."""
        return self.weights * self.jacobian_au


def quadrature_grid(e_excess_ev: float, n_quad: int = 200) -> QuadratureGrid:
    """Build the energy-sharing quadrature grid.

    Parameters
    ----------
    e_excess_ev:
        ``omega - DIP_f``, eV.  Must be positive; a closed channel is
        handled by :func:`integrate_state`, not here.
    n_quad:
        Number of Gauss-Legendre nodes.

    Returns
    -------
    QuadratureGrid

    Notes
    -----
    ``eps2 = E_excess * t**2`` so that ``k2 = sqrt(2*eps2) =
    t*sqrt(2*E_excess)`` is *linear* in the quadrature variable.  The
    shake-off block behaves as ``k2 * P(k2) ~ k2^(2l+1)`` as ``k2 -> 0``,
    and the Jacobian ``2*E_excess*t`` supplies the extra power of ``t``
    that makes the integrand polynomial-like near the origin, where the
    integrand is largest.
    """
    if not np.isfinite(e_excess_ev) or e_excess_ev <= 0.0:
        raise ModelError(
            "spectrum.quadrature_grid: e_excess_ev must be positive, got "
            f"{e_excess_ev!r}; closed channels are handled by "
            "integrate_state."
        )
    if n_quad < 2:
        raise ModelError(
            f"spectrum.quadrature_grid: n_quad must be >= 2, got {n_quad}."
        )

    x, w = np.polynomial.legendre.leggauss(int(n_quad))
    # [-1, 1] -> [0, 1]
    t = 0.5 * (x + 1.0)
    weights = 0.5 * w

    e_exc_au = ev_to_au(float(e_excess_ev))
    eps2_au = e_exc_au * t * t
    eps1_au = e_exc_au - eps2_au
    # Guard the endpoint against round-off producing eps1 < 0, which would
    # be handed to sigma_mu(eps1 + I_mu) below the AO threshold.
    eps1_au = np.clip(eps1_au, 0.0, None)
    k2_au = t * np.sqrt(2.0 * e_exc_au)
    jac = 2.0 * e_exc_au * t

    return QuadratureGrid(
        t=t,
        weights=weights,
        eps1_ev=au_to_ev(eps1_au),
        eps2_ev=au_to_ev(eps2_au),
        k2_au=k2_au,
        jacobian_au=jac,
        e_excess_au=e_exc_au,
    )


def prefactor(omega_ev: float) -> float:
    """Absolute-normalisation hook.  Returns ``1.0``.

    Parameters
    ----------
    omega_ev:
        Photon energy, eV.  Accepted and documented so that a future
        absolute normalisation is a one-line change here and nowhere else.

    Returns
    -------
    float
        ``1.0``, dimensionless.

    Notes
    -----
    The output of this package is a **relative** DPI intensity carrying the
    mixed unit Mb*a.u., not an absolute cross section in Mb.  The
    ``4*pi^2*omega/(3c)`` and ``k1*k2`` phase-space prefactors of Eq. (78)
    cancel against factors already implicit in the definitions of
    ``sigma^AO_mu`` (Eq. 81) and ``k2*P_nu`` (Eq. 92).  Recovering an
    absolute cross section additionally requires three convention
    inconsistencies in the notes to be resolved (REVIEW.md [P-6]):

    * the ``(2*pi)^-3`` normalisation of ``P^shake`` and its ``dOmega_k``
      integral-versus-average reading ([B-2]);
    * the factor-3 disagreement between Eqs. (113) and (116) in the weight
      of ``G_ij`` ([P-1]);
    * dimensional reconciliation of ``sigma^AO_mu`` in Mb against
      ``k2*P_nu`` in atomic units.

    Until those are settled, returning anything other than ``1.0`` here
    would attach a spurious absolute scale to a relative spectrum.  Ratios
    between states, channels and photon energies are unaffected, which is
    where the model's predictive content lies.
    """
    if not np.isfinite(omega_ev):
        raise ModelError(
            f"spectrum.prefactor: omega_ev must be finite, got {omega_ev!r}."
        )
    return 1.0


def _resolve_sigma(cfg: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable ``eps1_ev -> (nq, nbas)`` sigma grid in Mb."""
    fn = getattr(cfg, "sigma_at_eps1_grid", None)
    if callable(fn):
        return fn
    builder = getattr(cfg, "sigma_builder", None)
    if builder is not None and hasattr(builder, "at_eps1_grid"):
        return builder.at_eps1_grid
    raise ModelError(
        "spectrum.integrate_state: cfg exposes neither "
        "sigma_at_eps1_grid(eps1_ev) nor sigma_builder.at_eps1_grid; the "
        "per-AO cross sections must be resolvable once per state, not "
        "rebuilt per quadrature point."
    )


def _resolve_pshake(cfg: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable ``k_au -> (nq, nbas)`` shake-off density, a.u."""
    fn = getattr(cfg, "p_shake_at_k", None)
    if callable(fn):
        return fn
    basis = getattr(cfg, "basis", None)
    if basis is not None:
        from .shakeoff import p_shake  # local: keeps this module import-light

        return lambda k: p_shake(basis, k)
    raise ModelError(
        "spectrum.integrate_state: cfg exposes neither p_shake_at_k(k_au) "
        "nor basis; the shake-off probabilities cannot be evaluated."
    )


def build_grid_inputs(
    cfg: Any, grid: QuadratureGrid
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate ``sigma`` and ``P^shake`` on the whole quadrature grid.

    Parameters
    ----------
    cfg:
        Run configuration; see :func:`integrate_state`.
    grid:
        Quadrature grid from :func:`quadrature_grid`.

    Returns
    -------
    tuple[ndarray, ndarray]
        ``sigma_grid`` ``(nq, nbas)`` in Mb, already evaluated at the
        per-AO argument ``eps1 + I_mu``, and ``pshake_grid`` ``(nq, nbas)``
        in a.u.  Both are obtained in one vectorised call each: the old
        code reopened the HDF5 file and rebuilt the AO map once per
        quadrature point per state per channel ([C-4]).
    """
    sigma_grid = np.asarray(_resolve_sigma(cfg)(grid.eps1_ev), dtype=float)
    pshake_grid = np.asarray(_resolve_pshake(cfg)(grid.k2_au), dtype=float)
    nq = grid.t.size
    for name, arr in (("sigma_grid", sigma_grid),
                      ("pshake_grid", pshake_grid)):
        if arr.ndim != 2 or arr.shape[0] != nq:
            raise ModelError(
                f"spectrum.build_grid_inputs: {name} has shape "
                f"{arr.shape}, expected (nq, nbas) with nq = {nq}."
            )
    if sigma_grid.shape[1] != pshake_grid.shape[1]:
        raise ModelError(
            "spectrum.build_grid_inputs: sigma_grid has "
            f"{sigma_grid.shape[1]} AOs but pshake_grid has "
            f"{pshake_grid.shape[1]}; they must share the AO basis."
        )
    return sigma_grid, pshake_grid


def _cfg_terms(cfg: Any) -> TermSwitches:
    terms = getattr(cfg, "terms", None)
    if terms is None:
        raise ModelError(
            "spectrum.integrate_state: cfg has no 'terms' attribute; a "
            "TermSwitches instance is required so that the active model is "
            "recorded in the output header."
        )
    return terms


def _cfg_omega(cfg: Any) -> float:
    omega = getattr(cfg, "omega_ev", None)
    if omega is None:
        raise ModelError(
            "spectrum.integrate_state: cfg has no 'omega_ev'; the photon "
            "energy sets E_excess and cannot be defaulted."
        )
    return float(omega)


def integrate_state(
    dyson: Any,
    cfg: Any,
    dip_ev: float,
    channel: str,
    n_quad: int = 200,
    *,
    dyson_triplet: Any = None,
    index: int | None = None,
    label: str | None = None,
    e_dication_ev: float | None = None,
    e_shifted_ev: float | None = None,
) -> StateResult:
    """Integrate ``A_f`` over the energy sharing for one state and channel.

    Parameters
    ----------
    dyson:
        Dyson objects of the state (see
        :func:`dpi.amplitudes.build_blocks`).  Optional attributes
        ``p_i``, ``p_j`` (spectroscopic factors) and ``meta`` (a mapping
        that may supply ``index``, ``label``, ``e_dication_ev``,
        ``e_shifted_ev``) are copied into the result when the
        corresponding keyword is not given.
    cfg:
        Run configuration.  Required: ``omega_ev`` (photon energy, eV) and
        ``terms`` (:class:`dpi.amplitudes.TermSwitches`).  The per-AO cross
        sections come from ``cfg.sigma_at_eps1_grid(eps1_ev)`` or
        ``cfg.sigma_builder.at_eps1_grid``; the shake-off densities from
        ``cfg.p_shake_at_k(k_au)`` or ``cfg.basis``.  An optional
        ``display_shift_ev`` (eV) positions the state on the plot axis and
        never enters the physics.
    dip_ev:
        Double-ionization potential of *this* state, eV.  ``E_excess =
        omega - dip_ev``.  Passed explicitly rather than derived from any
        display shift.
    channel:
        ``'singlet'``, ``'triplet'``, ``'jc32'`` or ``'jc12'``.
    n_quad:
        Gauss-Legendre nodes.  200 is the production value; justify it with
        :func:`convergence`.

    Returns
    -------
    StateResult
        ``intensity`` and ``terms`` in Mb*a.u. (relative, see
        :func:`prefactor`).

    Notes
    -----
    A state whose ``E_excess <= 0`` is energetically **closed**: it is
    returned with ``open=False`` and zero intensity, keeping it in the
    state list rather than dropping it.

    A negative integral is reported through ``negative`` and never clamped
    ([B-3]).
    """
    if channel not in CHANNEL_LABELS:
        raise ModelError(
            f"spectrum.integrate_state: unknown channel {channel!r}; "
            f"expected one of {tuple(CHANNEL_LABELS)}."
        )

    meta = getattr(dyson, "meta", None) or {}
    if index is None:
        index = int(meta.get("index", -1))
    if label is None:
        label = meta.get("label")
    if e_dication_ev is None:
        e_dication_ev = float(meta.get("e_dication_ev", float("nan")))
    if e_shifted_ev is None:
        if "e_shifted_ev" in meta:
            e_shifted_ev = float(meta["e_shifted_ev"])
        else:
            # The display shift is cosmetic: it moves the stick on the
            # plotted axis and is deliberately kept out of E_excess.
            shift = float(getattr(cfg, "display_shift_ev", 0.0) or 0.0)
            base = e_dication_ev if np.isfinite(e_dication_ev) else dip_ev
            e_shifted_ev = float(base) + shift

    terms = _cfg_terms(cfg)
    omega_ev = _cfg_omega(cfg)
    dip_ev = float(dip_ev)
    e_excess_ev = omega_ev - dip_ev

    p_i = float(getattr(dyson, "p_i", float("nan")))
    p_j = float(getattr(dyson, "p_j", float("nan")))

    if e_excess_ev <= 0.0:
        return StateResult(
            index=index,
            channel=channel,
            label=label,
            e_dication_ev=float(e_dication_ev),
            e_shifted_ev=float(e_shifted_ev),
            dip_ev=dip_ev,
            e_excess_ev=float(e_excess_ev),
            intensity=0.0,
            terms={name: 0.0 for name in terms.active()},
            open=False,
            negative=False,
            amplitude_min=0.0,
            p_i=p_i,
            p_j=p_j,
            n_quad=int(n_quad),
        )

    grid = quadrature_grid(e_excess_ev, n_quad=n_quad)
    sigma_grid, pshake_grid = build_grid_inputs(cfg, grid)
    blk = amp.build_blocks(dyson, sigma_grid, pshake_grid, grid.k2_au, terms)

    # A jj-coupled peak is A(S=0) + A(S=1), and those two LS amplitudes
    # belong to different dication states with different relaxed orbitals.
    # `dyson` carries the S=0 set and `dyson_triplet` the S=1 set; both are
    # projected onto the SAME quadrature grid, which is why the second
    # Blocks is built here rather than by the caller.
    blk_t = None
    if dyson_triplet is not None:
        blk_t = amp.build_blocks(dyson_triplet, sigma_grid, pshake_grid,
                                 grid.k2_au, terms)
    elif channel in ("jc32", "jc12"):
        raise ModelError(
            f"spectrum.integrate_state: channel {channel!r} needs both the "
            f"S_dic=0 and S_dic=1 dication orbital sets, but only one was "
            f"given.  Pass the triplet Dyson objects as dyson_triplet."
        )

    measure = grid.measure_au
    pre = prefactor(omega_ev)

    breakdown = amp.term_breakdown(blk, channel, terms, blk_t)
    term_integrals = {
        name: float(pre * np.dot(measure, arr))
        for name, arr in breakdown.items()
    }
    a_total = amp.amplitude(blk, channel, terms, blk_t)
    intensity = float(pre * np.dot(measure, a_total))

    return StateResult(
        index=index,
        channel=channel,
        label=label,
        e_dication_ev=float(e_dication_ev),
        e_shifted_ev=float(e_shifted_ev),
        dip_ev=dip_ev,
        e_excess_ev=float(e_excess_ev),
        intensity=intensity,
        terms=term_integrals,
        open=True,
        negative=bool(intensity < 0.0),
        amplitude_min=float(min(0.0, float(np.min(a_total)))),
        p_i=p_i,
        p_j=p_j,
        n_quad=int(n_quad),
    )


def convergence(
    dyson: Any,
    cfg: Any,
    dip_ev: float,
    channel: str,
    n_quad: int = 200,
    *,
    dyson_triplet: Any = None,
) -> dict[str, float]:
    """Quadrature residual between ``n_quad`` and ``2*n_quad`` nodes.

    Parameters
    ----------
    dyson, cfg, dip_ev, channel:
        As in :func:`integrate_state`.
    n_quad:
        Node count whose adequacy is being tested.

    Returns
    -------
    dict[str, float]
        ``n_quad``, ``intensity`` (Mb*a.u. at ``n_quad``),
        ``intensity_2n`` (at ``2*n_quad``), ``abs_residual`` and
        ``rel_residual`` (dimensionless, ``nan`` for a closed channel or a
        vanishing intensity).  This is the number to quote when justifying
        the production ``n_quad``.
    """
    lo = integrate_state(
        dyson, cfg, dip_ev, channel, n_quad=n_quad,
        dyson_triplet=dyson_triplet)
    hi = integrate_state(
        dyson, cfg, dip_ev, channel, n_quad=2 * n_quad,
        dyson_triplet=dyson_triplet)
    abs_res = abs(hi.intensity - lo.intensity)
    denom = abs(hi.intensity)
    return {
        "n_quad": float(n_quad),
        "intensity": lo.intensity,
        "intensity_2n": hi.intensity,
        "abs_residual": abs_res,
        "rel_residual": abs_res / denom if denom > 0.0 else float("nan"),
    }


def _voigt_kernel(
    dx: np.ndarray, sigma_g: float, gamma_l: float, where: str
) -> np.ndarray:
    """Unit-area Voigt evaluated at displacements ``dx`` (eV).

    Shared by :func:`voigt` and :func:`broaden` so that the stick sum and a
    single profile cannot drift apart.  Returns 1/eV.
    """
    sigma_g = float(sigma_g)
    gamma_l = float(gamma_l)
    if sigma_g < 0.0 or gamma_l < 0.0:
        raise ModelError(
            f"spectrum.{where}: widths must be non-negative, got "
            f"sigma_g={sigma_g!r}, gamma_l={gamma_l!r}."
        )
    if sigma_g == 0.0 and gamma_l == 0.0:
        raise ModelError(
            f"spectrum.{where}: sigma_g and gamma_l are both zero; a "
            "zero-width area-normalised profile is a delta function and "
            "cannot be sampled on a grid."
        )
    # Both limits are taken exactly rather than by regularising a width, so
    # a run with a single broadening mechanism is not approximated.
    if sigma_g == 0.0:
        return gamma_l / (np.pi * (dx * dx + gamma_l * gamma_l))
    if gamma_l == 0.0:
        return np.exp(-0.5 * (dx / sigma_g) ** 2) / (
            sigma_g * np.sqrt(2.0 * np.pi)
        )
    z = (dx + 1j * gamma_l) / (sigma_g * np.sqrt(2.0))
    return np.real(wofz(z)) / (sigma_g * np.sqrt(2.0 * np.pi))


def voigt(
    x: np.ndarray | float,
    x0: float,
    area: float,
    sigma_g: float,
    gamma_l: float,
) -> np.ndarray:
    """Area-normalised Voigt profile.

    Parameters
    ----------
    x:
        Energy axis, eV (scalar or array of any shape).
    x0:
        Line centre, eV.
    area:
        Integrated area of the profile, in whatever units the stick
        intensity carries (Mb*a.u. here).  The returned profile has that
        area, so its values carry units of ``area``/eV.
    sigma_g:
        Gaussian standard deviation, eV.  ``0`` gives a pure Lorentzian.
    gamma_l:
        Lorentzian half-width at half-maximum, eV.  ``0`` gives a pure
        Gaussian.

    Returns
    -------
    ndarray
        Profile values, same shape as ``x``, units ``area``/eV.

    Notes
    -----
    Implemented as ``area * Re[w(z)] / (sigma_g*sqrt(2*pi))`` with
    ``z = ((x - x0) + i*gamma_l)/(sigma_g*sqrt(2))`` and ``w`` the Faddeeva
    function :func:`scipy.special.wofz`.  Both limits are taken explicitly
    rather than by regularising a width, so a run with a single broadening
    mechanism is exact.
    """
    x = np.asarray(x, dtype=float)
    dx = x - float(x0)
    return float(area) * _voigt_kernel(dx, sigma_g, gamma_l, where="voigt")


def broaden(
    e_grid_ev: np.ndarray,
    states: Iterable[StateResult] | Sequence[StateResult],
    sigma_g: float,
    gamma_l: float,
) -> np.ndarray:
    """Sum area-normalised Voigt profiles of a set of states.

    Parameters
    ----------
    e_grid_ev:
        ``(ne,)`` uniform energy grid, eV, on the *shifted*
        double-ionization energy axis.
    states:
        States to sum.  Each contributes a profile of area
        ``state.intensity`` centred at ``state.e_shifted_ev``.  Closed
        states carry zero intensity and therefore contribute nothing, but
        are not filtered out, so a run's state list can be passed whole.
        Negative intensities are summed *as they are*: clamping them would
        hide the model breakdown they signal ([B-3]).
    sigma_g:
        Gaussian standard deviation, eV.
    gamma_l:
        Lorentzian HWHM, eV.

    Returns
    -------
    ndarray
        ``(ne,)`` broadened spectrum, units Mb*a.u./eV.

    Notes
    -----
    Vectorised over the grid *and* the states: the profiles are evaluated
    on the ``(nstates, ne)`` outer grid in one call and reduced, rather
    than accumulated in a Python loop.  The result is linear in the stick
    intensities, which is a test.
    """
    e_grid_ev = np.asarray(e_grid_ev, dtype=float)
    if e_grid_ev.ndim != 1:
        raise ModelError(
            "spectrum.broaden: e_grid_ev must be one-dimensional, got shape "
            f"{e_grid_ev.shape}."
        )
    st = list(states)
    if not st:
        return np.zeros_like(e_grid_ev)

    centres = np.array([s.e_shifted_ev for s in st], dtype=float)
    areas = np.array([s.intensity for s in st], dtype=float)
    if not np.all(np.isfinite(centres)):
        bad = [s.index for s, ok in zip(st, np.isfinite(centres)) if not ok]
        raise ModelError(
            "spectrum.broaden: non-finite e_shifted_ev for state indices "
            f"{bad}; the display energies must be set before broadening."
        )

    # One (nstates, ne) evaluation, then a single matrix-vector reduction.
    dx = e_grid_ev[None, :] - centres[:, None]
    profiles = _voigt_kernel(dx, sigma_g, gamma_l, where="broaden")
    return areas @ profiles
