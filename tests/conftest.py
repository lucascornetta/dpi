"""Shared fixtures: synthetic Dyson objects and a minimal run config.

The physics-assembly layer is a pure function of numpy arrays, so the tests
never touch OpenMolcas output.  They build random but reproducible blocks of
the documented shapes and check the algebra against explicit loops and
hand-computed limits.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpi.amplitudes import TermSwitches  # noqa: E402


@dataclass
class FakeDyson:
    """Stand-in for dpi.dyson.DysonObjects with the SPEC.md shapes."""

    d_i: np.ndarray
    d_j: np.ndarray
    lam_i: np.ndarray | None = None
    lam_j: np.ndarray | None = None
    d2_ij: np.ndarray | None = None
    det_sb: float | None = None
    p_i: float = float("nan")
    p_j: float = float("nan")
    meta: dict | None = None


@dataclass
class FakeConfig:
    """Minimal cfg: photon energy, term switches and grid callables."""

    omega_ev: float
    terms: TermSwitches
    sigma_ao: np.ndarray          # (nbas,) Mb, energy independent
    pshake_ao: np.ndarray         # (nbas,) a.u., k independent
    display_shift_ev: float = 0.0

    def sigma_at_eps1_grid(self, eps1_ev: np.ndarray) -> np.ndarray:
        eps1_ev = np.asarray(eps1_ev, dtype=float)
        return np.broadcast_to(
            self.sigma_ao, (eps1_ev.size, self.sigma_ao.size)
        ).copy()

    def p_shake_at_k(self, k_au: np.ndarray) -> np.ndarray:
        k_au = np.asarray(k_au, dtype=float)
        return np.broadcast_to(
            self.pshake_ao, (k_au.size, self.pshake_ao.size)
        ).copy()


def antisym(rng: np.random.Generator, nbas: int) -> np.ndarray:
    """Random antisymmetric (nbas, nbas) matrix, as d2_ij must be."""
    a = rng.normal(size=(nbas, nbas))
    return a - a.T


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20240517)


@pytest.fixture
def nbas() -> int:
    return 8


@pytest.fixture
def nq() -> int:
    return 11


@pytest.fixture
def grid_inputs(rng, nbas, nq):
    """(sigma_grid, pshake_grid, k2) of the documented shapes and signs."""
    sigma_grid = rng.uniform(0.05, 2.0, size=(nq, nbas))
    pshake_grid = rng.uniform(0.01, 0.5, size=(nq, nbas))
    k2 = np.linspace(0.0, 3.0, nq)
    return sigma_grid, pshake_grid, k2


@pytest.fixture
def dyson(rng, nbas) -> FakeDyson:
    return FakeDyson(
        d_i=rng.normal(size=nbas),
        d_j=rng.normal(size=nbas),
        lam_i=rng.normal(size=(nbas, 3)),
        lam_j=rng.normal(size=(nbas, 3)),
        d2_ij=antisym(rng, nbas),
        det_sb=0.7314,
        p_i=0.91,
        p_j=0.83,
        meta={"index": 3, "label": "F1sT22", "e_dication_ev": 41.2},
    )


@pytest.fixture
def all_terms() -> TermSwitches:
    """Every term on, so the breakdown identities cover all branches."""
    return TermSwitches(
        direct=True,
        cross_dyson=True,
        indirect=True,
        aa_bb=True,
        dir_ind_interference=True,
        c_cross=True,
    )
