"""Single-photon double photoionization cross sections from OpenMolcas.

A Gelius-type one-centre intensity model: generalized Slater-Condon rules
for the non-orthogonal neutral/dication orbital sets give one- and
two-electron Dyson amplitudes, which are combined with tabulated atomic
subshell cross sections and analytic shake-off probabilities and integrated
over the energy sharing of the two emitted electrons.

Deliberately empty of submodule imports: every module must be importable on
its own and free of import-time side effects, so that a partially built
package still works and no reader, plot backend or HDF5 file is touched by
``import dpi``.
"""

from __future__ import annotations

__version__ = "2.0.0.dev0"

__all__ = ["__version__"]
