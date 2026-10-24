"""
quantizers.py — Interval quantizers q_i : [-1, 1] -> [-1, 1].

The strategic-CODA dynamics (Eq. (18)) are defined for any monotone
interval quantizer. In the paper every agent uses its L-bin credible code
(Theorem 4, Assumption 3); the other families below are provided for
exploration beyond the paper.

Families
--------
strategic(xi, L)   L-bin credible code of the reduced game Γ(ξ, U[-1,1])
                   (sinh-stretched thresholds, midpoint reconstruction levels)
uniform(L)         equally spaced bins with midpoint levels (rating scale)
sign()             two bins with extreme levels -1 / +1 (classical CODA)
custom(...)        arbitrary thresholds and reconstruction levels
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .equilibrium import compute_thresholds_closed_form

__all__ = ["Quantizer", "stack_quantizers", "build_quantizers"]


@dataclasses.dataclass(frozen=True)
class Quantizer:
    """A scalar interval quantizer q : [-1, 1] -> [-1, 1].

    Attributes
    ----------
    thresholds : np.ndarray, shape (L+1,)
        τ_0 = -1 < τ_1 < … < τ_L = 1.
    reconstruction_levels : np.ndarray, shape (L,)
        Strictly increasing levels r_1 < … < r_L in [-1, 1].

    Bin ℓ is B_ℓ = [τ_{ℓ-1}, τ_ℓ), the last bin being closed at 1, so an
    interior threshold belongs to the bin on its right.
    """

    thresholds: np.ndarray
    reconstruction_levels: np.ndarray

    def __post_init__(self):
        t = np.asarray(self.thresholds, dtype=float)
        r = np.asarray(self.reconstruction_levels, dtype=float)
        if t.ndim != 1 or r.ndim != 1 or len(t) != len(r) + 1:
            raise ValueError("need L+1 thresholds and L reconstruction levels")
        if not (np.isclose(t[0], -1.0) and np.isclose(t[-1], 1.0)):
            raise ValueError("thresholds must span [-1, 1]")
        if np.any(np.diff(t) <= 0) or (len(r) > 1 and np.any(np.diff(r) <= 0)):
            raise ValueError(
                "thresholds and reconstruction levels must be strictly increasing")
        if np.any(r < -1.0) or np.any(r > 1.0):
            raise ValueError("reconstruction levels must lie in [-1, 1]")
        object.__setattr__(self, "thresholds", t)
        object.__setattr__(self, "reconstruction_levels", r)

    # -------------------------------------------------- properties
    @property
    def L(self) -> int:
        """Number of bins (resolution of the public channel)."""
        return len(self.reconstruction_levels)

    @property
    def is_self_consistent(self) -> bool:
        """Whether every level r_ℓ lies in its own bin B_ℓ = [τ_{ℓ-1}, τ_ℓ)."""
        lower_ok = self.reconstruction_levels >= self.thresholds[:-1]
        upper_ok = self.reconstruction_levels < self.thresholds[1:]
        upper_ok[-1] = self.reconstruction_levels[-1] <= self.thresholds[-1]
        return bool(np.all(lower_ok & upper_ok))

    @property
    def is_strictly_self_consistent(self) -> bool:
        """Whether every level lies strictly inside its own bin."""
        return bool(np.all(
            (self.reconstruction_levels > self.thresholds[:-1])
            & (self.reconstruction_levels < self.thresholds[1:])
        ))

    @property
    def has_interior_levels(self) -> bool:
        """True iff the extreme levels r_1, r_L lie strictly inside (-1, 1)."""
        return bool(
            self.reconstruction_levels[0] > -1.0
            and self.reconstruction_levels[-1] < 1.0
        )

    # -------------------------------------------------- application
    def __call__(self, x) -> np.ndarray:
        """Apply q elementwise (same bin convention as the simulation kernels)."""
        x = np.asarray(x, dtype=float)
        idx = np.searchsorted(self.thresholds, x, side="right") - 1
        idx = np.clip(idx, 0, self.L - 1)
        return self.reconstruction_levels[idx]

    # -------------------------------------------------- constructors
    @classmethod
    def strategic(cls, xi: float, L: int) -> "Quantizer":
        """L-bin credible code of an agent with exaggeration factor ξ."""
        thr, levels, _ = compute_thresholds_closed_form(
            np.asarray([float(xi)]), int(L))
        return cls(thr[0], levels[0])

    @classmethod
    def uniform(cls, L: int) -> "Quantizer":
        """Equally spaced bins with midpoint levels (rating scale)."""
        t = np.linspace(-1.0, 1.0, int(L) + 1)
        return cls(t, (t[:-1] + t[1:]) / 2.0)

    @classmethod
    def sign(cls) -> "Quantizer":
        """Two bins with extreme levels -1 / +1 (classical CODA).

        Its levels are not interior, so boundary consensus states become
        reachable (cf. Theorem 5 of the paper, which assumes credible codes).
        """
        return cls(np.array([-1.0, 0.0, 1.0]), np.array([-1.0, 1.0]))

    @classmethod
    def custom(cls, thresholds, reconstruction_levels) -> "Quantizer":
        """Arbitrary quantizer from explicit thresholds and levels."""
        return cls(
            np.asarray(thresholds, float),
            np.asarray(reconstruction_levels, float),
        )


def stack_quantizers(quantizers: list[Quantizer]) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-agent quantizers into kernel-ready matrices.

    Heterogeneous resolutions L_i are supported by padding: agents with
    L_i < L_max get zero-width top bins (repeated threshold 1.0 and
    repeated top level), which are unreachable and do not change behavior.

    Returns
    -------
    thresholds : np.ndarray, shape (N, L_max + 1)
    reconstruction_levels : np.ndarray, shape (N, L_max)
    """
    N = len(quantizers)
    if N == 0:
        raise ValueError("need at least one quantizer")
    L_max = max(q.L for q in quantizers)
    thresholds = np.empty((N, L_max + 1))
    reconstruction_levels = np.empty((N, L_max))
    for i, q in enumerate(quantizers):
        pad = L_max - q.L
        thresholds[i] = np.concatenate([q.thresholds, np.full(pad, 1.0)])
        reconstruction_levels[i] = np.concatenate([
            q.reconstruction_levels,
            np.full(pad, q.reconstruction_levels[-1]),
        ])
    return thresholds, reconstruction_levels


def build_quantizers(family: str, N: int, L: int, xi=None) -> list[Quantizer]:
    """Build a per-agent quantizer list for a named family.

    Parameters
    ----------
    family : {'strategic', 'uniform', 'sign'}
    N : int
        Number of agents.
    L : int
        Number of bins (ignored for 'sign').
    xi : array-like, shape (N,), required for 'strategic'.
    """
    if family == "strategic":
        if xi is None:
            raise ValueError("strategic family requires xi")
        return [Quantizer.strategic(float(x), L) for x in np.asarray(xi)]
    if family == "uniform":
        q = Quantizer.uniform(L)
        return [q] * N
    if family == "sign":
        q = Quantizer.sign()
        return [q] * N
    raise ValueError(
        f"unknown quantizer family {family!r}; "
        "expected 'strategic', 'uniform' or 'sign' (or pass explicit "
        "quantizers to the simulator)")
