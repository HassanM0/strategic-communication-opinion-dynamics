"""
alpha.py — Inertia vector α = (α_1, …, α_N) construction.

Provides build_alpha_vector which supports three modes:
  'scalar'                — all agents share the same alpha
  'heterogeneous-random'  — iid uniform draws in [alpha_min, alpha_max]
  'manual'                — user-supplied comma-separated values
"""

from __future__ import annotations

import re

import numpy as np


def build_alpha_vector(
    num_agents: int,
    alpha: float = 0.5,
    alpha_mode: str = "scalar",
    alpha_seed: int = 31,
    alpha_min: float = 0.2,
    alpha_max: float = 0.8,
    alpha_manual: str = "",
) -> np.ndarray:
    """Build the per-agent inertia vector α ∈ [0, 1)^N (paper, Eq. (1)).

    Parameters
    ----------
    num_agents : int
        Number of agents N.
    alpha : float
        Common inertia used when alpha_mode='scalar'.
    alpha_mode : str
        Assignment mode:
        - 'scalar'                — all agents share *alpha*.
        - 'heterogeneous-random'  — iid uniform on [alpha_min, alpha_max].
        - 'manual'                — parse *alpha_manual* as N floats.
    alpha_seed : int
        RNG seed for 'heterogeneous-random' mode.
    alpha_min : float
        Lower bound for 'heterogeneous-random' mode.
    alpha_max : float
        Upper bound for 'heterogeneous-random' mode.
    alpha_manual : str
        Whitespace- or comma-separated floats for 'manual' mode.

    Returns
    -------
    alpha_vec : np.ndarray, shape (N,)
        Per-agent inertia α_i, all in [0, 1).

    Raises
    ------
    ValueError
        On invalid parameters or constraint violations.
    """
    N = int(num_agents)

    if alpha_mode == "scalar":
        alpha_vec = np.full(N, float(alpha), dtype=float)

    elif alpha_mode == "heterogeneous-random":
        alpha_min_f = float(alpha_min)
        alpha_max_f = float(alpha_max)
        if not (0.0 <= alpha_min_f < alpha_max_f < 1.0):
            raise ValueError(
                "alpha_min and alpha_max must satisfy 0 ≤ alpha_min < alpha_max < 1. "
                f"Got alpha_min={alpha_min_f}, alpha_max={alpha_max_f}."
            )
        rng = np.random.default_rng(int(alpha_seed))
        alpha_vec = rng.uniform(alpha_min_f, alpha_max_f, size=N)

    elif alpha_mode == "manual":
        tokens = [t for t in re.split(r"[\s,]+", str(alpha_manual).strip()) if t]
        if len(tokens) != N:
            raise ValueError(
                f"Manual alpha mode expects exactly N={N} values; got {len(tokens)}."
            )
        alpha_vec = np.array([float(t) for t in tokens], dtype=float)

    else:
        raise ValueError(
            "alpha_mode must be one of {'scalar', 'heterogeneous-random', 'manual'}. "
            f"Got '{alpha_mode}'."
        )

    if np.any((alpha_vec < 0.0) | (alpha_vec >= 1.0)):
        bad = np.where((alpha_vec < 0.0) | (alpha_vec >= 1.0))[0].tolist()
        raise ValueError(
            f"All alpha_i values must satisfy 0 ≤ alpha_i < 1. "
            f"Violations at indices: {bad[:8]}"
        )

    return alpha_vec
