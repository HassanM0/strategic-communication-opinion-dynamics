"""
opinions.py — Initial private opinions x(0) ∈ [-1, 1]^N.

Supported modes
---------------
'random'          — iid uniform on [-1, 1] (the paper's prior U[-1, 1])
'symmetric_zero'  — half agents at +scale, half at -scale
'manual'          — user-supplied comma-separated values
'identical'       — identical values for all agents
'eigen'           — scaled eigenvector of graph Laplacian (Fiedler or higher)
'eigen_noise'     — eigenvector + Gaussian noise at a specified SNR (dB)
"""

from __future__ import annotations

import re

import networkx as nx
import numpy as np


def build_initial_opinions(
    num_agents: int,
    mode: str = "random",
    opinion_seed: int = 13,
    symmetric_scale: float = 0.3,
    manual_values: str = "",
    identical_value: float = 0.0,
    base_graph: nx.Graph | None = None,
    eigen_index: int = 1,
    eigen_scale: float = 0.8,
    eigen_noise_snr: float = 10.0,
) -> np.ndarray:
    """Generate the initial private opinions x(0) ∈ [-1, 1]^N.

    Parameters
    ----------
    num_agents : int
        Number of agents N.
    mode : str
        Initialisation mode (see module docstring).
    opinion_seed : int
        RNG seed for 'random' and 'eigen_noise' modes.
    symmetric_scale : float
        Half-amplitude for 'symmetric_zero' mode.
    manual_values : str
        Comma-separated floats for 'manual' mode.
    identical_value : float
        Float for 'identical' mode.
    base_graph : nx.Graph or None
        Undirected base graph required for 'eigen' and 'eigen_noise' modes.
    eigen_index : int
        Which Laplacian eigenvector to use (0 = constant, 1 = Fiedler, …).
        Larger indices correspond to higher-frequency graph modes.
    eigen_scale : float
        Amplitude scaling factor applied to the eigenvector. After scaling,
        values are clipped to [-1, 1].
    eigen_noise_snr : float
        Signal-to-noise ratio in dB for 'eigen_noise' mode.
        Higher values add less noise.

    Returns
    -------
    x : np.ndarray, shape (N,)
        Initial opinions, all finite and in [-1, 1].

    Raises
    ------
    ValueError
        On invalid parameters or constraint violations.
    """
    N = int(num_agents)
    mode = str(mode)
    x_min, x_max = -1.0, 1.0  # opinion space [-1, 1]

    if mode == "random":
        rng = np.random.default_rng(int(opinion_seed))
        x = rng.uniform(x_min, x_max, size=N)

    elif mode == "symmetric_zero":
        scale = float(symmetric_scale)
        if scale < 0:
            raise ValueError("symmetric_scale must be non-negative.")
        x = np.ones(N, dtype=float) * scale
        x[: N // 2] = -scale
        if N % 2 != 0:
            x[N // 2] = 0.0

    elif mode == "manual":
        tokens = [t for t in re.split(r"[\s,]+", str(manual_values).strip()) if t]
        if len(tokens) != N:
            raise ValueError(
                f"Manual initial opinions expect exactly N={N} values; "
                f"got {len(tokens)}."
            )
        x = np.array([float(t) for t in tokens], dtype=float)

    elif mode == "identical":
        x = np.full(N, float(identical_value))

    elif mode in ("eigen", "eigen_noise"):
        if base_graph is None:
            raise ValueError(
                "base_graph must be provided for 'eigen' / 'eigen_noise' modes."
            )
        x = _eigen_opinions(
            base_graph=base_graph,
            N=N,
            eigen_index=eigen_index,
            eigen_scale=eigen_scale,
            x_min=x_min,
            x_max=x_max,
        )
        if mode == "eigen_noise":
            rng = np.random.default_rng(int(opinion_seed))
            signal_power = np.var(x) if np.var(x) > 0 else 1.0
            snr_linear = 10.0 ** (float(eigen_noise_snr) / 10.0)
            noise_std = np.sqrt(signal_power / snr_linear)
            x = x + rng.normal(0.0, noise_std, size=N)

    else:
        raise ValueError(
            "initial opinion mode must be one of "
            "{'random', 'symmetric_zero', 'manual', 'identical', 'eigen', 'eigen_noise'}. "
            f"Got '{mode}'."
        )

    # Final validation / clip
    if np.any(~np.isfinite(x)):
        raise ValueError("Initial opinions contain non-finite values.")

    # Clip to the opinion space [-1, 1] (eigen modes can slightly exceed it).
    x = np.clip(x, x_min, x_max)

    return x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eigen_opinions(
    base_graph: nx.Graph, # or nx.DiGraph
    N: int,
    eigen_index: int,
    eigen_scale: float,
    x_min: float,
    x_max: float,
) -> np.ndarray:
    """Return initial opinions based on a Laplacian eigenvector.

    The random walk normalized graph Laplacian L = D⁻¹(D - A) = I - W is computed, 
    its eigenvectors are sorted by eigenvalue, and the *eigen_index*-th vector
    is selected. The vector is then scaled so that its maximum absolute
    value equals *eigen_scale* × (x_max - x_min) / 2, and shifted to
    sit symmetrically around the midpoint of the opinion space.
    """
    
    # 1. Compute the Adjacency matrix and row-sums (out-degrees)
    A = nx.adjacency_matrix(base_graph).toarray()
    degrees = np.sum(A, axis=1)
    
    # Safely invert degrees to prevent divide-by-zero for isolated nodes
    with np.errstate(divide='ignore'):
        inv_degrees = np.where(degrees > 0, 1.0 / degrees, 0.0)
        
    # 2. Compute the Random Walk Laplacian: L = I - D^{-1}A
    # Notice that D^{-1}A is exactly the row-stochastic matrix W 
    W = np.diag(inv_degrees) @ A
    L = np.eye(N) - W

    # 3. Use eig because L might be asymmetric (directed graph)
    eigenvalues, eigenvectors = np.linalg.eig(L)
    
    # eig does not guarantee sorted output, so we must sort them manually
    sorted_indices = np.argsort(np.real(eigenvalues))
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    
    # 4. Extract the desired eigenvector
    idx = int(np.clip(eigen_index, 0, N - 1))
    v = eigenvectors_sorted[:, idx]
    
    # 5. Extract real part (since directed (hence asymmetric) graphs can yield complex eigenvectors)
    v = np.real(v)

    # Scaling 
    # Normalise to [-1, 1] then scale
    v_max = np.max(np.abs(v))
    if v_max > 1e-12:
        v = v / v_max
        
    # Scale to [-(half_range * scale), +(half_range * scale)] + midpoint
    half_range = 0.5 * (x_max - x_min)
    midpoint = 0.5 * (x_min + x_max)
    
    x = midpoint + float(eigen_scale) * half_range * v
    
    return x
