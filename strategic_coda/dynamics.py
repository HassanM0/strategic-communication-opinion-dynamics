"""
dynamics.py — High-performance closed-loop opinion dynamics.

Implements the closed-loop dynamics of the paper (Eq. (4)),

    x_i(t+1) = α_i x_i(t) + (1 - α_i) Σ_j w_ij s_j(x_j(t)),

for the effective signaling maps s_i studied in Sec. IV:

    quantized   s_i = q_i                 (Bayesian receivers, strategic CODA, Eq. (18))
    linear      s_i(x) = x                (aligned incentives)
    saturated   s_i(x) = sat(ξ_i x)       (naive receivers, Eq. (11) with δ_i = 0)
    tanh        s_i(x) = tanh(K x)        (extra, not in the paper)

Two execution paths are provided:
  1. Numba JIT (preferred) — compiled on first call, cached to disk.
     Falls back automatically if Numba is not installed.
  2. Pure NumPy fallback — always available, ~10× slower for large N.

Low-level kernel API
--------------------
run_simulation(opinions, W, alpha_vec, thresholds, reconstruction_levels,
               steps, track_bins, ...)
    → (history, bin_indices | None)

public_signals_numpy(opinions, thresholds, reconstruction_levels,
                     signal_mode, tanh_K, xi)
    → public signals s(x)
"""

from __future__ import annotations

import numpy as np

from enum import IntEnum


class SignalMode(IntEnum):
    """Effective signaling map selector for the simulation kernels."""

    QUANTIZED = 0    # s_i = q_i, interval quantizers (default)
    LINEAR = 1       # s_i(x) = x  (DeGroot averaging)
    TANH = 2         # s_i(x) = tanh(K x)  (not in the paper)
    SATURATED = 3    # s_i(x) = sat(ξ_i x)


#: SimConfig.signal_function value -> SignalMode (single source of truth)
SIGNAL_FUNCTIONS = {
    "quantized": SignalMode.QUANTIZED,
    "linear": SignalMode.LINEAR,
    "saturated": SignalMode.SATURATED,
    "tanh": SignalMode.TANH,
}

# ---------------------------------------------------------------------------
# Try to import Numba; fall back gracefully
# ---------------------------------------------------------------------------
try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NUMBA_AVAILABLE = False


# ===========================================================================
# NumPy fallback implementations (always available)
# ===========================================================================

def public_signals_numpy(
    opinions: np.ndarray,
    thresholds: np.ndarray,
    reconstruction_levels: np.ndarray,
    signal_mode: int = 0,
    tanh_K: float = 1.0,
    xi: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the public signals s_i(x_i) of all agents.

    Parameters
    ----------
    opinions : np.ndarray, shape (N,)
        Private opinions x.
    thresholds : np.ndarray, shape (N, L+1)
        Bin thresholds τ_{i,ℓ} (used by the quantized map).
    reconstruction_levels : np.ndarray, shape (N, L)
        Reconstruction levels r_{i,ℓ} (used by the quantized map).
    signal_mode : int
        See :class:`SignalMode` (0 quantized, 1 linear, 2 tanh, 3 saturated).
    tanh_K : float
        Slope K of the tanh map.
    xi : np.ndarray or None
        Exaggeration factors ξ_i (used by the saturated map).

    Returns
    -------
    signals : np.ndarray, shape (N,)
    """
    N = opinions.shape[0]
    if signal_mode == 1:
        return opinions.copy()
    elif signal_mode == 2:
        return np.tanh(tanh_K * opinions)
    elif signal_mode == 3:
        if xi is None:
            return np.clip(opinions, -1.0, 1.0)
        return np.clip(xi * opinions, -1.0, 1.0)
        
    L = reconstruction_levels.shape[1]
    signals = np.empty(N, dtype=np.float64)
    for i in range(N):
        idx = int(np.searchsorted(thresholds[i], opinions[i], side="right")) - 1
        idx = max(0, min(idx, L - 1))
        signals[i] = reconstruction_levels[i, idx]
    return signals


def _assign_bins_numpy(
    opinions: np.ndarray,
    thresholds: np.ndarray,
    L: int,
) -> np.ndarray:
    """Return the zero-based bin index of every agent."""
    N = opinions.shape[0]
    bins = np.empty(N, dtype=np.int32)
    for i in range(N):
        idx = int(np.searchsorted(thresholds[i], opinions[i], side="right")) - 1
        bins[i] = max(0, min(idx, L - 1))
    return bins


def run_simulation_numpy(
    opinions: np.ndarray,
    W: np.ndarray,
    alpha_vec: np.ndarray,
    thresholds: np.ndarray,
    reconstruction_levels: np.ndarray,
    steps: int,
    track_bins: bool = False,
    add_perturbation: bool = False,
    perturb_noise: np.ndarray | None = None,
    perturb_time: int = -1,
    signal_mode: int = 0,
    tanh_K: float = 1.0,
    xi: np.ndarray | None = None,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Pure-NumPy simulation loop (fallback when Numba is unavailable).

    Returns
    -------
    history : np.ndarray, shape (steps+1, N)
    bin_indices : np.ndarray shape (steps+1, N) of int32, or None
    """
    N = opinions.shape[0]
    L = reconstruction_levels.shape[1]
    history = np.empty((steps + 1, N), dtype=np.float64)
    history[0] = opinions.copy()
    x = opinions.copy()

    bins_out = None
    if track_bins:
        bins_out = np.empty((steps + 1, N), dtype=np.int32)
        bins_out[0] = _assign_bins_numpy(x, thresholds, L)

    for t in range(steps):
        sig = public_signals_numpy(x, thresholds, reconstruction_levels, signal_mode, tanh_K, xi)
        social = W @ sig
        x_new = alpha_vec * x + (1.0 - alpha_vec) * social
        # Forward-Euler relaxation; dt == 1 is the exact discrete map
        x = x_new if dt == 1.0 else (1.0 - dt) * x + dt * x_new
        if add_perturbation and t == perturb_time and perturb_noise is not None:
            x += perturb_noise
        history[t + 1] = x
        if track_bins:
            bins_out[t + 1] = _assign_bins_numpy(x, thresholds, L)

    return history, bins_out


# ===========================================================================
# Numba JIT implementations (compiled on first call, cached)
# ===========================================================================

if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _assign_bins_jit(
        opinions: np.ndarray,
        thresholds: np.ndarray,
        L: int,
    ) -> np.ndarray:
        """JIT-compiled bin assignment (inline bisect_right)."""
        N = opinions.shape[0]
        bins = np.empty(N, np.int32)
        for i in range(N):
            x = opinions[i]
            lo, hi = 0, thresholds.shape[1]
            while lo < hi:
                mid = (lo + hi) // 2
                if thresholds[i, mid] <= x:
                    lo = mid + 1
                else:
                    hi = mid
            idx = lo - 1
            if idx < 0:
                idx = 0
            if idx >= L:
                idx = L - 1
            bins[i] = idx
        return bins

    @njit(parallel=True, cache=True)
    def _quantize_signals_jit(
        opinions: np.ndarray,
        thresholds: np.ndarray,
        reconstruction_levels: np.ndarray,
    ) -> np.ndarray:
        """JIT-compiled (parallel) signal quantisation."""
        N = opinions.shape[0]
        L = reconstruction_levels.shape[1]
        signals = np.empty(N, np.float64)
        for i in prange(N):
            x = opinions[i]
            lo, hi = 0, thresholds.shape[1]
            while lo < hi:
                mid = (lo + hi) // 2
                if thresholds[i, mid] <= x:
                    lo = mid + 1
                else:
                    hi = mid
            idx = lo - 1
            if idx < 0:
                idx = 0
            if idx >= L:
                idx = L - 1
            signals[i] = reconstruction_levels[i, idx]
        return signals

    @njit(cache=True)
    def _run_simulation_jit(
        opinions: np.ndarray,
        W: np.ndarray,
        alpha_vec: np.ndarray,
        thresholds: np.ndarray,
        reconstruction_levels: np.ndarray,
        steps: int,
        add_perturbation: bool,
        perturb_noise: np.ndarray,
        perturb_time: int,
        signal_mode: int,
        tanh_K: float,
        xi: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """JIT-compiled simulation loop (without bin tracking).

        Returns
        -------
        history : np.ndarray, shape (steps+1, N)
        """
        N = opinions.shape[0]
        L = reconstruction_levels.shape[1]
        history = np.empty((steps + 1, N), np.float64)
        x = opinions.copy()
        history[0] = x

        for t in range(steps):
            # Generate signals
            sig = np.empty(N, np.float64)
            if signal_mode == 1:
                for i in range(N):
                    sig[i] = x[i]
            elif signal_mode == 2:
                for i in range(N):
                    sig[i] = np.tanh(tanh_K * x[i])
            elif signal_mode == 3:
                for i in range(N):
                    val = xi[i] * x[i]
                    if val > 1.0:
                        sig[i] = 1.0
                    elif val < -1.0:
                        sig[i] = -1.0
                    else:
                        sig[i] = val
            else:
                for i in range(N):
                    xi_val = x[i]
                    lo, hi = 0, thresholds.shape[1]
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if thresholds[i, mid] <= xi_val:
                            lo = mid + 1
                        else:
                            hi = mid
                    idx = lo - 1
                    if idx < 0:
                        idx = 0
                    if idx >= L:
                        idx = L - 1
                    sig[i] = reconstruction_levels[i, idx]

            # Social update: x_new = alpha * x + (1-alpha) * W @ sig
            social = W @ sig
            x_new = alpha_vec * x + (1.0 - alpha_vec) * social
            # Forward-Euler relaxation; dt == 1 is the exact discrete map
            if dt == 1.0:
                x = x_new
            else:
                x = (1.0 - dt) * x + dt * x_new

            if add_perturbation and t == perturb_time:
                for i in range(N):
                    x[i] += perturb_noise[i]


            history[t + 1] = x

        return history

    @njit(cache=True)
    def _run_simulation_with_bins_jit(
        opinions: np.ndarray,
        W: np.ndarray,
        alpha_vec: np.ndarray,
        thresholds: np.ndarray,
        reconstruction_levels: np.ndarray,
        steps: int,
        add_perturbation: bool,
        perturb_noise: np.ndarray,
        perturb_time: int,
        signal_mode: int,
        tanh_K: float,
        xi: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """JIT-compiled simulation loop with bin tracking.

        Returns
        -------
        history : np.ndarray, shape (steps+1, N)
        bins    : np.ndarray, shape (steps+1, N), dtype int32
            Zero-based bin indices.
        """
        N = opinions.shape[0]
        L = reconstruction_levels.shape[1]
        history = np.empty((steps + 1, N), np.float64)
        bins_out = np.empty((steps + 1, N), np.int32)
        x = opinions.copy()
        history[0] = x

        # Initial bin assignment
        if signal_mode == 0:
            for i in range(N):
                xi_val = x[i]
                lo, hi = 0, thresholds.shape[1]
                while lo < hi:
                    mid = (lo + hi) // 2
                    if thresholds[i, mid] <= xi_val:
                        lo = mid + 1
                    else:
                        hi = mid
                idx = lo - 1
                if idx < 0:
                    idx = 0
                if idx >= L:
                    idx = L - 1
                bins_out[0, i] = idx
        else:
            for i in range(N):
                bins_out[0, i] = 0

        for t in range(steps):
            sig = np.empty(N, np.float64)
            if signal_mode == 1:
                for i in range(N):
                    sig[i] = x[i]
            elif signal_mode == 2:
                for i in range(N):
                    sig[i] = np.tanh(tanh_K * x[i])
            elif signal_mode == 3:
                for i in range(N):
                    val = xi[i] * x[i]
                    if val > 1.0:
                        sig[i] = 1.0
                    elif val < -1.0:
                        sig[i] = -1.0
                    else:
                        sig[i] = val
            else:
                for i in range(N):
                    xi_val = x[i]
                    lo, hi = 0, thresholds.shape[1]
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if thresholds[i, mid] <= xi_val:
                            lo = mid + 1
                        else:
                            hi = mid
                    idx = lo - 1
                    if idx < 0:
                        idx = 0
                    if idx >= L:
                        idx = L - 1
                    sig[i] = reconstruction_levels[i, idx]

            social = W @ sig
            x_new = alpha_vec * x + (1.0 - alpha_vec) * social
            if dt == 1.0:
                x = x_new
            else:
                x = (1.0 - dt) * x + dt * x_new

            if add_perturbation and t == perturb_time:
                for i in range(N):
                    x[i] += perturb_noise[i]

            # Bin of the UPDATED state x(t+1), aligned with history[t+1].
            if signal_mode == 0:
                for i in range(N):
                    xv = x[i]
                    lo, hi = 0, thresholds.shape[1]
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if thresholds[i, mid] <= xv:
                            lo = mid + 1
                        else:
                            hi = mid
                    idx = lo - 1
                    if idx < 0:
                        idx = 0
                    if idx >= L:
                        idx = L - 1
                    bins_out[t + 1, i] = idx
            else:
                for i in range(N):
                    bins_out[t + 1, i] = 0

            history[t + 1] = x

        return history, bins_out


# ===========================================================================
# Unified public entry point
# ===========================================================================

def run_simulation(
    opinions: np.ndarray,
    W: np.ndarray,
    alpha_vec: np.ndarray,
    thresholds: np.ndarray,
    reconstruction_levels: np.ndarray,
    steps: int,
    track_bins: bool = False,
    add_perturbation: bool = False,
    perturb_type: str = "Random",
    perturb_scale: float = 0.1,
    perturb_time: int = -1,
    signal_mode: int = 0,
    tanh_K: float = 1.0,
    xi: np.ndarray | None = None,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run the closed-loop dynamics (Eq. (4)) for *steps* periods.

    Uses Numba JIT if available, otherwise falls back to pure NumPy.

    Parameters
    ----------
    opinions : np.ndarray, shape (N,)
        Initial private opinions x(0).
    W : np.ndarray, shape (N, N)
        Row-stochastic influence matrix, ``W[i, j] = w_ij``.
    alpha_vec : np.ndarray, shape (N,)
        Inertia α_i of every agent.
    thresholds : np.ndarray, shape (N, L+1)
        Bin thresholds τ_{i,ℓ} of the quantizers q_i.
    reconstruction_levels : np.ndarray, shape (N, L)
        Reconstruction levels r_{i,ℓ} of the quantizers q_i.
    steps : int
        Horizon T (number of periods).
    track_bins : bool
        Also return the bin index of every agent at every period
        (meaningful for the quantized map; zeros otherwise).
    add_perturbation, perturb_type, perturb_scale, perturb_time
        Optional one-off perturbation (not in the paper).
    signal_mode : int
        Effective signaling map, see :class:`SignalMode`.
    tanh_K : float
        Slope of the tanh map.
    xi : np.ndarray or None
        Exaggeration factors ξ_i (needed by the saturated map).
    dt : float
        Forward-Euler step of the relaxed update
        x <- (1 - dt) x + dt [alpha x + (1 - alpha) W q(x)].
        dt = 1 (default) is exactly the discrete-time map; dt < 1
        approximates the continuous-time flow dx/dt = -x + F(x).

    Returns
    -------
    history : np.ndarray, shape (steps+1, N)
        Private-opinion trajectory x(0), …, x(T).
    bin_indices : np.ndarray, shape (steps+1, N) or None
        Zero-based bin indices, or None when tracking is disabled.
    """
    # Ensure contiguous float64 arrays (required by Numba)
    opinions = np.ascontiguousarray(opinions, dtype=np.float64)
    W = np.ascontiguousarray(W, dtype=np.float64)
    alpha_vec = np.ascontiguousarray(alpha_vec, dtype=np.float64)
    thresholds = np.ascontiguousarray(thresholds, dtype=np.float64)
    reconstruction_levels = np.ascontiguousarray(
        reconstruction_levels, dtype=np.float64)

    # Precompute perturbation noise purely in Python so Numba doesn't crash on default_rng
    if add_perturbation:
        if perturb_type == "Random":
            rng_h = np.random.default_rng(42)
            p_noise = rng_h.normal(0.0, perturb_scale, size=opinions.shape[0])
        else:
            p_noise = np.full(opinions.shape[0], perturb_scale)
    else:
        p_noise = np.zeros_like(opinions)
    
    p_noise = np.ascontiguousarray(p_noise, dtype=np.float64)

    if xi is None:
        xi_arr = np.ones_like(opinions)
    else:
        xi_arr = np.ascontiguousarray(xi, dtype=np.float64)

    if _NUMBA_AVAILABLE:
        if track_bins:
            return _run_simulation_with_bins_jit(
                opinions, W, alpha_vec, thresholds, reconstruction_levels,
                int(steps),
                bool(add_perturbation), p_noise, int(perturb_time),
                int(signal_mode), float(tanh_K), xi_arr, float(dt),
            )
        else:
            history = _run_simulation_jit(
                opinions, W, alpha_vec, thresholds, reconstruction_levels,
                int(steps),
                bool(add_perturbation), p_noise, int(perturb_time),
                int(signal_mode), float(tanh_K), xi_arr, float(dt),
            )
            return history, None
    else:
        return run_simulation_numpy(
            opinions, W, alpha_vec, thresholds, reconstruction_levels,
            int(steps),
            track_bins=track_bins,
            add_perturbation=add_perturbation,
            perturb_noise=p_noise,
            perturb_time=perturb_time,
            signal_mode=signal_mode,
            tanh_K=tanh_K,
            xi=xi_arr,
            dt=float(dt),
        )
