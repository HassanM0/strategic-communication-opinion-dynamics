"""
equilibrium.py — Network-induced exaggeration factor and credible codes.

Notation follows the paper (arXiv:2607.16036):

  w̃_ki = (1 - α_k) w_ki          susceptibility weight (Sec. III-B)
  ξ_i                             exaggeration factor   (Definition 2, Eq. (7))
  η_i = arccosh(2ξ_i - 1)         (Eq. (14))
  τ_{i,ℓ}                         bin thresholds of the L-bin credible code
                                  (Theorem 4, Eq. (15))
  r_{i,ℓ} = (τ_{i,ℓ-1}+τ_{i,ℓ})/2 reconstruction levels (Eq. (12))

Opinions, messages and reconstruction levels live in [-1, 1].
"""

from __future__ import annotations

import numpy as np

__all__ = ["compute_xi", "compute_thresholds_closed_form"]


# ---------------------------------------------------------------------------
# Exaggeration factor
# ---------------------------------------------------------------------------

def compute_xi(
    W: np.ndarray,
    alpha_vec: np.ndarray,
    ensure_valid_graph: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the network-induced exaggeration factor ξ_i of every sender.

    Definition 2 of the paper:

        ξ_i = Σ_{k ∈ N_i^A} w̃_ki  /  Σ_{k ∈ N_i^A} w̃_ki²,
        w̃_ki = (1 - α_k) w_ki   (susceptibility weight).

    Parameters
    ----------
    W : np.ndarray, shape (N, N)
        Row-stochastic influence matrix. ``W[i, j] = w_ij > 0`` means
        receiver i listens to source j; the audience of sender i is
        N_i^A = {k : w_ki > 0}, i.e. the nonzero entries of column i.
    alpha_vec : np.ndarray, shape (N,)
        Inertia α_i ∈ [0, 1) of every agent.
    ensure_valid_graph : bool
        When True, raise ValueError if some ξ_i is undefined (empty
        audience) or smaller than 1.

    Returns
    -------
    xi : np.ndarray, shape (N,)
        Exaggeration factors ξ_i ≥ 1 (NaN for an agent with no audience
        when validation is off).
    w_tilde : np.ndarray, shape (N, N)
        Susceptibility weights, ``w_tilde[k, i] = (1 - α_k) w_ki``.
    """
    w_tilde = ((1.0 - alpha_vec)[:, np.newaxis]) * W

    # Sender i: sum over its audience members k (column i of w_tilde).
    numerator = w_tilde.sum(axis=0)
    denominator = (w_tilde ** 2).sum(axis=0)

    xi = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )

    if ensure_valid_graph:
        if np.any(~np.isfinite(xi)):
            bad = np.where(~np.isfinite(xi))[0].tolist()
            raise ValueError(
                f"ξ is undefined for agents with no audience: {bad[:8]}"
            )
        if np.any(xi < 1.0):
            bad = np.where(xi < 1.0)[0].tolist()
            raise ValueError(
                "Found ξ_i < 1.0 for some agents. Ensure 0 ≤ α_i < 1 and "
                f"valid positive audience weights. First indices: {bad[:8]}"
            )

    return xi, w_tilde


# ---------------------------------------------------------------------------
# Closed-form credible codes under the uniform prior
# ---------------------------------------------------------------------------

def compute_thresholds_closed_form(
    xi: np.ndarray,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form L-bin credible code of every agent (uniform prior).

    Theorem 4 of the paper: for ξ_i > 1 and η_i = arccosh(2ξ_i - 1),

        τ_{i,ℓ} = sinh(η_i (ℓ - L/2)) / sinh(η_i L/2),   ℓ = 0, …, L,

    and the reconstruction level of bin B_{i,ℓ} = [τ_{i,ℓ-1}, τ_{i,ℓ}) is
    its midpoint r_{i,ℓ} = (τ_{i,ℓ-1} + τ_{i,ℓ}) / 2 (Eq. (12)).
    In the limit ξ_i → 1 (η_i → 0) the bins become equally spaced.

    Parameters
    ----------
    xi : np.ndarray, shape (N,)
        Exaggeration factors ξ_i ≥ 1.
    num_bins : int
        Number of bins L ≥ 1 of the credible code.

    Returns
    -------
    thresholds : np.ndarray, shape (N, L+1)
        τ_{i,0} = -1 < τ_{i,1} < … < τ_{i,L} = 1 for every agent.
    reconstruction_levels : np.ndarray, shape (N, L)
        r_{i,ℓ}, ℓ = 1, …, L.
    eta_vec : np.ndarray, shape (N,)
        η_i = arccosh(2ξ_i - 1).
    """
    xi = np.asarray(xi, dtype=float)
    L = int(num_bins)
    if L < 1:
        raise ValueError("The number of bins L must be at least 1.")

    eta = np.arccosh(np.clip(2.0 * xi - 1.0, 1.0, None))  # shape (N,)

    half_L = L / 2.0
    centered_idx = np.arange(L + 1, dtype=float) - half_L  # ℓ - L/2

    den = np.sinh(eta * half_L)                                       # (N,)
    num_mat = np.sinh(eta[:, np.newaxis] * centered_idx[np.newaxis, :])  # (N, L+1)

    # η → 0 limit: sinh(ηa)/sinh(ηb) → a/b (equally spaced bins).
    small = np.isclose(den, 0.0, atol=1e-12)
    safe_den = np.where(small, 1.0, den)[:, np.newaxis]
    thresholds = num_mat / safe_den
    if np.any(small):
        thresholds[small] = centered_idx[np.newaxis, :] / half_L

    reconstruction_levels = 0.5 * (thresholds[:, :-1] + thresholds[:, 1:])
    return thresholds, reconstruction_levels, eta
