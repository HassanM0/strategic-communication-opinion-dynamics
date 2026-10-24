"""
simulator.py — StrategicCODASimulator: the central simulation object.

The class is purely numerical: it owns the network, the influence matrix W,
the inertia vector α, the exaggeration factors ξ, the quantizers q_i and the
private-opinion trajectory. All visualisation lives in plots.py.

Usage
-----
    from strategic_coda import SimConfig, StrategicCODASimulator

    cfg = SimConfig(N=30, alpha=0.5, L=4, steps=80, graph_type="star")
    sim = StrategicCODASimulator(cfg)
    history, bin_indices = sim.run(cfg.steps, track_bins=True)
"""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np

from .alpha import build_alpha_vector
from .config import SimConfig
from .dynamics import SIGNAL_FUNCTIONS, SignalMode, public_signals_numpy, run_simulation
from .equilibrium import compute_thresholds_closed_form, compute_xi
from .graph import build_directed_weight_matrix
from .opinions import build_initial_opinions
from .quantizers import Quantizer, build_quantizers, stack_quantizers

#: Signaling maps covered by the paper (Sec. IV).
PAPER_SIGNAL_FUNCTIONS = ("quantized", "linear", "saturated")


class StrategicCODASimulator:
    """Closed-loop opinion dynamics with strategic public signaling.

    Simulates x_i(t+1) = α_i x_i(t) + (1 - α_i) Σ_j w_ij s_j(x_j(t))
    (Eq. (4) of the paper), where the effective signaling map s_i is selected
    by ``config.signal_function``. With 'quantized' and the 'strategic'
    family, s_i = q_i is agent i's L-bin credible code (strategic CODA,
    Eq. (18)).

    Parameters
    ----------
    config : SimConfig
        All simulation parameters.
    quantizers : list[Quantizer] or None
        Explicit per-agent quantizers; overrides ``config.quantizer_family``.
    quantizer_overrides : dict[int, Quantizer] or None
        Replace the quantizer of selected (zero-based) agents.

    Attributes
    ----------
    config : SimConfig
    N : int
    L : int
    G : nx.DiGraph            communication digraph (arc j → i: j is a source of i)
    base_graph : nx.Graph     undirected base topology
    layout_pos : dict         node positions for plotting
    W : np.ndarray (N, N)     influence matrix, W[i, j] = w_ij
    alpha_vec : np.ndarray (N,)   inertia α_i
    xi : np.ndarray (N,)          exaggeration factors ξ_i
    w_tilde : np.ndarray (N, N)   susceptibility weights, w_tilde[k, i] = (1-α_k) w_ki
    eta_vec : np.ndarray (N,) or None   η_i = arccosh(2ξ_i - 1) (strategic family)
    thresholds_matrix : np.ndarray (N, L+1)          bin thresholds τ_{i,ℓ}
    reconstruction_levels_matrix : np.ndarray (N, L) reconstruction levels r_{i,ℓ}
    sources, audiences : list[list[int]]   N_i^S and N_i^A (zero-based)
    agent_colors : np.ndarray (N, 4)
    opinions : np.ndarray (N,)    current private opinions x(t)
    history : list[np.ndarray]    trajectory x(0), x(1), …
    """

    def __init__(self, config: SimConfig,
                 quantizers: "list[Quantizer] | None" = None,
                 quantizer_overrides: "dict[int, Quantizer] | None" = None) -> None:
        self.config = config
        cfg = config

        if cfg.signal_function not in SIGNAL_FUNCTIONS:
            raise ValueError(
                f"signal_function must be one of {sorted(SIGNAL_FUNCTIONS)}; "
                f"got {cfg.signal_function!r}")

        N = int(cfg.N)
        self.N = N
        self.L = int(cfg.L)

        # --- Inertia α ---
        self.alpha_vec = build_alpha_vector(
            num_agents=N,
            alpha=cfg.alpha,
            alpha_mode=cfg.alpha_mode,
            alpha_seed=cfg.alpha_seed,
            alpha_min=cfg.alpha_min,
            alpha_max=cfg.alpha_max,
            alpha_manual=cfg.alpha_manual,
        )

        # --- Network & influence matrix W ---
        graph_data = build_directed_weight_matrix(
            num_agents=N,
            graph_type=cfg.graph_type,
            seed=cfg.seed,
            regular_degree=cfg.regular_degree,
            directed=cfg.directed,
            directed_mode=cfg.directed_mode,
            weight_mode=cfg.weight_mode,
            ensure_valid_graph=cfg.ensure_valid_graph,
            weight_seed=cfg.seed + 997,
            layout_algo=cfg.layout_algo,
            sbm_sizes=cfg.sbm_sizes,
            sbm_p_in=cfg.sbm_p_in,
            sbm_p_out=cfg.sbm_p_out,
            ws_k=cfg.ws_k,
            ws_p=cfg.ws_p,
            group_sizes=cfg.group_sizes,
            cross_group_weight=cfg.cross_group_weight,
        )
        self.G = graph_data["graph"]
        self.base_graph = graph_data["base_graph"]
        self.layout_pos = graph_data["layout_pos"]
        self.W = graph_data["W"]
        self.in_degree = graph_data["in_degree"]
        self.out_degree = graph_data["out_degree"]
        self.is_weakly_connected = graph_data["is_weakly_connected"]
        self.is_strongly_connected = graph_data["is_strongly_connected"]
        self.scc_count = graph_data["scc_count"]
        self.largest_scc_size = graph_data["largest_scc_size"]
        self.has_one_way_edges = graph_data["has_one_way_edges"]

        # --- Exaggeration factors ξ (Definition 2) ---
        self.xi, self.w_tilde = compute_xi(
            self.W,
            self.alpha_vec,
            ensure_valid_graph=cfg.ensure_valid_graph,
        )

        # --- Quantizers q_i ---
        family = cfg.quantizer_family
        self.eta_vec = None
        if quantizers is not None:
            # explicit per-agent quantizers override the config family
            if len(quantizers) != N:
                raise ValueError(f"need {N} quantizers, got {len(quantizers)}")
            self.quantizers: list[Quantizer] | None = list(quantizers)
            self.thresholds_matrix, self.reconstruction_levels_matrix = (
                stack_quantizers(self.quantizers))
        elif family == "strategic":
            self.quantizers = None  # implicit; matrices are canonical here
            (self.thresholds_matrix,
             self.reconstruction_levels_matrix,
             self.eta_vec) = compute_thresholds_closed_form(
                xi=self.xi, num_bins=self.L)
        else:
            self.quantizers = build_quantizers(family, N, self.L)
            self.thresholds_matrix, self.reconstruction_levels_matrix = (
                stack_quantizers(self.quantizers))

        # --- Per-agent quantizer overrides (e.g. one agent with a custom map) ---
        if quantizer_overrides:
            qlist = self.quantizers
            if qlist is None:
                # strategic family: wrap the canonical matrix rows
                qlist = [Quantizer(self.thresholds_matrix[i],
                                   self.reconstruction_levels_matrix[i])
                         for i in range(N)]
            qlist = list(qlist)
            for idx, q_over in quantizer_overrides.items():
                idx = int(idx)
                if not 0 <= idx < N:
                    raise ValueError(
                        f"override agent index {idx} out of range 0..{N - 1}")
                qlist[idx] = q_over
            self.quantizers = qlist
            self.thresholds_matrix, self.reconstruction_levels_matrix = (
                stack_quantizers(qlist))

        # --- Sources N_i^S and audiences N_i^A ---
        self.sources: list[list[int]] = [
            np.flatnonzero(self.W[i] > 0).tolist() for i in range(N)
        ]
        self.audiences: list[list[int]] = [
            np.flatnonzero(self.W[:, i] > 0).tolist() for i in range(N)
        ]

        self._validate_model_profile(
            custom_quantizers=(
                quantizers is not None
                or bool(quantizer_overrides)
                or family == "custom"
            )
        )

        # --- Agent colours ---
        from .plots import get_agent_colors  # matplotlib colormap lookup only
        self.agent_colors = get_agent_colors(N, cmap_name=cfg.color_map)

        # --- Initial private opinions x(0) ---
        self._initial_opinions = build_initial_opinions(
            num_agents=N,
            mode=cfg.initial_opinion_mode,
            opinion_seed=cfg.opinion_seed,
            symmetric_scale=cfg.symmetric_scale,
            manual_values=cfg.initial_opinions_manual,
            identical_value=cfg.identical_value,
            base_graph=self.base_graph,
            eigen_index=cfg.eigen_index,
            eigen_scale=cfg.eigen_scale,
            eigen_noise_snr=cfg.eigen_noise_snr,
        )
        self.opinions = self._initial_opinions.copy()
        self.history: list[np.ndarray] = [self.opinions.copy()]
        self._bin_history: list[np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Model validation
    # ------------------------------------------------------------------

    def _effective_quantizer_rows(
        self,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return unpadded ``(thresholds, reconstruction_levels)`` per agent."""
        rows: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(self.N):
            thresholds = np.asarray(self.thresholds_matrix[i], dtype=float)
            levels = np.asarray(self.reconstruction_levels_matrix[i], dtype=float)
            active = np.flatnonzero(np.diff(thresholds) > 0.0)
            if active.size == 0:
                rows.append((thresholds[:1], levels[:0]))
                continue
            last = int(active[-1]) + 1
            rows.append((thresholds[:last + 1], levels[:last]))
        return rows

    @staticmethod
    def _is_self_consistent_row(
        thresholds: np.ndarray,
        levels: np.ndarray,
    ) -> bool:
        """Check r_ℓ ∈ B_ℓ = [τ_{ℓ-1}, τ_ℓ) for every bin."""
        if len(thresholds) != len(levels) + 1 or not len(levels):
            return False
        lower_ok = levels >= thresholds[:-1]
        upper_ok = levels < thresholds[1:]
        upper_ok[-1] = levels[-1] <= thresholds[-1]
        return bool(np.all(lower_ok & upper_ok))

    def _validate_model_profile(self, *, custom_quantizers: bool) -> None:
        """Warn (extended profile) or enforce the paper's assumptions.

        With ``model_profile='paper'`` the run must satisfy the modelling
        assumptions of the paper:

        * discrete-time dynamics (dt = 1) without perturbations;
        * a signaling map studied in the paper (linear, saturated, quantized);
        * every agent has at least one source and one audience member
          (Standing Assumption 1), so that ξ_i ≥ 1 is well defined;
        * for the quantized map, every agent uses its L-bin credible code
          (Assumption 3), i.e. the 'strategic' family without overrides.
        """
        cfg = self.config
        profile = cfg.model_profile
        if profile not in {"extended", "paper"}:
            raise ValueError("model_profile must be 'extended' or 'paper'")

        quantized = cfg.signal_function == "quantized"
        rows = self._effective_quantizer_rows()
        inconsistent_agents = [
            i + 1
            for i, (thresholds, levels) in enumerate(rows)
            if not self._is_self_consistent_row(thresholds, levels)
        ]

        if profile == "extended":
            if quantized and custom_quantizers and inconsistent_agents:
                warnings.warn(
                    "Custom reconstruction levels are not self-consistent "
                    f"for agents {inconsistent_agents} (paper indexing); "
                    "extended mode permits this.",
                    UserWarning,
                    stacklevel=3,
                )
            return

        violations: list[str] = []
        if cfg.continuous_time:
            violations.append("continuous_time must be False")
        if not np.isclose(float(cfg.dt), 1.0):
            violations.append("dt must equal 1")
        if cfg.add_perturbation:
            violations.append("add_perturbation must be False")
        if cfg.signal_function not in PAPER_SIGNAL_FUNCTIONS:
            violations.append(
                f"signal_function must be one of {list(PAPER_SIGNAL_FUNCTIONS)}")
        no_source = [i + 1 for i in range(self.N) if not self.sources[i]]
        no_audience = [i + 1 for i in range(self.N) if not self.audiences[i]]
        if no_source:
            violations.append(f"agents without sources: {no_source[:8]}")
        if no_audience:
            violations.append(f"agents without an audience: {no_audience[:8]}")
        if quantized:
            if cfg.quantizer_family != "strategic" or custom_quantizers:
                violations.append(
                    "quantized signaling must use the credible codes "
                    "(quantizer_family='strategic', no custom quantizers)")
            if inconsistent_agents:
                violations.append(
                    "reconstruction levels must be self-consistent for "
                    f"agents {inconsistent_agents}")
        if violations:
            details = "\n - ".join(violations)
            raise ValueError(
                "Paper model profile validation failed:\n - " + details
            )

    # ------------------------------------------------------------------
    # Core simulation methods
    # ------------------------------------------------------------------

    @property
    def signal_mode(self) -> SignalMode:
        """Kernel selector for ``config.signal_function``."""
        return SIGNAL_FUNCTIONS[self.config.signal_function]

    def reset(self) -> None:
        """Restore opinions to their initial state and clear the trajectory."""
        self.opinions = self._initial_opinions.copy()
        self.history = [self.opinions.copy()]
        self._bin_history = None

    def get_public_signals(
        self,
        opinions: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the public signals s(x) = (s_1(x_1), …, s_N(x_N)).

        Parameters
        ----------
        opinions : np.ndarray, shape (N,) or None
            Private opinions; if None, uses the current ``self.opinions``.
        """
        x = self.opinions if opinions is None else np.asarray(opinions, dtype=float)
        return public_signals_numpy(
            x,
            self.thresholds_matrix,
            self.reconstruction_levels_matrix,
            int(self.signal_mode),
            self.config.tanh_K,
            self.xi,
        )

    def get_social_input(
        self,
        opinions: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the social input Σ_j w_ij s_j(x_j), i.e. W s(x)."""
        return self.W @ self.get_public_signals(opinions)

    def update_map(self, opinions: np.ndarray) -> np.ndarray:
        """Evaluate one step of the closed-loop map without mutation."""
        x = np.asarray(opinions, dtype=float)
        if x.shape != (self.N,):
            raise ValueError(f"opinions must have shape ({self.N},)")
        discrete_update = (
            self.alpha_vec * x
            + (1.0 - self.alpha_vec) * self.get_social_input(x)
        )
        dt = float(self.config.dt)
        if dt != 1.0:
            return (1.0 - dt) * x + dt * discrete_update
        return discrete_update

    def step(self) -> np.ndarray:
        """Advance the simulation by one period (in place)."""
        self.opinions = self.update_map(self.opinions)
        self.history.append(self.opinions.copy())
        return self.opinions

    def run(
        self,
        steps: int,
        track_bins: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Run the simulation for *steps* periods using the fast kernel.

        The simulator is reset to x(0) first, so the returned history always
        starts at the initial opinions.

        Parameters
        ----------
        steps : int
            Number of periods to simulate.
        track_bins : bool
            Also return the bin index of every agent at every period.

        Returns
        -------
        history : np.ndarray, shape (steps+1, N)
            Private-opinion trajectory x(0), x(1), …, x(steps).
        bin_indices : np.ndarray, shape (steps+1, N) or None
            Zero-based bin index ℓ-1 of every agent and period.
        """
        cfg = self.config
        self.reset()

        dt = float(cfg.dt)
        perturb_iter = int(cfg.perturb_time)
        if cfg.continuous_time and dt > 0:
            # perturb_time is specified in continuous time; map to iteration
            perturb_iter = int(round(cfg.perturb_time / dt))

        history, bin_indices = run_simulation(
            opinions=self._initial_opinions,
            W=self.W,
            alpha_vec=self.alpha_vec,
            thresholds=self.thresholds_matrix,
            reconstruction_levels=self.reconstruction_levels_matrix,
            steps=int(steps),
            track_bins=track_bins,
            add_perturbation=cfg.add_perturbation,
            perturb_type=cfg.perturb_type,
            perturb_scale=cfg.perturb_scale,
            perturb_time=perturb_iter,
            signal_mode=int(self.signal_mode),
            tanh_K=cfg.tanh_K,
            xi=self.xi,
            dt=dt,
        )

        self.history = list(history)
        self.opinions = history[-1].copy()
        if track_bins:
            self._bin_history = (
                list(bin_indices) if bin_indices is not None else None)

        return history, bin_indices

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> np.ndarray:
        """Current private opinions x(t)."""
        return self.opinions

    @property
    def bin_history(self) -> list[np.ndarray] | None:
        """Tracked zero-based bin indices, if requested in :meth:`run`."""
        return self._bin_history

    @property
    def metadata(self) -> dict[str, Any]:
        """Summary dict of key model quantities (no large arrays)."""
        cfg = self.config
        return {
            "N": self.N,
            "L": self.L,
            "signal_function": cfg.signal_function,
            "quantizer_family": cfg.quantizer_family,
            "model_profile": cfg.model_profile,
            "graph_type": cfg.graph_type,
            "directed": cfg.directed,
            "directed_mode": cfg.directed_mode,
            "is_weakly_connected": self.is_weakly_connected,
            "is_strongly_connected": self.is_strongly_connected,
            "scc_count": self.scc_count,
            "largest_scc_size": self.largest_scc_size,
            "has_one_way_edges": self.has_one_way_edges,
            "alpha_mode": cfg.alpha_mode,
            "xi_min": float(np.nanmin(self.xi)),
            "xi_max": float(np.nanmax(self.xi)),
            "xi_mean": float(np.nanmean(self.xi)),
            "eta_min": float(self.eta_vec.min()) if self.eta_vec is not None else None,
            "eta_max": float(self.eta_vec.max()) if self.eta_vec is not None else None,
            "steps_run": len(self.history) - 1,
        }

    def __repr__(self) -> str:  # pragma: no cover
        cfg = self.config
        return (
            f"StrategicCODASimulator(N={self.N}, L={self.L}, "
            f"signal_function='{cfg.signal_function}', "
            f"graph='{cfg.graph_type}', steps_run={len(self.history) - 1})"
        )
