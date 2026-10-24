"""
config.py — SimConfig dataclass holding all simulation parameters.

A SimConfig fully describes one run; presets are SimConfigs saved as JSON.
Symbols follow the paper (arXiv:2607.16036): α_i inertia, W influence
matrix, L number of bins, ξ_i exaggeration factor, q_i quantizer,
s_i effective signaling map.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Literal


@dataclasses.dataclass
class SimConfig:
    """All parameters that control a strategic-CODA simulation run.

    Attributes
    ----------
    N : int
        Number of agents.
    alpha : float
        Common inertia α_i = alpha (used when alpha_mode='scalar'), in [0, 1).
    L : int
        Number of bins L of each agent's quantizer (L ≥ 1).
    steps : int
        Horizon T (number of periods). In continuous-time mode, the time
        horizon of the Euler integration.
    dt, continuous_time : float, bool
        Forward-Euler relaxation x ← (1-dt) x + dt f(x); dt = 1 is the
        exact discrete-time map of the paper (not in the paper otherwise).
    signal_function : str
        Effective signaling map s_i (Sec. IV):
        'quantized'  s_i = q_i, the strategic-CODA regime (Bayesian receivers),
        'linear'     s_i(x) = x (aligned incentives),
        'saturated'  s_i(x) = sat(ξ_i x) (naive receivers),
        'tanh'       s_i(x) = tanh(K x) (not in the paper).
    quantizer_family : str
        Quantizers used when signal_function='quantized':
        'strategic' (L-bin credible codes, Theorem 4; the paper's model),
        'uniform' (equally spaced rating scale), 'sign' (classical CODA,
        L is ignored), or 'custom' (per-agent rules from custom_rules;
        agents not covered use custom_base).
    custom_rules : str
        JSON list of {"agents", "thresholds", "reconstruction_levels"}.
        Agents use paper indexing ('all', '3', '1-11, 14'); later rules win.
    graph_type : str
        Network topology (see graph.build_base_graph).
    seed : int
        Random seed for graph generation, layout and edge orientation.
    regular_degree : int
        Degree d for 'd_regular' (and m for 'powerlaw_cluster').
    sbm_sizes, sbm_p_in, sbm_p_out
        Stochastic block model parameters.
    ws_k, ws_p
        Watts–Strogatz parameters.
    group_sizes, cross_group_weight
        'two_group' network: group sizes (e.g. "2,3") and the total weight ρ
        each receiver puts on the other group (with weight_mode='group_weights').
    directed, directed_mode
        Orient the base topology ('strongly_connected', 'random_asymmetric',
        'line_directed', 'ring_directed').
    weight_mode : str
        Influence weights w_ij of each receiver over its sources:
        'uniform_in', 'random_in', 'out_degree', 'in_degree', 'eigenvector',
        'karate_weights' (edge weights of the Karate Club data set),
        'group_weights' (two-group network with cross-group weight ρ).
    ensure_valid_graph : bool
        Require a connected (undirected) / strongly connected (directed)
        network with every agent having sources and an audience.
    alpha_mode, alpha_seed, alpha_min, alpha_max, alpha_manual
        Inertia assignment: 'scalar', 'heterogeneous-random' (iid uniform in
        [alpha_min, alpha_max]), or 'manual' (comma-separated N values).
    initial_opinion_mode : str
        'random' (iid U[-1, 1]), 'symmetric_zero', 'manual', 'identical',
        'eigen' (graph-Laplacian eigenvector), 'eigen_noise'.
    opinion_seed, symmetric_scale, initial_opinions_manual, identical_value,
    eigen_index, eigen_scale, eigen_noise_snr
        Parameters of the initial-opinion modes.
    model_profile : str
        'extended' allows every option of the simulator; 'paper' enforces
        the modelling assumptions of the paper (see
        StrategicCODASimulator._validate_model_profile).
    add_perturbation, perturb_type, perturb_scale, perturb_time
        Optional one-off perturbation of all opinions (not in the paper).
    Remaining fields only affect plotting and the interactive interface.
    """

    # --- Core model ---
    N: int = 20
    alpha: float = 0.5
    L: int = 4
    steps: int = 60
    dt: float = 1.0               # Euler step; 1 = exact discrete-time update
    continuous_time: bool = False  # integrate dx/dt = -x + f(x); steps = horizon T
    signal_function: Literal["quantized", "linear", "saturated", "tanh"] = "quantized"
    tanh_K: float = 1.0
    quantizer_family: Literal["strategic", "uniform", "sign", "custom"] = "strategic"
    custom_rules: str = ('[{"agents": "all", "thresholds": "-1, -0.5, 0, 0.5, 1", '
                         '"reconstruction_levels": "-0.75, -0.25, 0.25, 0.75"}]')
    custom_base: Literal["strategic", "uniform", "sign"] = "strategic"
    model_profile: Literal["extended", "paper"] = "extended"

    # --- Network ---
    graph_type: Literal[
        "watts_strogatz", "complete", "star", "wheel", "line", "grid", "ring",
        "stochastic_block", "d_regular", "bipartite", "powerlaw_cluster",
        "erdos_renyi", "karate_club", "petersen", "dodecahedral", "two_group",
    ] = "star"
    seed: int = 11
    regular_degree: int = 3
    sbm_sizes: str = "10,10"
    sbm_p_in: float = 0.3
    sbm_p_out: float = 0.06
    ws_k: int = 4
    ws_p: float = 0.1
    group_sizes: str = "2,3"
    cross_group_weight: float = 0.2
    directed: bool = False
    directed_mode: Literal[
        "strongly_connected", "random_asymmetric",
        "line_directed", "ring_directed",
    ] = "strongly_connected"
    weight_mode: Literal[
        "uniform_in", "random_in", "out_degree", "in_degree", "eigenvector",
        "karate_weights", "group_weights",
    ] = "uniform_in"
    ensure_valid_graph: bool = False

    # --- Inertia α_i ---
    alpha_mode: Literal["scalar", "heterogeneous-random", "manual"] = "scalar"
    alpha_seed: int = 23
    alpha_min: float = 0.2
    alpha_max: float = 0.8
    alpha_manual: str = ""

    # --- Initial private opinions x(0) ---
    initial_opinion_mode: Literal[
        "random", "symmetric_zero", "manual", "identical", "eigen", "eigen_noise"
    ] = "random"
    opinion_seed: int = 9
    symmetric_scale: float = 0.3
    initial_opinions_manual: str = ""
    identical_value: float = 0.0
    eigen_index: int = 1          # 1 = Fiedler vector (first non-trivial)
    eigen_scale: float = 0.8      # amplitude scaling in [0, 1]
    eigen_noise_snr: float = 10.0  # SNR in dB for eigen_noise mode

    # --- Perturbation (not in the paper) ---
    add_perturbation: bool = False
    perturb_type: Literal["Random", "Deterministic"] = "Random"
    perturb_scale: float = 0.05
    perturb_time: int = 30

    # --- Visualisation ---
    color_map: str = "turbo"
    show_edges: bool = True
    show_community: bool = False
    color_community_opinion: bool = False
    show_exag_profile: bool = False
    show_inverse_xi_profile: bool = False
    fig3_mode: Literal["identity", "opinion", "dual"] = "opinion"
    label_policy: Literal["auto", "always", "never"] = "auto"
    layout_algo: Literal[
        "spring", "circular", "shell", "spectral", "kamada_kawai"
    ] = "shell"
    fig_layout: Literal["split", "horizontal", "vertical"] = "split"

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict of all parameters."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SimConfig":
        """Construct a SimConfig from a plain dict (e.g. loaded from JSON).

        Unknown keys raise a ValueError so that a typo in a preset is not
        silently ignored.
        """
        valid = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(d) - valid)
        if unknown:
            raise ValueError(f"unknown SimConfig fields: {unknown}")
        return cls(**d)

    def save_json(self, path: str) -> None:
        """Serialise config to a JSON file."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")

    @classmethod
    def load_json(cls, path: str) -> "SimConfig":
        """Load a SimConfig from a JSON file."""
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def replace(self, **kwargs) -> "SimConfig":
        """Return a new SimConfig with selected fields overridden."""
        return dataclasses.replace(self, **kwargs)
