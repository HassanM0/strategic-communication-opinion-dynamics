"""Reproducibility checks for arXiv:2607.16036 (run with `pytest`)."""
import networkx as nx
import numpy as np
import pytest

from strategic_coda import (
    Quantizer,
    SimConfig,
    StrategicCODASimulator,
    compute_thresholds_closed_form,
    compute_xi,
    load_preset,
)


def _row_normalized(G: nx.Graph) -> np.ndarray:
    A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    return A / A.sum(axis=1, keepdims=True)


# --------------------------------------------------------------- Fig. 2 / Definition 2
@pytest.mark.parametrize("graph, expected", [
    (nx.star_graph(11), [2.0] + [22.0] * 11),
    (nx.cycle_graph(12), [4.0] * 12),
    (nx.random_regular_graph(4, 12, seed=42), [8.0] * 12),
])
def test_exaggeration_factor_examples(graph, expected):
    xi, _ = compute_xi(_row_normalized(graph), np.full(12, 0.5))
    assert np.allclose(xi, expected)


def test_xi_at_least_one():
    G = nx.watts_strogatz_graph(30, 4, 0.2, seed=1)
    xi, _ = compute_xi(_row_normalized(G), np.random.default_rng(0).uniform(0, 0.9, 30))
    assert np.all(xi >= 1.0)


# --------------------------------------------------------------- Theorem 4 / Prop. 3 / Cor. 2
@pytest.mark.parametrize("xi", [1.3, 2.0, 5.0, 40.0])
@pytest.mark.parametrize("L", [1, 2, 5, 9])
def test_closed_form_credible_code(xi, L):
    tau, r, eta = compute_thresholds_closed_form(np.array([xi]), L)
    tau, r = tau[0], r[0]
    assert np.isclose(tau[0], -1.0) and np.isclose(tau[-1], 1.0)          # boundary conditions
    assert np.allclose(tau, -tau[::-1])                                    # symmetry
    assert np.allclose(r, 0.5 * (tau[:-1] + tau[1:]))                      # Eq. (12)
    for l in range(1, L):                                                  # indifference, Eq. (13)
        assert np.isclose(tau[l + 1] - (4 * xi - 2) * tau[l] + tau[l - 1], 0.0, atol=1e-10)
        assert np.isclose(0.5 * (r[l - 1] + r[l]), xi * tau[l])
        assert abs(tau[l]) <= 1.0 / xi + 1e-12                              # Corollary 2
    assert np.isclose(eta[0], np.arccosh(2 * xi - 1))                      # Eq. (14)
    assert Quantizer(tau, r).is_strictly_self_consistent


# --------------------------------------------------------------- Fig. 4
def test_figure4_karate_club():
    cfg = load_preset("figure4_karate_club")
    expected = {  # (min x(T), max x(T)) as printed by notebooks/paper_figures.ipynb
        "linear": (0.134, 0.144),
        "saturated": (1.000, 1.000),
        "quantized": (-0.534, 0.535),
    }
    for signal_function, (lo, hi) in expected.items():
        sim = StrategicCODASimulator(cfg.replace(signal_function=signal_function))
        history, _ = sim.run(cfg.steps)
        assert history.shape == (31, 34)
        assert np.isclose(history[-1].min(), lo, atol=5e-4)
        assert np.isclose(history[-1].max(), hi, atol=5e-4)
    assert np.isclose(sim.xi.min(), 4.005, atol=5e-4)
    assert np.isclose(sim.xi.max(), 55.984, atol=5e-4)


# --------------------------------------------------------------- Fig. 5
def test_figure5_two_group_network():
    cfg = load_preset("figure5_two_group")
    sim = StrategicCODASimulator(cfg)
    rho = cfg.cross_group_weight
    expected_W = np.array([
        [0, 1 - rho, rho / 3, rho / 3, rho / 3],
        [1 - rho, 0, rho / 3, rho / 3, rho / 3],
        [rho / 2, rho / 2, 0, (1 - rho) / 2, (1 - rho) / 2],
        [rho / 2, rho / 2, (1 - rho) / 2, 0, (1 - rho) / 2],
        [rho / 2, rho / 2, (1 - rho) / 2, (1 - rho) / 2, 0],
    ])
    assert np.allclose(sim.W, expected_W)
    assert np.allclose(sim.history[0], [0.585, -0.459, -0.911, 0.730, -0.487])
    assert np.allclose(sim.xi, [1.728, 1.728, 2.987, 2.987, 2.987], atol=5e-4)

    # Quantized signaling: persistent period-2 public switching.
    history, _ = sim.run(cfg.steps)
    s = np.array([sim.get_public_signals(x) for x in history])
    assert np.allclose(s[-1], s[-3]) and not np.allclose(s[-1], s[-2])
    assert np.isclose(s[-1].min(), -0.605, atol=5e-4) and np.isclose(s[-1].max(), 0.605, atol=5e-4)

    # Linear and saturated signaling settle.
    for signal_function in ["linear", "saturated"]:
        h, _ = StrategicCODASimulator(cfg.replace(signal_function=signal_function)).run(cfg.steps)
        assert np.ptp(h[-1]) < 1e-2


# --------------------------------------------------------------- paper profile
def test_paper_profile_accepts_paper_presets():
    for name in ["figure4_karate_club", "figure5_two_group"]:
        cfg = load_preset(name)
        assert cfg.model_profile == "paper"
        for signal_function in ["linear", "saturated", "quantized"]:
            StrategicCODASimulator(cfg.replace(signal_function=signal_function))


@pytest.mark.parametrize("override", [
    {"signal_function": "tanh"},
    {"quantizer_family": "uniform"},
    {"add_perturbation": True},
    {"continuous_time": True, "dt": 0.1},
])
def test_paper_profile_rejects_extensions(override):
    cfg = load_preset("figure5_two_group").replace(**override)
    with pytest.raises(ValueError, match="Paper model profile"):
        StrategicCODASimulator(cfg)


def test_unknown_preset_field_is_rejected():
    with pytest.raises(ValueError, match="unknown SimConfig fields"):
        SimConfig.from_dict({"type_min": -1.0})
