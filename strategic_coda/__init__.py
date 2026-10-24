"""
strategic_coda — simulator for networked opinion dynamics with strategic
public communication.

Companion code of
  H. Munif, A. Couthures, V. S. Varma, S. Lasaulce,
  "Network-Induced Strategic Communication in Opinion Dynamics",
  arXiv:2607.16036.

Public API
----------
SimConfig                     : dataclass holding all simulation parameters
StrategicCODASimulator        : simulator of x(t+1) = D_α x(t) + (I - D_α) W s(x(t))
Quantizer, build_quantizers, stack_quantizers : interval quantizers q_i
compute_xi                    : network-induced exaggeration factor ξ_i (Definition 2)
compute_thresholds_closed_form: L-bin credible codes τ_{i,ℓ}, r_{i,ℓ} (Theorem 4)
plot_simulation, plot_public_signal_evolution, get_agent_colors : figures
load_preset, save_preset, list_presets : JSON presets in presets/
build_ui                      : interactive ipywidgets interface (needs the
                                'ui' extra: pip install -e ".[ui]")
"""

from .config import SimConfig
from .equilibrium import compute_thresholds_closed_form, compute_xi
from .quantizers import Quantizer, build_quantizers, stack_quantizers
from .simulator import StrategicCODASimulator
from .plots import get_agent_colors, plot_public_signal_evolution, plot_simulation
from .presets import list_presets, load_preset, save_preset

__version__ = "1.0.0"

__all__ = [
    "SimConfig",
    "StrategicCODASimulator",
    "Quantizer",
    "build_quantizers",
    "stack_quantizers",
    "compute_xi",
    "compute_thresholds_closed_form",
    "plot_simulation",
    "plot_public_signal_evolution",
    "get_agent_colors",
    "list_presets",
    "load_preset",
    "save_preset",
    "build_ui",
]


def build_ui(*args, **kwargs):
    """Build the interactive interface (imports ipywidgets on first use)."""
    try:
        from .widgets import build_ui as _build_ui
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The interactive interface needs ipywidgets: "
            'pip install -e ".[ui]"') from exc
    return _build_ui(*args, **kwargs)
