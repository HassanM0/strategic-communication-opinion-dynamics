"""
presets.py — Named SimConfig presets (JSON files) and save/load helpers.

Presets live as JSON files in the repository's ``presets/`` folder:

  figure4_karate_club   Fig. 4 of the paper (Karate Club, α_i = 0.5, L = 5, T = 30)
  figure5_two_group     Fig. 5 of the paper (five-node two-group network,
                        ρ = 0.2, α_i = 0.05, L = 5, T = 20)
  ring_cycle            directed 5-cycle with a perturbation (exploration)

Change ``signal_function`` to 'linear', 'saturated' or 'quantized' to switch
between the three regimes compared in the figures.

Usage
-----
    from strategic_coda.presets import list_presets, load_preset, save_preset

    cfg = load_preset("figure4_karate_club")
    cfg_linear = cfg.replace(signal_function="linear")
    save_preset(cfg_linear, "my_experiment")          # -> presets/my_experiment.json
"""

from __future__ import annotations

import os

from .config import SimConfig

__all__ = [
    "PRESET_DIR", "BUILTIN_PRESETS", "list_presets", "preset_path",
    "load_preset", "save_preset",
]

#: Folder holding the JSON presets (``presets/`` at the repository root).
PRESET_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "presets"))

#: Presets defined in code (always available, even without the presets folder).
BUILTIN_PRESETS: dict[str, SimConfig] = {
    "Default": SimConfig(),
}


def _safe_name(name: str) -> str:
    """Validate a preset name and strip an optional .json suffix."""
    name = str(name).strip()
    if name.lower().endswith(".json"):
        name = name[:-5].strip()
    if not name or any(c in name for c in '\\/:*?"<>|'):
        raise ValueError(f"invalid preset name: {name!r}")
    return name


def preset_path(name: str) -> str:
    """Full path of the JSON file backing the named preset."""
    return os.path.join(PRESET_DIR, f"{_safe_name(name)}.json")


def list_presets() -> list[str]:
    """Names of all JSON presets in PRESET_DIR."""
    if not os.path.isdir(PRESET_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(PRESET_DIR)
        if f.lower().endswith(".json")
    )


def _is_path(name_or_path: str) -> bool:
    text = str(name_or_path)
    return os.sep in text or "/" in text or os.path.isfile(text)


def load_preset(name_or_path: str) -> SimConfig:
    """Load a preset by name (from PRESET_DIR or BUILTIN_PRESETS) or by path."""
    if name_or_path in BUILTIN_PRESETS:
        return BUILTIN_PRESETS[name_or_path].replace()
    path = name_or_path if _is_path(name_or_path) else preset_path(name_or_path)
    return SimConfig.load_json(path)


def save_preset(config: SimConfig, name_or_path: str) -> str:
    """Save a SimConfig by name (into PRESET_DIR) or to a path; returns the path."""
    if _is_path(name_or_path):
        path = str(name_or_path)
    else:
        if _safe_name(name_or_path) in BUILTIN_PRESETS:
            raise ValueError(f"{name_or_path!r} is a built-in preset name")
        os.makedirs(PRESET_DIR, exist_ok=True)
        path = preset_path(name_or_path)
    config.save_json(path)
    return path
