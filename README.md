# Network-Induced Strategic Communication in Opinion Dynamics

Code and paper sources for

> H. Munif, A. Couthures, V. S. Varma, S. Lasaulce, T. Başar,
> **Network-Induced Strategic Communication in Opinion Dynamics**,
> arXiv:[2607.16036](https://arxiv.org/abs/2607.16036), 2026.

The repository contains `strategic_coda`, a Python simulator of opinion dynamics in which agents hold
private opinions and communicate through public signals. It reproduces the simulation figures of the
paper, and an interactive notebook lets you explore the model beyond them.

## Installation

Python 3.10 or newer is required. From the repository folder:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt   # exact versions used for the paper (Python 3.12.6)
pip install -e .                  # the strategic_coda package itself
```

The core package only needs NumPy, SciPy, NetworkX and Matplotlib; `pip install -e ".[fast,ui]"`
adds Numba (faster simulations) and the interface dependencies. Figs. 4 and 5 typeset their labels
with LaTeX, so a LaTeX distribution (TeX Live or MiKTeX) is needed to reproduce them exactly. Without
LaTeX the notebook falls back to Matplotlib's built-in math rendering.

## Exploring the model

[`simulator/interactive_explorer.ipynb`](simulator/interactive_explorer.ipynb) opens an interactive
interface (ipywidgets). In it you can:

- **Change the model:** pick the network, the inertia, the number of bins $L$ and the signaling map,
  then rerun with one click.
- **Use presets:** load `figure4_karate_club` or `figure5_two_group` from the *Presets* tab to start
  from the paper's settings.
- **Check against the paper:** set *Model profile* to *Paper assumptions* to have every run checked
  against the paper's modelling assumptions (Standing Assumption 1, Assumption 3).

Options labelled *beyond the paper* are extras for exploration that the paper does not analyse:
tanh signaling, uniform, sign and custom quantizers, continuous time, perturbations and directed
variants.

Programmatic use:

```python
from strategic_coda import StrategicCODASimulator, load_preset, plot_simulation

cfg = load_preset("figure5_two_group")              # settings of Fig. 5
for regime in ["linear", "saturated", "quantized"]:
    sim = StrategicCODASimulator(cfg.replace(signal_function=regime))
    history, _ = sim.run(cfg.steps)                 # x(0), ..., x(T)
    signals = [sim.get_public_signals(x) for x in history]
    plot_simulation(sim, history, layout="horizontal")
```

## Citation

If you use this code, please cite the paper (GitHub's *Cite this repository* button uses
[`CITATION.cff`](CITATION.cff)):

```bibtex
@article{munif2026network,
  title   = {Network-Induced Strategic Communication in Opinion Dynamics},
  author  = {Munif, Hassan and Couthures, Anthony and Varma, Vineeth S. and Lasaulce, Samson and Ba{\c{s}}ar, Tamer},
  journal = {arXiv preprint arXiv:2607.16036},
  year    = {2026}
}
```
