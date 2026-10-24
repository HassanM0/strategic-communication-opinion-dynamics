# Network-Induced Strategic Communication in Opinion Dynamics

Code and paper sources for

> H. Munif, A. Couthures, V. S. Varma, S. Lasaulce, T. Başar,
> **Network-Induced Strategic Communication in Opinion Dynamics**,
> arXiv:[2607.16036](https://arxiv.org/abs/2607.16036), 2026.

The repository contains `strategic_coda`, a Python simulator of opinion dynamics in which agents hold
private opinions and communicate through public signals. It reproduces the simulation figures of the
paper, and an interactive notebook lets you explore the model beyond them.

This repository is a frozen snapshot that matches the arXiv version of the paper.

## The model in one paragraph

Agents $i=1,\dots,N$ hold private opinions $x_i\in[-1,1]$ and listen to their sources with a
row-stochastic influence matrix $W=[w_{ij}]$. Each agent broadcasts a public signal $s_i(x_i)$ and updates

$$x_i(t+1)=\alpha_i x_i(t)+(1-\alpha_i)\sum_j w_{ij}\,s_j(x_j(t)) \qquad \text{(Eq. (4))},$$

where $\alpha_i$ is the agent's inertia. The network induces in each sender an **exaggeration factor**
$\xi_i\ge 1$ (Definition 2). The signaling map $s_i$ depends on the regime (Sec. IV):

| `signal_function` | Signaling map | Regime in the paper |
|---|---|---|
| `"linear"` | $s_i(x)=x$ | aligned incentives |
| `"saturated"` | $s_i(x)=\operatorname{sat}(\xi_i x)$ | persuasion with naive receivers (Eq. (11), with $\delta_i=0$ by Proposition 2) |
| `"quantized"` | $s_i=q_i$, the $L$-bin credible code (Theorem 4) | persuasion with Bayesian receivers: **strategic CODA** (Eq. (18)) |

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

## Reproducing the figures

Open [`notebooks/paper_figures.ipynb`](notebooks/paper_figures.ipynb) and run all cells. The figures
are written to `paper/figures_paper_1/`, which is where `paper/main.tex` reads them from, so
recompiling the paper picks up the freshly simulated figures.

| Figure | Content | Output file | Preset |
|---|---|---|---|
| Fig. 2 | $\xi_i$ on six example graphs ($N=12$, $\alpha_i=0.5$) | `fig_section3_xi_topology_gallery_compact.pdf` | — |
| Fig. 4 | Karate Club network under the three signaling maps | `fig_section7_karate_private_public_signature.pdf` | `presets/figure4_karate_club.json` |
| Fig. 5 | Two-group network: persistent period-2 public switching | `fig_section7_private_public_signature.pdf` | `presets/figure5_two_group.json` |

Figs. 1 and 3 are TikZ drawings inside `main.tex`.

In Fig. 4 the influence weights $w_{ij}$ are proportional to the interaction counts of Zachary's data set
(`weight_mode="karate_weights"`), normalized over each agent's sources.

`pytest` runs [`tests/test_paper.py`](tests/test_paper.py), which checks the closed-form credible codes
(Theorem 4, Proposition 3, Corollary 2) and the key numbers behind Figs. 2, 4 and 5.

`paper/main.pdf` is compiled from these sources and figures. To recompile it:
`cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## Exploring the model

[`notebooks/interactive_explorer.ipynb`](notebooks/interactive_explorer.ipynb) opens an interactive
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

## Notation: paper ↔ code

Agents are numbered $1,\dots,N$ in the paper, in plots and in custom-quantizer rules, while
arrays in the code are zero-based (agent $i$ is index `i-1`).

| Paper | Meaning | Code |
|---|---|---|
| $N$ | number of agents | `SimConfig.N`, `sim.N` |
| $\alpha_i$ | inertia | `SimConfig.alpha` / `alpha_mode`, `sim.alpha_vec` |
| $W=[w_{ij}]$ | influence of source $j$ on receiver $i$ | `sim.W[i, j]` |
| $\mathcal N_i^{\mathrm S}$, $\mathcal N_i^{\mathrm A}$ | sources, audience | `sim.sources[i]`, `sim.audiences[i]` |
| $\widetilde w_{ki}=(1-\alpha_k)w_{ki}$ | susceptibility weight | `sim.w_tilde[k, i]` |
| $\xi_i$ | exaggeration factor (Definition 2) | `compute_xi`, `sim.xi` |
| $\eta_i=\operatorname{arccosh}(2\xi_i-1)$ | Eq. (14) | `sim.eta_vec` |
| $L$ | number of bins | `SimConfig.L` |
| $\tau_{i,\ell}$ | bin thresholds (Theorem 4) | `sim.thresholds_matrix[i, ℓ]` |
| $r_{i,\ell}$ | reconstruction levels (Eq. (12)) | `sim.reconstruction_levels_matrix[i, ℓ-1]` |
| $\mathcal B_{i,\ell}$ | bin of agent $i$ | `sim.run(T, track_bins=True)` returns the bin indices $\ell-1$ |
| $q_i$ | induced quantizer (Eq. (16)) | `Quantizer`, `signal_function="quantized"` |
| $s_i$ | effective signaling map | `SimConfig.signal_function`, `sim.get_public_signals(x)` |
| $\mathbf x(t)$, $T$ | opinion profile, horizon | `history[t]`, `SimConfig.steps` |

## Repository layout

```
strategic_coda/              the simulator package
  simulator.py               StrategicCODASimulator
  equilibrium.py             exaggeration factor ξ_i and closed-form credible codes
  quantizers.py              interval quantizers q_i
  dynamics.py                simulation kernels (NumPy / Numba)
  graph.py, alpha.py, opinions.py   network W, inertia α, initial opinions x(0)
  config.py, presets.py      SimConfig and JSON presets
  plots.py, export.py        figures, CSV/JSON export, animation
  widgets.py                 interactive interface
presets/                     figure4_karate_club, figure5_two_group, ring_cycle
notebooks/                   paper_figures.ipynb, interactive_explorer.ipynb
paper/                       LaTeX sources of the arXiv version + figures_paper_1/
tests/                       reproducibility checks
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

## License

The code is released under the [MIT License](LICENSE). The paper in `paper/` is distributed under the
license chosen on arXiv.
