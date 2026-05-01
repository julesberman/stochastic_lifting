# Stochastic Lifting Demo Code

This repository contains demo code for the paper [Stochastic Lifting for Generating Trajectories of Stochastic Physical Systems](https://julesberman.github.io/files/Stochastic_Lifting.pdf) by Jules Berman, Tobias Blickhan, and Benjamin Peherstorfer.

The code is intended for running small stochastic lifting experiments in JAX. The reusable Python modules live in the installable `sl` package, and the notebook demos import from that package.

## What Stochastic Lifting Does

Stochastic lifting learns a stochastic one-step transition map from paired trajectory data. The training data consists of adjacent states,

```text
(x_t, x_{t+1})
```

where `x_t` may be a low-dimensional state, a spatial field, an image, or a short history of previous states. A deterministic model trained only as `x_t -> x_{t+1}` tends to predict an average next state when several outcomes are possible. Stochastic lifting instead assigns each observed transition an independent random label `xi` and trains

```text
F_theta(x_t, xi_t) ~= x_{t+1}.
```

At inference time, fresh labels are sampled at each step:

```text
x_{t+1} = F_theta(x_t, xi_t),   xi_t ~ N(0, I).
```

Repeating this autoregressively generates diverse trajectories from the same initial condition with one model evaluation per generated time step.

The key points used in these demos are:

- Use paired current-next transitions from trajectories.
- Give every training transition its own fixed random label.
- Train the transition map with a standard regression loss.
- During rollout, sample a fresh independent label at every step.
- For fields or images, use a Markov history such as the last three frames as the model input.

## Included Demo

- `wave.ipynb`: trains a conditional U-Net on wave-equation trajectories with random media and generates multiple rollouts from a shared initial wave frame.

## Setup

Use Python 3.11 or newer. The project uses `uv` for environment management.

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From the repository root, install the project and its dependencies:

```bash
uv sync
```

This installs the local `sl` package in editable mode inside the `uv` environment, so changes under `sl/` are visible to the notebooks without manual path edits.

Start JupyterLab:

```bash
uv run jupyter lab
```

If you prefer the classic notebook interface:

```bash
uv run jupyter notebook
```

Open notebooks from the Jupyter file browser. The notebooks live at the repository root and import reusable code with `from sl...` imports.

## Run The Wave U-Net Demo

1. Start Jupyter with `uv run jupyter lab`.
2. Open `wave.ipynb`.
3. Run the cells from top to bottom.
4. The notebook generates wave trajectories, builds three-frame Markov contexts, trains a label-conditioned U-Net, and samples many rollouts from the same initial condition.
5. The final cells display generated wave movies.

The wave demo can be compute-heavy. For a quicker local run, reduce these values in the first code cell:

```python
n_train = 32
n_rollout = 32
train_steps = 500
```

For better output quality, increase `n_train` and `train_steps`.

## Repository Layout

- `sl/`: installable Python package for stochastic lifting utilities.
- `sl/metric.py`: trajectory comparison metrics.
- `sl/opt.py`: small Optax training helper.
- `sl/plot.py`: plotting and animation helpers.
- `sl/sde.py`: Euler-Maruyama SDE simulation utilities.
- `sl/unet.py`: conditional Flax U-Net used by the wave demo.
- `sl/wave.py`: random-media wave-equation data generation.
- `wave.ipynb`: runnable wave-equation stochastic lifting demo.

## Paper Reference

For method details and experiments, see:

```text
Jules Berman, Tobias Blickhan, and Benjamin Peherstorfer.
Stochastic Lifting for Generating Trajectories of Stochastic Physical Systems.
Preprint, 2026.
```

PDF: <https://julesberman.github.io/files/Stochastic_Lifting.pdf>
