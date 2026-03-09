# The Sample Size Required in Importance Sampling

This repository contains a small Python codebase and notebook for reproducing numerical experiments around a central question in importance sampling:

> How large must the sample size `n` be for importance sampling to behave well?

The project studies this question in three settings:

1. **Independent spins** — a discrete Gibbs model where several quantities can be computed analytically.
2. **Ideal Gaussian shift case** — a well-behaved reference example.
3. **Bimodal Gaussian mixture** — a more difficult case showing the limitations of the same threshold idea.

The main workflow is organized in the notebook **`Run_Experiments.ipynb`**, which imports the Python modules, runs the experiments, and produces the plots.

---

## Repository structure

```text
.
├── Run_Experiments.ipynb                # Main notebook to reproduce all experiments and plots
├── The_Sample_Size_Required_in_Importance_Sampling.pdf
│                                       # Paper/report describing the project and theory
├── Presentation.pdf                    # Presentation slides
├── README.md                           # Project documentation
│
├── Independent_spins/
│   ├── Independent_spins.py            # Core experiment code for the independent-spins model
│   ├── utils.py                        # Importance-sampling helpers and variance diagnostics
│   ├── plot.py                         # Plotting functions for the independent-spins results
│   └── __init__.py
│
├── ideal_case/
│   ├── rare_event.py                   # Gaussian shift “ideal case” experiments
│   ├── plot.py                         # Generic plotting helpers used by the Gaussian experiments
│   └── __init__.py
│
├── Mixture_bimodale/
│   ├── Mixture_bimodale.py             # Bimodal Gaussian mixture experiments
│   └── __init__.py
│
└── __pycache__/                        # Python cache files (not needed for reproduction)
```

---

## What each part does

### 1) `Independent_spins/`
This folder contains the most complete experiment pipeline.

It implements:
- exact formulas for the partition function and its derivatives,
- IID sampling under the Gibbs distribution,
- a simple Gibbs/MCMC sampler,
- estimation of the threshold quantity `L`,
- repeated experiments across a grid of values of `c` where
  `log n = L + c sigma`,
- summary statistics such as:
  - median relative error,
  - median log error,
  - weight-dominance statistic `Q_n`,
  - variance diagnostic `v_n`.

Main entry point:
- **`Independent_spins.experiment(...)`**

Used in the notebook section:
- **Independent Spins**

### 2) `ideal_case/`
This folder implements the Gaussian shift model, used as an “ideal” or better-behaved reference case.

It provides:
- repeated Monte Carlo runs for different dimensions / shift amplitudes,
- computation of the theoretical threshold
  `L = (1/2) d a^2`,
- plots of error, weight dominance, and variance diagnostics.

Main entry point:
- **`ideal_case.rare_event.experiment(...)`**

Used in the notebook section:
- **Ideal Case**

### 3) `Mixture_bimodale/`
This folder studies a harder target distribution: a symmetric Gaussian mixture.

It provides:
- sampling and log-density evaluation for the mixture,
- Monte Carlo estimation of `L`,
- repeated IS runs across sample sizes,
- diagnostics showing when the threshold behavior becomes less reliable.

Main entry point:
- **`Mixture_bimodale.Mixture_bimodale.experiment_mixture(...)`**

Used in the notebook section:
- **High Limitation: Bi-modale case, rare event**

---

## Requirements

This project does **not** require compilation.
It is a pure Python project.

The code imports only:
- `numpy`
- `matplotlib`
- `jupyter` or `notebook` (to run the notebook)

A recent Python 3 version is recommended.
Python **3.10+** is a safe choice.

---

## Installation

### Option 1: create a virtual environment

#### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy matplotlib jupyter
```

#### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install numpy matplotlib jupyter
```

### Option 2: install directly in your current environment
```bash
pip install numpy matplotlib jupyter
```

---

## How to run the project

### Recommended: run the notebook
From the repository root, launch Jupyter:

```bash
jupyter notebook
```

Then open:
- **`Run_Experiments.ipynb`**

Run the notebook from top to bottom.
This is the easiest way to reproduce **all results and plots** in the repository.

The notebook is already organized into three sections:
1. **Independent Spins**
2. **Ideal Case**
3. **High Limitation: Bi-modale case, rare event**

---

## How to reproduce all results

### A) Independent spins experiments
The notebook runs:

```python
from Independent_spins import experiment

N_list = [2, 10, 20, 30]
beta0 = 1
beta = 0.5
c_grid = np.arange(-2, 2.5, 0.1)
reps = 200
seed = 42

results_iid, results_mcmc, results_L_estimate, results_iid_Lhat, results_mcmc_Lhat = experiment(
    N_list, beta0, beta, c_grid, reps, seed
)
```

Then it generates plots using:

```python
from Independent_spins.plot import (
    plot_results,
    plot_compare_L_Lhat,
    plot_Lhat_error_and_variance,
    plot_centered_transition,
)
```

This section compares:
- IID sampling,
- MCMC sampling,
- the theoretical threshold `L`,
- the estimated threshold `L_hat`.

### B) Ideal Gaussian shift case
The notebook runs:

```python
from ideal_case.rare_event import experiment
from ideal_case.plot import plot_results, plot_L_values

results = experiment(d=50, a_values=(0.3, 0.4, 0.5), ns=None, R=200, seed=0)
```

Default sample sizes are:

```python
ns = [10**k for k in range(1, 7)]
```

Then it plots the diagnostics with:

```python
plot_results(results, "Importance Sampling failure when μ ≠ ν")
plot_L_values(results)
```

### C) Bimodal Gaussian mixture case
The notebook runs:

```python
from Mixture_bimodale.Mixture_bimodale import experiment_mixture

results_mix = experiment_mixture(
    d=50,
    a_values=(0.3, 0.5, 0.7),
    ns=None,
    R=200,
    seed=0,
    M_L=50_000,
)
```

Then it plots the outputs with:

```python
from ideal_case.plot import plot_results
plot_results(results_mix, "Failure when ν is not concentrated around its mean (mixture target)")
```