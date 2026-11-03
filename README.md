# Transformer Thermal PINN

A physics-informed neural network (PINN) for modelling temperature along a power transformer's height over time. This project adapts the thermal model in **Bragone et al. (2022)** and compares PINN predictions with a finite volume method (FVM) reference using **synthetic data**.

The project builds on an earlier Burgers-equation practice implementation. It now includes a transformer heat-equation solver, time-dependent inputs, model training, temperature comparisons and error diagnostics.

## Approach

The implementation uses the numerical coefficients printed in the paper's scaled Eq. 12:

$$
1.8u_t = 0.05u_{xx} + 15 + 83K(t)^2 - (u-T_a(t)).
$$

Here, $u(x,t)$ is temperature, $T_a(t)$ is ambient temperature, and $K(t)$ is electrical load factor. The height coordinate spans $0\leq x\leq H$, with $H=1$.

The boundary conditions are:

$$
u(0,t)=T_a(t),\qquad u(H,t)=T_o(t),
$$

where $T_o(t)$ is the supplied top-oil temperature.

The numerical time coordinate advances by one per hourly observation. The coefficients reproduce the printed Eq. 12 literally; their equivalence to Table 1's SI coefficients converted to hours has **not** been established.

### Networks and training

- A temperature network takes height and time and predicts $\hat u(x,t)$.
- A time-only network predicts ambient temperature and squared load. These outputs cannot vary with height.
- Both networks use fully connected layers with `tanh` activations and are trained jointly with L-BFGS.
- The loss combines temperature boundary error, load-loss error, ambient-temperature error, the heat-equation residual and an initial-condition error.

For controlled verification, both methods use the same assumed linear initial temperature profile:

$$
u(x,0)=T_a(0)+[T_o(0)-T_a(0)]\frac{x}{H}.
$$

The PINN fits this profile through a loss term. The two-network architecture and added initial-condition constraint are adaptations of the paper's method, rather than an exact reproduction of its experiment.

## Files

| File | Purpose |
| --- | --- |
| `heat_model.py` | Shared numerical coefficients, cell-centred FVM solver with backward Euler time stepping, and an analytical steady-state solution for checking the solver. |
| `FVM-Reference.ipynb` | Checks solver convergence, generates synthetic input series, calculates the reference temperatures and saves them for the PINN notebook. |
| `Transformer-PINN.ipynb` | Builds and trains the PINN, evaluates it against FVM, plots temperatures and errors, and contains optional repeated-seed and hyperparameter experiments. |

Keep these three files together. The names above assume the notebooks have been renamed to remove download suffixes such as `(5)`.

## Setup

Clone the repository and enter its directory:

```bash
git clone https://github.com/Adele0001/transformer-thermal-pinn.git
cd transformer-thermal-pinn
```

Create and activate a Python environment on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies and launch Jupyter:

```bash
python -m pip install numpy scipy torch matplotlib pandas notebook ipykernel
python -m notebook
```

Use the notebook kernel associated with this environment. The PINN selects Apple's MPS backend when available and otherwise runs on CPU. Training time and results can vary across hardware and software versions; exact dependency versions are not pinned here.

## Run order

### 1. Generate the FVM reference

Open `FVM-Reference.ipynb` and run its cells from top to bottom.

With `USE_REAL_DATA = False`, it generates 100 synthetic observations for ambient temperature, load factor and top-oil temperature. The current top-oil series is an algebraic function of ambient temperature and load, with noise; it does not independently model realistic thermal lag.

The notebook checks spatial and temporal convergence, then generates:

- `fvm_reference.npz`: inputs, spatial/time grids, reference temperatures and parameter metadata.
- `fvm_solution.png`: reference temperature plots.

The reference uses 50 spatial cells and 10 internal time steps per observation interval. `.npz` is a binary NumPy archive: load it with `np.load(...)`, rather than opening it in Jupyter's text editor.

### 2. Train and evaluate the PINN

Open `Transformer-PINN.ipynb` and run its cells in order. It loads `fvm_reference.npz`; the FVM temperature field is used for evaluation, not supervised training.

The main run uses 100 sampled boundary observations, 10,000 sampled physics-check points plus those boundary points, and four hidden layers of 50 neurons in each network. Training also uses 50 initial-profile points.

Outputs include:

- `pinn_transformer_best.pt`: the single-run model's saved state. Despite the filename, this is the final saved model, not necessarily the best model across experiments.
- `pinn_fig4.png`: PINN heatmap and FVM/PINN temperature profiles.
- `pinn_fig5.png`: side-by-side temperature-field comparison.
- `pinn_top_oil.png`: predicted versus synthetic reference top-oil temperature.

The error-map cell displays the absolute field differences and reports their maximum location. It does not save that figure unless a `savefig` call is added.

Training can take tens of minutes or longer. L-BFGS progress counters count **loss evaluations**, which are not the same as optimiser iterations.

### 3. Optional experiments

The three-seed experiment repeats the same settings with seeds 1234, 2345 and 3456 to assess sensitivity to initial weights and sampled points. It performs three additional training runs. If its cell is not guarded by an enable flag, running all cells will execute it automatically.

The hyperparameter sweep compares 20 distinct configurations. It is disabled with `RUN_SWEEP = False`; enable it deliberately because it can take hours. Results are saved after each run to `sweep_results_time_only_ic.csv`, and completed configurations are skipped. Use a new CSV filename if the dataset, model implementation or training setup changes.

Restart the notebook kernel after changing `heat_model.py`, then rerun the FVM notebook before the PINN notebook.

## Preliminary results

The latest shared single-run notebook output reports:

| Metric | Result |
| --- | ---: |
| Temperature-field relative L2 error against FVM | 2.09% |
| Top-oil relative L2 error against the synthetic reference | 4.45% |
| Maximum absolute field difference on the evaluated grid | 5.545 °C |
| Location of maximum field difference | Time coordinate 38, height 0.99 m |

At the largest field difference, FVM gives 38.819 °C and PINN gives 44.364 °C. These results show close overall agreement while retaining a larger local discrepancy near the top boundary.

Relative L2 error is `norm(prediction - reference) / norm(reference)` across the evaluated values. It is not a bound on every individual temperature error.

Earlier checks of the supplied reference found approximately second-order spatial convergence at steady state. Refining from 50 cells/10 substeps to 200 cells/40 substeps changed the reference field by approximately 0.16% on the coarse evaluation grid.

## Interpretation and remaining work

These results support numerical agreement for one synthetic experiment. They do not establish real-transformer accuracy or exact reproduction of the paper's results.

- Top-oil evaluation includes observations used for boundary training, so it assesses reconstruction rather than prediction of future temperatures.
- Learned ambient temperature and load can differ from the interpolated functions used by FVM, particularly between observations.
- The linear initial profile is an assumption introduced for controlled verification.
- Three-seed repeatability results were not complete in the latest shared notebook.
- Further work includes examining local boundary errors, comparing learned forcing with reference inputs, completing repeated-seed tests and evaluating suitable real measurements when available.

The paper reports 14.5% field error and 2.59% top-oil error for its selected configuration. Those values are context, not a direct performance comparison: this project uses different data and modified training constraints and architecture.

## Reference

Bragone, F., Morozovska, K., Hilber, P., Laneryd, T., & Luvisotto, M. (2022). *Physics-informed neural networks for modelling power transformer's dynamic thermal behaviour*. Electric Power Systems Research, 211, 108447.

https://doi.org/10.1016/j.epsr.2022.108447
