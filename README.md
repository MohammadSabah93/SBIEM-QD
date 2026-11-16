# SBIEM-QD — Spectral Boundary-Integral Analysis and Validation Tools

Python research scripts for numerical analysis of fully dynamic (FD) and quasi-dynamic (QD) fault-rupture models, with particular emphasis on **spectral elastodynamic memory**, **rate-and-state friction**, **numerical passivity**, and **earthquake-cycle metrics**.

The repository collects tools for fitting and assessing relaxation-spectrum representations of wave-mediated fault traction, comparing FD and QD event characteristics, checking single-mode elastodynamic responses, and preparing quantitative research results. Its scientific context includes the SEAS BP1 benchmark and antiplane (mode III) and in-plane (mode II) fault mechanics.

## Research topics

- **Elastodynamic memory approximation:** fitting complex-conjugate relaxation spectra to spectral boundary-integral response functions.
- **Dissipation and passivity:** analytical and discrete-time checks of the stability and non-energy-generating behavior of reduced memory representations, including exact polynomial methods.
- **Dynamic rupture comparisons:** measures of rupture speed, peak slip rate, coseismic slip, and arrival-time differences between simulation outputs.
- **Earthquake cycles and fault segmentation:** event catalogues, recurrence intervals, stress drops, sequence delays, and ensemble statistics.
- **Verification and performance:** independent single-Fourier-mode checks, execution timing, and preparation of machine-readable supplementary research data.

## Repository layout

| Script | Purpose |
| --- | --- |
| `codes/generate_data/analysis.py` | Shared event loading and rupture metrics |
| `codes/generate_data/score.py` | Profile-based comparisons between reference and approximate simulations |
| `codes/generate_data/fit_memory_kernel.py` | Relaxation-spectrum fits to elastodynamic memory kernels |
| `codes/generate_data/fit_memory_passive.py` | Passivity-constrained memory fitting |
| `codes/generate_data/certify_discrete.py` | Continuous and discrete passivity certificates |
| `codes/generate_data/mode_pde_check.py` | Independent plane-strain single-mode time-domain check |
| `codes/generate_data/cycle_analysis.py` | Earthquake-cycle event catalogues and summary statistics |
| `codes/generate_data/seg_stats.py` | Segmented-fault event and sequence statistics |
| `codes/generate_data/seg_ensemble.py` | Ensemble statistics for segmented-fault configurations |
| `codes/generate_data/memory_remnant.py` | Assessment of residual dynamic memory between nearby events |
| `codes/generate_data/timing_study.py` | Performance characterization of numerical treatments |
| `codes/generate_data/make_supplement.py` | Assembly of reproducibility tables and numerical supplements |
| `codes/qd_paths.py` | Shared, repository-relative output paths |
| `requirements.txt` | Python dependencies |

## Python environment

Python 3.10 or later is recommended. Install the scientific dependencies with:

```bash
python -m pip install -r requirements.txt
```

The project uses NumPy and SciPy for numerical computing, Numba for accelerated numerical kernels, and SymPy for symbolic and exact-arithmetic calculations.

## Analysis workflows

The scripts are organized around computational experiments and their result files. For example, compare an FD event archive with one or more QD event archives:

```bash
python codes/generate_data/score.py FD.npz QD1.npz QD2.npz
```

Generate a catalogue of earthquake-cycle events from existing simulation output:

```bash
python codes/generate_data/cycle_analysis.py run.npz
```

Individual scripts contain documentation of their input assumptions, numerical formulations, and generated output paths. Some workflows use previously generated NPZ simulation results and JSON memory-fit coefficients in the `data/` tree; these are experiment-specific inputs.

## Scientific use

These scripts are intended for research, numerical-method assessment, and quantitative comparison of fault-rupture models. Scientific results should be evaluated using appropriate convergence tests, validation data, and explicitly documented numerical settings.

## Author

**Mohammad Sabah, PhD** — computational geomechanics, fault mechanics, and induced seismicity.

[GitHub profile](https://github.com/MohammadSabah93) · [ORCID](https://orcid.org/0000-0002-7260-6138)

## License

Released under the [MIT License](LICENSE).

Copyright (c) 2025 Mohammad Sabah.

For academic use, please cite this repository and identify the commit corresponding to the version used.
