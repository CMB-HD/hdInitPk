# hdInitPk

This repository contains the code used in the analysis and forecasts of [Cheslog et al. (2026)](https://arxiv.org/abs/XXXX.XXXXX). Please cite that work if you use this software or the associated data.

We constrain the primordial scalar power spectrum $\mathcal{P}(k)$ both as a power law, with an amplitude $A_\mathrm{s}$, spectral index $n_\mathrm{s}$, and running $\alpha_\mathrm{s}$, and as a general binned function of wavenumber from current CMB and BAO data (*Planck*, ACT, SPT, and DESI), and forecast the constraints achievable with SO-like and CMB-HD-like surveys.

# Installation

## Requirements

To use this software, you must have Python (version >=3) installed, along with the following Python packages:
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [CAMB](https://camb.readthedocs.io/en/latest/)
- [hdfisher](https://github.com/CMB-HD/hdfisher)
- [hdMockData](https://github.com/CMB-HD/hdMockData)
- [hdPk](https://github.com/CMB-HD/hdPk) (this provides `cmb_from_pk`, used for the binned primordial power spectrum calculation)
- [pandas](https://pandas.pydata.org/) (this is only required to run the Jupyter notebook)
- [matplotlib](https://matplotlib.org/) (this is only required to run the Jupyter notebook)
- [getdist](https://getdist.readthedocs.io/en/latest/intro.html) (this is only required to run the Jupyter notebook)
- [Cobaya](https://cobaya.readthedocs.io/en/latest/) (this is only required to run new MCMC chains)
- [hdlike](https://github.com/CMB-HD/hdlike) (this is only required to run new CMB-HD MCMC chains)
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/) (__optional__: the calculation of the derivatives used in the Fisher matrices can be parallelized, but this is not required.)

## Installation instructions

Simply clone this repository and install with `pip`:

```
git clone https://github.com/CMB-HD/hdInitPk.git
cd hdInitPk
pip install . --user
```

# Reproducing the plots and tables in Cheslog et al. (2026)

We provide one Jupyter notebook that can be run to reproduce the results of Cheslog et al. (2026):
- `hdInitPk_plots.ipynb`: Reproduces every figure and table in the paper, in two parts.
  - **Part 1: Current results** reproduces the CMB-PAS + DESI DR2 constraints (Figure 1 and the current-constraints table) from the thinned MCMC chains, along with the marginalized binned-$\mathcal{P}(k)$ chain statistics used later.
  - **Part 2: Forecasted results** reproduces Figures 2 through 11 and the remaining tables from the saved Fisher matrices, the saved Figure 6 points, and the chains loaded in Part 1.

The notebook reads the thinned chains, the forecasted Fisher matrices, the Figure 6 $P_\mathrm{lin}(k, z=0)$ points, and the binning files from `hdinitpk/data`. Each saved Fisher matrix already includes the $\tau$ prior and, where applicable, the `HMCode_logT_AGN` prior and the DESI BAO Fisher, exactly as applied in the paper, so the error tables follow from a simple matrix inversion.

The cells are meant to be executed in order from top to bottom, with the exception the bias calculation of Figures 7 and 8. This  additionally needs the CAMB and CLASS Fisher derivative directories, which we have not included with the data but may be rerun by following the example for creating new Fisher derivatives in the [hdfisher](https://github.com/CMB-HD/hdfisher) repository. 

# Running MCMCs for Current Constraints

The Cobaya input files used for every MCMC chain in the paper are in `hdinitpk/cobaya_yaml_files`:

- `PAS/` — The chains for a variety of sets of free parameters run on current data. 
- `binned_pk/` — the binned primordial $\mathcal{P}(k)$ chains run on current data, for the seven-bin scheme of Section III and the finer 30-bin scheme, with and without DESI BAO, and for both the CMB-PAS (`cmb-spa-*`) and P-ACT-LB (`p-pactlike-lb-*`) data combinations.

Run one with:

```
cobaya-run hdinitpk/cobaya_yaml_files/binned_pk/cmb-spa-desi-hd-pk.yaml
```

Some of the paths (written as `/path/to/...`) are placeholders: the external likelihood data (the *Planck* `clik` files, the SPT-3G D1 candl data set, etc.) and the directory this repository was cloned into. These will need to be filled in before they can be run.

# Running binned primordial P(k) with arbitrary binning

`arbitrary_Pkbinning.py` provides `BinnedPk`, a Cobaya `Theory` class that hands CAMB a primordial scalar power spectrum defined by an amplitude in each of a set of $k$ bins, rather than by a power law. The bin amplitudes are the cubic-spline values CAMB interpolates between, so with dense enough sampling any $\mathcal{P}(k)$ can be reproduced.

Configure it through these class options:

| Option | Meaning |
|---|---|
| `nbins` | The number of bins, and of free amplitudes `b1` … `b{nbins}`. |
| `k_min_bin`, `k_max_bin` | $\log_{10} k$ at the start and end of the log-spaced range, in Mpc<sup>-1</sup>. |
| `k_first` | A list of $k$ values (Mpc<sup>-1</sup>) to use as the leading bins with an alternate binning scheme. The remaining `nbins - len(k_first)` bins are log-spaced over the range above. Pass `None` or `[]` for an entirely evenly log-spaced binning; pass all `nbins` values to specify the grid exactly. |
| `scale` | The units of the bin amplitudes (default `1e-9`). |
| `bin_par` | A template `ParamDict` (prior, ref, proposal) applied to every bin, unless you give the bins individual priors in the `params` block. |

The sampled parameter for bin $i$ is `b_i`, related to the primordial spectrum by

$$b_i = e^{-2\tau} \ \mathcal{P}_i(k) \ / \ \texttt{scale}$$

since `BinnedPk` multiplies the amplitudes by `scale` and by $e^{2\tau}$ before passing them to CAMB. With the default `scale`, $b_i = 10^9 \, e^{-2\tau} \mathcal{P}_i(k)$. Sampling in $e^{-2\tau}\mathcal{P}(k)$ rather than $\mathcal{P}(k)$ minimizes the degeneracy between the optical depth and the overall amplitude.

The `*-arbitrary-binning.yaml` files in `hdinitpk/cobaya_yaml_files/binned_pk` are working examples: one for each chain in the paper, using this class with its bins set to the paper's own seven- and 30-bin schemes. Point `python_path` at your clone of this repository so Cobaya can import the module:

```yaml
theory:
  arbitrary_Pkbinning.BinnedPk:
    python_path: /path/to/hdInitPk
    nbins: 7
    k_first: [0.0018619358218086469, 0.006086768361185574, ...]
    scale: 1.0e-09
    params:
      b1:
        prior: {min: 0.0, max: 15.0}
        ...
  camb:
    external_primordial_pk: true
    ...
```

Note that `external_primordial_pk: true` must be set on the `camb` block, and that a proposal covariance built for a different binning (or for a different parameterization of the amplitudes) will not match these parameters.

# Obtaining linear P(k) today from primordial P(k)

`InitPk_to_LinearPk.py` converts constraints on the primordial spectrum into constraints on the linear matter power spectrum today, $P_\mathrm{lin}(k, z=0)$, which is plotted in Figure 6. It computes

$$P_\mathrm{lin}(k z=0) = 2 \pi^2 \ k \ T(k)^2 \ \mathcal{P}(k)$$

with one CAMB call per cosmology returning the $z = 0$ matter transfer function $T(k)$, and $$\mathcal{P}(k)$$ the sampled primordial power in each bin. The script runs in two modes, either or both of which can be enabled:

- **Fisher mode** draws Gaussian samples from the CMB-HD and SO-like binned-$\mathcal{P}(k)$ Fisher forecasts and computes $P_\mathrm{lin}$ for each. It needs the Fisher derivative directories, set at the top of the file as `HD_FISHER_DERIV_DIR` and `SO_FISHER_DERIV_DIR`.
- **Chain mode** computes $P_\mathrm{lin}$ for every post-burn-in sample of the binned-$\mathcal{P}(k)$ MCMC chain.

Both modes use the same transfer-function method, but each keeps the CAMB accuracy settings of the calculation it came from.

By default, chain mode reads the **thinned** chain distributed with the package, so that the script runs out of the box. **The results in the paper were obtained from the full, un-thinned chains.** To reproduce them exactly, point `chain_root` at the raw Cobaya chain root and set `BURN_IN_FRACTION` back to `0.5`; the file documents both. Note that the thinned chains had their burn-in removed *before* thinning, which is why `BURN_IN_FRACTION` is `0` for them.

The run is parallelized with `hdfisher.mpi`:

```
mpirun -n 32 python InitPk_to_LinearPk.py
```

Fisher mode splits each experiment's samples across all ranks; chain mode distributes whole chains across ranks, so request one task per chain for that part. The per-bin summaries it writes are what the `fig6_points_*.txt` files read by the notebook were made from.

# Running new Fisher forecasts

`hdinitpk.fisher.Fisher` extends [hdfisher](https://github.com/CMB-HD/hdfisher)'s `Fisher` with a binned primordial power spectrum and a kSZ template, and takes the same arguments plus `binned_pk`, `bin_edges`, `ksz`, and `pk_frac_step`. Code written with `hdfisher` only needs to change its import.

`example_calc_binnedPk_derivs.py` is a template for calculating the Fisher derivatives, and shows how to assemble a Fisher matrix from them afterwards:

```python
from hdinitpk import fisher

fisherlib = fisher.Fisher(
    fisher_dir, overwrite=True, binned_pk=True, bin_edges=bin_edges,
    pk_frac_step=0.05, fisher_steps_file=..., param_file=...)
fisherlib.calculate_fisher_derivs()
```

When `binned_pk=True`, the parameter file must give the number of bins (`nkbins`), the bin centers (`k1`, `k2`, …), the bin amplitudes (`Pk1`, `Pk2`, … or `eneg2tauPk1`, …), and an `effective_ns_for_nonlinear` entry; the step-size file must give a step for each bin amplitude; and `bin_edges` is required. Fiducial-parameter and step-size files for the paper's binnings are in `hdinitpk/data/fisher_fid_params` and `hdinitpk/data/fisher_steps`.

Because the derivatives are computed by perturbing the spectra inside one $k$ bin at a time, `pk_frac_step` must match the step size given for the `Pk` parameters in the step-size file — the spectra are perturbed by `pk_frac_step`, while the finite-difference denominator comes from the step-size file. The example pairs them correctly and runs the same calculation at 1%, 5%, and 10% so the derivatives can be checked for convergence
.
The binned-$\mathcal{P}(k)$ calculation uses CAMB (via `hdPk`) and cannot be combined with `use_class=True`. Any additional parameter accepted by CAMB's [`set_params`](https://camb.readthedocs.io/en/latest/camb.html#camb.set_params) may be varied.
