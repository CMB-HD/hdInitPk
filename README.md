# hdInitPk

This repository contains the code used in the analysis and forecasts of [Cheslog, Finson, MacInnis, Sehgal, Afshordi, Nerval, and Hložek (2026)](https://arxiv.org/abs/XXXX.XXXXX). (TO DO: add archive number) Please cite that work if you use this software or the associated data.

We constrain the primordial scalar power spectrum $\mathcal{P}(k)$ both as a power law, with an amplitude $A_\mathrm{s}$, spectral index $n_\mathrm{s}$, and running $\alpha_\mathrm{s}$, and as a general binned function of wavenumber. The current constraints come from *Planck* 2018, ACT DR6, SPT-3G D1, and the joint ACT-*Planck*-SPT (APS) CMB lensing reconstruction (the combination we call CMB-PAS), together with DESI DR2 BAO. We also forecast the constraints achievable with SO-like and CMB-HD-like surveys.

# Installation

## Requirements

To use this software, you must have Python (version 3.9 or later) installed, along with the following Python packages.

- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [CAMB](https://camb.readthedocs.io/en/latest/)
- [hdfisher](https://github.com/CMB-HD/hdfisher), the version with the CLASS option
- [hdMockData](https://github.com/CMB-HD/hdMockData), version `v1.2` or later. The CMB-HD theory spectra and bandpowers from CAMB and CLASS, the covariance matrices, the lensing reconstruction noise, and the CAMB and CLASS settings used here were added in v1.2.
- [hdPk](https://github.com/CMB-HD/hdPk), which provides `cmb_from_pk`, used for the binned primordial power spectrum calculation
- [CLASS](https://github.com/lesgourg/class_public) and its Python wrapper `classy`, only needed to recompute the CLASS Fisher forecasts or to run the CLASS CMB-HD chains
- [pandas](https://pandas.pydata.org/), [matplotlib](https://matplotlib.org/), and [getdist](https://getdist.readthedocs.io/en/latest/intro.html), only needed for the Jupyter notebooks
- [Cobaya](https://cobaya.readthedocs.io/en/latest/), only needed to run new MCMC chains
- [hdlike](https://github.com/CMB-HD/hdlike), only needed to run new CMB-HD MCMC chains
- the external CMB and BAO likelihoods, only needed to run new current-data MCMC chains. They are listed with links under [External likelihoods](#external-likelihoods) below.
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/), optional. The Fisher derivative calculation can be spread over several processes with it, but it is not required.

## Installing

Clone this repository and install it with `pip`.

```
git clone https://github.com/CMB-HD/hdInitPk.git
cd hdInitPk
pip install . --user
```

# What is in this repository

Everything in the paper can be reproduced from what is here. The three tables below list the complete contents of the repository, and the sections that follow explain them in more detail.

## Notebooks

| Notebook | Produces | Needs |
|---|---|---|
| `hdInitPk_plots.ipynb` | Figures 1 through 7 and 9 through 11, and Tables III, IV, V, VII, and VIII | nothing beyond what ships with the package |
| `run_hdInitPk_forecasts.ipynb` | the nine Fisher configurations and all twenty Fisher matrices behind the forecasts | Fisher derivatives, which it also shows how to generate |
| `run_hdInitPk_current.ipynb` | the twelve current-data MCMC chains behind Figures 1 and 3 and Tables III and VIII | Cobaya and the external likelihoods listed below |

## Scripts

| Script | Produces |
|---|---|
| `run_hdInitPk_forecasts.py` | the same nine Fisher configurations and twenty matrices as `run_hdInitPk_forecasts.ipynb`, as a batch job under MPI |
| `run_hdInitPk_to_LinearPk.py` | the $P_\mathrm{lin}(k, z=0)$ points of Figure 6, from either the Fisher forecasts or the binned-$\mathcal{P}(k)$ chains |
| `run_hdInitPk_verify_accuracy_settings.py` | Figure 8, the parameter bias from insufficient Boltzmann-code accuracy (Appendix A) |
| `example_calc_binnedPk_forecasts.py` | one complete binned-$\mathcal{P}(k)$ forecast, annotated line by line, with the derivatives, the Fisher matrix built from them, and the forecasted errors |
| `arbitrary_Pkbinning.py` | the Cobaya `Theory` class that hands CAMB a binned primordial spectrum. It is imported by the MCMC input files rather than run directly |

## Data and inputs

| Directory | Contents |
|---|---|
| `hdinitpk/data/` | the thinned MCMC chains, the twenty Fisher matrices, the Figure 6 points, the binning files, the cached binned-$\mathcal{P}(k)$ statistics, the MCMC proposal matrices, and the fiducial-parameter and step-size files of Tables I and II. The CMB-HD mock data itself comes from hdMockData |
| `hdinitpk/data/user_generated_data/` | empty at first. Everything you generate with the scripts and notebooks (Fisher derivatives and matrices, MCMC chains, and statistics computed from them) is written here and read back from here |
| `hdinitpk/cobaya_yaml_files/` | Cobaya input files for every MCMC in the paper. Five power-law runs on current data, seven binned-$\mathcal{P}(k)$ runs on current data, and four CMB-HD mock runs |

Table VI is a table of CAMB and CLASS computation times and is not reproduced by any of the above. Tables I and II are inputs rather than results. They give the $k$ bin centers and priors, and the fiducial parameters, step sizes and priors, which ship in `hdinitpk/data/fisher_fid_params` and `hdinitpk/data/fisher_steps`.

# Reproducing the figures and tables

## The plotting notebook

`hdInitPk_plots.ipynb` runs in two parts, and the cells are meant to be executed in order from top to bottom.

- Part 1, current results, reproduces Figure 1 and Table III from the thinned MCMC chains, along with the marginalized binned-$\mathcal{P}(k)$ statistics used in Part 2.
- Part 2, forecasted results, reproduces Figures 2 through 11 (all but Figure 8) and Tables IV, V, VII and VIII, from the saved Fisher matrices, the saved Figure 6 points, and the chains loaded in Part 1.

It reads the chains, the Fisher matrices, the Figure 6 points and the binning files from `hdinitpk/data`. Each saved Fisher matrix already includes the $\tau$ prior and, where applicable, the $\log_{10}(T_\mathrm{AGN})$ prior and the DESI BAO Fisher, exactly as applied in the paper, so the error tables follow from a matrix inversion.

By default the notebook uses the thinned chains distributed with the package. To use results you have run yourself with `run_hdInitPk_current.ipynb`, set

```python
chain_source = 'custom' # for the chains
fisher_source = 'custom' # for the fisher forecasts
fig6_source = 'custom_both' # for the figure 6 points
```

near the top. The notebook then reads the results from `hdinitpk/data/user_generated_data`, in the format the scripts or cobaya wrote them, and removes 0.5 burn-in from the chains. The Figure 6 points come in two halves, so `fig6_source` can also be `'custom_fisher'`, to use your own CMB-HD and SO-like points (from the fisher mode of `run_hdInitPk_to_LinearPk.py`) with the published CMB-PAS points, or `'custom_chain'`, to use your own CMB-PAS points (from its chain mode) with the published CMB-HD and SO-like points.

## Rebuilding the Fisher forecasts

Unless stated otherwise, the Fisher forecasts in the paper are calculated with CLASS. That includes Table IV and Figures 2 and 4. CAMB is used for the $w_0 w_a$CDM forecasts, where we found the two codes disagree, for the baryonic feedback forecast of Table V, and for the binned-$\mathcal{P}(k)$ forecasts, which go through hdPk and are CAMB only. A CAMB nine-parameter forecast is also kept for the comparison between the two codes in Appendix B.

`run_hdInitPk_forecasts.ipynb` rebuilds every Fisher forecast from scratch. It computes the numerical derivatives of the CMB and BAO theory, assembles the twenty Fisher matrices from them, reports the forecasted parameter errors, and compares them parameter by parameter against the matrices in `hdinitpk/data/fisher_matrices`. It follows the `example_calculate_fisher_matrices.ipynb` notebook of [hdfisher](https://github.com/CMB-HD/hdfisher), extended to the nine configurations this paper uses. These cover CAMB and CLASS, CMB-HD and SO-like, power-law and binned $\mathcal{P}(k)$, with and without baryonic feedback and a kSZ template.

The notebook shows how to run the derivatives in place, which works but takes hours per configuration. `run_hdInitPk_forecasts.py` runs the same nine configurations as a batch job.

```
mpirun -np 4 python run_hdInitPk_forecasts.py
```

Set `CALCULATE_DERIVS = True` at the top of that file. Either way the derivatives land in `hdinitpk/data/user_generated_data/fisher_derivs`, and both the notebook and the script pick them up from there. The derivative directories are far too large to distribute and are not included, so both need derivatives you have generated yourself.

## Rerunning the MCMC chains

`run_hdInitPk_current.ipynb` walks through the twelve Cobaya runs behind the current-data constraints. It shows what each input file contains, which external likelihoods you need, how to fill in the `/path/to/...` placeholders, the exact command for each run, and how to turn the finished chains into Table III and the marginalized binned-$\mathcal{P}(k)$ statistics of Table VIII. The chains and cached results go to `hdinitpk/data/user_generated_data`, in the layout the plotting notebook reads.

## Figure 6 and Figure 8

Two results come from scripts rather than the plotting notebook.

Figure 6, the linear matter power spectrum today, is produced by `run_hdInitPk_to_LinearPk.py` (see [Obtaining the linear P(k) today from the primordial P(k)](#obtaining-the-linear-pk-today-from-the-primordial-pk) below). The per-bin summaries it writes are what the `fig6_points_*.txt` files in `hdinitpk/data/fig6` were made from, and what the notebook plots.

Figure 8, the parameter bias from insufficient Boltzmann-code accuracy, is the only result that needs the CAMB and CLASS Fisher derivative directories. It lives in `run_hdInitPk_verify_accuracy_settings.py`.

```
python run_hdInitPk_verify_accuracy_settings.py
```

The script reads the `camb_9param` and `class_9param` derivatives from `hdinitpk/data/user_generated_data/fisher_derivs`, which is where `run_hdInitPk_forecasts.py` and `run_hdInitPk_forecasts.ipynb` write them. If they are not there it calculates them itself, which is slow. It computes


$$b_i = (F^{-1})_{ij} \ \partial_j C^T \ \mathrm{Cov}^{-1} \ (C^\mathrm{true} - C^\mathrm{fid})$$

at each accuracy setting, divides it by the forecasted error on each parameter, prints the result, and draws Figure 8. Everything else it needs (the accuracy-grid spectra, the high-accuracy reference spectra, and the eight-parameter Fisher matrices) ships with the package. Setting `RECALCULATE_SPECTRA = True` at the top of the script calculates the grid spectra and the reference again, with the settings of Appendix A, instead of reading the shipped ones. That is slow, since the reference alone takes hours, so run it under MPI. Figure 7, which compares the CAMB and CLASS spectra at the settings adopted for the paper, stays in the plotting notebook and needs nothing external.

# Running MCMC chains on current data

The Cobaya input files used for the MCMC chains in the paper are in `hdinitpk/cobaya_yaml_files`.

`current_data/` holds the power-law chains on CMB-PAS + DESI DR2, one per set of free parameters, for Figure 1 and Table III, plus the P-ACT-LB comparison chain.

| File | Model | Data | Thinned chain |
|---|---|---|---|
| `lcdm_nrun.yaml` | $\Lambda$CDM + $\alpha_\mathrm{s}$ | CMB-PAS + DESI DR2 | `chains/pas/pas_lcdm_nrun` |
| `w0wacdm_nrun.yaml` | $w_0w_a$CDM + $\alpha_\mathrm{s}$ | CMB-PAS + DESI DR2 | `chains/pas/pas_w0wacdm_nrun` |
| `lcdm_nrun_nnu.yaml` | $\Lambda$CDM + $\alpha_\mathrm{s}$ + $N_\mathrm{eff}$ | CMB-PAS + DESI DR2 | `chains/pas/pas_lcdm_nrun_nnu` |
| `lcdm_nrun_nnu_mnu.yaml` | $\Lambda$CDM + $\alpha_\mathrm{s}$ + $N_\mathrm{eff}$ + $\sum m_\nu$ | CMB-PAS + DESI DR2 | `chains/pas/pas_lcdm_nrun_nnu_mnu` |
| `p_act_lb.yaml` | $\Lambda$CDM + $\alpha_\mathrm{s}$ | P-ACT-LB | `chains/P-ACT-LB/pact_lb_nrun` |

`binned_pk/` holds the binned primordial $\mathcal{P}(k)$ chains run on current data, for Figure 3 and Table VIII (seven bins) and Figures 10 and 11 (30 bins), with and without DESI BAO, and for both the CMB-PAS and P-ACT-LB data combinations. They use the `BinnedPk` theory class in `arbitrary_Pkbinning.py` with the bins set to the paper's seven- and 30-bin schemes.

| File | Data | Bins |
|---|---|---|
| `cmb_pas_7_bins_arbitrary_binning.yaml` | CMB-PAS | 7 |
| `cmb_pas_30_bins_arbitrary_binning.yaml` | CMB-PAS | 30 |
| `cmb_pas_desi_7_bins_arbitrary_binning.yaml` | CMB-PAS + DESI DR2 | 7 |
| `cmb_pas_desi_30_bins_arbitrary_binning.yaml` | CMB-PAS + DESI DR2 | 30 |
| `p_act_lb_7_bins_no_sroll_arbitrary_binning.yaml` | P-ACT-LB, without `sroll2` | 7 |
| `p_act_lb_30_bins_arbitrary_binning.yaml` | P-ACT-LB | 30 |
| `p_act_lb_30_bins_no_sroll_arbitrary_binning.yaml` | P-ACT-LB, without `sroll2` | 30 |

`HD/` holds the CMB-HD mock chains used to validate the Fisher forecasts against a full MCMC (Figure 9), in six- and nine-parameter models, with theory spectra from either CAMB (`camb_*_likelihood.yaml`) or CLASS (`class_*_likelihood.yaml`). These use the [hdlike](https://github.com/CMB-HD/hdlike) likelihood.

Some of the paths in these files, written as `/path/to/...`, are placeholders. They stand for the external likelihood data (the *Planck* `clik` files, the SPT-3G D1 candl data set, and so on), the directory the chains are written to, and the directory this repository was cloned into. They need to be filled in before the files can be run. Each file has a header listing the placeholders it uses, and `run_hdInitPk_current.ipynb` fills them in for you.

Every chain in the paper was run as six chains, one per MPI process, on the SeaWulf computing system at Stony Brook University. The command for one run is

```
mpirun -np 6 cobaya-run mcmc_runs/cmb_pas_desi_7_bins_arbitrary_binning.yaml
```

where `mcmc_runs/` is the directory of prepared copies that `run_hdInitPk_current.ipynb` writes. That notebook lists the command for every input file. We remove the first half of each chain as burn-in and consider a run converged once its Gelman-Rubin statistic $R-1$ is below 0.01.

Each file's `sampler.mcmc.covmat` points at a proposal covariance in `hdinitpk/data/proposal_matrices/`, built from that run's own converged chain. These ship with the package.

## External likelihoods

These are not dependencies of `hdinitpk` itself. Install only the ones the runs you want require. The *Planck* low-$\ell$ and DESI BAO likelihoods are built into Cobaya and need no separate installation, though Cobaya must download their data (`cobaya-install`).

| Cobaya name in the YAML | What it is | Where to get it |
|---|---|---|
| `clipy_lowl_tt`, `clipy_highl_tt`, `clipy_highl_te`, `clipy_highl_ee` | *Planck* 2018 PR3 `commander` low-$\ell$ TT and `plik` high-$\ell$ TT/TE/EE, read through `clipy`, a pure-Python reimplementation of `clik` | [benabed/clipy](https://github.com/benabed/clipy). The `.clik` data files come from the [Planck Legacy Archive](https://pla.esac.esa.int/) |
| `candl_like` (SPT-3G D1 T&E, `lite` variant) | SPT-3G D1 temperature and E-mode power spectra, via the differentiable likelihood framework `candl` | [Lbalkenhol/candl](https://github.com/Lbalkenhol/candl) and the data in [SouthPoleTelescope/spt_candl_data](https://github.com/SouthPoleTelescope/spt_candl_data) |
| `act_dr6_cmbonly.ACTDR6CMBonly`, `act_dr6_cmbonly.PlanckActCut` | ACT DR6 foreground-marginalized CMB-only likelihood (ACT-lite) | [ACTCollaboration/DR6-ACT-lite](https://github.com/ACTCollaboration/DR6-ACT-lite) |
| `act_dr6_lenslike.ACTDR6LensLike` | ACT DR6 CMB lensing (used with `variant: actplanck_baseline` in the P-ACT-LB run) | [ACTCollaboration/act_dr6_lenslike](https://github.com/ACTCollaboration/act_dr6_lenslike) |
| `act_dr6_spt_lenslike.ACTDR6LensLike` | Joint ACT + *Planck* + SPT-3G CMB lensing (used with `variant: actplanckspt3g_extended` in the CMB-PAS runs) | [qujia7/spt_act_likelihood](https://github.com/qujia7/spt_act_likelihood) |
| `planck_2018_lowl.TT`, `planck_2018_lowl.EE_sroll2` | *Planck* 2018 low-$\ell$ TT and the `sroll2` low-$\ell$ EE reanalysis | built into Cobaya ([likelihood docs](https://cobaya.readthedocs.io/en/latest/likelihood_planck.html)) |
| `bao.desi_dr2.desi_bao_all`, `bao.desi_2024_bao_all` | DESI DR2 and DESI 2024 (DR1) BAO | built into Cobaya ([likelihood docs](https://cobaya.readthedocs.io/en/latest/likelihood_bao.html)) |
| `hdlike.hdlike.HDLike` | the CMB-HD mock likelihood, used only by the `HD/` files | [CMB-HD/hdlike](https://github.com/CMB-HD/hdlike) |
| `bao.generic` | the CMB-HD mock DESI BAO data, used only by the `HD/` files | ships with [CMB-HD/hdlike](https://github.com/CMB-HD/hdlike) |

Two theory-side requirements are worth calling out.

- Every current-data file sets `recombination_model: CosmoRec`. CAMB must be built against [CosmoRec](https://www.jb.man.ac.uk/~jchluba/Science/CosmoRec/CosmoRec.html), which is not the default. See CAMB's [recombination models](https://camb.readthedocs.io/en/latest/recombination.html) documentation.
- `HD/class_*_likelihood.yaml` use `classy` rather than `camb`, so they need [CLASS](https://github.com/lesgourg/class_public) and its Python wrapper, modified as described in Appendix A of the paper. The BBN table those files point at ships with hdMockData, so the `/path/to/hdMockData` placeholder in them must point at your hdMockData checkout. They set `use_class: True` so that hdlike compares the theory to the CMB-HD bandpowers calculated with CLASS, which also come from hdMockData.

# Running a binned primordial P(k) with arbitrary binning

`arbitrary_Pkbinning.py` provides `BinnedPk`, a Cobaya `Theory` class that hands CAMB a primordial scalar power spectrum defined by an amplitude in each of a set of $k$ bins, rather than by a power law. The bin amplitudes are the cubic-spline values CAMB interpolates between, so with dense enough sampling any $\mathcal{P}(k)$ can be reproduced.

Configure it through these class options.

| Option | Meaning |
|---|---|
| `nbins` | The number of bins, and of free amplitudes `b1` to `b{nbins}`. |
| `k_min_bin`, `k_max_bin` | $\log_{10} k$ at the start and end of the log-spaced range, in Mpc<sup>-1</sup>. |
| `k_first` | A list of $k$ values (Mpc<sup>-1</sup>) to use as the leading bins with an alternate binning scheme. The remaining `nbins - len(k_first)` bins are log-spaced over the range above. Pass `None` or `[]` for an entirely evenly log-spaced binning, or all `nbins` values to specify the grid exactly. |
| `scale` | The units of the bin amplitudes (default `1e-9`). |
| `bin_par` | A template `ParamDict` (prior, ref, proposal) applied to every bin, unless you give the bins individual priors in the `params` block. |

The sampled parameter for bin $i$ is `b_i`, related to the primordial spectrum by

$$b_i = e^{-2\tau} \ \mathcal{P}_i(k) \ / \ \texttt{scale}$$

since `BinnedPk` multiplies the amplitudes by `scale` and by $e^{2\tau}$ before passing them to CAMB. With the default `scale`, $b_i = 10^9 \ e^{-2\tau} \mathcal{P}_i(k)$. Sampling in $e^{-2\tau}\mathcal{P}(k)$ rather than $\mathcal{P}(k)$ minimizes the degeneracy between the optical depth and the overall amplitude.

The files in `hdinitpk/cobaya_yaml_files/binned_pk` are working examples, one for each binned chain in the paper, using this class with its bins set to the paper's own seven- and 30-bin schemes. Point `python_path` at your clone of this repository so Cobaya can import the module.

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

# Obtaining the linear P(k) today from the primordial P(k)

`run_hdInitPk_to_LinearPk.py` converts constraints on the primordial spectrum into constraints on the linear matter power spectrum today, $P_\mathrm{lin}(k, z=0)$, which is plotted in Figure 6. It computes

$$P_\mathrm{lin}(k, z=0) = 2 \pi^2 \ k \ T(k)^2 \ \mathcal{P}(k)$$

with one CAMB call per cosmology returning the $z = 0$ matter transfer function $T(k)$, and $\mathcal{P}(k)$ the sampled primordial power in each bin. The script runs in two modes, either or both of which can be enabled.

- Fisher mode draws Gaussian samples from the CMB-HD and SO-like binned-$\mathcal{P}(k)$ Fisher forecasts and computes $P_\mathrm{lin}$ for each. It needs the binned-$\mathcal{P}(k)$ Fisher derivative directories, which `HD_FISHER_DERIV_DIR` and `SO_FISHER_DERIV_DIR` at the top of the file point at. They default to the `hd_binned_pk_feedback` and `so_binned_pk` directories that `run_hdInitPk_forecasts.py` writes, so running that first leaves nothing to fill in.
- Chain mode computes $P_\mathrm{lin}$ for every post-burn-in sample of a binned-$\mathcal{P}(k)$ MCMC chain.

Both modes use the same transfer-function method, but each keeps the CAMB accuracy settings of the calculation it came from.

Chain mode takes a `CHAIN_SOURCE` at the top of the file. `'packaged'` (the default) reads the thinned chain distributed with the package, so the script runs out of the box. `'custom'` reads a chain you have run yourself with `run_hdInitPk_current.ipynb`, from `hdinitpk/data/user_generated_data/chains`. `'raw'` reads your own full Cobaya output from `RAW_CHAIN_ROOT`. The results in the paper were obtained from the full chains, so `'raw'` or `'custom'` is what reproduces them. The burn-in follows the source automatically. The packaged chain had its burn-in removed before thinning, so nothing more is removed there, and the first half of each chain is dropped for the other two.

The run is parallelized with `hdfisher.mpi`.

```
mpirun -n 32 python run_hdInitPk_to_LinearPk.py
```

Fisher mode splits each experiment's samples across all ranks. Chain mode distributes whole chains across ranks, so request one task per chain for that part. The per-bin summaries it writes are what the `fig6_points_*.txt` files read by the notebook were made from.

# Running new Fisher forecasts

`run_hdInitPk_forecasts.ipynb` already does all of this for the nine configurations in the paper, and `run_hdInitPk_forecasts.py` does the same as a batch job. This section is for forecasting something the paper does not cover.

`hdinitpk.hdinitPkfisher.Fisher` extends [hdfisher](https://github.com/CMB-HD/hdfisher)'s `Fisher` with a binned primordial power spectrum and a kSZ template, and takes the same arguments plus `binned_pk`, `bin_edges`, `ksz`, and `pk_frac_step`. Code written with `hdfisher` only needs to change its import. The theory is calculated to the maximum multipole of the CMB-HD mock data in hdMockData (24,000 for v1.2) for every experiment, and the covariance matrices, the lensing reconstruction noise, the binning, and the BBN table that CLASS reads all come from hdMockData. The fiducial-parameter files in `hdinitpk/data/fisher_fid_params` hold the cosmology and the parameters that are varied; their accuracy settings are the ones in hdMockData.

`example_calc_binnedPk_forecasts.py` is a template for calculating the Fisher derivatives, and shows how to assemble a Fisher matrix from them afterwards.

```python
from hdinitpk import hdinitPkfisher

fisherlib = hdinitPkfisher.Fisher(
    fisher_dir, overwrite=True, binned_pk=True, bin_edges=bin_edges,
    pk_frac_step=0.05, fisher_steps_file=..., param_file=...)
fisherlib.calculate_fisher_derivs()
```

When `binned_pk=True`, the parameter file must give the number of bins (`nkbins`), the bin centers (`k1`, `k2`, and so on), the bin amplitudes (`Pk1`, `Pk2`, ... or `eneg2tauPk1`, ...), and an `effective_ns_for_nonlinear` entry. The step-size file must give a step for each bin amplitude, and `bin_edges` is required. Fiducial-parameter and step-size files for the paper's binnings are in `hdinitpk/data/fisher_fid_params` and `hdinitpk/data/fisher_steps`.

Because the derivatives are computed by perturbing the spectra inside one $k$ bin at a time, `pk_frac_step` must match the step size given for the `Pk` parameters in the step-size file. The spectra are perturbed by `pk_frac_step`, while the finite-difference denominator comes from that file. The example pairs them correctly. Rerunning it with the 1% and 10% step-size files, and the matching `pk_frac_step`, is how the derivatives were checked for convergence.

The derivatives are large. The example writes them to `hdinitpk/data/user_generated_data/fisher_derivs`.

The binned-$\mathcal{P}(k)$ calculation uses CAMB (via `hdPk`) and cannot be combined with `use_class=True`. Any additional parameter accepted by CAMB's [`set_params`](https://camb.readthedocs.io/en/latest/camb.html#camb.set_params) may be varied.
