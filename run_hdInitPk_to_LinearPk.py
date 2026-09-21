r"""Compute P_lin(k, z=0) samples for the Figure 6 points, from both sources:

1. FISHER MODE: draw `N_SAMPLES` Gaussian samples from the CMB-HD and
   SO-like binned-P(k) Fisher forecasts and compute Plin for each sample.

2. CHAIN MODE: compute Plin for every post-burn-in sample of the CMB-PAS
   7-bin MCMC chain, deduplicating repeated cosmologies so CAMB is called
   once per unique (ombh2, omch2, H0, tau). This mode uses ALL chain
   samples; `N_SAMPLES` does not apply here. It reads the THINNED chain
   distributed with the package by default; set `CHAIN_SOURCE = 'raw'`
   below to use your own full, un-thinned chains, which is what the paper
   used.

Both modes use the SAME transfer-function method: one CAMB call per
cosmology returns the z = 0 matter transfer T(k), and

    P_lin(k, z=0) [Mpc^3] = 2 pi^2 k T(k)^2 \mathcal{P}(k)   (Eq. 11, G(0)=1)

with \mathcal{P}(k) the sampled primordial power in each bin: the Pk_i Fisher
parameters directly, or e^{+2 tau} * eneg2tauPk_i (each sample's own tau)
for the chains. The primordial enters only as that final per-bin factor, so
the chain-mode cosmology dedup applies to T^2 while the primordial
rescaling stays per-row.

The CAMB ACCURACY SETTINGS are NOT shared: each mode keeps exactly the
settings of the script it came from (`make_camb_params_fisher` = Listing 1;
`make_camb_params_chain` = ACT DR6 settings). NOTE that chain mode's kmax=10
covers the 7-bin scheme (k <= 0.41 Mpc^-1) but not the 11-bin scheme (k up
to 27.4 Mpc^-1).

Parallelization (hdfisher.mpi): Fisher mode splits the samples of each
experiment across all ranks; chain mode distributes whole chains across
ranks (request one task per chain for that part).
"""
import os
import re
import glob

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import camb
from getdist.gaussian_mixtures import GaussianND
from hdfisher import utils, mpi
from hdfisher import theory as hdtheory
from hd_mock_data import hd_data

import hdinitpk
from hdinitpk import hdinitPkfisher


# The directory this script lives in, so that the paths below are relative
# to the script rather than to wherever it is run from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def script_relative(*parts):
    """A path below this script's own directory."""
    return os.path.join(SCRIPT_DIR, *parts)


# ======================================================================
# The Fisher derivative directories.
# ======================================================================
# Fisher mode needs the binned-P(k) Fisher DERIVATIVES for CMB-HD and for
# SO-like. Those are far too large to distribute, so they are not included
# with this package. The defaults below are where run_hdInitPk_forecasts.py
# and run_hdInitPk_forecasts.ipynb write them, from the `binned_pk_hd_feedback`
# and `binned_pk_so` configurations. example_calc_binnedPk_forecasts.py writes
# the SO one too, and shows how the calculation works:
#
#     fisherlib = hdinitpk.hdinitPkfisher.Fisher(<output dir>, overwrite=True,
#                                        binned_pk=True, bin_edges=...,
#                                        pk_frac_step=..., ...)
#     fisherlib.calculate_fisher_derivs()
#
# Each directory below is the `fisher_dir` that was passed to that `Fisher`
# constructor. `build_hd_fisher` and `build_so_fisher` further down re-open
# them with `overwrite=False`, so the saved derivatives are reused rather
# than recomputed, and assemble the Fisher matrices with `get_fisher`. The
# constructor arguments there (`binned_pk`, `bin_edges`, `ksz`,
# `hd_data_version`, and the step-size and parameter files) MUST match the
# ones used when the derivatives were calculated.
#
# These resolve inside the repository, which is what you want when running
# from a clone. If hdinitpk is installed somewhere else, the derivatives are
# under hdinitpk.user_data_path('fisher_derivs') instead, so use that here.
DERIV_DIR = script_relative('hdinitpk', 'data', 'user_generated_data',
                            'fisher_derivs')

HD_FISHER_DERIV_DIR = os.path.join(DERIV_DIR, 'hd_binned_pk_feedback')
SO_FISHER_DERIV_DIR = os.path.join(DERIV_DIR, 'so_binned_pk')


# ----------------------------------------------------------------------
# Paths and run configuration.
# ----------------------------------------------------------------------
# Everything else the script reads ships with the `hdinitpk` package and is
# located with `hdinitpk.data_path`.
fisher_steps_path = hdinitpk.data_path('fisher_steps')
fisher_params_path = hdinitpk.data_path('fisher_fid_params')
binning_path = hdinitpk.data_path('binning')

# Output directories, all under `hdinitpk/data/user_generated_data`. Set
# OVERWRITE = False to skip any experiment or chain whose combined samples
# file already exists.
#
# The two `plin_z0` directories hold the per-rank shards and the combined
# sample arrays, which are bulky. `out_dir_fig6` holds the small per-bin
# summaries that the plotting notebook reads, in the same five-column
# format and under the same names as the `fig6_points_*.txt` files
# distributed with the package, so that setting `fig6_source` there to
# `'custom_both'` (or to `'custom_fisher'` or `'custom_chain'`, for the
# points from one mode only) picks them up with no other change.
out_dir_fisher = hdinitpk.user_data_path('plin_z0', 'from_fisher')
out_dir_chains = hdinitpk.user_data_path('plin_z0', 'from_chains')
out_dir_fig6 = hdinitpk.user_data_path('fig6')

RUN_FISHER_MODE = True
RUN_CHAIN_MODE = True

hd_data_version = 'v1.2'   # the CMB-HD mock data version, from hdMockData
use_H0 = True

# Fisher mode only: number of Gaussian samples per experiment.
N_SAMPLES = 10000
SEED = 20260728
OVERWRITE = True

# ----- chain mode -----
# Which binned-P(k) chain to read, and where it comes from. The three
# sources are the same ones hdInitPk_plots.ipynb offers, plus the raw
# cobaya output that the notebook has no use for:
#
#   'packaged'  the thinned chain distributed with hdinitpk. Runs out of
#               the box, and is the default.
#   'custom'    a chain you ran yourself from one of the input files in
#               `hdinitpk/cobaya_yaml_files/binned_pk`, following
#               run_hdInitPk_current.ipynb. Full cobaya output, read from
#               `hdinitpk/data/user_generated_data/chains`.
#   'raw'       your own FULL, un-thinned cobaya output. This lets you specify
#               your own path to chains you have written anywhere (RAW_CHAIN_ROOT)
#
# `find_chain_files` handles either layout: a getdist root (`<root>.txt`
# with `<root>.paramnames` beside it), or a set of raw cobaya files
# (`<root>.1.txt` ... `<root>.N.txt`).
CHAIN_SOURCE = 'packaged'

# The chain to read, for the two thinned sources. The six binned-P(k)
# chains are named `{pas,pas_desi,pact_lb}_{7,30}bin`; the 30-bin ones need
# a larger `kmax`, see the module docstring.
CHAIN_NAME = 'pas_7bin'

# CHAIN_SOURCE = 'custom': the directory holding your own chains. The runs
# prepared by run_hdInitPk_current.ipynb write there under the name of
# their input file, which is what CUSTOM_CHAIN_NAMES maps to.
CUSTOM_DATA_PATH = hdinitpk.user_data_path()
CUSTOM_CHAIN_NAMES = {
    'pas_7bin':       'cmb_pas_7_bins_arbitrary_binning',
    'pas_desi_7bin':  'cmb_pas_desi_7_bins_arbitrary_binning',
    'pact_lb_7bin':   'p_act_lb_7_bins_no_sroll_arbitrary_binning',
    'pas_30bin':      'cmb_pas_30_bins_arbitrary_binning',
    'pas_desi_30bin': 'cmb_pas_desi_30_bins_arbitrary_binning',
    'pact_lb_30bin':  'p_act_lb_30_bins_arbitrary_binning',
}

# CHAIN_SOURCE = 'raw': the cobaya `output:` root of your own run.
# Unlike 'custom', this takes the root itself rather than a name to look
# up, so it reaches a chain written anywhere. An absolute path works. The
# default is relative to this script, and points where 'custom' would for
# the 7-bin CMB-PAS run.
RAW_CHAIN_ROOT = script_relative(
    'hdinitpk', 'data', 'user_generated_data', 'chains',
    'cmb_pas_7_bins_arbitrary_binning')

CHAIN_ROOTS = {
    'packaged': hdinitpk.data_path('chains', 'binned_pk', CHAIN_NAME),
    'custom': os.path.join(CUSTOM_DATA_PATH, 'chains',
                           CUSTOM_CHAIN_NAMES[CHAIN_NAME]),
    'raw': RAW_CHAIN_ROOT,
}
chain_root = CHAIN_ROOTS[CHAIN_SOURCE]

# Burn-in fraction discarded from the start of each chain. The packaged
# chain had its burn-in removed before it was thinned, so nothing more is
# removed there. The other two sources are full cobaya output and still
# carry it, so the first half of each chain is dropped, as in Section III
# of the paper.
BURN_IN_FRACTION = 0.0 if (CHAIN_SOURCE == 'packaged') else 0.5


# ----------------------------------------------------------------------
# k bins, from the binning file (Table I).
# ----------------------------------------------------------------------
# The 11-bin forecast scheme: log-spaced edges from 0.00367 to 38.3 Mpc^-1.
binning = utils.load_from_file(
    os.path.join(binning_path, 'hd_pk_wavenumbers_11bins.txt'),
    ['k bin center [Mpc^-1]', 'lower bin edge [Mpc^-1]',
     'upper bin edge [Mpc^-1]'])
K_CENTERS_11 = np.asarray(binning['k bin center [Mpc^-1]'], dtype=float)
BIN_EDGES_11 = np.append(binning['lower bin edge [Mpc^-1]'],
                         binning['upper bin edge [Mpc^-1]'][-1])

# Seven-bin scheme: wide first bin from 0.0000562, then the first six of the
# log-spaced bins above (up to 0.571 Mpc^-1).
BIN_EDGES_7 = np.concatenate(([5.623413251903490700e-05], BIN_EDGES_11[:7]))
K_CENTERS_7 = 0.5 * (BIN_EDGES_7[:-1] + BIN_EDGES_7[1:])

SO_MAX_BIN = 7

# Map number of Pk columns found in a chain -> (k centers, scheme label).
_BINNING_BY_NBINS = {
    7: (K_CENTERS_7, 'seven-bin current-data scheme (Sec. III)'),
    11: (K_CENTERS_11, 'eleven-bin forecast scheme (Sec. IV A)'),
}


# ----------------------------------------------------------------------
# Fiducial parameters, from the same file the Fisher derivatives used.
# ----------------------------------------------------------------------
# This gives the cosmological parameters, the kSZ and feedback parameters,
# and the fiducial bin amplitudes `Pk1` ... `Pk11` of Eq. 2 with alpha_s = 0
# (Table II).
FID_PARAM_FILE = os.path.join(fisher_params_path,
                              'hd_binned_pk_fiducial_params_feedback.yaml')
FID_PARAMS = hdtheory.get_params(FID_PARAM_FILE)

_AS_FID = np.exp(FID_PARAMS['logA']) * 1e-10
_NS_FID = FID_PARAMS['ns']
_K_PIVOT = FID_PARAMS.get('pivot_scalar', 0.05)  # Mpc^-1


def hd_Pk(k):
    """The fiducial power-law primordial spectrum."""
    return _AS_FID * (np.asarray(k, dtype=float) / _K_PIVOT) ** (_NS_FID - 1.0)


# ----------------------------------------------------------------------
# CAMB parameters: one builder per mode, each with that mode's
# accuracy settings.
# ----------------------------------------------------------------------
# The theory is calculated to the lmax of the CMB-HD mock data (24,000).
LMAX = hd_data.HDMockData(version=hd_data_version).theo_lmax

# Fisher mode (Listing 1):
LISTING1_ACCURACY = {
    'AccuracyBoost': 1.1,
    'lSampleBoost': 3.0,
    'lAccuracyBoost': 3.0,
    'DoLateRadTruncation': False,
    'min_l_logl_sampling': 10000,
}

# Chain mode :
CHAIN_CAMB_ACCURACY_SETTINGS = {
    'kmax': 10,
    'k_per_logint': 130,
    'nonlinear': True,
    'lens_potential_accuracy': 8,
    'lens_margin': 2050,
    'lAccuracyBoost': 1.2,
    'min_l_logl_sampling': 6000,
    'DoLateRadTruncation': False,
}


def make_camb_params_fisher(ombh2, omch2, H0, tau):
    """CAMBparams for one Fisher-mode sample, following Listing 1.

    The InitPower values are irrelevant: only the transfer functions are
    read, and they are normalized to unit primordial curvature.
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau,
                       num_massive_neutrinos=3, mnu=0.06, nnu=3.044,
                       bbn_predictor='PRIMAT_Yp_DH_ErrorMC_2021.dat')
    pars.set_classes(recombination_model='Recfast')
    pars.InitPower.set_params(As=_AS_FID, ns=_NS_FID, pivot_scalar=_K_PIVOT)
    pars.set_matter_power(redshifts=[0.0], kmax=100, k_per_logint=130)
    pars.set_for_lmax(LMAX + 500, lens_potential_accuracy=30, lens_margin=2050)
    pars.set_accuracy(**LISTING1_ACCURACY)
    pars.NonLinear = camb.model.NonLinear_both
    pars.NonLinearModel.set_params('mead2016')
    pars.WantCls = False
    pars.WantTransfer = True
    return pars


def make_camb_params_chain(ombh2, omch2, H0, tau):
    """CAMBparams for one chain-mode cosmology"""
    pars = camb.set_params(
        H0=H0, ombh2=ombh2, omch2=omch2, tau=tau,
        num_massive_neutrinos=3, mnu=0.06, nnu=3.044,
        bbn_predictor='PRIMAT_Yp_DH_ErrorMC_2021.dat',
        As=_AS_FID, ns=_NS_FID, pivot_scalar=_K_PIVOT,
        redshifts=[0.0], WantCls=False, WantTransfer=True,
        lmax=LMAX,   # lens_potential_accuracy/lens_margin act through lmax
        **CHAIN_CAMB_ACCURACY_SETTINGS,
    )
    return pars


# ----------------------------------------------------------------------
# Transfer functions (shared by both modes).
# ----------------------------------------------------------------------
# 'delta_tot' = CDM + baryons + massive neutrinos. Row 0 of
# MatterTransferData.transfer_data is k/h, so the row index is the position
# of the name in transfer_names.
TRANSFER_VAR = 'delta_tot'
_TRANSFER_ROW = list(camb.model.transfer_names).index(TRANSFER_VAR)


def transfer_k_and_Tsq(results, H0):
    r"""(k [Mpc^-1], T(k)^2 [Mpc^4]) at z = 0 on CAMB's own q grid.

    T is the density contrast per unit primordial curvature divided by k^2,
    with k in Mpc^-1, so that P_lin(k) = 2 pi^2 k T(k)^2 \mathcal{P}(k).

    With nonlinear lensing active, CAMB adds its own redshifts to the
    requested z = 0, so the z = 0 slice is looked up rather than assumed.
    """
    td = results.get_matter_transfer_data().transfer_data
    tr = results.Params.Transfer
    zs = np.asarray(tr.PK_redshifts, dtype=float)[:int(tr.PK_num_redshifts)]
    iz = int(np.argmin(np.abs(zs)))
    h = H0 / 100.0
    k = np.asarray(td[0, :, iz], dtype=float) * h   # k/h [h/Mpc] -> Mpc^-1
    T = np.asarray(td[_TRANSFER_ROW, :, iz], dtype=float)
    order = np.argsort(k)
    return k[order], (T[order]) ** 2


def interp_Tsq(k_grid, Tsq_grid, k_out):
    """Cubic spline of log T^2 in log k, evaluated at k_out."""
    spline = CubicSpline(np.log(k_grid), np.log(Tsq_grid))
    return np.exp(spline(np.log(np.atleast_1d(np.asarray(k_out, dtype=float)))))


def camb_Tsq(ombh2, omch2, H0, tau, k_centers, params_builder=None):
    """T(k)^2 [Mpc^4] at k_centers for one cosmology. `params_builder`
    selects the mode's settings (default: `make_camb_params_fisher`). tau
    has no effect on the z = 0 transfer; it is passed through for interface
    consistency."""
    if params_builder is None:
        params_builder = make_camb_params_fisher
    results = camb.get_transfer_functions(params_builder(ombh2, omch2, H0, tau))
    k_grid, Tsq_grid = transfer_k_and_Tsq(results, H0)
    return interp_Tsq(k_grid, Tsq_grid, k_centers)


def plin_from_Tsq(k, Tsq, primordial):
    r"""P_lin(k, z=0) [Mpc^3] = 2 pi^2 k T(k)^2 \mathcal{P}(k). `primordial`
    is the dimensionless curvature power in each bin. Broadcasts over
    leading sample axes."""
    return 2.0 * np.pi ** 2 * np.asarray(k) * np.asarray(Tsq) * np.asarray(primordial)


# ----------------------------------------------------------------------
# Fisher matrices (Fisher mode).
# ----------------------------------------------------------------------
def build_hd_fisher():
    params = ([f'Pk{i+1}' for i in range(11)]
              + ['H0', 'tau', 'ombh2', 'omch2',
                 'HMCode_logT_AGN', 'n_ksz', 'A_ksz'])
    # `overwrite=False`: re-open the directory of derivatives produced by
    # `calculate_fisher_derivs` and reuse them, rather than recomputing.
    fisherlib = hdinitPkfisher.Fisher(
        HD_FISHER_DERIV_DIR,
        overwrite=False, use_H0=use_H0,
        hd_data_version=hd_data_version, pol_only_lensing=True,
        binned_pk=True, ksz=True, bin_edges=BIN_EDGES_11, pk_frac_step=0.05,
        fisher_steps_file=os.path.join(
            fisher_steps_path, 'binned_pk_steps_5_percent_feedback.yaml'),
        param_file=FID_PARAM_FILE)
    return fisherlib.get_fisher(
        'lensed', priors={'tau': 0.005, 'HMCode_logT_AGN': 0.0006 * 7.8},
        use_H0=use_H0, with_desi=True, params=params)


def build_so_fisher():
    params = ([f'Pk{i+1}' for i in range(SO_MAX_BIN)]
              + ['H0', 'tau', 'ombh2', 'omch2'])
    fisherlib = hdinitPkfisher.Fisher(
        SO_FISHER_DERIV_DIR,
        overwrite=False, exp='so', use_H0=use_H0,
        hd_data_version=hd_data_version,
        binned_pk=True, bin_edges=BIN_EDGES_11, pk_frac_step=0.05,
        fisher_steps_file=os.path.join(
            fisher_steps_path, 'binned_pk_steps_5_percent.yaml'),
        param_file=os.path.join(
            fisher_params_path, 'hd_binned_pk_fiducial_params.yaml'))
    return fisherlib.get_fisher(
        'lensed', priors={'tau': 0.005}, use_H0=use_H0, with_desi=False,
        params=params)


EXPERIMENTS = {
    'hd': {'tag': 'fisher_hd_feedback_pol_11bins',
           'build': build_hd_fisher,
           'n_bins': 11},
    'so': {'tag': 'fisher_so_7bins',
           'build': build_so_fisher,
           'n_bins': SO_MAX_BIN},
}


# ----------------------------------------------------------------------
# Fisher mode: Gaussian sampling.
# ----------------------------------------------------------------------
def draw_samples(fisher_matrix, fisher_params, n_samples, seed):
    """Draw from the multivariate Gaussian defined by the Fisher matrix
    (getdist GaussianND centered on the fiducials, inverse Fisher as the
    covariance). Samples CAMB cannot be called on (nonpositive density,
    negative tau, nonpositive H0) are dropped."""
    params = list(fisher_params)
    mean = np.array([FID_PARAMS[p] for p in params])
    cov = np.linalg.inv(fisher_matrix)

    gauss = GaussianND(mean, cov, names=params)
    gauss_samples = gauss.MCSamples(size=n_samples, names=params,
                                    random_state=seed)
    p = gauss_samples.getParams()
    samples = np.column_stack([getattr(p, name) for name in params])

    idx = {name: i for i, name in enumerate(params)}
    ok = ((samples[:, idx['ombh2']] > 0) & (samples[:, idx['omch2']] > 0)
          & (samples[:, idx['tau']] > 0) & (samples[:, idx['H0']] > 0))
    n_bad = int((~ok).sum())
    if n_bad and mpi.rank == 0:
        print(f'[rank 0] dropped {n_bad}/{len(samples)} samples that CAMB '
              f'cannot be called on', flush=True)
    return samples[ok], params


def compute_plin_for_experiment(exp_key):
    """Draw from one experiment's Fisher forecast, run CAMB on this rank's
    share of the samples, and write the shard."""
    cfg = EXPERIMENTS[exp_key]
    tag, n_bins = cfg['tag'], cfg['n_bins']
    k_centers = K_CENTERS_11[:n_bins]

    final_file = os.path.join(out_dir_fisher, f'plin_z0_samples_{tag}.npy')
    if os.path.exists(final_file) and not OVERWRITE:
        if mpi.rank == 0:
            print(f'[rank 0] {tag}: {final_file} exists and OVERWRITE is '
                  f'False, skipping', flush=True)
        return

    # every rank builds the Fisher: the constructor is collective
    fisher_matrix, fisher_params = cfg['build']()

    if mpi.rank == 0:
        samples, params = draw_samples(fisher_matrix, fisher_params,
                                       N_SAMPLES, SEED)
    else:
        samples, params = None, None
    samples = mpi.comm.bcast(samples, root=0)
    params = mpi.comm.bcast(params, root=0)

    idx = {p: i for i, p in enumerate(params)}
    ombh2, omch2 = samples[:, idx['ombh2']], samples[:, idx['omch2']]
    H0, tau = samples[:, idx['H0']], samples[:, idx['tau']]
    pk_samp = np.column_stack([samples[:, idx[f'Pk{i+1}']]
                               for i in range(n_bins)])

    my_idxs = mpi.distribute(len(samples), mpi.size, mpi.rank)
    print(f'[rank {mpi.rank}] {tag}: {len(my_idxs)} of {len(samples)} '
          f'samples, {n_bins} bins', flush=True)

    plin = np.empty((len(my_idxs), n_bins))
    Tsq_all = np.empty((len(my_idxs), n_bins))
    for j, i in enumerate(my_idxs):
        Tsq_all[j] = camb_Tsq(ombh2[i], omch2[i], H0[i], tau[i], k_centers,
                              params_builder=make_camb_params_fisher)
        # this sample's primordial in each bin is just its Pk_i value
        plin[j] = plin_from_Tsq(k_centers, Tsq_all[j], pk_samp[i])
        if (j + 1) % 100 == 0:
            print(f'[rank {mpi.rank}] {tag}: {j + 1}/{len(my_idxs)} CAMB '
                  f'calls', flush=True)

    os.makedirs(out_dir_fisher, exist_ok=True)
    shard = f'{tag}_part{mpi.rank + 1}'
    np.save(os.path.join(out_dir_fisher, f'plin_z0_samples_{shard}.npy'), plin)
    np.save(os.path.join(out_dir_fisher, f'Tsq_{shard}.npy'), Tsq_all)
    np.save(os.path.join(out_dir_fisher, f'k_centers_{shard}.npy'), k_centers)
    print(f'[rank {mpi.rank}] {tag}: wrote shard {shard}', flush=True)

    mpi.comm.barrier()
    if mpi.rank == 0:
        combine_shards(tag, mpi.size, k_centers, fisher_matrix, fisher_params,
                       n_bins)


def combine_shards(tag, n_shards, k_centers, fisher_matrix, fisher_params,
                   n_bins):
    """Concatenate the per-rank shards and write the combined samples and
    the per-bin error summary."""
    parts, tparts = [], []
    for r in range(n_shards):
        f = os.path.join(out_dir_fisher, f'plin_z0_samples_{tag}_part{r + 1}.npy')
        if os.path.exists(f):
            parts.append(np.load(f))
        ft = os.path.join(out_dir_fisher, f'Tsq_{tag}_part{r + 1}.npy')
        if os.path.exists(ft):
            tparts.append(np.load(ft))
    plin = np.concatenate(parts, axis=0)

    np.save(os.path.join(out_dir_fisher, f'plin_z0_samples_{tag}.npy'), plin)
    np.save(os.path.join(out_dir_fisher, f'k_centers_{tag}.npy'), k_centers)
    if tparts:
        np.save(os.path.join(out_dir_fisher, f'Tsq_{tag}.npy'),
                np.concatenate(tparts, axis=0))

    mean = plin.mean(axis=0)
    std = plin.std(axis=0, ddof=1)
    lo = np.percentile(plin, 15.865, axis=0)
    hi = np.percentile(plin, 84.135, axis=0)

    cov = np.linalg.inv(fisher_matrix)
    pk_frac = np.array([
        np.sqrt(cov[fisher_params.index(f'Pk{i+1}'),
                    fisher_params.index(f'Pk{i+1}')])
        / FID_PARAMS[f'Pk{i+1}'] for i in range(n_bins)])

    np.savetxt(
        os.path.join(out_dir_fisher, f'plin_z0_errors_{tag}.txt'),
        np.column_stack([k_centers, mean, std, std / mean,
                         mean - lo, hi - mean, pk_frac]),
        header=('k bin center [Mpc^-1]    Plin_mean [Mpc^3]    '
                'Plin_1sigma [Mpc^3]    fractional_error    '
                'lower_68_err [Mpc^3]    upper_68_err [Mpc^3]    '
                'Pk_bin_fractional_error'),
        fmt=['%.6e'] * 7,
    )
    print(f'[rank 0] {tag}: combined {plin.shape[0]} samples, {n_bins} bins; '
          f'fractional Plin error per bin: '
          f'{np.array2string(std / mean, precision=4)}', flush=True)


# ----------------------------------------------------------------------
# Chain mode.
# ----------------------------------------------------------------------
# Bin-amplitude columns. A chain run with `binnedPk_theory` names them
# `eneg2tauPk<n>` and stores them in absolute units; one run with
# `arbitrary_Pkbinning.BinnedPk` names them `b<n>` and stores them in units
# of that class's `scale`, so those are multiplied back up on read.
_PK_COL_RES = {'eneg2tauPk': re.compile(r'^eneg2tauPk(\d+)$'),
               'b': re.compile(r'^b(\d+)$')}
ARBITRARY_BINNING_SCALE = 1e-9


def find_chain_files(root):
    """Return the list of sample files belonging to `root`. Handles both
    chain layouts:

      * a getdist root, `<root>.txt` with its column names in
        `<root>.paramnames` (what the thinned chains distributed with the
        package look like), a single merged file;
      * numbered parts, `<root>.1.txt` ... `<root>.N.txt`. These are
        either raw cobaya output, each carrying its own '#'-prefixed
        header line, or one getdist chain split into pieces small enough
        to distribute, which share a single `<root>.paramnames`.
        `read_chain` tells them apart by that header line.
    """
    numbered = sorted(
        (p for p in glob.glob(f'{root}.*.txt')
         if os.path.basename(p)[len(os.path.basename(root)) + 1:-4].isdigit()),
        key=lambda p: int(os.path.basename(p).rsplit('.', 2)[-2]))
    if numbered:
        return numbered
    if os.path.exists(f'{root}.txt'):
        return [f'{root}.txt']
    return []


def read_chain(path):
    """Read one chain file into a DataFrame with named columns.

    A raw cobaya file carries its column names on a '#'-prefixed first
    line. A getdist `<root>.txt` does not: its names live in the sibling
    `<root>.paramnames`, whose first two columns of the sample file are
    always `weight` and `minuslogpost`.
    """
    with open(path) as f:
        first = f.readline()
    if first.lstrip().startswith('#'):
        colnames = first.lstrip('#').split()
        return pd.read_csv(path, sep=r'\s+', skiprows=1, names=colnames)

    # getdist format: names from <root>.paramnames. A chain saved in parts
    # is `<root>.1.txt`, `<root>.2.txt`, ... sharing one
    # `<root>.paramnames`, so strip a trailing numeric segment before
    # looking for it.
    stem = path[:-4]
    head, _, last = stem.rpartition('.')
    if head and last.isdigit():
        stem = head
    with open(stem + '.paramnames') as f:
        names = [line.split()[0].rstrip('*') for line in f
                 if line.strip() and not line.startswith('#')]
    colnames = ['weight', 'minuslogpost'] + names
    return pd.read_csv(path, sep=r'\s+', comment='#', names=colnames)


def get_pk_columns(df):
    """Return the bin-amplitude column names, sorted by bin number, and the
    factor that converts them to absolute e^{-2 tau} P(k).

    Accepts either naming: `eneg2tauPk<n>` (absolute, factor 1) or `b<n>`
    (in units of `arbitrary_Pkbinning.BinnedPk`'s `scale`, factor `scale`).
    """
    for prefix, regex in _PK_COL_RES.items():
        found = sorted((int(m.group(1)), c)
                       for c in df.columns
                       for m in [regex.match(c)] if m)
        if found:
            unit = 1.0 if prefix == 'eneg2tauPk' else ARBITRARY_BINNING_SCALE
            return [c for _, c in found], unit
    raise ValueError(f'No bin-amplitude columns found (looked for eneg2tauPk<n> '
                     f'and b<n>); got {list(df.columns)}')


def compute_plin_for_chain(chain_path, tag, progress_every=1000):
    """Compute Plin(k, z=0) for every post-burn-in sample in one chain: one
    CAMB call per unique cosmology gives T^2(k), and each row's Plin is
    Eq. 11 with that row's own sampled primordial."""
    final_file = os.path.join(out_dir_chains, f'plin_z0_samples_{tag}.npy')
    if os.path.exists(final_file) and not OVERWRITE:
        print(f'[rank {mpi.rank}] {tag}: {final_file} exists and OVERWRITE '
              f'is False, skipping', flush=True)
        return

    df = read_chain(chain_path)
    n_raw = len(df)
    if BURN_IN_FRACTION > 0.0:
        n_burn = int(np.floor(BURN_IN_FRACTION * n_raw))
        df = df.iloc[n_burn:].reset_index(drop=True)
        print(f'[rank {mpi.rank}] {tag}: dropped {n_burn}/{n_raw} rows as '
              f'burn-in ({BURN_IN_FRACTION:.0%}), {len(df)} remain', flush=True)

    pk_cols, pk_unit = get_pk_columns(df)
    n_bins = len(pk_cols)
    k_centers, scheme = _BINNING_BY_NBINS[n_bins]
    print(f'[rank {mpi.rank}] {tag}: {n_bins} Pk bins -> {scheme}', flush=True)

    weight = df['weight'].to_numpy(float)
    ombh2 = df['ombh2'].to_numpy(float)
    omch2 = df['omch2'].to_numpy(float)
    H0 = df['H0'].to_numpy(float)
    tau = df['tau'].to_numpy(float)
    # The chain columns are e^{-2 tau} P(k) with no factor of 1e9 (the 1e9
    # in Table I is display-only). Invert with each sample's OWN tau:
    pk_samp = (df[pk_cols].to_numpy(float) * pk_unit
               / np.exp(-2.0 * tau)[:, None])

    # One CAMB call per unique cosmology.
    cosmo = np.column_stack([ombh2, omch2, H0, tau])
    _, first_idx, inverse = np.unique(cosmo, axis=0, return_index=True,
                                      return_inverse=True)
    n_samples, n_unique = len(df), len(first_idx)
    print(f'[rank {mpi.rank}] {tag}: {n_samples} rows, {n_unique} unique '
          f'cosmologies ({n_samples / max(n_unique, 1):.1f}x fewer CAMB calls)',
          flush=True)

    Tsq_unique = np.empty((n_unique, n_bins))
    for j, row in enumerate(first_idx):
        Tsq_unique[j] = camb_Tsq(ombh2[row], omch2[row], H0[row], tau[row],
                                 k_centers,
                                 params_builder=make_camb_params_chain)
        if progress_every and (j + 1) % progress_every == 0:
            print(f'[rank {mpi.rank}] {tag}: {j + 1}/{n_unique} CAMB calls',
                  flush=True)

    # Broadcast T^2 back to one row per sample, then apply that row's own
    # sampled primordial (the rescaling is per-row, outside the dedup):
    plin = plin_from_Tsq(k_centers, Tsq_unique[inverse], pk_samp)

    os.makedirs(out_dir_chains, exist_ok=True)
    np.save(os.path.join(out_dir_chains, f'plin_z0_samples_{tag}.npy'), plin)
    np.save(os.path.join(out_dir_chains, f'weights_{tag}.npy'), weight)
    np.save(os.path.join(out_dir_chains, f'k_centers_{tag}.npy'), k_centers)
    np.save(os.path.join(out_dir_chains, f'Tsq_{tag}.npy'), Tsq_unique[inverse])

    wmean = np.average(plin, axis=0, weights=weight)
    wstd = np.sqrt(np.average((plin - wmean) ** 2, axis=0, weights=weight))
    np.savetxt(
        os.path.join(out_dir_chains, f'plin_z0_errors_{tag}.txt'),
        np.column_stack([k_centers, wmean, wstd, wstd / wmean]),
        header=('k bin center [Mpc^-1]    Plin_weighted_mean [Mpc^3]    '
                'Plin_weighted_1sigma [Mpc^3]    fractional_error'),
        fmt=['%.6e'] * 4,
    )
    print(f'[rank {mpi.rank}] wrote {tag}: {n_samples} samples, {n_bins} bins',
          flush=True)


def run_chain_mode():
    """Distribute whole chains across ranks and process each."""
    chain_files = find_chain_files(chain_root)
    if not chain_files:
        raise FileNotFoundError(
            f'no chain files found for root {chain_root}. Expected either '
            f'{chain_root}.txt (a getdist root, with its .paramnames beside '
            f'it) or {chain_root}.1.txt ... (raw cobaya output).')

    run_label = os.path.basename(chain_root)
    if mpi.rank == 0:
        print(f'[rank 0] chain root {chain_root}: {len(chain_files)} file(s)',
              flush=True)

    chain_idxs = mpi.distribute(len(chain_files), mpi.size, mpi.rank)
    for ci in chain_idxs:
        path = chain_files[ci]
        tag = (run_label if len(chain_files) == 1
               else f'{run_label}_chain{ci + 1}')
        compute_plin_for_chain(path, tag=tag)


# ----------------------------------------------------------------------
# The Figure 6 points.
# ----------------------------------------------------------------------
# Both modes write a `plin_z0_errors_*.txt` of their own, in the format
# each calculation naturally produces. The plotting notebook wants one
# five-column file per experiment (k bin center, central value, symmetric
# 1 sigma, lower 68% error, upper 68% error), which is what these build.

def _weighted_quantile(values, weights, q):
    """The `q`-quantile of `values` under `weights`."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cumulative = np.cumsum(w) - 0.5 * w
    cumulative /= np.sum(w)
    return np.interp(q, cumulative, v)


def _write_fig6_points(name, k, pk, sigma, lower, upper):
    os.makedirs(out_dir_fig6, exist_ok=True)
    path = os.path.join(out_dir_fig6, f'fig6_points_{name}.txt')
    np.savetxt(
        path, np.column_stack([k, pk, sigma, lower, upper]),
        header=('columns: k bin center [Mpc^-1]   P_lin(k, z=0) [Mpc^3]   '
                'sigma   lower_err   upper_err\n'
                'raw values: no x-offsets and no snapping to the theory '
                'grid (the notebook applies those)'))
    print(f'[rank 0] wrote {path}', flush=True)


def write_fig6_points():
    """Write `fig6_points_<hd|so|pas>.txt` from whichever modes have run.

    The CMB-HD and SO-like points come from fisher mode, whose error file
    already holds every column needed. The CMB-PAS points come from chain
    mode, where the samples carry importance weights, so the mean and the
    68% interval are recomputed here from the saved samples rather than
    read from its error file, which only records a symmetric width.
    """
    if RUN_FISHER_MODE:
        for name in ('hd', 'so'):
            path = os.path.join(
                out_dir_fisher,
                f'plin_z0_errors_{EXPERIMENTS[name]["tag"]}.txt')
            k, mean, sigma, _frac, lower, upper, _pk_frac = np.loadtxt(
                path, unpack=True)
            _write_fig6_points(name, k, mean, sigma, lower, upper)

    if RUN_CHAIN_MODE:
        run_label = os.path.basename(chain_root)
        sample_files = sorted(glob.glob(os.path.join(
            out_dir_chains, f'plin_z0_samples_{run_label}*.npy')))
        samples, weights = [], []
        for sample_file in sample_files:
            tag = os.path.basename(sample_file)[len('plin_z0_samples_'):-4]
            samples.append(np.load(sample_file))
            weights.append(np.load(os.path.join(out_dir_chains, f'weights_{tag}.npy')))
            k_centers = np.load(os.path.join(out_dir_chains, f'k_centers_{tag}.npy'))
        samples = np.concatenate(samples, axis=0)
        weights = np.concatenate(weights, axis=0)

        mean, sigma, lower, upper = [], [], [], []
        for j in range(len(k_centers)):
            column = samples[:, j]
            m = np.average(column, weights=weights)
            p16 = _weighted_quantile(column, weights, 0.16)
            p84 = _weighted_quantile(column, weights, 0.84)
            mean.append(m)
            lower.append(m - p16)
            upper.append(p84 - m)
            sigma.append(0.5 * ((m - p16) + (p84 - m)))
        _write_fig6_points('pas', k_centers, mean, sigma, lower, upper)


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------
if __name__ == '__main__':
    if mpi.rank == 0:
        print(f'[rank 0] {mpi.size} ranks; fisher mode: {RUN_FISHER_MODE} '
              f'({N_SAMPLES} samples/experiment, seed {SEED}); chain mode: '
              f'{RUN_CHAIN_MODE} (all post-burn-in samples of '
              f'{os.path.basename(chain_root)})', flush=True)
        if RUN_FISHER_MODE:
            os.makedirs(out_dir_fisher, exist_ok=True)
        if RUN_CHAIN_MODE:
            os.makedirs(out_dir_chains, exist_ok=True)
    mpi.comm.barrier()

    if RUN_FISHER_MODE:
        for exp_key in ['hd', 'so']:
            compute_plin_for_experiment(exp_key)
            mpi.comm.barrier()

    if RUN_CHAIN_MODE:
        run_chain_mode()
        mpi.comm.barrier()

    if mpi.rank == 0:
        write_fig6_points()
        print('All requested modes complete.', flush=True)
