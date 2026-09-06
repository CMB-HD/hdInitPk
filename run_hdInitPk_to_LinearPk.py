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
cosmology returns the z = 0 matter transfer T(k) (density contrast per unit
primordial curvature over k^2), and

    P_lin(k, z=0) [Mpc^3] = 2 pi^2 k T(k)^2 \mathcal{P}(k)   (Eq. 11, G(0)=1)

with \mathcal{P}(k) the sampled primordial power in each bin: the Pk_i Fisher
parameters directly, or e^{+2 tau} * eneg2tauPk_i (each sample's own tau)
for the chains. The primordial enters only as that final per-bin factor, so
the chain-mode cosmology dedup applies to T^2 while the primordial
rescaling stays per-row.

The CAMB ACCURACY SETTINGS are NOT shared: each mode keeps exactly the
settings of the script it came from (`make_camb_params_fisher` = Listing 1;
`make_camb_params_chain` = the original chain script). Each builder gets
its own startup normalization check. NOTE that chain mode's kmax=10 covers
the 7-bin scheme (k <= 0.41 Mpc^-1) but not the 11-bin scheme (k up to
27.4 Mpc^-1); an 11-bin chain raises a clear error instead of silently
extrapolating.

Parallelization (hdfisher.mpi): Fisher mode splits the samples of each
experiment across all ranks; chain mode distributes whole chains across
ranks (request one task per chain for that part).
"""
import os
import re
import glob
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import camb
from getdist.gaussian_mixtures import GaussianND
from hdfisher import utils, mpi

import hdinitpk
from hdinitpk import hdinitPkfisher


# ======================================================================
# SET THESE: the Fisher derivative directories.
# ======================================================================
# Fisher mode needs the binned-P(k) Fisher DERIVATIVES for CMB-HD and for
# SO-like. Those are far too large to distribute, so they are not included
# with this package. You generate them yourself and point the two
# variables below at the result. `example_calc_binnedPk_forecasts.py` in the
# repository root shows how to produce them:
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
# FILL THESE IN with the paths to your own derivative directories.
HD_FISHER_DERIV_DIR = '/path/to/derivs/amanda_binned_pk_11_bins_5_percent_feedback'
SO_FISHER_DERIV_DIR = '/path/to/derivs/binned_pk_so_fisher_output'


# ----------------------------------------------------------------------
# Paths and run configuration.
# ----------------------------------------------------------------------
# Everything else the script reads ships with the `hdinitpk` package and is
# located with `hdinitpk.data_path`; override the directory they are read
# from with the `HDINITPK_DATA` environment variable.
data_path = hdinitpk.DATA_DIR
fisher_steps_path = hdinitpk.data_path('fisher_steps')
fisher_params_path = hdinitpk.data_path('fisher_fid_params')
binning_path = hdinitpk.data_path('binning')

# Output directories. The combined per-bin summaries written here are what
# the `<data>/fig6/fig6_points_*.txt` files distributed with the package
# were made from. Set OVERWRITE = False to skip any experiment/chain whose
# combined samples file already exists.
#
# Chain-mode output goes into the packaged data directory. Fisher mode
# writes one shard per MPI rank plus a 50000-sample array per experiment,
# which is bulkier scratch, so it gets its own path for you to FILL IN;
# point `out_dir_fisher` at `hdinitpk.data_path('plin_z0_from_fisher')`
# instead if you would rather keep the two together.
out_dir_fisher = hdinitpk.user_data_path('plin_z0', 'from_fisher')
out_dir_chains = hdinitpk.data_path('plin_z0_from_chains')

RUN_FISHER_MODE = True
RUN_CHAIN_MODE = True

hd_data_version = 'v1.2'   # the CMB-HD mock data version, from hdMockData
use_H0 = True

# Fisher mode only: number of Gaussian samples per experiment.
N_SAMPLES = 50000
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
#   'raw'       your own FULL, un-thinned cobaya output. THIS IS WHAT THE
#               PAPER USED. Thinning changes the sampling of the posterior
#               slightly, so the P_lin(k) errors from the two thinned
#               sources can differ from the published ones at the level of
#               the thinning noise.
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

# CHAIN_SOURCE = 'raw': the cobaya `output:` root of your own run, e.g.
# '/path/to/chains/pk_chains/pas/tau_prior_cmb_pas_7_bins'.
RAW_CHAIN_ROOT = '/path/to/chains/pk_chains/pas/tau_prior_cmb_pas_7_bins'

if CHAIN_SOURCE == 'packaged':
    chain_root = hdinitpk.data_path('chains', 'binned_pk', CHAIN_NAME)
elif CHAIN_SOURCE == 'custom':
    chain_root = os.path.join(CUSTOM_DATA_PATH, 'chains',
                              CUSTOM_CHAIN_NAMES[CHAIN_NAME])
elif CHAIN_SOURCE == 'raw':
    chain_root = RAW_CHAIN_ROOT
else:
    raise ValueError(f"CHAIN_SOURCE must be 'packaged', 'custom' or 'raw', "
                     f"not {CHAIN_SOURCE!r}")

# Burn-in fraction discarded from the start of each chain. The packaged
# chain had its burn-in removed before it was thinned, so nothing more is
# removed there. The other two sources are full cobaya output and still
# carry it, so the first half of each chain is dropped, as in Section III
# of the paper.
BURN_IN_FRACTION = 0.0 if (CHAIN_SOURCE == 'packaged') else 0.5


# ----------------------------------------------------------------------
# Transfer-function options.
# ----------------------------------------------------------------------
# 'delta_tot' = CDM + baryons + massive neutrinos.
TRANSFER_VAR = 'delta_tot'

# Leave as set unless the startup validation says otherwise.
TRANSFER_K_IN_HUNITS = False

# get_transfer_functions is the supported way to get transfers without
# computing power spectra; set False to go through get_results instead.
USE_GET_TRANSFER_FUNCTIONS = True

# Force NonLinear_none for the transfer runs (one flag per mode). Leave
# False to keep each original script's NonLinear setup exactly; the
# normalization checks verify the transfer is still linear either way.
FORCE_LINEAR_NONLINEAR_NONE = False        # fisher mode
FORCE_LINEAR_NONLINEAR_NONE_CHAIN = False  # chain mode

# Apply set_for_lmax(..., lens_potential_accuracy=30, ...) in the fisher
# builder, as in Listing 1. With WantCls = False this should not affect the
# z = 0 matter transfer, and turning it off is much faster.
APPLY_CMB_LMAX_SETUP = True

# Run the normalization checks on rank 0 before any sampling (one per
# enabled mode, since the modes use different accuracy settings).
VALIDATE_NORMALIZATION = True
VALIDATION_RTOL = 2e-3


# ----------------------------------------------------------------------
# k bins.
# ----------------------------------------------------------------------
# Log-spaced edges of the 11-bin forecast scheme: 0.00367 to 38.3 Mpc^-1.
BIN_EDGES_11 = np.array([
    3.667637511098258835e-03, 8.505899211272888866e-03,
    1.972668268698882926e-02, 4.574966151931515734e-02,
    1.061015459285717805e-01, 2.460682259622835044e-01,
    5.706756795889451617e-01, 1.323497700691443235e+00,
    3.069424940269466440e+00, 7.118538595893407539e+00,
    1.650914836730800417e+01, 3.828763111167506139e+01,
])

# Seven-bin scheme: wide first bin from 0.0000562, then the first six of the
# log-spaced bins above (up to 0.571 Mpc^-1).
BIN_EDGES_7 = np.concatenate(([5.623413251903490700e-05], BIN_EDGES_11[:7]))

# Full 12-element edge list (lower edge + the 11-bin edges), as used by the
# Fisher constructors:
PK_BIN_EDGES = np.concatenate(([5.623413251903490700e-05], BIN_EDGES_11))

binning = utils.load_from_file(
    os.path.join(binning_path, 'hd_pk_wavenumbers_11bins.txt'),
    ['k bin center [Mpc^-1]', 'lower bin edge [Mpc^-1]',
     'upper bin edge [Mpc^-1]'])
K_CENTERS_11 = np.asarray(binning['k bin center [Mpc^-1]'], dtype=float)

SO_MAX_BIN = 7


def bin_centers_from_edges(edges):
    """Arithmetic midpoint of each bin, matching Table I."""
    edges = np.asarray(edges, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


K_CENTERS_7 = bin_centers_from_edges(BIN_EDGES_7)

# Sanity check against Table I (5% tolerance: the table rounds to two
# significant figures, so the first bins differ by up to ~3.4%).
_TABLE_I_7 = np.array([0.0018, 0.0060, 0.0141, 0.0327, 0.0759, 0.176, 0.408])
_TABLE_I_11 = np.array([0.0060, 0.0141, 0.0327, 0.0759, 0.176, 0.408,
                        0.947, 2.20, 5.09, 11.8, 27.4])
for _name, _computed, _quoted in (
        ('7-bin', K_CENTERS_7, _TABLE_I_7),
        ('11-bin', bin_centers_from_edges(BIN_EDGES_11), _TABLE_I_11)):
    _frac = np.abs(_computed - _quoted) / _quoted
    if np.any(_frac > 0.05):
        warnings.warn(f'{_name} centers disagree with Table I by >5% in bins '
                      f'{np.where(_frac > 0.05)[0].tolist()}. Check the '
                      f'binning convention.')

# Map number of Pk columns found in a chain -> (k centers, scheme label).
_BINNING_BY_NBINS = {
    7: (K_CENTERS_7, 'seven-bin current-data scheme (Sec. III)'),
    11: (K_CENTERS_11, 'eleven-bin forecast scheme (Sec. IV A)'),
}


# ----------------------------------------------------------------------
# Fiducial primordial spectrum, Eq. 2 with alpha_s = 0 and the Table II
# fiducial values. Used only for the fiducial Pk_i values; it does not
# appear in the Plin calculation itself.
# ----------------------------------------------------------------------
_AS_FID = np.exp(3.044) * 1e-10
_NS_FID = 0.9649
_K_PIVOT = 0.05  # Mpc^-1


def hd_Pk(k):
    return _AS_FID * (np.asarray(k, dtype=float) / _K_PIVOT) ** (_NS_FID - 1.0)


# ----------------------------------------------------------------------
# CAMB parameters: one builder per mode, each with that mode's original
# accuracy settings.
# ----------------------------------------------------------------------
LMAX = 24000

# Fisher mode (Listing 1):
LISTING1_ACCURACY = {
    'AccuracyBoost': 1.1,
    'lSampleBoost': 3.0,
    'lAccuracyBoost': 3.0,
    'DoLateRadTruncation': False,
    'min_l_logl_sampling': 10000,
}

# Chain mode (the original chain script):
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


def make_camb_params_fisher(ombh2, omch2, H0, tau, force_linear=None):
    """CAMBparams for one Fisher-mode sample, following Listing 1.

    The InitPower values are irrelevant: only the transfer functions are
    read, and they are normalized to unit primordial curvature.
    `force_linear` overrides FORCE_LINEAR_NONLINEAR_NONE for this call; the
    validation uses it to build a guaranteed-linear reference.
    """
    if force_linear is None:
        force_linear = FORCE_LINEAR_NONLINEAR_NONE
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau,
                       num_massive_neutrinos=3, mnu=0.06, nnu=3.044,
                       bbn_predictor='PRIMAT_Yp_DH_ErrorMC_2021.dat')
    pars.set_classes(recombination_model='Recfast')
    pars.InitPower.set_params(As=_AS_FID, ns=_NS_FID, pivot_scalar=_K_PIVOT)
    pars.set_matter_power(redshifts=[0.0], kmax=100, k_per_logint=130)
    if APPLY_CMB_LMAX_SETUP:
        pars.set_for_lmax(LMAX + 500, lens_potential_accuracy=30,
                          lens_margin=2050)
    pars.set_accuracy(**LISTING1_ACCURACY)
    if force_linear:
        # after set_for_lmax, which switches nonlinear lensing on
        pars.NonLinear = camb.model.NonLinear_none
    else:
        pars.NonLinear = camb.model.NonLinear_both
        pars.NonLinearModel.set_params('mead2016')
    pars.WantCls = False
    pars.WantTransfer = True
    return pars


def make_camb_params_chain(ombh2, omch2, H0, tau, force_linear=None):
    """CAMBparams for one chain-mode cosmology, using exactly the accuracy
    settings of the original chain script (via camb.set_params)."""
    if force_linear is None:
        force_linear = FORCE_LINEAR_NONLINEAR_NONE_CHAIN
    pars = camb.set_params(
        H0=H0, ombh2=ombh2, omch2=omch2, tau=tau,
        num_massive_neutrinos=3, mnu=0.06, nnu=3.044,
        bbn_predictor='PRIMAT_Yp_DH_ErrorMC_2021.dat',
        As=_AS_FID, ns=_NS_FID, pivot_scalar=_K_PIVOT,
        redshifts=[0.0], WantCls=False, WantTransfer=True,
        lmax=LMAX,   # lens_potential_accuracy/lens_margin act through lmax
        **CHAIN_CAMB_ACCURACY_SETTINGS,
    )
    if force_linear:
        pars.NonLinear = camb.model.NonLinear_none
    return pars


# ----------------------------------------------------------------------
# Transfer functions (shared by both modes).
# ----------------------------------------------------------------------
# Row of MatterTransferData.transfer_data holding TRANSFER_VAR: row 0 is
# k/h, so the row index is the position of the name in transfer_names.
_TRANSFER_ROW = list(camb.model.transfer_names).index(TRANSFER_VAR)


def _run_camb(pars):
    if USE_GET_TRANSFER_FUNCTIONS:
        return camb.get_transfer_functions(pars)
    return camb.get_results(pars)


def _z0_index(results, n_z):
    """Index of z = 0 along the redshift axis of transfer_data.

    Do not assume there is only one output redshift: with nonlinear lensing
    active, CAMB replaces the requested PK_redshifts with its own grid for
    the lensing calculation, so set_matter_power(redshifts=[0.0]) can come
    back as tens of redshifts (stored decreasing, z = 0 last). Read the
    actual list rather than trusting that ordering.
    """
    try:
        tr = results.Params.Transfer
        zs = np.asarray(tr.PK_redshifts, dtype=float)[:int(tr.PK_num_redshifts)]
    except Exception:
        zs = None
    if zs is None or len(zs) != n_z:
        return n_z - 1
    iz = int(np.argmin(np.abs(zs)))
    if abs(zs[iz]) > 1e-8:
        raise RuntimeError(
            f'no z = 0 among the {n_z} output redshifts (closest is '
            f'{zs[iz]:.6g}); the requested PK_redshifts were overridden')
    return iz


def transfer_k_and_Tsq(results, H0, hunits=None):
    r"""(k [Mpc^-1], T(k)^2 [Mpc^4]) at z = 0 on CAMB's own q grid.

    T is the density contrast per unit primordial curvature divided by k^2,
    with k in Mpc^-1, so that P_lin(k) = 2 pi^2 k T(k)^2 \mathcal{P}(k).
    """
    if hunits is None:
        hunits = TRANSFER_K_IN_HUNITS
    td = results.get_matter_transfer_data().transfer_data
    iz = _z0_index(results, td.shape[2])
    h = H0 / 100.0
    k = np.asarray(td[0, :, iz], dtype=float) * h   # k/h [h/Mpc] -> Mpc^-1
    T = np.asarray(td[_TRANSFER_ROW, :, iz], dtype=float)
    if hunits:
        # stored T is Delta / (k/h)^2, i.e. (Mpc/h)^2; convert to Mpc^2
        T = T / h ** 2
    order = np.argsort(k)
    return k[order], (T[order]) ** 2


def interp_Tsq(k_grid, Tsq_grid, k_out):
    """Cubic spline of log T^2 in log k, evaluated at k_out."""
    k_out = np.atleast_1d(np.asarray(k_out, dtype=float))
    if k_out.min() < k_grid.min() or k_out.max() > k_grid.max():
        raise ValueError(
            f'bin centers span [{k_out.min():.4e}, {k_out.max():.4e}] Mpc^-1 '
            f'but CAMB returned transfers on [{k_grid.min():.4e}, '
            f'{k_grid.max():.4e}] Mpc^-1; raise kmax')
    if not np.all(Tsq_grid > 0):
        raise ValueError('nonpositive T^2 on the CAMB grid; cannot take logs')
    spline = CubicSpline(np.log(k_grid), np.log(Tsq_grid))
    return np.exp(spline(np.log(k_out)))


def camb_Tsq(ombh2, omch2, H0, tau, k_centers, params_builder=None):
    """T(k)^2 [Mpc^4] at k_centers for one cosmology. `params_builder`
    selects the mode's settings (default: `make_camb_params_fisher`). tau
    has no effect on the z = 0 transfer; it is passed through for interface
    consistency."""
    if params_builder is None:
        params_builder = make_camb_params_fisher
    results = _run_camb(params_builder(ombh2, omch2, H0, tau))
    k_grid, Tsq_grid = transfer_k_and_Tsq(results, H0)
    return interp_Tsq(k_grid, Tsq_grid, k_centers)


def plin_from_Tsq(k, Tsq, primordial):
    r"""P_lin(k, z=0) [Mpc^3] = 2 pi^2 k T(k)^2 \mathcal{P}(k). `primordial`
    is the dimensionless curvature power in each bin. Broadcasts over
    leading sample axes."""
    return 2.0 * np.pi ** 2 * np.asarray(k) * np.asarray(Tsq) * np.asarray(primordial)


# ----------------------------------------------------------------------
# Normalization check (run once per enabled mode).
# ----------------------------------------------------------------------
def validate_transfer_normalization(params_builder=None, k_test=None,
                                    rtol=VALIDATION_RTOL, label='fisher',
                                    force_linear_flag_name='FORCE_LINEAR_NONLINEAR_NONE'):
    r"""Compare 2 pi^2 k T^2 \mathcal{P} against CAMB's own linear P(k) at the
    fiducial cosmology, for one mode's parameter builder. Raises if the
    configured h-convention is off by more than rtol (reporting whether the
    other convention would have worked), and if the production path applies
    nonlinear ratios to the transfer data."""
    if params_builder is None:
        params_builder = make_camb_params_fisher
    if k_test is None:
        k_test = K_CENTERS_11
    k_test = np.asarray(k_test, dtype=float)
    fid = (FID_PARAMS['ombh2'], FID_PARAMS['omch2'], FID_PARAMS['H0'],
           FID_PARAMS['tau'])

    # Reference: one internally consistent linear run (NonLinear_none is
    # forced whatever the mode's flag says).
    res_lin = camb.get_results(params_builder(*fid, force_linear=True))
    PK = res_lin.get_matter_power_interpolator(
        nonlinear=False, hubble_units=False, k_hunit=False,
        var1=TRANSFER_VAR, var2=TRANSFER_VAR)
    p_ref = PK.P(0.0, k_test)

    k_grid, Tsq_lin = transfer_k_and_Tsq(res_lin, FID_PARAMS['H0'])
    devs = {}
    for hunits in (True, False):
        kg, Tsq_grid = transfer_k_and_Tsq(res_lin, FID_PARAMS['H0'],
                                          hunits=hunits)
        try:
            Tsq = interp_Tsq(kg, Tsq_grid, k_test)
        except ValueError:
            devs[hunits] = np.inf
            continue
        p_tf = plin_from_Tsq(k_test, Tsq, hd_Pk(k_test))
        devs[hunits] = float(np.max(np.abs(p_tf / p_ref - 1.0)))

    dev = devs[TRANSFER_K_IN_HUNITS]
    other = devs[not TRANSFER_K_IN_HUNITS]
    print(f'[rank 0] ({label}) normalization check: '
          f'max |P_transfer/P_camb - 1| = {dev:.3e} with '
          f'TRANSFER_K_IN_HUNITS={TRANSFER_K_IN_HUNITS} '
          f'({other:.3e} with the other convention)', flush=True)
    if dev > rtol:
        msg = (f'transfer-derived P(k) disagrees with CAMB by {dev:.3e} '
               f'(tolerance {rtol:.1e}).')
        if other <= rtol:
            msg += (f' TRANSFER_K_IN_HUNITS={not TRANSFER_K_IN_HUNITS} agrees '
                    f'to {other:.3e}; flip it.')
        raise RuntimeError(msg)

    # Production path vs linear reference: with NonLinear active, CAMB can
    # apply the HMCode ratios to the stored transfer data through
    # get_results; get_transfer_functions does not.
    Tsq_ref = interp_Tsq(k_grid, Tsq_lin, k_test)
    Tsq_prod = camb_Tsq(*fid, k_test, params_builder=params_builder)
    prod_dev = float(np.max(np.abs(Tsq_prod / Tsq_ref - 1.0)))
    print(f'[rank 0] ({label}) production path vs linear reference: '
          f'max |T^2 ratio - 1| = {prod_dev:.3e}', flush=True)
    if prod_dev > rtol:
        raise RuntimeError(
            f'the production transfer differs from the linear one by '
            f'{prod_dev:.3e}: the nonlinear ratios have almost certainly '
            f'been applied to the transfer data. Set '
            f'{force_linear_flag_name} = True.')
    return dev


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
        overwrite=False, use_H0=use_H0, use_class=False,
        hd_data_version=hd_data_version, pol_only_lensing=True,
        binned_pk=True, ksz=True, bin_edges=PK_BIN_EDGES[1:],
        fisher_steps_file=os.path.join(
            fisher_steps_path, 'binned_pk_steps_5_percent_feedback.yaml'),
        param_file=os.path.join(
            fisher_params_path, 'hd_binned_pk_fiducial_params_feedback.yaml'))
    return fisherlib.get_fisher(
        'lensed', priors={'tau': 0.005, 'HMCode_logT_AGN': 0.0006 * 7.8},
        use_H0=use_H0, with_desi=True, params=params)


def build_so_fisher():
    params = ([f'Pk{i+1}' for i in range(SO_MAX_BIN)]
              + ['H0', 'tau', 'ombh2', 'omch2'])
    fisherlib = hdinitPkfisher.Fisher(
        SO_FISHER_DERIV_DIR,
        overwrite=False, exp='so', use_H0=use_H0, use_class=False,
        hd_data_version=hd_data_version,
        binned_pk=True, bin_edges=PK_BIN_EDGES[1:],
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

FID_PARAMS = {'ombh2': 0.02237, 'omch2': 0.1200, 'tau': 0.0544, 'H0': 67.36,
              'HMCode_logT_AGN': 7.8, 'A_ksz': 1.0, 'n_ksz': 0.0}
for _n, _pk in enumerate(hd_Pk(K_CENTERS_11), start=1):
    FID_PARAMS[f'Pk{_n}'] = _pk


# ----------------------------------------------------------------------
# Fisher mode: Gaussian sampling.
# ----------------------------------------------------------------------
def draw_samples(fisher_matrix, fisher_params, n_samples, seed):
    """Draw from the multivariate Gaussian defined by the Fisher matrix
    (getdist GaussianND centered on the fiducials, inverse Fisher as the
    covariance). Samples CAMB cannot be called on (nonpositive density,
    negative tau, nonpositive H0) are dropped and counted. Samples with a
    nonpositive P(k) bin are counted but NOT dropped, matching the original
    script; those give a negative Plin in that bin."""
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
        frac = n_bad / len(samples)
        msg = (f'dropped {n_bad}/{len(samples)} ({frac:.2%}) samples that '
               f'CAMB cannot be called on')
        if frac > 0.01:
            warnings.warn(msg + ' check the Fisher matrix.')
        else:
            print(f'[rank 0] {msg}', flush=True)

    if mpi.rank == 0:
        pk_cols = [idx[c] for c in params if c.startswith('Pk')]
        n_negpk = int((samples[ok][:, pk_cols] <= 0).any(axis=1).sum())
        if n_negpk:
            print(f'[rank 0] {n_negpk}/{int(ok.sum())} kept samples have a '
                  f'nonpositive P(k) bin (kept, not dropped)', flush=True)

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
    """Return the list of sample files belonging to `root`, and a label for
    them. Handles both chain layouts:

      * a getdist root, `<root>.txt` with its column names in
        `<root>.paramnames` (what the thinned chains distributed with the
        package look like), a single merged file;
      * numbered parts, `<root>.1.txt` ... `<root>.N.txt`. These are
        either raw cobaya output, each carrying its own '#'-prefixed
        header line, or one getdist chain split into pieces small enough
        to distribute, which share a single `<root>.paramnames`.
        `read_chain` tells them apart by that header line.

    Returns
    -------
    files : list of str
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
    paramnames_file = stem + '.paramnames'
    if not os.path.exists(paramnames_file):
        raise ValueError(
            f'{path} has no header line and no {paramnames_file} beside it, '
            'so its columns cannot be named.')
    with open(paramnames_file) as f:
        names = [line.split()[0].rstrip('*') for line in f
                 if line.strip() and not line.startswith('#')]
    colnames = ['weight', 'minuslogpost'] + names
    df = pd.read_csv(path, sep=r'\s+', comment='#', names=colnames)
    if df.shape[1] != len(colnames):
        raise ValueError(
            f'{path} has {df.shape[1]} columns but {paramnames_file} implies '
            f'{len(colnames)}; check that the two files match.')
    return df


def get_pk_columns(df):
    """Return the bin-amplitude column names, sorted by bin number, and the
    factor that converts them to absolute e^{-2 tau} P(k).

    Accepts either naming: `eneg2tauPk<n>` (absolute, factor 1) or `b<n>`
    (in units of `arbitrary_Pkbinning.BinnedPk`'s `scale`, factor `scale`).

    Returns
    -------
    columns : list of str
    unit : float
        Multiply the columns by this to get e^{-2 tau} P(k).
    """
    matches = {}
    for prefix, regex in _PK_COL_RES.items():
        found = sorted((int(m.group(1)), c)
                       for c in df.columns
                       for m in [regex.match(c)] if m)
        if found:
            matches[prefix] = found
    if not matches:
        raise ValueError(
            f'No bin-amplitude columns found (looked for eneg2tauPk<n> and '
            f'b<n>); got {list(df.columns)}')
    if len(matches) > 1:
        raise ValueError(
            f'Ambiguous chain: it has both {" and ".join(matches)} columns. '
            'Keep only one set of bin amplitudes.')

    prefix, found = matches.popitem()
    nums = [n for n, _ in found]
    if nums != list(range(1, len(nums) + 1)):
        raise ValueError(f'Expected contiguous {prefix}1..{prefix}N, got {nums}')
    unit = 1.0 if prefix == 'eneg2tauPk' else ARBITRARY_BINNING_SCALE
    return [c for _, c in found], unit


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
    if BURN_IN_FRACTION == 0.0:
        print(f'[rank {mpi.rank}] {tag}: BURN_IN_FRACTION = 0, keeping all '
              f'{n_raw} rows (correct for a thinned chain, whose burn-in was '
              f'removed before thinning; set it to 0.5 for raw cobaya '
              f'output)', flush=True)
    if BURN_IN_FRACTION > 0.0:
        n_burn = int(np.floor(BURN_IN_FRACTION * n_raw))
        df = df.iloc[n_burn:].reset_index(drop=True)
        print(f'[rank {mpi.rank}] {tag}: dropped {n_burn}/{n_raw} rows as '
              f'burn-in ({BURN_IN_FRACTION:.0%}), {len(df)} remain', flush=True)
    if len(df) == 0:
        raise ValueError(f'{chain_path} has no rows left after burn-in.')

    pk_cols, pk_unit = get_pk_columns(df)
    n_bins = len(pk_cols)
    if pk_unit != 1.0:
        print(f'[rank {mpi.rank}] {tag}: bin amplitudes are named '
              f'{pk_cols[0]}..{pk_cols[-1]}, so they are in units of '
              f'{pk_unit:g}; converting to absolute e^-2tau P(k)', flush=True)
    if n_bins not in _BINNING_BY_NBINS:
        raise ValueError(
            f'Chain {chain_path} has {n_bins} Pk bins, matching neither the '
            f'7-bin nor the 11-bin scheme; refusing to guess the k centers.')
    k_centers, scheme = _BINNING_BY_NBINS[n_bins]
    print(f'[rank {mpi.rank}] {tag}: {n_bins} Pk bins -> {scheme}', flush=True)

    for col in ('weight', 'ombh2', 'omch2', 'H0', 'tau'):
        if col not in df.columns:
            raise ValueError(f'{chain_path} missing required column {col!r}')

    weight = df['weight'].to_numpy(float)
    ombh2 = df['ombh2'].to_numpy(float)
    omch2 = df['omch2'].to_numpy(float)
    H0 = df['H0'].to_numpy(float)
    tau = df['tau'].to_numpy(float)
    # The chain columns are e^{-2 tau} P(k) with no factor of 1e9 (the 1e9
    # in Table I is display-only). Invert with each sample's OWN tau:
    pk_samp = (df[pk_cols].to_numpy(float) * pk_unit
               / np.exp(-2.0 * tau)[:, None])

    # One CAMB call per unique cosmology. T^2 does not depend on tau, but
    # tau stays in the dedup key for exact parity with the original script.
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
        # do not let a wrong path silently produce a "successful" empty run
        raise FileNotFoundError(
            f'no chain files found for root {chain_root}. Expected either '
            f'{chain_root}.txt (a getdist root, with its .paramnames beside '
            f'it) or {chain_root}.1.txt ... (raw cobaya output).')

    run_label = os.path.basename(chain_root)
    if mpi.rank == 0:
        merged = len(chain_files) == 1
        print(f'[rank 0] chain root {chain_root}: {len(chain_files)} file(s)'
              + (' (a single merged/thinned chain, so only one rank has work '
                 'to do here)' if merged else ''), flush=True)

    chain_idxs = mpi.distribute(len(chain_files), mpi.size, mpi.rank)
    print(f'[rank {mpi.rank}/{mpi.size}] chains: {chain_idxs}', flush=True)

    for ci in chain_idxs:
        path = chain_files[ci]
        tag = (run_label if len(chain_files) == 1
               else f'{run_label}_chain{ci + 1}')
        compute_plin_for_chain(path, tag=tag)


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

    if VALIDATE_NORMALIZATION:
        err = None
        if mpi.rank == 0:
            try:
                if RUN_FISHER_MODE:
                    validate_transfer_normalization(
                        params_builder=make_camb_params_fisher,
                        k_test=K_CENTERS_11, label='fisher',
                        force_linear_flag_name='FORCE_LINEAR_NONLINEAR_NONE')
                if RUN_CHAIN_MODE:
                    # chain settings have kmax=10; validate on the 7-bin
                    # centers (the k range the PAS chains actually use)
                    validate_transfer_normalization(
                        params_builder=make_camb_params_chain,
                        k_test=K_CENTERS_7, label='chain',
                        force_linear_flag_name='FORCE_LINEAR_NONLINEAR_NONE_CHAIN')
            except Exception as e:      # report on rank 0, abort everywhere
                err = repr(e)
        err = mpi.comm.bcast(err, root=0)
        if err is not None:
            if mpi.rank == 0:
                print(f'[rank 0] normalization check failed: {err}', flush=True)
            raise SystemExit(1)
        mpi.comm.barrier()

    if RUN_FISHER_MODE:
        for exp_key in ['hd', 'so']:
            compute_plin_for_experiment(exp_key)
            mpi.comm.barrier()

    if RUN_CHAIN_MODE:
        run_chain_mode()
        mpi.comm.barrier()

    if mpi.rank == 0:
        print('All requested modes complete.', flush=True)
