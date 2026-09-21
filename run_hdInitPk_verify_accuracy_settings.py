"""Check the CAMB and CLASS accuracy settings against the parameter errors.

This reproduces Figure 8 of Cheslog et al. (2026), the bias on each
cosmological parameter from computing the theory spectra with accuracy
settings that are not fully converged, divided by the forecasted error on
that parameter. A value well below 1 means the setting is good enough.

The bias is the standard linear estimate

    b_i = (F^-1)_ij  d_j C^T  Cov^-1  (C^true - C^fid)

where d_j C is the derivative of the binned spectra with respect to
parameter j, Cov is the CMB-HD bandpower covariance, and F is the Fisher
matrix. It is evaluated over a grid of accuracy settings (CAMB's
lens_potential_accuracy and CLASS's P_k_max_h/Mpc) against a high-accuracy
CAMB reference spectrum. The grid spectra, the reference, and the
eight-parameter Fisher matrices ship with the package in hdinitpk/data. The
bias calculation fixes the running, which is why it uses the
eight-parameter matrices. The binning and the covariance matrix come from
hdMockData. Set RECALCULATE_SPECTRA = True to calculate the grid spectra
and the reference again instead of reading the shipped ones.

The one input that does not ship is the Fisher derivatives, which are far
too large. The script reads the CAMB and CLASS nine-parameter derivatives
from hdinitpk/data/user_generated_data/fisher_derivs, where
run_hdInitPk_forecasts.py and run_hdInitPk_forecasts.ipynb write them. If
they are not there yet, it calculates them first. That is two Boltzmann
calls per varied parameter and takes hours, so run it under MPI when the
derivatives still need computing.

    mpirun -np 4 python run_hdInitPk_verify_accuracy_settings.py

With the derivatives already on disk, plain `python` is enough and the
script takes seconds.
"""
import os

import numpy as np

from hdfisher import fisher, utils, mpi
from hd_mock_data import hd_data

import hdinitpk
from hdinitpk import hdinitPkfisher, theory


# The derivative directories. These default to where the forecast script
# and notebook write, so nothing needs filling in.
CAMB_DERIV_DIR = hdinitpk.user_data_path('fisher_derivs', 'camb_9param')
CLASS_DERIV_DIR = hdinitpk.user_data_path('fisher_derivs', 'class_9param')

# Draw the Figure 8 plot at the end (needs matplotlib).
MAKE_FIGURE = True
SAVE_FIGURE = True

# Recalculate the spectra used in the bias calculation (the CAMB and CLASS
# accuracy grids and the high-accuracy CAMB reference) instead of reading
# the ones that ship with the package. The grids use the fiducial CAMB and
# CLASS settings with one accuracy setting varied at a time, and the
# reference uses the CAMB settings with the changes in
# TRUE_UNIVERSE_SETTINGS below, as in Appendix A. This is slow (the reference
# alone takes hours), so the calculations are spread over the MPI ranks.
# The spectra are written to hdinitpk/data/user_generated_data/spectra_for_bias
# and read from there.
RECALCULATE_SPECTRA = False


# ----------------------------------------------------------------------
# Settings, matching the notebook.
# ----------------------------------------------------------------------
use_H0 = True

# The CMB-HD mock data version, used both for the binning and covariance
# matrix read below and for what hdfisher reads for itself. Pinned rather
# than left at 'latest' so that a new hdMockData release does not move the
# forecasts.
hd_data_version = 'v1.2'

# lmax of the binning used for the bias itself, and of the spectra files:
BIAS_LMAX = 20100
SPECTRA_LMAX = 24000

# accuracy grids
lens_potential_values = [8, 10, 15, 20, 25, 30, 35, 40]   # CAMB
P_k_max_values = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]  # CLASS

# the settings of the high-accuracy CAMB reference, on top of the fiducial
# CAMB settings
TRUE_UNIVERSE_SETTINGS = {'AccuracyBoost': 3.0, 'lAccuracyBoost': 5.0,
                          'lens_potential_accuracy': 40, 'lSampleBoost': 5.0}

# the columns of the spectra files, starting at ell = 0
SPECTRA_COLS = ['tt', 'te', 'ee', 'bb', 'kk']

# the bias calculation fixes the running, so eight parameters:
eight_params = ['mnu', 'tau', 'logA', 'H0', 'ombh2', 'omch2', 'ns', 'nnu']

# CAMB -> CLASS parameter names, as in the notebook. The CLASS Fisher and
# derivatives carry native CLASS names; everything here is keyed by the CAMB
# name, so the CLASS results are translated back on load.
camb_to_class_fisher_params = {
    'ombh2': 'omega_b', 'omch2': 'omega_cdm', 'theta': 'theta_s_100',
    'tau': 'tau_reio', 'logA': 'ln_A_s_1e10', 'As': 'A_s', 'ns': 'n_s',
    'H0': 'H0', 'nnu': 'Neff', 'mnu': 'sum_m_ncdm', 'nrun': 'alpha_s',
    'omk': 'Omega_k', 'w': 'w0_fld', 'wa': 'wa_fld',
    'HMCode_logT_AGN': 'log10T_heat_hmcode',
}
class_to_camb_fisher_params = {v: k for k, v in camb_to_class_fisher_params.items()}
class_to_camb_fisher_params['Neff'] = 'nnu'


def from_class_fisher(result):
    """`get_fisher` output -> CAMB names. Only the names change."""
    matrix, params = result
    return matrix, [class_to_camb_fisher_params.get(p, p) for p in params]


def from_class_derivs(result):
    """`load_cmb_fisher_derivs` output -> CAMB names."""
    ells, derivs = result
    out = {}
    for cmb_type, per_param in derivs.items():
        out[cmb_type] = {class_to_camb_fisher_params.get(p, p): spectra
                         for p, spectra in per_param.items()}
    return ells, out


# ======================================================================
# The bias calculation itself.
# ======================================================================

def calc_bias(fid_spec, true_spec, covmat, derivs, fisher_matrix):
    """Take in the components of the bias equation and calculate the bias.

    Args:
        fid_spec (numpy array): 1d np array of the binned fiducial spectra,
            concatenated in order TT, TE, EE, BB, kk
        true_spec (numpy array): 1d np array of the binned true universe
            spectra, concatenated in order TT, TE, EE, BB, kk
        covmat (numpy array): a 2d array containing the CMB-HD covmat
        derivs (numpy array): an nd array (where n is the number of params)
            of the derivatives of the spectra with respect to each param.
            The derivs should be binned and concatenated in the same spectra
            order as the true/fid spec, and should have the same params in
            the same order as the fisher matrix.
        fisher_matrix (numpy array): a n x n fisher matrix, where n is the
            number of params.

    Returns:
        bias (np array): A 1d array of the biases on each parameter, in the
            same order as the params in the fisher matrix/derivs.
    """
    difference = true_spec - fid_spec
    first_product = np.linalg.solve(covmat, difference)
    second_product = derivs @ first_product
    bias = np.linalg.solve(fisher_matrix, second_product)
    return bias


def order_spectra(spectra):
    """Take in a dict of spectra and return a concatenated array of spectra
    in order TT, TE, EE, BB, kk."""
    return np.concatenate([spectra['tt'], spectra['te'], spectra['ee'],
                           spectra['bb'], spectra['kk']])


def order_derivs(derivs, param_order):
    """Take in a dict of derivs of spectra and return them in the order
    `param_order`, spectra concatenated in TT, TE, EE, BB, kk.

    Returns an array of shape (num params, num bins).
    """
    return np.stack([order_spectra(derivs[p]) for p in param_order])


def bin_spec(cl, matrix, lmax):
    """Take in a dict of unbinned spectra, the binning matrix, and the
    maximum ell to go to, and return the binned spectra."""
    slice_end = lmax + 1
    return {s: (matrix @ cl[s][2:slice_end])
            for s in ('tt', 'te', 'ee', 'bb', 'kk')}


# ======================================================================
# Loading the inputs.
# ======================================================================

def spectra_fname(spectra_path, code, value=None):
    """The file holding one set of spectra: the CAMB reference (`code`
    `'true'`), or the CAMB or CLASS spectra at one accuracy setting."""
    if code == 'true':
        return os.path.join(spectra_path,
                            f'lensed_lmax={SPECTRA_LMAX}_true_universe_spectra.txt')
    if code == 'camb':
        return os.path.join(spectra_path, 'camb',
                            f'lensed_lmax={SPECTRA_LMAX}_lens_potential_accuracy={value}'
                            '_camb_spectra.txt')
    return os.path.join(spectra_path, 'class',
                        f'lensed_lmax={SPECTRA_LMAX}_P_k_max_h_Mpc={value}'
                        '_class_spectra.txt')


def calculate_spectra(spectra_path):
    """Calculate the lensed spectra used in the bias calculation with
    `hdinitpk.theory.Theory`, one setting at a time, spread over the MPI
    ranks, and save each as five columns (TT, TE, EE, BB, kk) from
    ell = 0 to `SPECTRA_LMAX`."""
    for sub in ['camb', 'class']:
        os.makedirs(os.path.join(spectra_path, sub), exist_ok=True)
    tasks = ([('true', None)] + [('camb', p) for p in lens_potential_values]
             + [('class', p) for p in P_k_max_values])
    for i in mpi.distribute(len(tasks), mpi.size, mpi.rank):
        code, value = tasks[i]
        print(f'[rank {mpi.rank}] calculating {code} spectra'
              + ('' if value is None else f' at {value}'), flush=True)
        if code == 'class':
            theolib = theory.Theory(
                SPECTRA_LMAX, spectra_path, use_class=True,
                param_file=hdinitPkfisher.fiducial_param_file(use_class=True),
                **{'P_k_max_h/Mpc': value})
        else:
            settings = (TRUE_UNIVERSE_SETTINGS if code == 'true'
                        else {'lens_potential_accuracy': value})
            theolib = theory.Theory(
                SPECTRA_LMAX, spectra_path,
                param_file=hdinitPkfisher.fiducial_param_file(), **settings)
        spectra = theolib.get_theory(cmb_types=['lensed'])['lensed']
        utils.save_to_file(spectra_fname(spectra_path, code, value), spectra,
                           keys=SPECTRA_COLS)
    mpi.comm.barrier()


def ensure_derivs(fisherlib, label):
    """Calculate the derivatives in `fisherlib`'s directory if there are
    none there yet that match `use_H0`. Runs on every rank, so that the
    calculation is shared under MPI."""
    cmb_types, found = fisher.get_available_cmb_fisher_derivs(fisherlib.derivs_dir,
                                                              use_H0=use_H0)
    if cmb_types and found:
        if mpi.rank == 0:
            print(f'  {label}: {len(found)} parameters found in '
                  f'{fisherlib.derivs_dir}', flush=True)
        return
    if mpi.rank == 0:
        print(f'  {label}: no derivatives in {fisherlib.derivs_dir}, '
              'calculating them now. This takes hours.', flush=True)
    fisherlib.calculate_fisher_derivs()
    mpi.comm.barrier()


def load_inputs():
    """Build the Fisher libraries, calculate the derivatives if they are
    not there, then (on rank 0) load the derivatives, the Fisher matrices,
    the covariance matrix, and the spectra. Returns None on other ranks."""
    # `overwrite=False`, so that existing derivatives are reused. With no
    # parameter or step-size file given, these use the CAMB and CLASS
    # nine-parameter files provided with hdinitpk, the same ones the
    # forecast script uses.
    camb_fisherlib = hdinitPkfisher.Fisher(
        CAMB_DERIV_DIR, overwrite=False, use_H0=use_H0, use_class=False,
        hd_data_version=hd_data_version)
    class_fisherlib = hdinitPkfisher.Fisher(
        CLASS_DERIV_DIR, overwrite=False, use_H0=use_H0, use_class=True,
        hd_data_version=hd_data_version)

    if mpi.rank == 0:
        print('Fisher derivatives:', flush=True)
    ensure_derivs(camb_fisherlib, 'CAMB')
    ensure_derivs(class_fisherlib, 'CLASS')

    # Spectra: the high-accuracy CAMB reference, and the two accuracy grids.
    # These ship with the package, or are calculated here if asked.
    if RECALCULATE_SPECTRA:
        spectra_path = hdinitpk.user_data_path('spectra_for_bias')
        calculate_spectra(spectra_path)
    else:
        spectra_path = hdinitpk.data_path('spectra_for_bias')
    if mpi.rank != 0:
        return None

    camb_derivs = camb_fisherlib.load_cmb_fisher_derivs(binned=True)
    class_derivs = from_class_derivs(
        class_fisherlib.load_cmb_fisher_derivs(cmb_types=['lensed'],
                                               binned=True))

    # Eight-parameter Fisher matrices, from the packaged data directory.
    # These already have the tau prior and the DESI BAO Fisher applied. The
    # CLASS file keeps native CLASS parameter names, translated here.
    load = lambda n: fisher.load_fisher_matrix(
        hdinitpk.data_path('fisher_matrices', f'{n}.txt'))
    camb_fisher_8 = load('hd_camb_lensed_8param')
    class_fisher_8 = from_class_fisher(load('hd_class_lensed_8param'))
    camb_errs_8 = fisher.get_fisher_errors(*camb_fisher_8)
    class_errs_8 = fisher.get_fisher_errors(*class_fisher_8)

    # Binning matrix and covariance matrix, both from hdMockData. The
    # covariance is the binned lensed one, ordered TT, TE, EE, BB, kk.
    hd_datalib = hd_data.HDMockData(version=hd_data_version)
    binning_matrix = hd_datalib.binning_matrix(lmin=30, lmax=BIAS_LMAX)
    covmat = hd_datalib.block_covmat('lensed')

    true_spectra = utils.load_from_file(spectra_fname(spectra_path, 'true'), SPECTRA_COLS)
    camb_spectra = {p: utils.load_from_file(spectra_fname(spectra_path, 'camb', p), SPECTRA_COLS)
                    for p in lens_potential_values}
    class_spectra = {p: utils.load_from_file(spectra_fname(spectra_path, 'class', p), SPECTRA_COLS)
                     for p in P_k_max_values}

    return dict(camb_derivs=camb_derivs, class_derivs=class_derivs,
                camb_fisher_8=camb_fisher_8, class_fisher_8=class_fisher_8,
                camb_errs_8=camb_errs_8, class_errs_8=class_errs_8,
                binning_matrix=binning_matrix, covmat=covmat,
                true_spectra=true_spectra, camb_spectra=camb_spectra,
                class_spectra=class_spectra)


# ======================================================================
# Main.
# ======================================================================

def main():
    d = load_inputs()
    if d is None:   # not rank 0
        return None

    # Bin and concatenate the reference spectra once.
    binned_true = bin_spec(d['true_spectra'], d['binning_matrix'], BIAS_LMAX)
    cat_true_spec = order_spectra(binned_true)

    # Concatenate the derivs in the Fisher-matrix parameter order, so that
    # `calc_bias` returns the biases in that same order.
    ordered_camb_derivs = order_derivs(d['camb_derivs'][1]['lensed'],
                                       d['camb_fisher_8'][1])
    ordered_class_derivs = order_derivs(d['class_derivs'][1]['lensed'],
                                        d['class_fisher_8'][1])

    camb_bias, class_bias = {}, {}
    for p in lens_potential_values:   # CAMB
        cat_spec = order_spectra(bin_spec(d['camb_spectra'][p],
                                          d['binning_matrix'], BIAS_LMAX))
        camb_bias[p] = calc_bias(cat_spec, cat_true_spec, d['covmat'],
                                 ordered_camb_derivs, d['camb_fisher_8'][0])
    for p in P_k_max_values:          # CLASS
        cat_spec = order_spectra(bin_spec(d['class_spectra'][p],
                                          d['binning_matrix'], BIAS_LMAX))
        class_bias[p] = calc_bias(cat_spec, cat_true_spec, d['covmat'],
                                  ordered_class_derivs, d['class_fisher_8'][0])

    # Divide the bias by the forecasted error on each parameter.
    camb_bias_over_error, class_bias_over_error = {}, {}
    for i, p in enumerate(d['camb_fisher_8'][1]):
        camb_bias_over_error[p] = [camb_bias[a][i] / d['camb_errs_8'][p]
                                   for a in lens_potential_values]
    for i, p in enumerate(d['class_fisher_8'][1]):
        class_bias_over_error[p] = [class_bias[a][i] / d['class_errs_8'][p]
                                    for a in P_k_max_values]

    # Report: the largest |bias / error| over all parameters, per setting.
    print('\nCAMB, |bias / error| (max over the 8 parameters):')
    for j, a in enumerate(lens_potential_values):
        worst = max(abs(camb_bias_over_error[p][j]) for p in eight_params)
        print(f'  lens_potential_accuracy = {a:4d} : {worst:.4f}')
    print('\nCLASS, |bias / error| (max over the 8 parameters):')
    for j, a in enumerate(P_k_max_values):
        worst = max(abs(class_bias_over_error[p][j]) for p in eight_params)
        print(f'  P_k_max_h/Mpc = {a:5d} : {worst:.4f}')

    camb_bias_over_error['lens_potential_accuracy'] = lens_potential_values
    class_bias_over_error['P_k_max_h/Mpc'] = P_k_max_values
    bias_plt_data = {'camb': camb_bias_over_error,
                     'class': class_bias_over_error}

    if MAKE_FIGURE:
        make_figure8(bias_plt_data)
    return bias_plt_data


def make_figure8(bias_plt_data):
    """Reproduce Figure 8: bias / error against the accuracy setting."""
    import matplotlib.pyplot as plt
    from hdinitpk import plotting_utilities
    from hdinitpk.plotting_utilities import param_labels

    plotting_utilities.set_plot_style()

    # parameters to plot, and the order to plot them in:
    fig8_params = ['mnu', 'tau', 'logA', 'H0', 'ombh2', 'omch2', 'ns', 'nnu']
    fig8_param_labels = {p: f'${param_labels[p]}$' for p in fig8_params}
    fig8_colors = {'mnu': 'g', 'tau': 'tab:grey', 'logA': 'y', 'H0': 'b',
                   'ombh2': 'tab:brown', 'omch2': 'tab:pink', 'ns': 'm',
                   'nnu': 'r'}

    # x-axis for CAMB or CLASS:
    accuracy_param = {'camb': 'lens_potential_accuracy',
                      'class': 'P_k_max_h/Mpc'}
    fig8_xlabels = {'camb': r'$\mathtt{lens\_potential\_accuracy}$',
                    'class': r'$\mathtt{P\_k\_max\_h/Mpc}$'}
    fig8_xticks = {'camb': [8, *range(10, 45, 5)],
                   'class': [50, *range(200, 1200, 200)]}

    # text box to label CAMB or CLASS:
    txt_bbox = {'edgecolor': 'k', 'facecolor': 'w', 'alpha': 0.85, 'lw': 1.5,
                'boxstyle': 'round', 'pad': 0.4}
    txt_kwargs = {'fontsize': 18, 'va': 'top', 'ha': 'center', 'bbox': txt_bbox}

    fig, axs = plt.subplots(figsize=(12, 5), dpi=300, ncols=2)

    for i, key in enumerate(['camb', 'class']):
        ax = axs[i]
        xvals = bias_plt_data[key][accuracy_param[key]]

        ax.margins(0.025)
        for param in fig8_params:
            ax.plot(xvals, bias_plt_data[key][param], '.-', lw=1,
                    label=fig8_param_labels[param], color=fig8_colors[param])

        ax.text(0.3, 0.9, key.upper(), transform=ax.transAxes, **txt_kwargs)
        ax.legend(fontsize=14)
        ax.set_xticks(fig8_xticks[key])
        ax.set_xlabel(fig8_xlabels[key], fontsize=16)
        ax.set_ylabel('Bias / Error', fontsize=16)
        ax.tick_params(labelsize=12)
        ax.grid(alpha=0.25)

    plotting_utilities.save_figure('fig8', enabled=SAVE_FIGURE)
    plt.show()


if __name__ == '__main__':
    main()
