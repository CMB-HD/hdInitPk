"""Recalculate every Fisher derivative and Fisher matrix used in the paper.

There are nine Fisher configurations behind the forecasts in Cheslog et al.
(2026), and twenty Fisher matrices built from them. This script does both,
in two phases.

    CALCULATE_DERIVS    steps every parameter and writes the derivatives.
                        Hours per configuration. Run it under MPI.
    BUILD_MATRICES      assembles the twenty matrices from those
                        derivatives, saves them, and compares them against
                        the ones shipped in hdinitpk/data/fisher_matrices.
                        Seconds, and serial.

    mpirun -np 4 python run_hdInitPk_forecasts.py

Most of the forecasts in the paper are made with CLASS. The two CLASS
configurations, `class` for CMB-HD and `so` for SO-like, are the ones behind
Table IV and Figures 2 and 4. CAMB is used for the w0waCDM forecasts, where
we found the two codes disagree, for the baryonic feedback forecast of
Table V, and for the binned P(k) forecasts, which go through hdPk and so
are CAMB only. The CAMB nine-parameter configuration is kept for the
comparison with CLASS in Appendix B and for the accuracy study of
Appendix A.

`ONLY_JOBS` restricts a run to some of the configurations, which is how to
pick up after one of them times out. Phase 2 then builds only the matrices
those configurations back.

The step-size and fiducial-parameter files come from the installed hdinitpk
package, so this runs anywhere the package is importable. The output goes
to hdinitpk/data/user_generated_data, where the notebooks and the other
scripts look for it. The CMB-HD mock data (the lensing reconstruction noise
and the covariance matrices) comes from hdMockData.

`run_hdInitPk_forecasts.ipynb` walks through the same calculation
interactively, one configuration at a time.
"""
import os

from hdfisher import fisher as hdfisher_fisher
from hdfisher import mpi
from hdinitpk import hdinitPkfisher as hdinitpk_fisher
import hdinitpk


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
STEPS_DIR = hdinitpk.data_path('fisher_steps')
PARAMS_DIR = hdinitpk.data_path('fisher_fid_params')
REFERENCE_MATRIX_DIR = hdinitpk.data_path('fisher_matrices')

# Set HDINITPK_USER_DATA to put the output somewhere else.
DERIV_DIR = hdinitpk.user_data_path('fisher_derivs')
MATRIX_OUT_DIR = hdinitpk.user_data_path('fisher_matrices')

# The CLASS parameter files name the sBBN table, and CLASS opens that file
# itself, so the `/path/to/hdInitPk` placeholder in the shipped copies has
# to be replaced with a real path. `params` writes the resolved copy here.
RESOLVED_PARAMS_DIR = hdinitpk.user_data_path('resolved_params')
PLACEHOLDER = '/path/to/hdInitPk/hdinitpk/data'


def steps(name):
    return os.path.join(STEPS_DIR, name)


def params(name):
    """A fiducial-parameter file from the package, with the sBBN path
    filled in if the file has one. Only rank 0 writes the copy."""
    src = os.path.join(PARAMS_DIR, name)
    with open(src, encoding='utf-8') as f:
        text = f.read()
    if PLACEHOLDER not in text:
        return src
    out = os.path.join(RESOLVED_PARAMS_DIR, name)
    if mpi.rank == 0:
        with open(out, 'w', encoding='utf-8') as f:
            f.write(text.replace(PLACEHOLDER, hdinitpk.DATA_DIR))
    mpi.comm.barrier()
    return out


# ----------------------------------------------------------------------
# What to run
# ----------------------------------------------------------------------
# Every job below is constructed with overwrite=True, so phase 1 discards
# whatever derivatives are already in DERIV_DIR.
CALCULATE_DERIVS = False
BUILD_MATRICES = True

# Run only some of the configurations, by key from `JOBS`. None means all
# of them. For example
#
#     ONLY_JOBS = ['binned_pk_so', 'binned_pk_hd_feedback']
ONLY_JOBS = None


# ----------------------------------------------------------------------
# Shared settings
# ----------------------------------------------------------------------
use_H0 = True   # marginalize over H0 rather than theta

# The CMB-HD mock data version, passed to every job rather than left at
# hdfisher's default of 'latest'. v1.2 is what the paper used.
hd_data_version = 'v1.2'

TAU_PRIOR = {'tau': 0.005}
FEEDBACK_PRIOR = {'tau': 0.005, 'HMCode_logT_AGN': 0.0006 * 7.8}

# k bin edges [Mpc^-1] for the 11-bin binned-P(k) scheme.
PK_BIN_EDGES = [3.667637511098258835e-03, 8.505899211272888866e-03,
                1.972668268698882926e-02, 4.574966151931515734e-02,
                1.061015459285717805e-01, 2.460682259622835044e-01,
                5.706756795889451617e-01, 1.323497700691443235e+00,
                3.069424940269466440e+00, 7.118538595893407539e+00,
                1.650914836730800417e+01, 3.828763111167506139e+01]


# ----------------------------------------------------------------------
# Parameter sets
# ----------------------------------------------------------------------
nine_params = ['mnu', 'tau', 'logA', 'H0', 'ombh2', 'omch2', 'ns', 'nnu', 'nrun']
eight_params = ['mnu', 'tau', 'logA', 'H0', 'ombh2', 'omch2', 'ns', 'nnu']
feedback_params = nine_params + ['HMCode_logT_AGN', 'n_ksz', 'A_ksz']
w0wa_params = ['ombh2', 'omch2', 'H0', 'tau', 'logA', 'ns', 'nrun', 'w', 'wa']

binned_pk_params = ([f'Pk{i+1}' for i in range(11)]
                    + ['H0', 'tau', 'ombh2', 'omch2'])
binned_pk_params_so = ([f'Pk{i+1}' for i in range(7)]
                       + ['H0', 'tau', 'ombh2', 'omch2'])
binned_pk_params_feedback = (binned_pk_params
                             + ['HMCode_logT_AGN', 'n_ksz', 'A_ksz'])

ext_model_params = {
    'lcdm_nrun':          ['ombh2', 'omch2', 'H0', 'tau', 'logA', 'ns', 'nrun'],
    'lcdm_nrun_neff':     ['ombh2', 'omch2', 'H0', 'tau', 'logA', 'ns', 'nrun', 'nnu'],
    'lcdm_nrun_neff_mnu': ['ombh2', 'omch2', 'H0', 'tau', 'logA', 'ns', 'nrun',
                           'nnu', 'mnu'],
}

# The CLASS parameter files use CLASS names, so parameter lists and priors
# are translated on the way in. The matrices are saved with CLASS names,
# and the plotting notebook translates them back.
camb_to_class = {
    'ombh2': 'omega_b',   'omch2': 'omega_cdm',  'theta': 'theta_s_100',
    'tau': 'tau_reio',    'logA': 'ln_A_s_1e10', 'As': 'A_s',
    'ns': 'n_s',          'H0': 'H0',            'nnu': 'Neff',
    'mnu': 'sum_m_ncdm',  'nrun': 'alpha_s',     'omk': 'Omega_k',
    'w': 'w0_fld',        'wa': 'wa_fld',
    'HMCode_logT_AGN': 'log10T_heat_hmcode',
}


def to_class(params_):
    """CAMB parameter names to CLASS. Takes a list or a dict of priors."""
    if isinstance(params_, dict):
        return {camb_to_class.get(p, p): v for p, v in params_.items()}
    return [camb_to_class.get(p, p) for p in params_]


# ----------------------------------------------------------------------
# The nine configurations
# ----------------------------------------------------------------------
# Jobs with binned_pk or ksz need hdinitpk's Fisher, which subclasses
# hdfisher's. The rest use hdfisher's directly.
JOBS = {
    # CMB-HD, CLASS. Behind Table IV and Figures 2 and 4.
    'class': {
        'dirname': 'class_9param',
        'cls': hdfisher_fisher.Fisher,
        'kwargs': dict(
            overwrite=True, use_H0=use_H0, use_class=True,
            fisher_steps_file=steps('class_fiducial_step_sizes.yaml'),
            param_file=params('class_fiducial_params.yaml')),
    },

    # SO-like, CLASS. Backs the 9-parameter SO forecast and the three
    # extended models of Table IV.
    'so': {
        'dirname': 'so_9param',
        'cls': hdfisher_fisher.Fisher,
        'kwargs': dict(
            overwrite=True, use_H0=use_H0, use_class=True, exp='so',
            fisher_steps_file=steps('class_fiducial_step_sizes.yaml'),
            param_file=params('class_fiducial_params.yaml')),
    },

    # CMB-HD, CAMB. For the comparison with CLASS (Appendix B) and the
    # accuracy study (Appendix A).
    'camb': {
        'dirname': 'camb_9param',
        'cls': hdfisher_fisher.Fisher,
        'kwargs': dict(
            overwrite=True, use_H0=use_H0, use_class=False,
            fisher_steps_file=steps('camb_fiducial_step_sizes.yaml'),
            param_file=params('camb_fiducial_params.yaml')),
    },

    # 12 parameters, the nine plus baryonic feedback and the kSZ template.
    # Table V, with delensed spectra.
    'camb_feedback_ksz': {
        'dirname': 'camb_feedback_ksz',
        'cls': hdinitpk_fisher.Fisher,
        'kwargs': dict(
            overwrite=True, use_H0=use_H0, use_class=False,
            binned_pk=False, ksz=True,
            fisher_steps_file=steps('fiducial_step_sizes_feedback_ksz.yaml'),
            param_file=params('fiducial_params_feedback_ksz.yaml')),
    },

    # w0waCDM + alpha_s, with CAMB. Both need the w0wa step-size file,
    # since hdfisher's default steps do not vary w and wa. The parameter
    # file's lmax of 24000 takes precedence over the experiment's, so the
    # SO theory is computed to 24000 and then binned into SO's ell ranges,
    # as it was for the published forecasts.
    'w0wa_hd': {
        'dirname': 'hd_w0wa',
        'cls': hdfisher_fisher.Fisher,
        'kwargs': dict(
            overwrite=True, use_H0=use_H0, use_class=False,
            fisher_steps_file=steps('camb_w0wa_fiducial_step_sizes.yaml'),
            param_file=params('camb_w0wa_fiducial_params.yaml')),
    },

    'w0wa_so': {
        'dirname': 'so_w0wa',
        'cls': hdfisher_fisher.Fisher,
        'kwargs': dict(
            overwrite=True, use_H0=use_H0, use_class=False, exp='so',
            fisher_steps_file=steps('camb_w0wa_fiducial_step_sizes.yaml'),
            param_file=params('camb_w0wa_fiducial_params.yaml')),
    },

    # Binned P(k), with CAMB through hdPk.
    'binned_pk_hd': {
        'dirname': 'hd_binned_pk',
        'cls': hdinitpk_fisher.Fisher,
        'kwargs': dict(
            overwrite=True, use_H0=use_H0, use_class=False,
            binned_pk=True, ksz=False, bin_edges=PK_BIN_EDGES,
            fisher_steps_file=steps('binned_pk_steps_5_percent.yaml'),
            param_file=params('hd_binned_pk_fiducial_params.yaml')),
    },

    'binned_pk_so': {
        'dirname': 'so_binned_pk',
        'cls': hdinitpk_fisher.Fisher,
        'kwargs': dict(
            overwrite=True, use_H0=use_H0, use_class=False, exp='so',
            binned_pk=True, ksz=False, bin_edges=PK_BIN_EDGES,
            fisher_steps_file=steps('binned_pk_steps_5_percent.yaml'),
            param_file=params('hd_binned_pk_fiducial_params.yaml')),
    },

    'binned_pk_hd_feedback': {
        'dirname': 'hd_binned_pk_feedback',
        'cls': hdinitpk_fisher.Fisher,
        'kwargs': dict(
            overwrite=True, use_H0=use_H0, use_class=False,
            binned_pk=True, ksz=True, bin_edges=PK_BIN_EDGES,
            fisher_steps_file=steps('binned_pk_steps_5_percent_feedback.yaml'),
            param_file=params('hd_binned_pk_fiducial_params_feedback.yaml')),
    },
}


# ----------------------------------------------------------------------
# The twenty Fisher matrices
# ----------------------------------------------------------------------
# Each is (name, job, extra constructor kwargs, get_fisher kwargs). The
# extra kwargs only apply when a matrix is read back, such as the
# polarization-only lensing option and the parameter subset the SO
# extended-model library uses.
#
# Four of the six SO matrices come from the CLASS `so` job, so their
# parameter lists and priors go through `to_class` and the saved matrices
# carry CLASS names. `so_w0wa_lensed` and the binned P(k) forecast stay on
# CAMB. All the SO matrices use lensed spectra, which come from hdfisher's
# own `data/covmats` directory, since hdMockData carries CMB-HD data only.
MATRICES = [
    ('hd_class_lensed_9param', 'class', {},
     dict(cmb_type='lensed', priors=to_class(TAU_PRIOR), with_desi=True,
          params=to_class(nine_params))),
    ('hd_class_lensed_8param', 'class', {},
     dict(cmb_type='lensed', priors=to_class(TAU_PRIOR), with_desi=True,
          params=to_class(eight_params))),

    ('so_class_lensed_9param', 'so', {},
     dict(cmb_type='lensed', priors=to_class(TAU_PRIOR), with_desi=True,
          params=to_class(nine_params))),

    ('hd_camb_lensed_9param', 'camb', {},
     dict(cmb_type='lensed', priors=TAU_PRIOR, with_desi=True,
          params=nine_params)),
    ('hd_camb_delensed_9param', 'camb', {},
     dict(cmb_type='delensed', priors=TAU_PRIOR, with_desi=True,
          params=nine_params)),
    # 8 parameters, for the accuracy study behind Figure 8.
    ('hd_camb_lensed_8param', 'camb', {},
     dict(cmb_type='lensed', priors=TAU_PRIOR, with_desi=True,
          params=eight_params)),

    ('hd_camb_delensed_12param_feedback', 'camb_feedback_ksz', {},
     dict(cmb_type='delensed', priors=FEEDBACK_PRIOR, with_desi=True,
          params=feedback_params)),
    ('hd_camb_delensed_9param_from_feedback', 'camb_feedback_ksz', {},
     dict(cmb_type='delensed', priors=TAU_PRIOR, with_desi=True,
          params=nine_params)),

    ('hd_w0wa_lensed', 'w0wa_hd', {},
     dict(cmb_type='lensed', priors=TAU_PRIOR, with_desi=True,
          params=w0wa_params)),
    ('so_w0wa_lensed', 'w0wa_so', {},
     dict(cmb_type='lensed', priors=TAU_PRIOR, with_desi=True,
          params=w0wa_params)),

    ('hd_binned_pk_lensed_11bins', 'binned_pk_hd', {},
     dict(cmb_type='lensed', priors=TAU_PRIOR, with_desi=False,
          params=binned_pk_params)),
    ('so_binned_pk_lensed_7bins', 'binned_pk_so', {},
     dict(cmb_type='lensed', priors=TAU_PRIOR, with_desi=False,
          params=binned_pk_params_so)),

    ('hd_binned_pk_feedback_lensed', 'binned_pk_hd_feedback', {},
     dict(cmb_type='lensed', priors=FEEDBACK_PRIOR, with_desi=True,
          params=binned_pk_params_feedback)),
    # Same derivatives, polarization-only lensing reconstruction.
    ('hd_binned_pk_feedback_pol_lensed', 'binned_pk_hd_feedback',
     dict(pol_only_lensing=True),
     dict(cmb_type='lensed', priors=FEEDBACK_PRIOR, with_desi=True,
          params=binned_pk_params_feedback)),
]

# LCDM + alpha_s, then + N_eff, then + sum m_nu, for CMB-HD and SO.
for _model, _params in ext_model_params.items():
    MATRICES.append(
        (f'hd_ext_lensed_{_model}', 'class', {},
         dict(cmb_type='lensed', priors=to_class(TAU_PRIOR), with_desi=True,
              params=to_class(_params))))
    MATRICES.append(
        (f'so_ext_lensed_{_model}', 'so',
         dict(fisher_params=to_class(['ombh2', 'omch2', 'logA', 'ns', 'tau',
                                      'H0', 'nnu', 'mnu', 'nrun'])),
         dict(cmb_type='lensed', priors=to_class(TAU_PRIOR), with_desi=True,
              params=to_class(_params))))

ACTIVE_JOBS = list(ONLY_JOBS) if ONLY_JOBS else list(JOBS)
ACTIVE_MATRICES = [m for m in MATRICES if m[1] in ACTIVE_JOBS]


# ----------------------------------------------------------------------
# Phase 1: derivatives
# ----------------------------------------------------------------------
def job_dir(name):
    return os.path.join(DERIV_DIR, JOBS[name]['dirname'])


def run_job(name):
    """Build one Fisher library and calculate its derivatives."""
    job = JOBS[name]
    if mpi.rank == 0:
        print(f'\n=== {name} -> {job_dir(name)}', flush=True)
    kwargs = dict(job['kwargs'], hd_data_version=hd_data_version)
    fisherlib = job['cls'](job_dir(name), **kwargs)
    fisherlib.calculate_fisher_derivs()
    mpi.comm.barrier()


# ----------------------------------------------------------------------
# Phase 2: matrices
# ----------------------------------------------------------------------
def build_matrix(name, job_key, extra, get_kwargs):
    """One Fisher matrix, from derivatives already on disk."""
    job = JOBS[job_key]
    # overwrite=False here, so that reading the derivatives back does not
    # remove them.
    kwargs = dict(job['kwargs'], overwrite=False,
                  hd_data_version=hd_data_version, **extra)
    fisherlib = job['cls'](job_dir(job_key), **kwargs)
    return fisherlib.get_fisher(use_H0=use_H0, save=False, **get_kwargs)


def reference_errors(name):
    """1-sigma errors of the shipped matrix of the same name, or None."""
    path = os.path.join(REFERENCE_MATRIX_DIR, f'{name}.txt')
    if not os.path.isfile(path):
        return None
    return hdfisher_fisher.get_fisher_errors(
        *hdfisher_fisher.load_fisher_matrix(path))


def build_matrices():
    """Build the matrices, save them, and print the largest fractional
    difference between each one's errors and the shipped matrix."""
    os.makedirs(MATRIX_OUT_DIR, exist_ok=True)
    print(f'\n=== {len(ACTIVE_MATRICES)} Fisher matrices -> {MATRIX_OUT_DIR}',
          flush=True)
    for name, job_key, extra, get_kwargs in ACTIVE_MATRICES:
        matrix, matrix_params = build_matrix(name, job_key, extra, get_kwargs)
        hdfisher_fisher.save_fisher_matrix(
            os.path.join(MATRIX_OUT_DIR, f'{name}.txt'), matrix, matrix_params)

        reference = reference_errors(name)
        if reference is None:
            print(f'  {name:40s} saved', flush=True)
            continue
        errors = hdfisher_fisher.get_fisher_errors(matrix, matrix_params)
        worst = max(abs(errors[p] / reference[p] - 1)
                    for p in errors if p in reference)
        print(f'  {name:40s} saved, largest |sigma/sigma_paper - 1| = '
              f'{worst:.2e}', flush=True)


# ----------------------------------------------------------------------
# On reading the comparison
# ----------------------------------------------------------------------
# Agreement at 1e-6 or better means the derivatives reproduce the shipped
# matrix exactly. Differences below a percent are what recomputing with a
# different CAMB or CLASS build gives. 

if __name__ == '__main__':
    if CALCULATE_DERIVS:
        for job_name in ACTIVE_JOBS:
            run_job(job_name)
        mpi.comm.barrier()

    # Assembling the matrices is serial, so one rank does it.
    if BUILD_MATRICES and mpi.rank == 0:
        build_matrices()
