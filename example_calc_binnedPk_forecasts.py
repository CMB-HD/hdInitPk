"""One binned-P(k) Fisher forecast, for example

This is the smallest complete example of what `hdinitpk` adds to
`hdfisher`: it calculates the Fisher derivatives for a binned primordial
power spectrum, assembles one Fisher matrix from them, and prints the
forecasted errors. The configuration here is the SO-like binned-P(k)
forecast of Cheslog et al. (2026), the `so_binned_pk_lensed_7bins`
matrix in `hdinitpk/data/fisher_matrices`.

`hdinitpk.hdinitPkfisher.Fisher` is a drop-in replacement for
`hdfisher.fisher.Fisher` that adds the `binned_pk`, `bin_edges`, `ksz`,
and `pk_frac_step` options; everything else below is the ordinary hdfisher
workflow. Instead of a power-law P(k) with a single amplitude, spectral
index, and running, the spectrum is a set of independent bin amplitudes
`Pk1` ... `Pk11`, and the derivative with respect to `Pk<n>` is taken by
perturbing the spectrum inside k bin n only.

Run it under MPI to spread the parameter variations over several tasks:

    mpirun -n 6 python example_calc_binnedPk_forecasts.py

All nine configurations in the paper, and all twenty Fisher matrices built
from them, are in `run_hdInitPk_forecasts.py`;
`run_hdInitPk_forecasts.ipynb` walks through the same thing
interactively. This file is the one-configuration version, for reading.
"""
import os

import numpy as np
from hdfisher import fisher as hdfisher_fisher
from hdfisher import mpi

import hdinitpk
from hdinitpk import hdinitPkfisher


# Where the derivatives go. They land under
# `hdinitpk/data/user_generated_data/fisher_derivs`, alongside everything
# else this repository generates; set HDINITPK_USER_DATA to move that
# somewhere else, if you prefer.
fisher_dir = hdinitpk.user_data_path('fisher_derivs', 'so_binned_pk')

# The fiducial parameters and the step sizes ship with the package, in
# `hdinitpk/data/fisher_fid_params` and `hdinitpk/data/fisher_steps`.
# Between them they define what gets varied: the fiducial file gives the
# 11 bin centers (`k1` ... `k11`), the 11 fiducial bin amplitudes
# (`Pk1` ... `Pk11`), `nkbins`, and `effective_ns_for_nonlinear`; the
# steps file gives a step size for each of them, plus the cosmological
# parameters marginalized over.
param_file = hdinitpk.data_path('fisher_fid_params',
                                'hd_binned_pk_fiducial_params.yaml')
steps_file = hdinitpk.data_path('fisher_steps',
                                'binned_pk_steps_5_percent.yaml')

# `binned_pk=True` also needs the bin EDGES (12 of them for 11 bins),
# which is what tells the theory code where to stop and start perturbing
# the spectrum. The centers in the parameter file are the arithmetic
# midpoints of these; both are tabulated in the binning file (Table I).
k_bins = np.loadtxt(
    hdinitpk.data_path('binning', 'hd_pk_wavenumbers_11bins.txt'),
    usecols=(1, 2))
bin_edges = np.append(k_bins[:, 0], k_bins[-1, 1])

# The one thing that is easy to get wrong: `pk_frac_step` is the fraction
# by which the spectrum inside the varied bin is perturbed, while the
# denominator of the finite difference comes from the step given for
# `Pk<n>` in the steps file. The two MUST agree, so this 0.05 goes with
# the `..._5_percent.yaml` above. (A mismatch raises a warning from
# `hdinitpk.theory.build_binned_pk_transfer` rather than an error, and the
# derivatives come out wrong) Rerunning
# with the 1% and 10% steps files (and 0.01 and 0.10 here) is how the
# derivatives in the paper were checked for convergence.
pk_frac_step = 0.05

fisherlib = hdinitPkfisher.Fisher(
    fisher_dir,
    overwrite=True,           # discards anything already in fisher_dir
    exp='so',                 # SO-like noise and sky coverage
    use_H0=True,              # marginalize over H0 rather than theta
    use_class=False,          # binned P(k) goes through CAMB and hd_pk
    hd_data_version='v1.2',   # the mock data version, from hdMockData
    binned_pk=True,
    bin_edges=bin_edges,
    pk_frac_step=pk_frac_step,
    ksz=False,                # no kSZ template in this configuration
    fisher_steps_file=steps_file,
    param_file=param_file)

# The expensive part: two CAMB calls per parameter, for every parameter in
# the steps file. Takes hours without MPI. The
# derivatives are written into `fisher_dir` along with copies of the
# fiducial-parameter and step-size files, so a later `Fisher` pointed at
# the same directory reads them back without recomputing.
fisherlib.calculate_fisher_derivs()
mpi.comm.barrier()

# Fisher matrices, which are comparively fast to make. `get_fisher` assembles the matrix from the
# derivatives and the covariance, and returns it together with the list of
# parameters it is indexed by. The paper's SO forecast keeps the first
# seven bin amplitudes and marginalizes over the background parameters;
# the four highest-k bins are past what SO constrains. `priors` adds the
# tau prior as an independent Gaussian, and `with_desi=False` leaves out
# the DESI BAO Fisher, matching the shipped matrix of the same name.
if mpi.rank == 0:
    params = [f'Pk{i + 1}' for i in range(7)] + ['H0', 'tau', 'ombh2', 'omch2']

    matrix, matrix_params = fisherlib.get_fisher(
        cmb_type='lensed', priors={'tau': 0.005}, with_desi=False,
        params=params, use_H0=True, save=False)

    # Marginalized 1-sigma errors: the square root of the diagonal of the
    # inverse, as a dict keyed by parameter name.
    errors = hdfisher_fisher.get_fisher_errors(matrix, matrix_params)

    print(f'\nSO binned P(k), {len(matrix_params)} parameters:')
    for param in matrix_params:
        print(f'  sigma({param}) = {errors[param]:.4g}')

    # Written in the same format as the matrices in
    # `hdinitpk/data/fisher_matrices`, so it can be compared against
    # `so_binned_pk_lensed_7bins.txt` there, or read back with
    # `hdfisher.fisher.load_fisher_matrix`.
    out_file = os.path.join(fisher_dir, 'so_binned_pk_lensed_7bins.txt')
    hdfisher_fisher.save_fisher_matrix(out_file, matrix, matrix_params)
    print(f'\nwrote {out_file}')
