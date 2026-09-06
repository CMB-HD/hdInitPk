"""Fisher matrix construction for the binned primordial power spectrum P(k)
and the kSZ template, used in Cheslog et. al. (2026).

`hdinitpk.hdinitPkfisher.Fisher` extends `hdfisher.fisher.Fisher` (via
the subclass seams added there) with the `binned_pk`, `ksz`, and
`bin_edges` options of the old merged hdfisher code, so code written against the merged version
only needs to change its import. All of the derivative bookkeeping, MPI
handling, covariance-matrix loading (including the temporary external
covmats), and Fisher assembly is inherited unchanged from hdfisher.
"""
import numpy as np
from hdfisher import fisher as hdfisher_fisher
from . import theory


class Fisher(hdfisher_fisher.Fisher):
    """Extends `hdfisher.fisher.Fisher` with the binned primordial power
    spectrum and the kSZ template. Accepts the same arguments as the base
    class, plus the following (matching the merged hdfisher code):

    Parameters
    ----------
    binned_pk : bool, default=False
        If `True`, the primordial power spectrum is described by an
        amplitude in each of a set of k-bins rather than by a power law.
        The `param_file` must then contain an `nkbins` entry, the bin
        centers (`k1`, `k2`, ...), the bin amplitudes (`Pk1`, `Pk2`, ... or
        `eneg2tauPk1`, ...), and an `effective_ns_for_nonlinear` entry, and
        the `fisher_steps_file` must contain a step size for each bin
        amplitude. Requires `bin_edges`, and requires `use_class=False`.
    bin_edges : str, array_like of float, or None, default=None
        The k-bin edges used when `binned_pk=True`, given either as an
        array or as the name (including the absolute path) of a text file
        that can be read with `numpy.loadtxt`. Required when
        `binned_pk=True`, and ignored otherwise.
    ksz : bool, default=False
        If `True`, a kinematic SZ template is added to the theory TT
        spectrum, scaled by an amplitude `A_ksz` and tilt `n_ksz`. Both
        must be given fiducial values in the `param_file`, and may be
        varied like any other parameter by including them in the
        `fisher_steps_file`.
    pk_frac_step : float, default=0.05
        The fractional step applied inside the varied k-bin when computing
        the perturbed spectra; see
        `hdinitpk.theory.build_binned_pk_transfer`. NOTE: this must be
        consistent with the bin-amplitude step sizes in the
        `fisher_steps_file` (a warning is issued if they disagree).

    Raises
    ------
    ValueError
        If both `use_class=True` and `binned_pk=True`; or if
        `binned_pk=True` but no `bin_edges` were given.
    """

    def __init__(self, fisher_dir, exp='hd', overwrite=False, param_file=None,
                 fisher_steps_file=None, feedback=False, fisher_params=None,
                 use_H0=False, hd_lmax=None, include_fg=True,
                 hd_data_version='latest', use_class=False,
                 pol_only_lensing=False,
                 binned_pk=False, bin_edges=None, ksz=False,
                 pk_frac_step=0.05):
        self.binned_pk = binned_pk
        if use_class and binned_pk:
            err_msg = "`use_class=True` and `binned_pk=True` cannot be combined: the binned primordial power spectrum is only implemented for CAMB. Set one of them to `False`."
            raise ValueError(err_msg)
        if bin_edges is None:
            self.bin_edges = None
        elif isinstance(bin_edges, str):
            self.bin_edges = np.loadtxt(bin_edges)
        else:
            self.bin_edges = np.asarray(bin_edges, dtype=float)
        if self.binned_pk and self.bin_edges is None:
            raise ValueError("`bin_edges` (array or file path) is required when `binned_pk=True`.")
        self.ksz = ksz
        self.pk_frac_step = pk_frac_step

        super().__init__(fisher_dir, exp=exp, overwrite=overwrite,
                         param_file=param_file,
                         fisher_steps_file=fisher_steps_file,
                         feedback=feedback, fisher_params=fisher_params,
                         use_H0=use_H0, hd_lmax=hd_lmax,
                         include_fg=include_fg,
                         hd_data_version=hd_data_version,
                         use_class=use_class,
                         pol_only_lensing=pol_only_lensing)


    def _make_theory(self, param, **cosmo_params):
        """Construct the `hdinitpk.theory.Theory` instance used to compute
        the spectra when `param` is varied away from its fiducial value.

        The varied parameter name is passed through as `varied_param` (the
        binned-Pk calculation uses it to work out which k-bin is being
        stepped), and, when `ksz=True`, the fiducial kSZ template
        parameters are added to `cosmo_params` unless they are the varied
        parameter (in which case the varied value is already there).
        """
        if self.ksz:
            if 'A_ksz' not in cosmo_params:
                cosmo_params['A_ksz'] = self.fid_params['A_ksz']
            if 'n_ksz' not in cosmo_params:
                cosmo_params['n_ksz'] = self.fid_params['n_ksz']
        return theory.Theory(self.lmax, self.theo_dir,
                             param_file=self.param_file, nlkk=self.nlkk,
                             recon_lmin=self.Lmin, recon_lmax=self.Lmax,
                             use_H0=self.use_H0, use_class=self.use_class,
                             binned_pk=self.binned_pk,
                             varied_param=param, ksz=self.ksz,
                             bin_edges=self.bin_edges,
                             pk_frac_step=self.pk_frac_step,
                             **cosmo_params)
