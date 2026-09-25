"""Fisher matrices for the binned primordial power spectrum and the kSZ
template used in Cheslog et. al. (2026).

`hdinitpk.hdinitPkfisher.Fisher` extends `hdfisher.fisher.Fisher` with the
`binned_pk`, `bin_edges`, `ksz`, and `pk_frac_step` options, and uses
`hdinitpk.theory.Theory` to calculate the spectra. The derivatives, the
Fisher matrices, the MPI handling, and the CMB-HD mock data (the
covariance matrices, the lensing reconstruction noise, and the binning,
all from hdMockData) are inherited from hdfisher as they are.
"""
import os
import numpy as np
from hdfisher import fisher as hdfisher_fisher
from hdfisher import config, dataconfig, utils
from . import data_path, theory


def fiducial_param_file(use_class_or_camb='camb', feedback=False):
    """The fiducial parameter file provided with hdinitpk for CAMB or
    CLASS, with or without the HMCode 2020 baryonic feedback model.

    The cosmological parameters and the accuracy settings match those in
    hdMockData (version v1.2), with the parameters varied in Cheslog et.
    al. (2026) added: the running of the spectral index, the Hubble
    constant, and (with feedback) the kSZ template amplitude and tilt.
    """
    code = 'class' if theory.use_class_flag(use_class_or_camb) else 'camb'
    feedback_info = '_feedback' if feedback else ''
    return data_path('fisher_fid_params', f'{code}_fiducial_params{feedback_info}.yaml')


def fiducial_fisher_steps_file(use_class_or_camb='camb', feedback=False, ksz=False):
    """The step-size file provided with hdinitpk for CAMB or CLASS, with or
    without the baryonic feedback parameter and the kSZ template
    parameters."""
    code = 'class' if theory.use_class_flag(use_class_or_camb) else 'camb'
    feedback_info = '_feedback' if feedback else ''
    ksz_info = '_ksz' if ksz else ''
    return data_path('fisher_steps', f'{code}_fiducial_step_sizes{feedback_info}{ksz_info}.yaml')


class Fisher(hdfisher_fisher.Fisher):
    """Calculate Fisher derivatives and matrices, with the option of a
    binned primordial power spectrum and a kSZ template.

    This takes the same arguments as `hdfisher.fisher.Fisher` (including
    `exp`, `hd_data_version`, and `pol_only_lensing`, which go to
    `hdfisher.fisher.FisherData`), except that the Boltzmann code is
    chosen by name with `use_class_or_camb` instead of the `use_class`
    flag, plus the ones below.

    Parameters
    ----------
    use_class_or_camb : str, default='camb'
        The Boltzmann code, `'camb'` or `'class'` (either case). hdfisher
        is given `use_class=True` for `'class'` and `False` for `'camb'`.
    binned_pk : bool, default=False
        If `True`, the primordial power spectrum is a set of amplitudes in
        k bins instead of a power law. See `hdinitpk.theory.Theory` for
        what the fiducial parameters must then contain. The step sizes
        must include one for each bin amplitude. Requires `bin_edges`,
        and only works with CAMB.
    bin_edges : str or array_like of float, default=None
        The k bin edges in Mpc^-1, either as an array or as the name of a
        text file that `numpy.loadtxt` can read. Needed when
        `binned_pk=True`.
    ksz : bool, default=False
        If `True`, add the kSZ template to the theory TT spectrum. Its
        amplitude `A_ksz` and tilt `n_ksz` must have fiducial values in
        the fiducial parameters, and can be varied like any other
        parameter by giving them step sizes.
    pk_frac_step : float, default=0.05
        The fraction by which the power inside a k bin is changed when its
        amplitude is varied. Must match the relative step size of the bin
        amplitudes in the step sizes; see `hdinitpk.theory.Theory`.
    theo_lmax : str or int or None, default='hd'
        The maximum multipole the theory is calculated to. `'hd'` is the
        maximum multipole of the CMB-HD mock data (24,000 for version
        v1.2), for every experiment, which is what Cheslog et. al. (2026)
        used for the SO-like forecasts as well. `None` keeps the value
        hdfisher sets for the experiment (5,000 for the SO-like and S4-like
        configurations). A number is used as it is.

    Notes
    -----
    The theory is calculated out to `theo_lmax` and then cut to the
    multipole range of the experiment.

    The parameter files provided with hdinitpk use the CAMB 1.x name
    `lens_margin`. The entry is renamed to `lens_output_margin` when CAMB
    2.0.0 or later is installed; see `hdinitpk.theory.camb_param_names`.

    If no fiducial parameters or step sizes are given (as `fiducial_params`
    or `param_file`, and `step_sizes` or `fisher_steps_file`), the copies
    saved in `fisher_dir` by an earlier run are used, or else the files
    provided with hdinitpk; see `fiducial_param_file` and
    `fiducial_fisher_steps_file`.
    """

    def __init__(self, fisher_dir, fiducial_params=None, step_sizes=None,
                 fisher_params=None, use_H0=False, feedback=False,
                 use_class_or_camb='camb', overwrite=False, param_file=None,
                 fisher_steps_file=None, binned_pk=False, bin_edges=None,
                 ksz=False, pk_frac_step=0.05, theo_lmax='hd', **kwargs):
        use_class = theory.use_class_flag(use_class_or_camb)
        # these are used by `_get_fid_params` and `_get_step_sizes`, which
        # the base class calls during its initialization
        self.use_class_or_camb = 'class' if use_class else 'camb'
        self.feedback = feedback
        self.binned_pk = binned_pk
        self.ksz = ksz
        self.pk_frac_step = pk_frac_step
        # the k bin edges are kept as `pk_bin_edges`, since hdfisher uses
        # `bin_edges` for the multipole bins of the covariance matrix
        if isinstance(bin_edges, str):
            bin_edges = np.loadtxt(bin_edges)
        self.pk_bin_edges = None if (bin_edges is None) else np.asarray(bin_edges, dtype=float)
        super().__init__(fisher_dir, fiducial_params=fiducial_params,
                         step_sizes=step_sizes, fisher_params=fisher_params,
                         use_H0=use_H0, feedback=feedback, use_class=use_class,
                         overwrite=overwrite, param_file=param_file,
                         fisher_steps_file=fisher_steps_file, **kwargs)
        # the paper calculates the theory to the CMB-HD lmax for every
        # experiment; `theo_lmax=None` keeps hdfisher's value instead
        if isinstance(theo_lmax, str):
            if theo_lmax.lower() != 'hd':
                raise ValueError(f"theo_lmax must be 'hd', None, or a number, not {theo_lmax!r}.")
            self.theo_lmax = self.data.hd_datalib.theo_lmax
        elif theo_lmax is not None:
            self.theo_lmax = int(theo_lmax)


    def _get_fid_params(self, fiducial_params=None, param_file=None, feedback=False):
        """The fiducial parameters: the ones passed, or else the copy saved
        in `fisher_dir` by an earlier run, or else the file provided with
        hdinitpk for this Boltzmann code. The CAMB settings are named for
        the installed version of CAMB."""
        if (fiducial_params is None) and (param_file is None):
            saved = self.param_file_name()
            if os.path.exists(saved) and (not self.overwrite):
                param_file = saved
            else:
                param_file = fiducial_param_file(use_class_or_camb=self.use_class_or_camb,
                                                 feedback=feedback)
        params = fiducial_params if (fiducial_params is not None) else param_file
        return theory.get_param_dict(params, use_class=self.use_class)


    def _get_step_sizes(self, step_sizes=None, fisher_steps_file=None):
        """The step sizes: the ones passed, or else the copy saved in
        `fisher_dir` by an earlier run, or else the file provided with
        hdinitpk for this Boltzmann code."""
        if (step_sizes is None) and (fisher_steps_file is None):
            saved = self.step_sizes_file_name()
            if os.path.exists(saved) and (not self.overwrite):
                fisher_steps_file = saved
            else:
                fisher_steps_file = fiducial_fisher_steps_file(
                    use_class_or_camb=self.use_class_or_camb, feedback=self.feedback,
                    ksz=self.ksz)
        return hdfisher_fisher.get_step_sizes_dict(steps_dict_or_file=step_sizes,
                                                   use_class=self.use_class,
                                                   fisher_steps_file=fisher_steps_file)


    def calculate_theory_for_deriv(self, param, value, use_H0=None):
        """Calculate and save the CMB and BAO theory when `param` has the
        given `value`, with the other parameters at their fiducial values.

        Same as `hdfisher.fisher.Fisher.calculate_theory_for_deriv`, but
        with the theory calculated by `hdinitpk.theory.Theory`.
        """
        if param is None:
            cosmo_params = {}
            step_direction = None
        else:
            cosmo_params = {param: value}
            step_direction = 'up' if (value > self.fid_params[param]) else 'down'
            if param == 'logA': # so that `logA` is used rather than `As`
                cosmo_params['As'] = None
        use_H0 = self.use_H0 if (use_H0 is None) else use_H0
        theolib = theory.Theory(self.theo_lmax, self.theo_dir,
                                params=self.fid_params, nlkk=self.nlkk,
                                recon_lmin=self.Lmin, recon_lmax=self.Lmax,
                                use_H0=use_H0, use_class_or_camb=self.use_class_or_camb,
                                binned_pk=self.binned_pk,
                                bin_edges=self.pk_bin_edges, ksz=self.ksz,
                                pk_frac_step=self.pk_frac_step,
                                hd_data_version=self.hd_data_version,
                                **cosmo_params)
        cmb_theo = theolib.get_theory(cmb_types=self.cmb_types, save=False,
                                      output_lmax=self.lmax)
        z = dataconfig.desi_redshifts()
        rs_dv = theolib.get_rs_dv(z, save=False)
        # save the theory
        header_info = f'{param} = {value}\n'
        for cmb_type in self.cmb_types:
            cmb_theo_fname = config.fisher_cmb_theo_fname(
                self.theo_dir, cmb_type, param, step_direction, use_H0=use_H0)
            utils.save_to_file(cmb_theo_fname, cmb_theo[cmb_type],
                               keys=config.theo_cols, extra_header_info=header_info)
        bao_theo_fname = config.fisher_bao_theo_fname(
            self.theo_dir, param, step_direction, use_H0=use_H0)
        utils.save_to_file(bao_theo_fname, {'z': z, 'rs_dv': rs_dv},
                           keys=['z', 'rs_dv'], extra_header_info=header_info)
