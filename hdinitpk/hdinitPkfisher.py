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


def fiducial_param_file(use_class=False, feedback=False):
    """The fiducial parameter file provided with hdinitpk for CAMB or
    CLASS, with or without the HMCode 2020 baryonic feedback model.

    The cosmological parameters and the accuracy settings match those in
    hdMockData (version v1.2), with the parameters varied in Cheslog et.
    al. (2026) added: the running of the spectral index, the Hubble
    constant, and (with feedback) the kSZ template amplitude and tilt.
    """
    code = 'class' if use_class else 'camb'
    feedback_info = '_feedback' if feedback else ''
    return data_path('fisher_fid_params', f'{code}_fiducial_params{feedback_info}.yaml')


def fiducial_fisher_steps_file(use_class=False, feedback=False, ksz=False):
    """The step-size file provided with hdinitpk for CAMB or CLASS, with or
    without the baryonic feedback parameter and the kSZ template
    parameters."""
    code = 'class' if use_class else 'camb'
    feedback_info = '_feedback' if feedback else ''
    ksz_info = '_ksz' if ksz else ''
    return data_path('fisher_steps', f'{code}_fiducial_step_sizes{feedback_info}{ksz_info}.yaml')


class Fisher(hdfisher_fisher.Fisher):
    """Calculate Fisher derivatives and matrices, with the option of a
    binned primordial power spectrum and a kSZ template.

    This takes the same arguments as `hdfisher.fisher.Fisher` (including
    `exp`, `hd_data_version`, and `pol_only_lensing`, which go to
    `hdfisher.fisher.FisherData`), plus the ones below.

    Parameters
    ----------
    binned_pk : bool, default=False
        If `True`, the primordial power spectrum is a set of amplitudes in
        k bins instead of a power law. See `hdinitpk.theory.Theory` for
        what the `param_file` must then contain. The `fisher_steps_file`
        must give a step size for each bin amplitude. Requires `bin_edges`,
        and only works with CAMB.
    bin_edges : str or array_like of float, default=None
        The k bin edges in Mpc^-1, either as an array or as the name of a
        text file that `numpy.loadtxt` can read. Needed when
        `binned_pk=True`.
    ksz : bool, default=False
        If `True`, add the kSZ template to the theory TT spectrum. Its
        amplitude `A_ksz` and tilt `n_ksz` must have fiducial values in the
        `param_file`, and can be varied like any other parameter by giving
        them step sizes in the `fisher_steps_file`.
    pk_frac_step : float, default=0.05
        The fraction by which the power inside a k bin is changed when its
        amplitude is varied. Must match the relative step size of the bin
        amplitudes in the `fisher_steps_file`; see
        `hdinitpk.theory.Theory`.

    Notes
    -----
    The theory is calculated out to the maximum multipole of the CMB-HD
    mock data (24,000 for version v1.2) for every experiment, and then cut
    to the multipole range of the experiment, as in Cheslog et. al. (2026).

    If no `param_file` or `fisher_steps_file` is given, the files provided
    with hdinitpk are used; see `fiducial_param_file` and
    `fiducial_fisher_steps_file`.
    """

    def __init__(self, fisher_dir, param_file=None, fisher_steps_file=None,
                 fisher_params=None, use_H0=False, use_class=False,
                 feedback=False, overwrite=False, binned_pk=False,
                 bin_edges=None, ksz=False, pk_frac_step=0.05, **kwargs):
        # these are used by `get_param_file` and `get_fisher_steps_file`,
        # which the base class calls during its initialization
        self.use_class = use_class
        self.binned_pk = binned_pk
        self.ksz = ksz
        self.pk_frac_step = pk_frac_step
        if isinstance(bin_edges, str):
            bin_edges = np.loadtxt(bin_edges)
        self.bin_edges = None if (bin_edges is None) else np.asarray(bin_edges, dtype=float)
        super().__init__(fisher_dir, param_file=param_file,
                         fisher_steps_file=fisher_steps_file,
                         fisher_params=fisher_params, use_H0=use_H0,
                         feedback=feedback, overwrite=overwrite,
                         use_class=use_class, **kwargs)
        self.use_class = use_class
        # CLASS does not calculate delensed spectra
        if use_class:
            self.cmb_types = ['lensed', 'unlensed']
        # the theory is calculated to the CMB-HD lmax for every experiment
        self.theo_lmax = self.data.hd_datalib.theo_lmax


    def get_param_file(self, feedback=False):
        """The fiducial parameter file to use when none was given: the copy
        saved in `fisher_dir` by an earlier run, or else the file provided
        with hdinitpk for this Boltzmann code."""
        input_param_file = os.path.join(self.fisher_dir, 'fiducial_params.yaml')
        if os.path.exists(input_param_file) and (not self.overwrite):
            return input_param_file
        return fiducial_param_file(use_class=self.use_class, feedback=feedback)


    def get_fisher_steps_file(self, feedback=False):
        """The step-size file to use when none was given: the copy saved in
        `fisher_dir` by an earlier run, or else the file provided with
        hdinitpk for this Boltzmann code."""
        input_steps_file = os.path.join(self.fisher_dir, 'step_sizes.yaml')
        if os.path.exists(input_steps_file) and (not self.overwrite):
            return input_steps_file
        return fiducial_fisher_steps_file(use_class=self.use_class,
                                          feedback=feedback, ksz=self.ksz)


    def calculate_theory_for_deriv(self, param, value):
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
        theolib = theory.Theory(self.theo_lmax, self.theo_dir,
                                param_file=self.param_file, nlkk=self.nlkk,
                                recon_lmin=self.Lmin, recon_lmax=self.Lmax,
                                use_H0=self.use_H0, use_class=self.use_class,
                                binned_pk=self.binned_pk,
                                bin_edges=self.bin_edges, ksz=self.ksz,
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
                self.theo_dir, cmb_type, param, step_direction, use_H0=self.use_H0)
            utils.save_to_file(cmb_theo_fname, cmb_theo[cmb_type],
                               keys=config.theo_cols, extra_header_info=header_info)
        bao_theo_fname = config.fisher_bao_theo_fname(
            self.theo_dir, param, step_direction, use_H0=self.use_H0)
        utils.save_to_file(bao_theo_fname, {'z': z, 'rs_dv': rs_dv},
                           keys=['z', 'rs_dv'], extra_header_info=header_info)
