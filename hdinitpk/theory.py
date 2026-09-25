"""Theory spectra for the binned primordial power spectrum and the kSZ
template used in Cheslog et. al. (2026).

`hdinitpk.theory.Theory` extends `hdfisher.theory.Theory` in two ways.

- The primordial power spectrum can be a set of amplitudes in k bins
  (`Pk1`, `Pk2`, ... or `eneg2tauPk1`, ...) instead of a power law. The
  CMB and lensing spectra are then calculated with CAMB and `hd_pk`,
  whose `calculate_clkk` integral gives the lensing spectrum. The
  delensed spectra are made from it with the lensing reconstruction
  noise passed in, as hdfisher does for a power law.
- A kinematic SZ template, loaded from hdMockData, can be added to the
  theory TT spectrum, with an amplitude `A_ksz` and a tilt `n_ksz`.

Everything else (the spectra for a power-law P(k) with CAMB or CLASS, the
loading and saving of the spectra, and the BAO theory) is done by hdfisher.
"""
import os
import re
import warnings
import numpy as np
import camb
from scipy.interpolate import interp1d
from hd_mock_data import hd_data
from hdfisher import theory as hdtheory
from hdfisher import utils, config
from hd_pk import cmb_from_pk


# the order CAMB returns the CMB spectra in
CAMB_SPECTRA = ['tt', 'ee', 'bb', 'te']

# entries of the parameter file that neither CAMB nor CLASS takes: the kSZ
# template parameters, and the description of the binned P(k) (the number
# of bins, the bin centers `k<n>`, the bin amplitudes `Pk<n>` or
# `eneg2tauPk<n>`, and the effective spectral index used for the
# non-linear correction)
KSZ_PARAMS = ['A_ksz', 'n_ksz']
BINNED_PK_KEYS = ['nkbins', 'effective_ns_for_nonlinear']
BIN_PARAM = re.compile(r'^(k|Pk|eneg2tauPk)(\d+)$')


# ----- the kSZ template -----

def get_cl_ksz(lmax, A_ksz, n_ksz, hd_data_version='latest'):
    """The kSZ template from hdMockData, scaled by the amplitude `A_ksz`
    and the tilt `n_ksz` about ell = 3000, from ell = 0 to `lmax`.

    Parameters
    ----------
    lmax : int
        The maximum multipole of the returned spectrum.
    A_ksz, n_ksz : float
        The amplitude and the tilt of the template, so that
        cl_ksz(ell) = A_ksz * (ell / 3000)^n_ksz * template(ell).
    hd_data_version : str, default='latest'
        The version of the CMB-HD mock data the template is read from.

    Returns
    -------
    cl_ksz : array_like of float
        The scaled template, in uK^2, starting at ell = 0.
    """
    ells, template = hd_data.HDMockData(version=hd_data_version).cl_ksz(output_lmax=lmax)
    with np.errstate(divide='ignore', invalid='ignore'):
        cl_ksz = A_ksz * (ells / 3000)**n_ksz * template
    cl_ksz[:2] = 0
    return cl_ksz


# ----- the Boltzmann code and the parameter file -----

def use_class_flag(use_class_or_camb):
    """The `use_class` flag hdfisher takes, from the name of the Boltzmann
    code: `True` for `'class'` and `False` for `'camb'`, in either case.
    Anything else raises a `ValueError`."""
    code = str(use_class_or_camb).lower()
    if code not in ['camb', 'class']:
        raise ValueError(f"use_class_or_camb must be 'camb' or 'class', not {use_class_or_camb!r}.")
    return code == 'class'


def camb_param_names(params):
    """A copy of the `params` dict with the CAMB settings named the way the
    installed version of CAMB names them.

    CAMB 2.0.0 renamed `lens_margin` to `lens_output_margin`. The parameter
    files provided with hdinitpk use the old name. This renames it for CAMB
    2.0.0 or later, and renames the new one back for older versions, so
    either name works with either version. It is the same check hdMockData
    makes to choose between its CAMB parameter files.

    Parameters
    ----------
    params : dict
        A dictionary of parameter names and values.

    Returns
    -------
    dict
        A copy with the renamed entry. If both names are present, the one
        the installed CAMB takes is kept.
    """
    old, new = 'lens_margin', 'lens_output_margin'
    use_new = int(camb.__version__.split('.')[0]) >= 2
    wanted, unwanted = (new, old) if use_new else (old, new)
    params = dict(params)
    if unwanted in params:
        value = params.pop(unwanted)
        if wanted not in params:
            params[wanted] = value
    return params


def get_param_dict(params=None, use_class=False):
    """The dictionary of parameter names and values that hdfisher builds
    from a dictionary or a YAML file (with its parameter aliases added),
    with the CAMB settings renamed for the installed version of CAMB; see
    `camb_param_names`.

    Parameters
    ----------
    params : str or dict or None, default=None
        A dictionary of parameter names and values, or the path to a YAML
        file holding them. If `None`, hdfisher's fiducial parameters.
    use_class : bool, default=False
        Whether the parameters are for CLASS instead of CAMB.

    Returns
    -------
    dict
        The parameter names and values.
    """
    param_dict = hdtheory.get_param_dict(param_dict_or_file=params, use_class=use_class)
    if (param_dict is not None) and (not use_class):
        param_dict = camb_param_names(param_dict)
    return param_dict


def class_file_path(path):
    """The form in which to give CLASS the path of a data file it opens
    itself, such as the BBN table (`'sBBN file'`).

    Up to version 3.3.1, CLASS opens the path it is given as it is, so an
    absolute path works. From version 3.3.2 on, CLASS puts its `base_path`
    in front of the path of every data file it opens, and classy sets
    `base_path` to the directory classy is installed in. An absolute path
    then ends up inside that directory, and CLASS cannot find the file. For
    those versions the path is given relative to the classy directory,
    starting with '/', which is the form CLASS's own default paths take
    (for example '/external/bbn/sBBN_2025.dat').

    Parameters
    ----------
    path : str
        The path to the file, absolute or relative to the current directory.

    Returns
    -------
    str
        The path to pass to CLASS. It is returned unchanged if classy is not
        installed, or if its version is older than 3.3.2.
    """
    try:
        import classy
    except ImportError:
        return path
    version = str(getattr(classy, '__version__', '')).lstrip('v')
    try:
        version = tuple(int(n) for n in version.split('.')[:3])
    except ValueError:
        return path
    if version < (3, 3, 2):
        return path
    # the directory classy uses as `base_path`, found the same way classy does
    try:
        import importlib.resources
        base_path = os.path.abspath(str(importlib.resources.files('classy')))
    except Exception:
        base_path = os.path.dirname(os.path.abspath(classy.__file__))
    relative = os.path.relpath(os.path.abspath(path), base_path)
    return '/' + relative.replace(os.sep, '/')


def remove_extra_params(params):
    """Returns a copy of the `params` dict without the entries that
    describe the binned P(k) or the kSZ template, which neither CAMB nor
    CLASS takes."""
    return {key: value for key, value in params.items()
            if (key not in KSZ_PARAMS + BINNED_PK_KEYS)
            and (BIN_PARAM.match(key) is None)}


# ----- the binned P(k) -----

def binned_pk_camb_params(camb_params, bin_edges=None, varied_bin=None,
                          step=0.0, effective_ns=None):
    """A copy of `camb_params` whose primordial power spectrum is the
    fiducial power law, tabulated on a fine k grid, with the power inside
    one k bin scaled by `1 + step`.

    The fiducial power law (`As`, `ns`, and the pivot scale) is read from
    `camb_params.InitPower`. The tabulated spectrum is passed to CAMB even
    when no bin is varied, so that the fiducial and the varied spectra are
    calculated the same way.

    Parameters
    ----------
    camb_params : camb.model.CAMBparams
        The CAMB parameters with the fiducial power-law primordial spectrum.
    bin_edges : array_like of float, default=None
        The k bin edges in Mpc^-1: element `i` is the lower edge of bin `i`
        and element `i + 1` its upper edge. Only needed if a bin is varied.
    varied_bin : int, default=None
        The index of the bin to scale, counting from zero, or `None` to
        leave the power law as it is.
    step : float, default=0.0
        The fractional change of the power inside the varied bin.
    effective_ns : float, default=None
        The spectral index CAMB uses for the non-linear correction. By
        default, the `ns` of the power law.

    Returns
    -------
    pars : camb.model.CAMBparams
        The CAMB parameters with the tabulated primordial spectrum.
    pk_transfer_function : callable or None
        A function of (k, z) that returns the factor applied to the power
        law, for use with `hd_pk.cmb_from_pk.calculate_clkk`, or `None` if
        no bin is varied.
    """
    As = camb_params.InitPower.As
    ns = camb_params.InitPower.ns
    k_pivot = camb_params.InitPower.pivot_scalar
    if effective_ns is None:
        effective_ns = ns
    k_fine = np.logspace(-6, 2, 10000)
    pk_fine = As * (k_fine / k_pivot)**(ns - 1)
    pk_transfer_function = None
    if varied_bin is not None:
        kmin = bin_edges[varied_bin]
        kmax = bin_edges[varied_bin + 1]

        def pk_transfer_function(k, z):
            k = np.atleast_1d(np.asarray(k, dtype=float))
            factor = np.ones_like(k)
            factor[(k >= kmin) & (k <= kmax)] = 1 + step
            return factor

        pk_fine = pk_fine * pk_transfer_function(k_fine, None)
    pk_func = interp1d(k_fine, pk_fine, kind='linear', bounds_error=False,
                       fill_value=(pk_fine[0], pk_fine[-1]))
    pars = camb_params.copy()
    pars.set_initial_power_function(pk_func, effective_ns_for_nonlinear=effective_ns,
                                    kmin=1e-6, kmax=100)
    return pars, pk_transfer_function


def binned_pk_spectra(lmax, camb_results, clkk, cmb_types=['lensed', 'unlensed'],
                      nlkk=None, Lmin=None, Lmax=None):
    """The lensed, unlensed, and delensed CMB spectra for a binned
    primordial power spectrum, given the CAMB results for that spectrum
    and its lensing power spectrum. This follows
    `hd_pk.cmb_from_pk.calculate_theory_spectra`, but takes the lensing
    spectrum and the lensing reconstruction noise as arguments instead of
    calculating and loading them itself, so the lensing spectrum is
    integrated once and the delensing uses the noise of the experiment
    being forecast.

    Parameters
    ----------
    lmax : int
        The maximum multipole of the returned spectra.
    camb_results : camb.results.CAMBdata
        The CAMB results for the binned primordial spectrum.
    clkk : array_like of float
        The lensing convergence power spectrum for the binned primordial
        spectrum, C_L^kk = [L(L+1)]^2 C_L^phiphi / 4, from L = 0 to at
        least the maximum multipole CAMB calculated.
    cmb_types : list of str, default=['lensed', 'unlensed']
        Any of `'lensed'`, `'unlensed'`, and `'delensed'`.
    nlkk : array_like of float, default=None
        The lensing reconstruction noise, in the same convention as `clkk`,
        from L = 0 to at least `Lmax`. Needed for the delensed spectra.
    Lmin, Lmax : int, default=None
        The range of lensing multipoles used in the reconstruction. Needed
        for the delensed spectra. `Lmax` defaults to `lmax`.

    Returns
    -------
    theo : dict of dict of array_like of float
        The spectra for each requested CMB type, with keys `'ells'`,
        `'tt'`, `'te'`, `'ee'`, `'bb'`, and `'kk'`, starting at ell = 0.
        The CMB spectra are C_ell's in uK^2.
    """
    ells = np.arange(lmax + 1)
    theo = {}
    for cmb_type in cmb_types:
        if cmb_type == 'lensed':
            cl = clkk
        elif cmb_type == 'unlensed':
            cl = np.zeros_like(clkk)
        elif cmb_type == 'delensed':
            if (nlkk is None) or (Lmin is None):
                raise ValueError("The delensed spectra need the lensing reconstruction noise `nlkk` and the minimum multipole `Lmin` of the reconstruction.")
            if Lmax is None:
                Lmax = lmax
            # the lensing power left after the reconstruction is removed
            cl = hdtheory.get_residual_lensing(clkk, nlkk, Lmin, Lmax, len(clkk) - 1)
        else:
            raise ValueError(f"cmb_types must contain only 'lensed', 'unlensed', or 'delensed', not {cmb_type!r}.")
        cls = camb_results.get_lensed_cls_with_spectrum(
            cl * 4 / (2 * np.pi), lmax=lmax, CMB_unit='muK', raw_cl=True)
        theo[cmb_type] = {'ells': ells.copy(), 'kk': clkk[:lmax+1].copy()}
        for i, s in enumerate(CAMB_SPECTRA):
            theo[cmb_type][s] = cls[:, i].copy()
    return theo


# ----- the Theory class -----

class Theory(hdtheory.Theory):
    """Calculate the CMB and lensing power spectra and the BAO theory, with
    the option of a binned primordial power spectrum and a kSZ template.

    This takes the same arguments as `hdfisher.theory.Theory`, except that
    the Boltzmann code is chosen by name with `use_class_or_camb` instead
    of the `use_class` flag, plus the ones below. Without them it does
    exactly what hdfisher does.

    Parameters
    ----------
    use_class_or_camb : str, default='camb'
        The Boltzmann code, `'camb'` or `'class'` (either case). hdfisher
        is given `use_class=True` for `'class'` and `False` for `'camb'`.
    binned_pk : bool, default=False
        If `True`, the primordial power spectrum is a set of amplitudes in
        k bins instead of a power law. The parameters (`params` or
        `param_file`) must then give the fiducial power law (`logA` or
        `As`, and `ns`), the number of bins (`nkbins`), the bin centers
        (`k1`, `k2`, ...), the bin amplitudes (`Pk1`, `Pk2`, ... or
        `eneg2tauPk1`, ...), and `effective_ns_for_nonlinear`. Requires
        `bin_edges`, and only works with CAMB.
    bin_edges : str or array_like of float, default=None
        The k bin edges in Mpc^-1, either as an array or as the name of a
        text file that `numpy.loadtxt` can read. Needed when
        `binned_pk=True`.
    ksz : bool, default=False
        If `True`, add the kSZ template to the theory TT spectrum. Its
        amplitude `A_ksz` and tilt `n_ksz` are taken from `cosmo_params`
        if given there, and from the parameters otherwise.
    pk_frac_step : float, default=0.05
        The fraction by which the power inside a k bin is changed when the
        amplitude of that bin is varied. This must match the relative step
        size given for the bin amplitudes in the Fisher step-size file,
        since the finite difference divides by that step. A warning is
        issued if the two disagree.
    hd_data_version : str, default='latest'
        The version of the CMB-HD mock data, used for the kSZ template and
        the BBN table CLASS reads.

    Notes
    -----
    When a bin amplitude `Pk<n>` is passed in `cosmo_params` with a value
    different from its fiducial value, the power inside bin `n` is scaled
    by `1 + pk_frac_step` (or `1 - pk_frac_step`, if the value is below
    the fiducial one). The spectra are then calculated with CAMB and
    `hd_pk`, following MacInnis & Sehgal (2024). The lensing spectrum is
    the `hd_pk` integral over the matter power spectrum, done once, and
    the delensed spectra are made from it with the lensing reconstruction
    noise `nlkk` passed here, in the same way hdfisher delenses a power
    law. Only one bin amplitude can be varied at a time.

    The binned P(k) and kSZ entries of the parameters are taken out before
    the rest is handed to hdfisher, which sets up CAMB or CLASS as usual,
    with the spectra calculated to `lmax + 500` and kept to `lmax`.

    CAMB 2.0.0 renamed the `lens_margin` setting to `lens_output_margin`.
    The setting is renamed to match the installed version of CAMB (see
    `camb_param_names`), so the parameter files work with either. The path
    of the BBN table is given to CLASS in the form its version expects; see
    `class_file_path`.
    """

    def __init__(self, lmax, output_dir, output_root=None, params=None,
                 nlkk=None, recon_lmin=None, recon_lmax=None, use_H0=False,
                 use_class_or_camb='camb', param_file=None, binned_pk=False,
                 bin_edges=None, ksz=False, pk_frac_step=0.05,
                 hd_data_version='latest', **cosmo_params):
        use_class = use_class_flag(use_class_or_camb)
        self.use_class_or_camb = 'class' if use_class else 'camb'
        self.binned_pk = binned_pk
        self.pk_frac_step = pk_frac_step
        self.hd_data_version = hd_data_version
        if isinstance(bin_edges, str):
            bin_edges = np.loadtxt(bin_edges)
        self.pk_bin_edges = None if (bin_edges is None) else np.asarray(bin_edges, dtype=float)
        # which bin is varied (counting from zero) and by how much; filled
        # in by `_setup_boltzmann_params` when a bin amplitude is passed
        self.varied_bin = None
        self.pk_step = 0.0
        # the fiducial values, used for the kSZ template and to tell which
        # way a bin amplitude has been varied:
        if params is None:
            params = param_file
        self.fid_params = get_param_dict(params, use_class=use_class)
        self.ksz = ksz
        self.A_ksz = cosmo_params.pop('A_ksz', self.fid_params.get('A_ksz'))
        self.n_ksz = cosmo_params.pop('n_ksz', self.fid_params.get('n_ksz'))
        # the CAMB results and lensing spectrum for the binned P(k), filled
        # in the first time they are needed:
        self._binned_pk_results = None
        # the base class sets up the Boltzmann code parameters through
        # `_setup_boltzmann_params` below, which uses the attributes above
        super().__init__(lmax, output_dir, output_root=output_root,
                         params=params, nlkk=nlkk, recon_lmin=recon_lmin,
                         recon_lmax=recon_lmax, use_H0=use_H0,
                         use_class=use_class, **cosmo_params)


    def _setup_boltzmann_params(self, params=None, use_H0=False, **cosmo_params):
        """Set up `self.camb_params` or `self.class_params` with hdfisher,
        after taking out the binned P(k) and kSZ entries that the Boltzmann
        code does not take. For a binned P(k), also work out which bin (if
        any) is varied, and in which direction."""
        if self.use_class and self.binned_pk:
            raise ValueError("The binned P(k) is only calculated with CAMB; set `use_class_or_camb='camb'`.")
        params = remove_extra_params(self.fid_params)
        overrides = remove_extra_params(cosmo_params)
        if self.use_class:
            # CLASS reads the BBN table itself, so it needs its location
            overrides.setdefault(
                'sBBN file', hd_data.HDMockData(version=self.hd_data_version).class_sbbn_file)
        else:
            overrides = camb_param_names(overrides)
        super()._setup_boltzmann_params(params=params, use_H0=use_H0, **overrides)
        if self.use_class:
            # in the form the installed version of CLASS expects
            self.class_params['sBBN file'] = class_file_path(self.class_params['sBBN file'])
        if self.binned_pk:
            self._find_varied_bin(cosmo_params)


    def _find_varied_bin(self, cosmo_params):
        """Set `varied_bin` and `pk_step` from the bin amplitude in
        `cosmo_params` (if there is one) whose value differs from its
        fiducial value. Only one bin can be varied at a time."""
        varied = []
        for name, value in cosmo_params.items():
            match = BIN_PARAM.match(name)
            if (match is None) or (match.group(1) == 'k'):
                continue
            if value != self.fid_params[name]:
                varied.append((name, value, int(match.group(2)) - 1))
        if len(varied) > 1:
            names = ', '.join(name for name, _, _ in varied)
            raise ValueError(f"Only one bin amplitude can be varied at a time, but {names} were all changed from their fiducial values.")
        if not varied:
            return
        name, value, self.varied_bin = varied[0]
        fid = self.fid_params[name]
        self.pk_step = self.pk_frac_step if (value > fid) else -self.pk_frac_step
        # the finite difference in the Fisher derivative divides by the
        # step in the step-size file, so the two steps must agree
        implied_step = abs(value / fid - 1)
        if abs(implied_step - self.pk_frac_step) > 1e-3 * self.pk_frac_step:
            msg = (f"The value of {name} is {implied_step:.6f} of its fiducial value away from it, but the power in that bin is changed by pk_frac_step = {self.pk_frac_step}. The derivative with respect to {name} will be off by the ratio of the two. Set pk_frac_step to the relative step size of the bin amplitudes in the step-size file.")
            warnings.warn(msg)


    def get_binned_pk_results(self):
        """The CAMB results and the lensing power spectrum for the binned
        primordial spectrum, calculated once and kept. The lensing spectrum
        is the `hd_pk` integral over the matter power spectrum, which is
        slow, so it is done here once and used for the lensed and the
        delensed spectra alike."""
        if self._binned_pk_results is None:
            effective_ns = self.fid_params.get('effective_ns_for_nonlinear')
            pars_fid, _ = binned_pk_camb_params(self.camb_params, effective_ns=effective_ns)
            if self.varied_bin is None:
                pars, pk_transfer_function = pars_fid, None
            else:
                pars, pk_transfer_function = binned_pk_camb_params(
                    self.camb_params, bin_edges=self.pk_bin_edges,
                    varied_bin=self.varied_bin, step=self.pk_step,
                    effective_ns=effective_ns)
            results = camb.get_results(pars)
            clkk = cmb_from_pk.calculate_clkk(pars_fid, pk_transfer_function=pk_transfer_function)
            self._binned_pk_results = (results, clkk)
        return self._binned_pk_results


    def calculate_spectra(self, cmb_types):
        """Calculate the theory spectra for the requested `cmb_types`
        (`'lensed'`, `'unlensed'`, and/or `'delensed'`): with `hd_pk` for a
        binned P(k), and with hdfisher otherwise. The kSZ template is added
        to TT if `ksz=True`."""
        if self.binned_pk:
            if 'delensed' in cmb_types:
                self.check_delensing_vars()
            results, clkk = self.get_binned_pk_results()
            # delensed with the lensing reconstruction noise passed at
            # initialization, which is that of the experiment being forecast
            theo = binned_pk_spectra(self.lmax, results, clkk, cmb_types=cmb_types,
                                     nlkk=self.nlkk, Lmin=self.Lmin, Lmax=self.Lmax)
        else:
            # hdfisher's own calculation, with CAMB or CLASS, done fresh
            theo = {}
            if ('lensed' in cmb_types) or ('unlensed' in cmb_types):
                theo.update(super().get_theory_spectra(overwrite=True, save=False))
            if 'delensed' in cmb_types:
                theo['delensed'] = super().get_delensed_spectra(overwrite=True, save=False)
            theo = {cmb_type: theo[cmb_type] for cmb_type in cmb_types}
        if self.ksz:
            cl_ksz = get_cl_ksz(self.lmax, self.A_ksz, self.n_ksz,
                                hd_data_version=self.hd_data_version)
            for cmb_type in theo:
                theo[cmb_type]['tt'] = theo[cmb_type]['tt'] + cl_ksz
        return theo


    def _stored_spectra(self, cmb_type, overwrite=False):
        """The spectra for one `cmb_type` from memory or from the file in
        `output_dir`, or `None` if they have to be calculated."""
        if overwrite:
            return None
        if all(s in self.theo.get(cmb_type, {}) for s in config.theo_cols):
            return self.theo[cmb_type].copy()
        fname = self.theo_fnames.get(cmb_type)
        if (fname is not None) and os.path.exists(fname):
            print(f'loading {cmb_type} theory from {fname}')
            return utils.load_from_file(fname, config.theo_cols)
        return None


    def get_theory_spectra(self, overwrite=False, save=False):
        """The lensed and unlensed CMB spectra and the lensing spectrum.

        For a power-law P(k) without the kSZ template this is hdfisher's
        method. Otherwise it does the same loading and saving, with the
        spectra calculated by `calculate_spectra`.
        """
        if not (self.binned_pk or self.ksz):
            return super().get_theory_spectra(overwrite=overwrite, save=save)
        theo = {}
        to_calculate = []
        for cmb_type in ['lensed', 'unlensed']:
            stored = self._stored_spectra(cmb_type, overwrite=overwrite)
            if stored is None:
                to_calculate.append(cmb_type)
            else:
                theo[cmb_type] = stored
        if to_calculate:
            theo.update(self.calculate_spectra(to_calculate))
        for cmb_type in ['lensed', 'unlensed']:
            if save:
                fname = self.theo_fnames[cmb_type]
                print(f'saving {cmb_type} theory to {fname}')
                utils.save_to_file(fname, theo[cmb_type], keys=config.theo_cols)
            self.theo[cmb_type] = theo[cmb_type].copy()
        return theo


    def get_delensed_spectra(self, save=False, overwrite=False):
        """The delensed CMB spectra and the lensing spectrum, given the
        lensing reconstruction noise passed at initialization.

        For a power-law P(k) without the kSZ template this is hdfisher's
        method. Otherwise it does the same loading and saving, with the
        spectra calculated by `calculate_spectra`.
        """
        if not (self.binned_pk or self.ksz):
            return super().get_delensed_spectra(save=save, overwrite=overwrite)
        delensed = self._stored_spectra('delensed', overwrite=overwrite)
        if delensed is None:
            self.check_delensing_vars()
            delensed = self.calculate_spectra(['delensed'])['delensed']
        if save:
            fname = self.theo_fnames['delensed']
            print(f'saving delensed theory to {fname}')
            utils.save_to_file(fname, delensed, keys=config.theo_cols)
        self.theo['delensed'] = delensed.copy()
        return delensed
