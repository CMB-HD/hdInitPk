"""Theory calculation for the binned primordial power spectrum P(k) and the
kSZ template, used in Cheslog et. al. (2026).

This module extends `hdfisher.theory` (via the subclass seams added there)
with the pieces that were removed from the merged hdfisher code:

- a binned primordial power spectrum, described by an amplitude in each of a
  set of k-bins (parameters `Pk1`, `Pk2`, ... or `eneg2tauPk1`, ...) rather
  than by a power law, calculated with `hd_pk.cmb_from_pk`;
- a kinematic SZ template added to the theory TT spectrum, scaled by an
  amplitude `A_ksz` and tilt `n_ksz`, loaded from `hd_mock_data`.

`hdinitpk.theory.Theory` accepts the same arguments as the old merged
`hdfisher.theory.Theory`, so code written against the merged version only
needs to change its import.
"""
import os
import re
import warnings
import numpy as np
import camb
from scipy.interpolate import interp1d
from hdfisher import theory as hdtheory

LMAX_BUFFER = hdtheory.LMAX_BUFFER


def _import_cmb_from_pk():
    """Import and return `hd_pk.cmb_from_pk`, with a clear error message if
    the `hd_pk` package is not installed. The import is done lazily so that
    the kSZ-only functionality of this module works without `hd_pk`.
    """
    try:
        from hd_pk import cmb_from_pk
    except ImportError as err:
        raise ImportError(
            "The binned-Pk calculation requires the `hd_pk` package "
            "(`cmb_from_pk`), which could not be imported. Install hdPk and "
            "its dependencies, then try again.") from err
    return cmb_from_pk


def get_cl_ksz(lmax, A_ksz, n_ksz):
    """Returns the kSZ template power spectrum, scaled by the amplitude
    `A_ksz` and tilt `n_ksz`, from ell = 0 to `lmax`.

    The template is loaded from `hd_mock_data`, normalized at ell = 3000,
    and the first two multipoles are set to zero:
        cl_ksz(ell) = A_ksz * (ell / 3000)^n_ksz * template(ell).

    Parameters
    ----------
    lmax : int
        The maximum multipole of the returned spectrum.
    A_ksz : float
        The template amplitude.
    n_ksz : float
        The template tilt.

    Returns
    -------
    cl_ksz : array_like of float
        The scaled kSZ template, starting at ell = 0.

    Raises
    ------
    ValueError
        If `A_ksz` or `n_ksz` is `None`, or if the template does not reach
        `lmax`.
    """
    if A_ksz is None:
        raise ValueError("Must provide `A_ksz` when `ksz=True`")
    if n_ksz is None:
        raise ValueError("Must provide `n_ksz` when `ksz=True`")
    from hd_mock_data import hd_data
    datalib = hd_data.HDMockData()
    # Ask for the template out to `lmax` explicitly. Left to itself,
    # `cl_ksz` returns it only as far as the data version's own lmaxTT,
    # which is 40000 for v1.1 but 20100 from v1.2 on, and the
    # multiplication below then fails to broadcast. The underlying template
    # file is the same one in every version, so asking for `lmax` makes
    # this independent of which version is installed.
    ksz_ells, cl_ksz_template = datalib.cl_ksz(output_lmax=lmax)
    if len(cl_ksz_template) < lmax + 1:
        raise ValueError(
            f"the kSZ template from hd_mock_data reaches only ell = "
            f"{len(cl_ksz_template) - 1}, but the theory is being computed "
            f"to lmax = {lmax}. Lower `lmax` in the parameter file, or "
            "supply a template that goes further.")
    ells_for_ksz = np.arange(lmax + 1)
    cl_ksz = A_ksz * (ells_for_ksz / 3000)**n_ksz * cl_ksz_template[:lmax+1]
    cl_ksz[:2] = 0
    return cl_ksz


# ----- cosmological parameters: -----

def set_cosmo_params(param_file=None, use_H0=False, binned_pk=False, **cosmo_params):
    """Load the parameter values saved in the `param_file` and return a
    dictionary that can be passed to `camb.set_params()`, optionally
    converting the binned-Pk parameters into the internal form used by
    `set_camb_params` and `build_binned_pk_transfer`.

    This wraps `hdfisher.theory.set_cosmo_params` and adds back the
    binned-Pk and kSZ handling that was removed from hdfisher: when
    `binned_pk=True`, the k-bin centers (`k1`, ..., `kN`) and bin amplitudes
    (`Pk1`, ... or `eneg2tauPk1`, ..., converted with exp(2 tau)) are
    collected into a `'primordial_scalar_pk'` entry, the fiducial `'As'` is
    moved to `'_fid_As'`, and the kSZ template parameters `'A_ksz'` and
    `'n_ksz'` (which are not CAMB parameters) are always removed.

    Parameters
    ----------
    param_file : str, default=None
        The file name, including the absolute path, of a YAML file that
        contains the parameter names and values. When `binned_pk=True`, the
        file must contain an `'nkbins'` entry, the bin centers, the bin
        amplitudes, and an `'effective_ns_for_nonlinear'` entry.
    use_H0 : bool, default=False
        Pass the Hubble constant instead of CosmoMC theta to CAMB, if both
        are present in the parameter file.
    binned_pk : bool, default=False
        Whether the primordial power spectrum is described by binned
        amplitudes rather than by a power law.
    **cosmo_params : dict of float
        An optional dictionary of parameter names and values to override
        the values loaded from the YAML file.

    Returns
    -------
    p : dict
        A dictionary of the parameter names and values.
    """
    p = hdtheory.set_cosmo_params(param_file=param_file, use_H0=use_H0,
                                  **cosmo_params)
    if binned_pk:
        p['ks'] = np.array([p['k{}'.format(b + 1)] for b in range(p['nkbins'])])
        pk = np.zeros_like(p['ks'])
        if 'eneg2tauPk1' in p.keys():
            for b in range(p['nkbins']):
                pk[b] = p['eneg2tauPk%s' % (b + 1)]
            pk *= np.exp(2 * p['tau'])
        elif 'Pk1' in p.keys():
            for b in range(p['nkbins']):
                pk[b] = p['Pk%s' % (b + 1)]
        p['primordial_scalar_pk'] = {'k': p['ks'],
                                         'Pk': pk, 'log_regular': False}
        p['_fid_As'] = p['As'] 
        p.pop('As')
    # remove ksz params that are not CAMB parameters
    for ksz_key in ['A_ksz', 'n_ksz']:
        p.pop(ksz_key, None)
    return p


def set_camb_params(lmax, param_file=None, use_H0=False, binned_pk=False, **cosmo_params):
    """Returns a `CAMBparams` instance with the requested cosmological and
    accuracy parameters, with support for the binned primordial power
    spectrum and for parameter files containing kSZ template parameters.

    This reproduces the behavior of the merged hdfisher `set_camb_params`:
    for `binned_pk=False` it matches `hdfisher.theory.set_camb_params`
    exactly, except that `'A_ksz'` and `'n_ksz'` entries are stripped before
    the parameters are passed to CAMB (the base hdfisher function would
    reject them). For `binned_pk=True`, the binned-Pk parameters are
    removed from the CAMB input, the initial power spectrum is set to the
    fiducial power law via `set_initial_power_function`, and the perturbed
    bin amplitudes are recorded on the returned instance in the private
    `_primordial_pk` attribute for use by `build_binned_pk_transfer`.

    Parameters
    ----------
    lmax : int
        The maximum multipole for the theory spectra.
    param_file : str, default=None
        The file name, including the absolute path, of a YAML file that
        contains the parameter names and values.
    use_H0 : bool, default=False
        Pass the Hubble constant instead of CosmoMC theta to CAMB, if both
        are present in the parameter file.
    binned_pk : bool, default=False
        Whether the primordial power spectrum is described by binned
        amplitudes rather than by a power law.
    **cosmo_params : dict of float
        An optional dictionary of parameter names and values to override
        the values loaded from the YAML file.

    Returns
    -------
    pars_out : camb.model.CAMBparams
        The CAMB parameters, carrying the private attributes `_fid_As`,
        `_fid_ns`, and `_fid_kpivot` (and, when `binned_pk=True`,
        `_primordial_pk`).
    """
    input_params = set_cosmo_params(param_file=param_file, use_H0=use_H0, binned_pk=binned_pk, **cosmo_params).copy()
    if binned_pk == True:
        primordial_scalar_pk = input_params['primordial_scalar_pk']
        effective_ns_for_nonlinear = input_params['effective_ns_for_nonlinear']
        ks = input_params['ks']
        fid_As = input_params.pop('_fid_As')
        input_params.pop('primordial_scalar_pk')
        input_params.pop('ks')
        bad_keys = [k for k in input_params if re.match(r'^k\d+$', k) or k.startswith('Pk') or k.startswith('eneg2tauPk') or k == 'nkbins' or k == 'effective_ns_for_nonlinear']
        for k in bad_keys:
            input_params.pop(k, None)
    # the parameter file's `lmax` (if present) sets the multipole to which
    # the theory is saved; CAMB computes `LMAX_BUFFER` multipoles beyond it
    file_lmax = input_params.get('lmax')
    theory_lmax = int(file_lmax) if file_lmax is not None else int(lmax)
    input_params['lmax'] = theory_lmax + LMAX_BUFFER
    # only pass `redshifts` to CAMB if the parameter file provided them;
    # otherwise drop the key so CAMB uses its own default
    if input_params.get('redshifts') is None:
        input_params.pop('redshifts', None)
    pars = camb.set_params(**input_params)
    if binned_pk == True:
        k_fine  = np.logspace(-6, 2, 10000)
        pk_fine = fid_As * (k_fine / input_params.get('pivot_scalar', 0.05)) ** (input_params.get('ns', 0.965) - 1)
        Pk_func = interp1d(k_fine, pk_fine, kind='linear',
                           bounds_error=False,
                           fill_value=(pk_fine[0], pk_fine[-1]))
        pars.set_initial_power_function(
            Pk_func,
            effective_ns_for_nonlinear=effective_ns_for_nonlinear,
            kmin=1e-6, kmax=100
        )

    pars_out = pars.copy()

    if binned_pk == True:
        pars_out._fid_As = fid_As
        pars_out._fid_ns      = input_params.get('ns')
        pars_out._fid_kpivot  = input_params.get('pivot_scalar', 0.05)
        pars_out._primordial_pk = primordial_scalar_pk
    else:
        pars_out._fid_As     = input_params.get('As')
        pars_out._fid_ns     = input_params.get('ns')
        pars_out._fid_kpivot = input_params.get('pivot_scalar', 0.05)
    return pars_out


# ----- the binned-Pk transfer function: -----

def build_binned_pk_transfer(camb_params, camb_params_cdm, bin_edges,
                             varied_param='fid', frac_step=0.05):
    """Build CAMB results and a primordial-Pk transfer function for a
    binned-Pk calculation when `varied_param` is varied. The lensed,
    unlensed, and delensed spectra all respond identically because they are
    all computed from the same results and transfer function.

    The transfer function multiplies the fiducial power-law primordial
    spectrum by `(1 + step)` inside the varied k-bin, where
    `step = sign * frac_step` and the sign is inferred by comparing the
    perturbed bin amplitude recorded in `camb_params._primordial_pk` to the
    fiducial power law at the bin center.

    Parameters
    ----------
    camb_params : camb.model.CAMBparams
        The CAMB parameters carrying the (possibly perturbed) bin
        amplitudes in the private `_primordial_pk` attribute, as returned
        by `set_camb_params` with `binned_pk=True`.
    camb_params_cdm : camb.model.CAMBparams
        The CAMB parameters carrying the fiducial power-law attributes
        `_fid_As`, `_fid_ns`, and `_fid_kpivot`, used both to define the
        fiducial power law and as the base for the perturbed CAMB run.
    bin_edges : array_like of float
        The k-bin edges: element `i` is the lower edge of bin `i`, and
        element `i + 1` is its upper edge.
    varied_param : str or None, default='fid'
        The name of the varied bin amplitude (`'Pk<n>'` or
        `'eneg2tauPk<n>'`, 1-indexed), or `'fid'`/`None` for the fiducial
        (no bin varied).
    frac_step : float, default=0.05
        The fractional step applied inside the varied bin. NOTE: this must
        be consistent with the step sizes in the Fisher steps file, since
        the finite-difference denominator comes from the steps file while
        the spectra are perturbed by `frac_step`. A warning is issued if
        the fractional step implied by the perturbed bin amplitude differs
        from `frac_step` by more than 0.1 percent (relative).

    Returns
    -------
    camb_results : camb.results.CAMBdata
        The CAMB results for the perturbed (or fiducial) primordial
        spectrum.
    pk_transfer_function : callable(k, z) or None
        The transfer function to apply to the fiducial primordial spectrum,
        or `None` for the fiducial case.
    """
    pk_bins_arr = np.array(camb_params._primordial_pk['Pk'])
    k_bins_arr = np.array(camb_params._primordial_pk['k'])
    As = camb_params_cdm._fid_As
    ns = camb_params_cdm._fid_ns
    k_pivot = camb_params_cdm._fid_kpivot

    step_size = 0.0
    varied_bin_idx = None
    if varied_param not in (None, 'fid'):
        match = re.match(r'^(?:Pk|eneg2tauPk)(\d+)$', varied_param)
        if match is not None:
            bin_num = int(match.group(1)) - 1
            pk_fid_at_bin = As * (k_bins_arr[bin_num] / k_pivot) ** (ns - 1)
            sign = 1.0 if pk_bins_arr[bin_num] >= pk_fid_at_bin else -1.0
            step_size = sign * frac_step
            varied_bin_idx = bin_num
            # consistency check: the spectra are perturbed by `frac_step`,
            # but the finite-difference denominator in the Fisher derivative
            # comes from the steps file (via the perturbed bin amplitude).
            # If the two disagree, the derivatives are silently wrong by
            # their ratio, so warn loudly:
            implied_frac = (pk_bins_arr[bin_num] - pk_fid_at_bin) / pk_fid_at_bin
            if abs(abs(implied_frac) - frac_step) > 1e-3 * frac_step:
                msg = (f"The perturbed amplitude of bin {bin_num + 1} implies "
                       f"a fractional step of {implied_frac:+.6f}, but the "
                       f"spectra will be perturbed by frac_step = "
                       f"{step_size:+.6f}. The Fisher derivative for "
                       f"'{varied_param}' will be wrong by the ratio of the "
                       "two. Make `frac_step` (the `pk_frac_step` argument "
                       "of `hdinitpk.hdinitPkfisher.Fisher`) consistent with the "
                       "step sizes in the Fisher steps file.")
                warnings.warn(msg)

    if varied_bin_idx is not None and abs(step_size) > 0:
        kmin = bin_edges[varied_bin_idx]
        kmax = bin_edges[varied_bin_idx + 1]

        def pk_transfer_function(k, z):
            k = np.atleast_1d(np.asarray(k, dtype=float))
            result = np.ones_like(k)
            in_bin = (k >= kmin) & (k <= kmax)
            result[in_bin] = 1.0 + step_size
            return result

        k_fine = np.logspace(np.log10(1e-6), np.log10(100), 10000)
        pk_fine = As * (k_fine / k_pivot) ** (ns - 1) * pk_transfer_function(k_fine, None)
        Pk_func = interp1d(k_fine, pk_fine, kind='linear', bounds_error=False,
                           fill_value=(pk_fine[0], pk_fine[-1]))
        camb_params_binned = camb_params_cdm.copy()
        camb_params_binned.set_initial_power_function(
            Pk_func, effective_ns_for_nonlinear=ns, kmin=1e-6, kmax=100)
        camb_results = camb.get_results(camb_params_binned)
    else:
        pk_transfer_function = None
        camb_results = camb.get_results(camb_params_cdm)

    return camb_results, pk_transfer_function


# ----- the Theory subclass: -----

class Theory(hdtheory.Theory):
    """Extends `hdfisher.theory.Theory` with the binned primordial power
    spectrum and the kSZ template. Accepts the same arguments as the base
    class, plus the following (matching the merged hdfisher code):

    Parameters
    ----------
    binned_pk : bool, default=False
        If `True`, the primordial power spectrum is described by an
        amplitude in each of a set of k-bins rather than by a power law.
        The `param_file` must then contain an `nkbins` entry, the bin
        centers (`k1`, `k2`, ...), the bin amplitudes (`Pk1`, `Pk2`, ...
        or `eneg2tauPk1`, ...), and an `effective_ns_for_nonlinear` entry.
        Requires `bin_edges`, and requires `use_class=False`. The spectra
        are calculated with `hd_pk.cmb_from_pk`.
    varied_param : str or None, default='fid'
        The name of the parameter that is being varied away from its
        fiducial value. Used by the binned-Pk calculation to work out which
        k-bin is being stepped, and in which direction. Pass `'fid'` or
        `None` for the fiducial theory. Ignored when `binned_pk=False`.
    ksz : bool, default=False
        If `True`, add a kinematic SZ template to the theory TT spectrum.
        The amplitude `A_ksz` and tilt `n_ksz` must then be provided in
        `cosmo_params` (they are removed before anything is passed to the
        Boltzmann code). For CAMB, the template is added to every requested
        CMB type; for CLASS, it is added to the lensed spectra only.
    bin_edges : str, array_like of float, or None, default=None
        The k-bin edges used when `binned_pk=True`, given either as an
        array or as the name (including the absolute path) of a text file
        that can be read with `numpy.loadtxt`. Required when
        `binned_pk=True`, and ignored otherwise.
    pk_frac_step : float, default=0.05
        The fractional step applied inside the varied k-bin; see
        `build_binned_pk_transfer`.

    Raises
    ------
    ValueError
        If `ksz=True` but `A_ksz` or `n_ksz` was not provided; if
        `binned_pk=True` but no `bin_edges` were given; or if both
        `use_class=True` and `binned_pk=True`.
    NotImplementedError
        If delensed spectra are requested with both `binned_pk=True` and
        `ksz=True`.
    """

    def __init__(self, lmax, output_dir, output_root=None, param_file=None,
                 nlkk=None, recon_lmin=None, recon_lmax=None, use_H0=False,
                 use_class=False, binned_pk=False, varied_param='fid',
                 ksz=False, bin_edges=None, pk_frac_step=0.05, **cosmo_params):
        self.binned_pk = binned_pk # whether or not you are using binned pk
        self.varied_param = varied_param
        self.pk_frac_step = pk_frac_step

        self.ksz = ksz
        self.A_ksz = cosmo_params.pop('A_ksz', None)
        self.n_ksz = cosmo_params.pop('n_ksz', None)
        if self.ksz and (self.A_ksz is None or self.n_ksz is None):
            raise ValueError("Must provide `A_ksz` and `n_ksz` when `ksz=True`")
        if use_class and binned_pk:
            err_msg = "`use_class=True` and `binned_pk=True` cannot be combined: the binned primordial power spectrum is only implemented for CAMB. Set one of them to `False`."
            raise ValueError(err_msg)

        self.bin_edges = None
        if bin_edges is not None:
            self.bin_edges = np.loadtxt(bin_edges) if isinstance(bin_edges, str) else np.asarray(bin_edges)
        if self.binned_pk and self.bin_edges is None:
            raise ValueError("You must pass `bin_edges` (an array or a file path) when `binned_pk=True`.")

        # the base `__init__` calls the `_setup_boltzmann_params` override
        # below, which reads `self.binned_pk`, so the attributes above must
        # be set first:
        super().__init__(lmax, output_dir, output_root=output_root,
                         param_file=param_file, nlkk=nlkk,
                         recon_lmin=recon_lmin, recon_lmax=recon_lmax,
                         use_H0=use_H0, use_class=use_class, **cosmo_params)


    def _setup_boltzmann_params(self, param_file=None, use_H0=False, **cosmo_params):
        """Set up the Boltzmann-code parameters, using this module's
        `set_camb_params` (which understands the binned-Pk and kSZ
        parameter-file entries) instead of hdfisher's.

        When `binned_pk=True`, a second `CAMBparams` instance
        (`self.camb_params_cdm`) is set up carrying the fiducial power law,
        used by `build_binned_pk_transfer` as the base for the perturbed
        CAMB run.
        """
        if self.use_class:
            # the kSZ parameters were already removed from `cosmo_params` in
            # `__init__`, so the base CLASS setup can be used directly:
            super()._setup_boltzmann_params(param_file=param_file,
                                            use_H0=use_H0, **cosmo_params)
        else:
            self.class_params = None
            self.camb_params = set_camb_params(
                self.lmax, param_file=param_file, use_H0=use_H0,
                binned_pk=self.binned_pk, **cosmo_params)
            if self.binned_pk:
                self.camb_params_cdm = set_camb_params(
                    self.lmax, param_file=param_file, use_H0=use_H0,
                    binned_pk=self.binned_pk, **cosmo_params)
            else:
                self.camb_params_cdm = None


    def _get_cl_ksz(self):
        """Returns the scaled kSZ template up to `self.lmax`."""
        return get_cl_ksz(self.lmax, self.A_ksz, self.n_ksz)


    def _compute_spectra(self, cmb_type, save=False):
        """Compute the (lensed or unlensed) theory spectra for a single
        `cmb_type`, using `hd_pk.cmb_from_pk` when `binned_pk=True`, and
        adding the kSZ template to TT when `ksz=True`.
        """
        if self.binned_pk:
            cmb_from_pk = _import_cmb_from_pk()
            camb_results_binned, pk_transfer_function = build_binned_pk_transfer(
                self.camb_params, self.camb_params_cdm, self.bin_edges,
                varied_param=self.varied_param, frac_step=self.pk_frac_step)
            theo = cmb_from_pk.calculate_theory_spectra(
                self.lmax, self.camb_params_cdm,
                camb_results=camb_results_binned,
                pk_transfer_function=pk_transfer_function,
                cmb_types=[cmb_type])[cmb_type]
            if self.ksz:
                theo['tt'] += self._get_cl_ksz()
            return theo
        theo = super()._compute_spectra(cmb_type, save=save)
        if self.ksz:
            # for CAMB, the template is added to every requested CMB type;
            # for CLASS, only to the lensed spectra (matching the merged
            # hdfisher behavior):
            if (not self.use_class) or (cmb_type == 'lensed'):
                theo['tt'] += self._get_cl_ksz()
        return theo


    def _compute_delensed_spectra(self, save=False):
        """Compute the delensed theory spectra, using `hd_pk.cmb_from_pk`
        when `binned_pk=True`.
        """
        if self.binned_pk:
            cmb_from_pk = _import_cmb_from_pk()
            camb_results, pk_transfer_function = build_binned_pk_transfer(
                self.camb_params, self.camb_params_cdm, self.bin_edges,
                varied_param=self.varied_param, frac_step=self.pk_frac_step)
            theo = cmb_from_pk.calculate_theory_spectra(
                self.lmax, self.camb_params_cdm,
                camb_results=camb_results,
                pk_transfer_function=pk_transfer_function,
                cmb_types=['delensed'],
                nlkk=self.nlkk, recon_lmin=self.Lmin, recon_lmax=self.Lmax)
            delensed_theo = theo['delensed']
        else:
            delensed_theo = super()._compute_delensed_spectra(save=save)
        if self.ksz:
            delensed_theo['tt'] += self._get_cl_ksz()
        return delensed_theo
