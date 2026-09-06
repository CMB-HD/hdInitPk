import numpy as np
from typing import Any
from copy import deepcopy
from cobaya.model import get_model
from cobaya.theory import Theory
from cobaya.likelihood import LikelihoodInterface, Likelihood
from cobaya.typing import InputDict, empty_dict, ParamDict


# Modified version for CMB-HD
class BinnedPk(Theory):
    # Example splined power spectrum exp(-2 tau) P(k) based on bin values.
    # Can pass dense sampling to CAMB to reproduce any function; here bins are
    # directly the cubic-spline values used by CAMB.
    # Note: need wide k bounds, or a recent CAMB (which takes values beyond the
    # start and end bins to equal the end bins).
    #
    # This version optionally allows N non-log-regular bins as the leading
    # bins. Set `k_first` to a list of linear k values (Mpc^-1); those become
    # the first len(k_first) bins, and the remaining nbins - len(k_first) bins
    # are log-spaced over [10**k_min_bin, 10**k_max_bin]. The total number of
    # free amplitudes stays equal to nbins, so the dynamically generated params
    # (b1..bnbins) are unchanged: b1..b{N} map to the custom k values (in the
    # order given) and the rest map to the log grid. If `k_first` is None or
    # empty, behaviour is identical to the original (all nbins bins log-spaced).
    # A single float is also accepted as shorthand for a one-element list.
    nbins: int = 20
    k_min_bin: float = np.log10(0.001)  # log10(k) at the start of the log-spaced range
    k_max_bin: float = np.log10(0.35)   # log10(k) at the end of the log-spaced range
    k_first: Any = None                 # list of linear k (Mpc^-1) for the leading non-log bins; None/[] disables
    scale: float = 1e-9
    bin_par: ParamDict = {"prior": {"min": 0, "max": 100}}

    def initialize(self):
        if self.k_first is None or len(np.atleast_1d(self.k_first)) == 0:
            # Original behaviour: all nbins bins log-spaced.
            self.ks = np.logspace(self.k_min_bin, self.k_max_bin, self.nbins)
        else:
            # N custom (non-log) leading bins, then the rest log-spaced.
            k_first = np.atleast_1d(np.asarray(self.k_first, dtype=float))
            n_first = k_first.size
            if n_first > self.nbins:
                raise ValueError(
                    "Got %d custom bins in k_first but only nbins=%d total bins."
                    % (n_first, self.nbins)
                )
            n_log = self.nbins - n_first
            if n_log > 0:
                log_ks = np.logspace(self.k_min_bin, self.k_max_bin, n_log)
                self.ks = np.concatenate([k_first, log_ks])
            else:
                # All bins are custom; no log-spaced bins remain.
                self.ks = k_first

    def get_requirements(self):
        return {"tau"}

    def calculate(self, state, want_derived=True, **params_values_dict):
        pk = np.zeros_like(self.ks)
        for b in range(self.nbins):
            pk[b] = params_values_dict["b%s" % (b + 1)]
        pk *= self.scale * np.exp(2 * self.provider.get_param("tau"))
        # Use log_regular: True for speed only if the binning is log regular.
        # A non-log first bin (or testing the non-regular path for coverage)
        # requires log_regular: False, so the explicit k array is used.
        state["primordial_scalar_pk"] = {"k": self.ks, "Pk": pk, "log_regular": False}
        state["primordial_tensor_pk"] = {
            "k": self.ks,
            "Pk": pk * 0.1,
            "log_regular": False,
        }

    def get_primordial_scalar_pk(self):
        return self.current_state["primordial_scalar_pk"]

    def get_primordial_tensor_pk(self):
        return self.current_state["primordial_tensor_pk"]

    @classmethod
    def get_class_options(cls, input_options=empty_dict):
        # Dynamically generate defaults for params based on nbins.
        options = super().get_class_options().copy()
        nbins = input_options["nbins"]
        bin_par = input_options.get("bin_par", cls.bin_par)
        params = {}
        for b in range(nbins):
            par = deepcopy(bin_par)
            par["latex"] = "b_%s" % (b + 1)
            params["b%s" % (b + 1)] = par
        options["params"] = params
        return options