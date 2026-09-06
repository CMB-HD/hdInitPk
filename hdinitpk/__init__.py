"""hdInitPk: binned primordial power spectrum forecasts (Cheslog et. al. 2026).

Extends the hdfisher Fisher pipeline with a binned primordial power spectrum
P(k) (computed with hd_pk) and a kSZ template. See `hdinitpk.theory` and
`hdinitpk.hdinitPkfisher`.

Data products
-------------
The small data products that ship with the package (Fisher matrices, the
Figure 6 points, the binning files, the fiducial-parameter and step-size
YAML files, and the spectra used by the bias calculation) live in
`hdinitpk/data`. Use `hdinitpk.data_path(...)` to build a path into that
directory instead of hard-coding one, so that scripts and the notebook work
regardless of the current working directory or of where the package was
installed:

    >>> from hdinitpk import data_path
    >>> data_path('fisher_matrices', 'hd_camb_lensed_9param.txt')

Set the environment variable `HDINITPK_DATA` to point `data_path` somewhere
else (for example a copy of the data directory outside the installed
package).

The large products that cannot ship with the package (the Fisher
derivative output directories, the CMB-HD covariance matrices, and the raw
MCMC chains) are NOT resolved here. Each script that needs one declares it
as a literal path at the top of the file, written as `/path/to/...` for you
to fill in.
"""
import os

from . import theory, hdinitPkfisher

# plotting_utilities needs matplotlib, pandas, and getdist (the `plots`
# extra); keep the core package importable without them:
try:
    from . import plotting_utilities
except ImportError:
    plotting_utilities = None


# ----- locating the data products -----

#: Directory holding the data products distributed with the package.
DATA_DIR = os.environ.get(
    'HDINITPK_DATA',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))

def data_path(*parts):
    """Returns the absolute path to a file or directory inside the packaged
    data directory (`hdinitpk.DATA_DIR`), which can be overridden with the
    `HDINITPK_DATA` environment variable."""
    return os.path.join(DATA_DIR, *parts)


#: Directory holding everything the user generates: Fisher derivatives and
#: matrices, MCMC chains, cached statistics. It sits under the packaged
#: data directory so that a script writes its output next to the data it
#: read, and every other script and notebook knows where to find it without
#: being told. Override it with `HDINITPK_USER_DATA`, which is worth doing
#: if the package is installed somewhere read-only, or if you would rather
#: the (large) derivative directories not live inside site-packages.
USER_DATA_DIR = os.environ.get('HDINITPK_USER_DATA',
                               os.path.join(DATA_DIR, 'user_generated_data'))


def user_data_path(*parts, make=True):
    """Returns the absolute path to a file or directory inside
    `hdinitpk.USER_DATA_DIR`, creating the directories along the way.

    This is where the scripts and notebooks put what they produce, and
    where they look for it afterwards:

        fisher_derivs/<configuration>/   derivative directories
        fisher_matrices/<name>.txt       rebuilt Fisher matrices
        chains/<name>.*                  chains you have run
        cached_results/<name>.txt        statistics computed from them
        plin_z0/                         Figure 6 samples

    Parameters
    ----------
    *parts : str
        Path components below `USER_DATA_DIR`.
    make : bool, default=True
        Create the parent directory if it does not exist. Pass `False` to
        build a path without touching the filesystem (when checking whether
        something is there, for instance).

    Returns
    -------
    str
        The absolute path.

    Raises
    ------
    OSError
        If the directory cannot be created, which usually means the package
        is installed read-only. Set `HDINITPK_USER_DATA` to somewhere
        writable in that case.
    """
    path = os.path.join(USER_DATA_DIR, *parts)
    if make:
        parent = os.path.dirname(path) if os.path.splitext(path)[1] else path
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as err:
            raise OSError(
                f"could not create {parent}: {err}. This is the directory "
                "hdInitPk writes its output to. If the package is installed "
                "somewhere read-only, set the HDINITPK_USER_DATA environment "
                "variable to a writable location.") from err
    return path
