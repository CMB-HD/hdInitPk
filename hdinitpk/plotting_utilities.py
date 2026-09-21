"""Plotting and formatting utilities for the Cheslog et. al. (2026)
figures and tables.

This module collects the purely cosmetic and formatting machinery used by
the `reproduce_cheslog_et_al_2026` notebook: the matplotlib style, the
LaTeX parameter labels, the color palettes, and small generic helpers
(legends, log axes, box-style error bars, table formatting). Nothing in
here affects any numerical result; everything with non-negligible
importance to the results stays in the notebook, where the user can see it
being done.

Requires matplotlib, numpy, pandas, and getdist (declared as the `plots`
extra of the hdinitpk package).
"""
import os
import math
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from getdist.gaussian_mixtures import GaussianND

try:
    from IPython.display import display as _display
except ImportError:  # fall back to plain printing outside IPython
    _display = print


# ======================================================================
# Matplotlib style
# ======================================================================

# The rcParams used for every figure in the paper:
PAPER_RC_PARAMS = {
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 500,
}


def set_plot_style(**overrides):
    """Apply the paper's matplotlib rcParams (`PAPER_RC_PARAMS`), with any
    keyword arguments applied on top as overrides."""
    params = dict(PAPER_RC_PARAMS)
    params.update(overrides)
    plt.rcParams.update(params)


# ======================================================================
# Parameter labels and table formatting
# ======================================================================

# LaTeX labels for each parameter (without surrounding $...$, as expected by
# getdist). Wrap in $...$ for use in matplotlib legends or table code.
param_labels = {
    'ombh2': r'\Omega_\mathrm{b} h^2',
    'omch2': r'\Omega_\mathrm{c} h^2',
    'H0': r'H_0',
    'tau': r'\tau',
    'logA': r'\ln \left(10^{10} A_\mathrm{s}\right)',
    'ns': r'n_\mathrm{s}',
    'nrun': r'\alpha_\mathrm{s}',
    'nnu': r'N_\mathrm{eff}',
    'mnu': r'\sum m_\nu',
    'theta': r'100\theta_\mathrm{MC}',
    'omk': r'\Omega_k',
    'w': r'w_0',
    'wa': r'w_a',
    'HMCode_logT_AGN': r'\log_{10}\left({T_\mathrm{AGN}}/{\mathrm{K}}\right)',
    'A_ksz': r'A_\mathrm{kSZ}',
    'n_ksz': r'n_\mathrm{kSZ}',
}

# Significant figures used for the 1-sigma errors in the LaTeX tables
# (Omega_b h^2 gets 2 significant figures, everything else gets 3);
# ratio columns always use `ratio_decimals` decimal places.
err_sig_figs = {
    'ombh2': 2, 'omch2': 3, 'H0': 3, 'tau': 3, 'logA': 3,
    'ns': 3, 'nrun': 3, 'nnu': 3, 'mnu': 3,
    'A_ksz': 3, 'n_ksz': 3, 'HMCode_logT_AGN': 3,
}
ratio_decimals = 3


def sigfig(x, n):
    """Format `x` to `n` significant figures in plain (non-scientific)
    decimal notation, preserving trailing zeros (e.g. 0.045 -> '0.0450')."""
    if x == 0 or not np.isfinite(x):
        return f'{0:.{max(n - 1, 0)}f}'
    decimals = max(n - 1 - math.floor(math.log10(abs(x))), 0)
    s = f'{x:.{decimals}f}'
    # Rounding can bump the value across a power of ten (e.g. 0.00997 at
    # 2 significant figures would render as '0.010', i.e. 3 figures);
    # re-round in that case.
    d2 = max(n - 1 - math.floor(math.log10(abs(float(s)))), 0) if float(s) != 0 else decimals
    return f'{x:.{d2}f}' if d2 != decimals else s


def ratio(dict1, dict2, fill_value=None):
    """Returns a dictionary with the ratio of the values in `dict1` to those
    in `dict2` for each key they share in common. If the keys don't match, you
    can pass a `fill_value` as a placeholder for any keys not in both
    dictionaries; otherwise leave it as `None` to exclude those keys from the
    returned dictionary."""
    rdict = {}
    all_keys = set(dict1.keys()) | set(dict2.keys())
    for key in all_keys:
        if (key in dict1) and (key in dict2):
            rdict[key] = dict1[key] / dict2[key]
        elif fill_value is not None:
            rdict[key] = fill_value
    return rdict


def print_table(list_of_dicts, list_of_labels, title=None, list_of_keys=None):
    """
    Display a table of keys and values held in each dictionary in the
    `list_of_dicts`.

    Parameters
    ----------
    list_of_dicts : list of dict
        A list of dictionaries.
    list_of_labels : list of str
        A list holding a label for each dictionary in `list_of_dicts`, in
        the same order that they appear in `list_of_dicts`.
    title : str or None, default=None
        A title that is printed out above the table.
    list_of_keys : list or None, default=None
        A list of keys to use in the table. The table will only contain
        values corresponding to the keys in `list_of_keys`, and in the same
        order. If `None`, all keys are used. In either case, the dictionaries
        in `list_of_dicts` do not need to contain the same set of keys.
    """
    table = pd.DataFrame(list_of_dicts, index=list_of_labels, columns=list_of_keys)
    table = table.T
    if title is not None:
        print('\n', title)
    _display(table)


# ======================================================================
# getdist helpers
# ======================================================================

def getdist_samples(fisher_matrix, fisher_params, fid_params, labels=None):
    """Wrap a Fisher matrix as a getdist `GaussianND` distribution centered
    on the fiducial parameter values, so it can be plotted alongside MCMC
    samples. The Fisher matrix is passed as the INVERSE covariance
    (`is_inv_cov=True`). Labels default to `param_labels`."""
    if labels is None:
        labels = param_labels
    fids = [fid_params[p] for p in fisher_params]
    plot_labels = [labels[p] for p in fisher_params]
    samples = GaussianND(fids, fisher_matrix, is_inv_cov=True,
                         names=fisher_params, labels=plot_labels)
    return samples


# ======================================================================
# Figure helpers
# ======================================================================

def save_figure(fig_name, enabled=True, **savefig_kwargs):
    """Save the current figure to the working directory as
    `<fig_name>-<month>-<day>-<year>.png`, dated with today's date, but only
    if `enabled` is True. Any keyword arguments are passed straight through
    to `plt.savefig`."""
    if not enabled:
        return
    today = datetime.date.today()
    date_tag = f'{today.strftime("%B").lower()}-{today.day}-{today.year}'
    filename = f'{fig_name}-{date_tag}.png'
    plt.savefig(filename, **savefig_kwargs)
    print(f'saved {os.path.join(os.getcwd(), filename)}')


def legend_below(ax, prev_legend, handles, ncol, **legend_kwargs):
    """Draw a legend just beneath `prev_legend`'s lower-left corner."""
    ax.figure.canvas.draw()
    bbox = prev_legend.get_window_extent()
    y = ax.transAxes.inverted().transform((bbox.x0, 1.01 * bbox.y0))[1]
    leg = ax.legend(handles=handles, fontsize=10, columnspacing=1,
                    bbox_to_anchor=(0, y), loc='upper left', ncol=ncol,
                    **legend_kwargs)
    ax.add_artist(leg)
    return leg


def set_loglog(ax):
    """Set both axes of `ax` to a logarithmic scale."""
    ax.set_xscale('log')
    ax.set_yscale('log')


def add_box_errors(ax, x, y, yerr_low, yerr_high, color, alpha, edges):
    """Draw the (asymmetric) error on each point as a shaded box spanning the
    full width of its k bin."""
    boxes = []
    for i, (yi, ylo, yhi) in enumerate(zip(y, yerr_low, yerr_high)):
        x_left = edges[i]
        x_right = edges[i + 1]
        width = x_right - x_left
        y_bottom = yi - ylo
        height = ylo + yhi
        boxes.append(mpatches.Rectangle((x_left, y_bottom), width, height))
    pc = PatchCollection(boxes, facecolor=color, alpha=alpha,
                         edgecolor='none', zorder=2)
    ax.add_collection(pc)


# ======================================================================
# Colors and shared style dictionaries
# ======================================================================

# Colorblind-friendly palette (used for the inflation-model markers):
cb_orange = '#E69F00'
cb_skyblue = '#56B4E9'
cb_blue = '#0072B2'
cb_vermillion = '#D55E00'
cb_pink = '#CC79A7'

# Named colors shared by Figures 2 and 4:
red = '#d62728'
green = '#3cc03c'
yellow = '#fdb700'
darkyellow = '#b38c00'
magenta = '#d627d6'
brightblue = '#005cfa'
lightgray = '0.75'
blue = '#56B4E9'
blueishgreen = '#009E73'
magenta2 = '#CC79A7'

# One color per cosmological model, for the current-data contours (Figure 1):
fig1_colors = {'lcdm_nrun': '#ff81c0',          # pink
               'w0wacdm_nrun': '#b467bd',       # purple
               'lcdm_nrun_nnu': '#77becf',      # cyan
               'lcdm_nrun_nnu_mnu': '#fdb700'}  # yellow

# One color and label per data combination ('real' data or forecasts),
# shared by Figures 2 and 4:
exp_colors = {'pas': yellow, 'so': green, 'hd': red}
exp_labels = {'pas': 'CMB-PAS + DESI DR2 (data)',
              'so': 'SO-like + DESI BAO (forecast)',
              'hd': 'CMB-HD + DESI BAO (forecast)'}

# Labels, colors, and z-orders for the linear-matter-power data sets of
# Figure 6:
mpk_labels = {'hd': r'CMB-HD', 'so': r'SO-like', 'pas': 'CMB-PAS',
              'uvlf': 'HST UV LF', 'sdss': 'SDSS DR7 LRG',
              'lya': r'eBOSS DR14 Ly-$\alpha$ forest',
              'des': 'DES Y1 cosmic shear'}
mpk_colors = {'hd': 'tab:red', 'so': 'tab:green', 'pas': 'tab:olive',
              'uvlf': 'darkcyan', 'sdss': '#d477a5', 'lya': '#7d53a3',
              'des': '#a18257'}
mpk_zorders = {'hd': 10, 'so': 5, 'pas': 4, 'uvlf': 2, 'sdss': -5,
               'lya': -4, 'des': 2}

# Style of the guide lines at nrun = 0 and ns = 1 (Figures 1 and 2):
param_line_kwargs = {'ls': '-.', 'lw': 1, 'color': '0.75'}

# Frameless legend style used throughout:
flat_legend = {'borderaxespad': 0, 'frameon': False}
