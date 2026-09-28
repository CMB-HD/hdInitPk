Everything hdInitPk generates is written here, and read back from here.

Nothing in this directory ships with the package. It starts empty, and the
scripts and notebooks fill it in as you run them.

    fisher_derivs/<configuration>/   Fisher derivative directories, written
                                     by run_hdInitPk_forecasts.py and by
                                     run_hdInitPk_forecasts.ipynb. These
                                     are large, tens of GB in total.
    fisher_matrices/<name>.txt       Fisher matrices rebuilt from those
                                     derivatives, in the same format as the
                                     ones in ../fisher_matrices, so the two
                                     sets can be compared directly.
    chains/<name>.*                  MCMC chains you have run from the
                                     input files prepared by
                                     run_hdInitPk_current.ipynb, at full
                                     length and in cobaya's own format,
                                     named after the input file.
    cached_results/<name>.txt        Statistics computed from those chains.
    plin_z0/                         P_lin(k, z=0) samples from
                                     run_hdInitPk_to_LinearPk.py.
    fig6/                            The Figure 6 points made from them.

To use what you have generated instead of what ships with the package, set

    chain_source = 'custom'

in hdInitPk_plots.ipynb, or CHAIN_SOURCE = 'custom' in
run_hdInitPk_to_LinearPk.py.
