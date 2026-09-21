Thinned MCMC chains, in getdist format, loaded by `load_thinned_chain` in
hdInitPk_plots.ipynb. The burn-in (ignore_rows = 0.5) was removed BEFORE
thinning, so the notebook loads them with ignore_rows = 0.

`load_thinned_chain` resolves a chain by name: it accepts either
`chains/<name>` or `chains/<subdir>/<name>`, so the subdirectories below
may be organized freely.

    pas/            CMB-PAS (+ DESI DR2) chains for the four cosmological
                    models, loaded as:
                        pas_lcdm_nrun
                        pas_w0wacdm_nrun
                        pas_lcdm_nrun_nnu
                        pas_lcdm_nrun_nnu_mnu

    P-ACT-LB/       the P-ACT-LB (LCDM + alpha_s) comparison chain, loaded
                    as:
                        pact_lb_nrun

    hd/             the CMB-HD + DESI BAO validation chains of Figure 9,
                    loaded as:
                        hd_camb_mcmc_9param
                        hd_class_mcmc_9param

    binned_pk/      the binned-P(k) chains, for the three data
                    combinations at both binnings:
                        pas_7bin        pas_30bin
                        pas_desi_7bin   pas_desi_30bin
                        pact_lb_7bin    pact_lb_30bin

                    The notebook itself reads only their precomputed
                    marginalized statistics (see ../cached_results);
                    run_hdInitPk_to_LinearPk.py reads `pas_7bin` directly, and
                    any of the six can be substituted by changing its
                    `chain_root`.

                    Their bin-amplitude columns keep whatever the run used:
                    `eneg2tauPk<n>` (absolute units) for the chains in the
                    paper, or `b<n>` (units of 1e-9) for a chain run from
                    one of the *_arbitrary_binning.yaml files.
                    `run_hdInitPk_to_LinearPk.get_pk_columns` accepts either.

Plus one plain text file in this directory:

    chain_convergence.txt
        Gelman-Rubin R-1 of each FULL chain, recorded before thinning (it
        cannot be recomputed from a merged thinned chain). One row per
        chain: "<root name> <R-1 value>".
