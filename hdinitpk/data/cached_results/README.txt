Marginalized statistics of the binned-P(k) MCMC chains, computed from the
FULL (un-thinned) chains and read by `load_saved_results` in
hdInitPk_plots.ipynb. Format: one row per quantity, the row name followed
by its values, with '#' comment lines ignored.

Expected files:

    binned_pk_stats_7bin.txt
        Rows: p_act_means, p_act_lower_errs, p_act_upper_errs,
        p_act_tot_err, pas_desi_means, pas_desi_lower_errs,
        pas_desi_upper_errs, pas_desi_tot_err, pas_means, pas_lower_errs,
        pas_upper_errs, pas_tot_err, and the corresponding *_pk_* rows for
        the derived raw P(k) (p_act_pk_means, ... , pas_pk_tot_err).
        Each row holds 7 values (one per k bin).

    binned_pk_stats_30bin.txt
        Rows: p_act_means, p_act_lower_errs, p_act_upper_errs,
        pas_desi_means, pas_desi_lower_errs, pas_desi_upper_errs,
        pas_means, pas_lower_errs, pas_upper_errs, official_means,
        official_lower_errs, official_upper_errs.
        Each row holds 30 values (one per k bin).
