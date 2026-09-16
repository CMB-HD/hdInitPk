MCMC proposal covariance matrices, named by the `sampler.mcmc.covmat`
entries of the files in ../../cobaya_yaml_files/. They are small enough to
distribute, so they belong here rather than in the external data directory.

All sixteen are present. Each was built by save_proposal_matrices.py from
its own run's full, converged chain, so every Cobaya input file gets the
proposal from the chain it produced:

    CMB-HD mock chains (Figure 9)
        hd_camb_6param.covmat      HD/camb_nrun_likelihood.yaml
        hd_camb_9param.covmat      HD/camb_lcdm_likelihood.yaml
        hd_class_6param.covmat     HD/class_lcdm_likelihood.yaml
        hd_class_9param.covmat     HD/class_nrun_likelihood.yaml

    binned P(k), CMB-PAS
        cmb_pas_7_bins.covmat        binned_pk/cmb_pas_7_bins*.yaml
        cmb_pas_30_bins.covmat       binned_pk/cmb_pas_30_bins*.yaml
        cmb_pas_desi_7_bins.covmat   binned_pk/cmb_pas_desi_7_bins*.yaml
        cmb_pas_desi_30_bins.covmat  binned_pk/cmb_pas_desi_30_bins*.yaml

    binned P(k), P-ACT-LB
        act_7_bins_no_sroll.covmat   binned_pk/p_act_lb_7_bins_no_sroll*.yaml
        act_30_bins.covmat           binned_pk/p_act_lb_30_bins*.yaml
        act_30_bins_no_sroll.covmat  binned_pk/p_act_lb_30_bins_no_sroll*.yaml

    current data, power law
        cmb_pas_desi_nrun.covmat           current_data/lcdm_nrun.yaml
        cmb_pas_desi_nrun_w0wa.covmat      current_data/w0wacdm_nrun.yaml
        cmb_pas_desi_nrun_neff.covmat      current_data/lcdm_nrun_nnu.yaml
        cmb_pas_desi_nrun_neff_mnu.covmat  current_data/lcdm_nrun_nnu_mnu.yaml
        p_act_lb_nrun.covmat               current_data/p_act_lb.yaml

The bin amplitudes in the binned-P(k) covmats are named `b1`..`bN`, in
units of `arbitrary_Pkbinning.BinnedPk`'s `scale` (1e-9), so they are a full
match for the `*_arbitrary_binning.yaml` runs: every parameter those files
sample appears in the covmat.

The seven original (non-arbitrary) binned_pk files sample `eneg2tauPk1`..
`eneg2tauPkN` instead. They get the cosmology and nuisance entries from the
same covmats but not the bins, and cobaya falls back to each bin's own
`proposal` for those. To build a matching matrix, set
OLD_BIN_PREFIX = NEW_BIN_PREFIX = 'eneg2tauPk' in save_proposal_matrices.py,
which disables both the rename and the rescaling.
