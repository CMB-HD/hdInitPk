MCMC proposal covariance matrices, named by the `sampler.mcmc.covmat`
entries of the files in ../../cobaya_yaml_files/.

Each was built from its own run's full, converged chain, so every Cobaya input file gets the
proposal from the chain it produced:

    CMB-HD mock chains (Figure 9)
        hd_camb_6param.covmat      HD/camb_lcdm_likelihood.yaml
        hd_camb_9param.covmat      HD/camb_nrun_likelihood.yaml
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
