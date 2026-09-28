Cobaya input files for the current-data MCMC chains: the CMB-PAS + DESI DR2
runs for the four cosmological models of Figure 1 and the
current-constraints table, plus the P-ACT-LB comparison chain.

All five are present. Each produced one of the thinned chains distributed
in ../../data/chains/:

    lcdm_nrun.yaml          LCDM + alpha_s                     pas/pas_lcdm_nrun
    w0wacdm_nrun.yaml       w0waCDM + alpha_s                  pas/pas_w0wacdm_nrun
    lcdm_nrun_nnu.yaml      LCDM + alpha_s + N_eff             pas/pas_lcdm_nrun_nnu
    lcdm_nrun_nnu_mnu.yaml  LCDM + alpha_s + N_eff + sum m_nu  pas/pas_lcdm_nrun_nnu_mnu
    p_act_lb.yaml           LCDM + alpha_s, on P-ACT-LB        P-ACT-LB/pact_lb_nrun

The four CMB-PAS files use the same likelihood blocks as the binned-P(k)
files in ../binned_pk/ (candl for SPT-3G D1, clipy for the Planck clik
likelihoods, the ACT DR6 CMB-only likelihood, the joint ACT+Planck+SPT-3G
lensing likelihood, and the DESI DR2 BAO likelihood), but with a power-law
primordial spectrum rather than the binned theory, so they carry
logA/ns/nrun in place of the eneg2tauPk<n> or b<n> amplitudes.

p_act_lb.yaml is a lighter combination: Planck low-l TT and low-l EE
(sroll2), ACT DR6 CMB-only with the Planck cut, ACT DR6 + Planck lensing,
and DESI 2024 BAO.

Every absolute path has been replaced by a /path/to/... placeholder, in the
same convention used by ../binned_pk/ and ../HD/; each file carries a
header listing the placeholders it uses. Each sampler.mcmc.covmat points at
a proposal matrix in ../../data/proposal_matrices/ built from that run's own
converged chain. All five are exact matches, covering every parameter the
run samples.

The external likelihoods these need, with links, are listed in the
repository README under "External likelihoods". Note that all five set
recombination_model: CosmoRec, which requires a CAMB built against
CosmoRec.
