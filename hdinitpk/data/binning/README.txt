Binning files for the primordial power spectrum.

    hd_pk_wavenumbers_11bins.txt   
        The 11-bin CMB-HD forecast binning: k bin center, lower bin edge,
        and upper bin edge, in Mpc^-1. The centers are the arithmetic
        midpoints of the edges and match the k1..k11 entries of
        ../fisher_fid_params/hd_binned_pk_fiducial_params.yaml exactly.

    binning_62bins.txt             
        Two columns (k bin center [Mpc^-1] and the fiducial P(k)) for
        the 62-bin (kmax = 50 Mpc^-1) ACT-style binning. The notebook uses
        the first 30 rows for Figure 10.

    act_bin_edges_kmax50.txt      
        The bin edges [Mpc^-1] of that same binning, used to draw the
        box-style error bars in Figure 10.
