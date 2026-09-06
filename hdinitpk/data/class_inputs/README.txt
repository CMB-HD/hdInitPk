Inputs that CLASS needs by absolute path.

    PRIMAT21_class_format.dat
        The PRIMAT 2021 BBN table, converted to the three-column CLASS
        format (ombh2, DeltaN, Yp) from CAMB's
        `PRIMAT_Yp_DH_ErrorMC_2021.dat`.

        Referenced as `sBBN file:` by
        ../fisher_fid_params/class_fiducial_params.yaml and by the CLASS
        entries in ../../cobaya_yaml_files/HD/. CLASS requires an absolute
        path, so those files carry a `/path/to/hdinitpk/...` placeholder
        that must be edited to point here.
