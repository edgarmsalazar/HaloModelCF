from warnings import filterwarnings

filterwarnings("ignore")

from hmcf.utils import timer


@timer
def catalogue() -> None:
    from hmcf.catalogue import (generate_halo_mass_bin_masks,
                               generate_particle_mass_bin_masks)

    generate_halo_mass_bin_masks()
    generate_particle_mass_bin_masks()

    return None


@timer
def hmcf() -> None:
    from hmcf.cosmology import COSMO, COSMO_QUIJOTE
    from hmcf.tpcf import compute_hmcf, compute_mmcf
    from hmcf.tpcf_split import (compute_hmcf_cov_split, compute_hmcf_split,
                                extend_xi_inf)
    from hmcf.zeldovich_cf import compute_zeldovich_approx_cf

    compute_mmcf()
    compute_hmcf()
    compute_hmcf_split()
    compute_hmcf_cov_split()
    extend_xi_inf()
    compute_zeldovich_approx_cf(COSMO, "xi_zel")
    compute_zeldovich_approx_cf(COSMO_QUIJOTE, "xi_zel_quijote")

    return None


@timer
def model_fitting() -> None:
    from hmcf.config import NMBINS
    from hmcf.mass import mass_correction
    from hmcf.mcmc import BinMC, FullMC
    from hmcf.model import lnpost_inf, lnpost_inf_1, lnpost_orb, lnpost_orb_smooth

    # mass_correction()

    # for i in range(NMBINS):
    #     bmc = BinMC(
    #         ndim=4,
    #         nwalkers=40,
    #         nsteps=10_000,
    #         burnin=10_000,
    #         log_posterior=lnpost_orb,
    #         mbin_i=i,
    #         ptype="orb",
    #     )
    #     print("Initial point ", bmc.pars_init)
    #     bmc.run_chain()
    #     bmc.summary()
        
    # fmc = FullMC(
    #     ndim=6,
    #     nwalkers=100,
    #     nsteps=10_000,
    #     burnin=10_00,
    #     log_posterior=lnpost_orb_smooth,
    #     ptype="orb",
    # )
    # print("Initial point ", fmc.pars_init)
    # fmc.run_chain()
    # fmc.summary()

    # NOTE: The infall profile takes rh as argument. So the orbiting profile
    # must be fit first.
    for i in range(NMBINS):
        # if i == 0: continue
        bmc = BinMC(
            ndim=6,
            nwalkers=100,
            nsteps=10_000,
            burnin=10_000,
            log_posterior=lnpost_inf,
            mbin_i=i,
            ptype="inf",
        )
        print("Initial point ", bmc.pars_init)
        bmc.run_chain()
        bmc.summary()
        break

    return None


@timer
def main() -> None:
    # catalogue()         # Takes ~1 min
    # hmcf()              # Takes ~1 hr
    model_fitting()     # Takes ~2 hr

    return None


if __name__ == "__main__":
    main()
#
