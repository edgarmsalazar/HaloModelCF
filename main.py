from warnings import filterwarnings

filterwarnings("ignore")

from physhalo.utils import timer


@timer
def catalogue() -> None:
    from physhalo.catalogue import (generate_halo_mass_bin_masks,
                                    generate_particle_mass_bin_masks)

    generate_halo_mass_bin_masks()
    generate_particle_mass_bin_masks()

    return None


@timer
def physhalo() -> None:
    from physhalo.cosmology import COSMO, COSMO_QUIJOTE
    from physhalo.hmcorrfunc.tpcf import compute_hmcf, compute_mmcf
    from physhalo.hmcorrfunc.tpcf_split import (compute_hmcf_cov_split,
                                                compute_hmcf_split,
                                                extend_xi_inf)
    from physhalo.hmcorrfunc.zeldovich_cf import compute_zeldovich_approx_cf

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
    from physhalo.config import NMBINS
    from physhalo.hmcorrfunc.mass import mass_correction
    from physhalo.hmcorrfunc.mcmc import BinMC, FullMC
    from physhalo.hmcorrfunc.model import (lnpost_inf, lnpost_inf_1,
                                           lnpost_orb,
                                           lnpost_orb_smooth_a_pl_ai_pl,
                                           lnpost_orb_smooth_ai_pl,
                                           lnpost_orb_smooth_a_pl)

    # mass_correction()
    # for i in range(NMBINS):
    #     bmc = BinMC(
    #         ndim=4,
    #         nwalkers=100,
    #         nsteps=10_000,
    #         burnin=5_000,
    #         log_posterior=lnpost_orb,
    #         mbin_i=i,
    #         ptype="orb",
    #     )
    #     print("Initial point ", bmc.pars_init)
    #     bmc.run_chain()
    #     bmc.summary()
        # break
    
    # Fit smooth model with all parameters power-laws of mass
    # fmc = FullMC(
    #     ndim=7,
    #     nwalkers=100,
    #     nsteps=10_000,
    #     burnin=2_000,
    #     log_posterior=lnpost_orb_smooth_a_pl_ai_pl,
    #     ptype="orb",
    # )
    # print("Initial point ", fmc.pars_init)
    # fmc.run_chain()
    # fmc.summary()
    
    # Fit smooth model with a mass-independent
    # fmc = FullMC(
    #     ndim=6,
    #     nwalkers=100,
    #     nsteps=5_000,
    #     burnin=2_000,
    #     log_posterior=lnpost_orb_smooth_ai_pl,
    #     ptype="orb",
    # )
    # print("Initial point ", fmc.pars_init)
    # fmc.run_chain()
    # fmc.summary()
    
    # Fit smooth model with alpha_infty mass-independent
    fmc = FullMC(
        ndim=6,
        nwalkers=100,
        nsteps=5_000,
        burnin=2_000,
        log_posterior=lnpost_orb_smooth_a_pl,
        ptype="orb",
    )
    print("Initial point ", fmc.pars_init)
    fmc.run_chain()
    fmc.summary()
    


    # # NOTE: The infall profile takes rh as argument. So the orbiting profile
    # # must be fit first.
    # for i in range(NMBINS):
    #     # if i == 0: continue
    #     bmc = BinMC(
    #         ndim=6,
    #         nwalkers=100,
    #         nsteps=10_000,
    #         burnin=10_000,
    #         log_posterior=lnpost_inf,
    #         mbin_i=i,
    #         ptype="inf",
    #     )
    #     print("Initial point ", bmc.pars_init)
    #     bmc.run_chain()
    #     bmc.summary()
    #     break

    return None


@timer
def main() -> None:
    # catalogue()         # Takes ~1 min
    # physhalo()              # Takes ~1 hr
    model_fitting()     # Takes ~2 hr

    return None


if __name__ == "__main__":
    main()
#
