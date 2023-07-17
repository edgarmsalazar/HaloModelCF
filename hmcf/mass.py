from typing import List, Tuple, Union

import h5py as h5
import numpy as np
from scipy.integrate import quad, simps
from scipy.optimize import minimize

from hmcf.config import MBINSTRS, NMBINS, SRC_PATH
from hmcf.cosmology import RHOM, RSOFT
from hmcf.model import rho_orb_model_norm


def mass_correction() -> None:
    """Compute a mass correction model."""

    def chi2_cost(
        pars: Union[List[float], Tuple[float, ...]],
        *data,
    ) -> float:
        # Unpack data
        x, y, covy = data
        # Compute model deviation from data
        u = y - rho_orb_model_norm(x, *pars)
        return 0.5 * np.dot(u, np.linalg.solve(covy, u))

    # Load mean mass in the mass bin.
    with h5.File(SRC_PATH + "/data/mass_bin_haloes.h5", "r") as hdf:
        mass_mean = hdf["mean"][()]
    mass_corrected = np.zeros_like(mass_mean)
    mass_from_fit = np.zeros_like(mass_mean)

    print("logA\t rh\t ainf\t a")
    with h5.File(SRC_PATH + "/data/xihm_split.h5", "r") as hdf:
        rbins = hdf["rbins"][()]
        r_mask = rbins > 6 * RSOFT
        # Load density profiles and rbins.
        for i, mbin in enumerate(MBINSTRS):
            x = rbins[r_mask]
            y = hdf[f"rho/orb/{mbin}"][r_mask]
            covy = RHOM**2 * hdf[f"xi_cov/all/{mbin}"][()][r_mask, :][:, r_mask]

            # Fit orbiting model with the mean mass.
            res = minimize(
                chi2_cost,
                [np.log10(mass_mean[i]), 1.0, 2.0, 0.1],
                args=(x, y, covy),
                method="Nelder-Mead",
                options={"maxiter": 10_000},
            )
            # print(res)
            print("".join("{:>6.3f}\t".format(i) for i in res.x))

            # Evaluate model at each rbin < 6*rsoft
            rbins_inner = rbins[~r_mask]
            rho_data = hdf[f"rho/orb/{mbin}"][~r_mask]

            # Compute total integrated mass for the model:
            #                   Morb = int(rho_model*dV)
            int1 = quad(
                func=lambda x, pars: x**2 * rho_orb_model_norm(x, *pars),
                a=0,
                b=np.inf,
                args=(res.x[:]),
            )[0]
            mass_from_fit[i] = 4 * np.pi * int1

            # Evaluate rho_model at small scales
            rho_model = rho_orb_model_norm(rbins_inner, *res.x)

            # Integrate (rho_model - rho_data)*dV
            int2 = simps(rbins_inner**2 * (rho_model - rho_data), rbins_inner)
            correction = 4 * np.pi * int2

            # Corrected mass is <M>s = <Morb> + correction, where <Morb> is the
            # mean mass in the bin. This should be equal to the total integrated
            # mass <M>s = Morb
            mass_corrected[i] = mass_mean[i] + correction
            # break
    print("\n")
    print("<Morb>\t Morb\t <M>_s\t Perc Diff\t\t")
    for i in range(NMBINS):
        print(
            f"{np.log10(mass_mean[i]):.4f}\t{np.log10(mass_from_fit[i]):.4f}\t"
            f"{np.log10(mass_corrected[i]):.4f}\t"
            f"{(mass_from_fit[i]-mass_corrected[i])/mass_from_fit[i]: .2%}\t"
            f"{(mass_from_fit[i]-mass_mean[i])/mass_from_fit[i]: .2%}"
        )
    with h5.File(SRC_PATH + "/data/mass.h5", "w") as hdf_save:
        hdf_save.create_dataset("mass", data=mass_corrected)

    return


if __name__ == "__main__":
    pass
#
