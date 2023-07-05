import h5py as h5
import numpy as np
from nbodykit.lab import cosmology

from src.config import SRC_PATH


def compute_zeldovich_approx_cf(cosmo: dict, fname: str) -> None:
    # We compute the linear correlation function for completeness.
    c = cosmology.Cosmology(
        h=cosmo["h"],
        Omega0_b=cosmo["Ob0"],
        Omega0_cdm=cosmo["Om0"] - cosmo["Ob0"],
        n_s=cosmo["ns"],
    ).match(sigma8=cosmo["sigma8"])

    # Power spectra
    pk_lin = cosmology.LinearPower(c, redshift=0, transfer="CLASS")
    pk_zel = cosmology.ZeldovichPower(c, redshift=0)
    # Correlation functions
    cf_lin = cosmology.CorrelationFunction(pk_lin)
    cf_zel = cosmology.CorrelationFunction(pk_zel)

    k = np.logspace(-3, 1, 1_000)
    p_lin = pk_lin(k)
    p_zel = pk_zel(k)

    r = np.logspace(-2, np.log10(150), 1_000)
    xi_lin = cf_lin(r)
    xi_zel = cf_zel(r)

    with h5.File(SRC_PATH + f"/data/{fname}.h5", "w") as hdf:
        hdf.create_dataset("k", data=k)  # h/Mpc
        hdf.create_dataset("r", data=r)  # Mpc//h
        hdf.create_dataset("pk_lin", data=p_lin)  # (Mpc/h)^3
        hdf.create_dataset("pk_zel", data=p_zel)  # (Mpc/h)^3
        hdf.create_dataset("xi_lin", data=xi_lin)
        hdf.create_dataset("xi_zel", data=xi_zel)
    return


if __name__ == "__main__":
    pass
#