from os.path import join

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from chainconsumer import ChainConsumer
from emcee import EnsembleSampler
from emcee.backends import HDFBackend
from matplotlib.colorbar import ColorbarBase
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize

from physhalo.config import MBINEDGES, MBINSTRS, NMBINS, SRC_PATH
from physhalo.cosmology import RHOM, RSOFT
from physhalo.hmcorrfunc.model import (error_function, power_law,
                                       rho_orb_model, xi_inf_model)
from physhalo.plot.config import (CMAP, COLOR_GRAY, NORM, SIZE_LABELS,
                                  SIZE_LEGEND, SIZE_TICKS, LEGEND_3)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "savefig.facecolor": "white",
        "figure.dpi": 150,
    }
)

def fits_hm_profile():
    # Load data
    with h5.File(join(SRC_PATH, "data/xihm.h5"), "r") as hdf_load:
        r = hdf_load["rbins"][()]
        xi = np.zeros((NMBINS, len(r)))
        xi_err = np.zeros((NMBINS, len(r)))
        for k, mbin in enumerate(MBINSTRS):
            xi[k, :] = hdf_load[f"xi/100/{mbin}"][()]
            xi_err[k, :] = np.sqrt(np.diag(hdf_load[f"cov/100/{mbin}"][()]))

    with h5.File(join(SRC_PATH, "data/xihm_split.h5"), "r") as hdf_load:
        r_orb = hdf_load["rbins"][()]
        r_inf = hdf_load["rbins_ext"][()]
        xi_orb = np.zeros((NMBINS, len(r_orb)))
        xi_inf = np.zeros((NMBINS, len(r_inf)))
        for k, mbin in enumerate(MBINSTRS):
            xi_orb[k, :] = hdf_load[f"xi/orb/{mbin}"][()]
            xi_inf[k, :] = hdf_load[f"xi_ext/inf/{mbin}"][()]
            
    # Setup canvas
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    # Create a color bar to display mass ranges.
    cax = fig.add_axes([1.01, 0.2, 0.02, 0.7])
    cbar = ColorbarBase(cax, cmap=CMAP.name, norm=NORM, orientation="vertical")
    cbar.set_label(r"$\log_{10} M_{\rm orb}$", fontsize=SIZE_LEGEND)
    cbar.set_ticklabels([f"${m:.2f}$" for m in MBINEDGES], fontsize=SIZE_TICKS)

    plt.sca(ax)
    # Plot all mass bins.
    for k, mbin in enumerate(MBINSTRS):
        # Plot data
        plt.errorbar(
            r,
            r ** (2) * xi[k, :],
            yerr=r ** (2) * xi_err[k, :],
            fmt=".",
            elinewidth=0.5,
            capsize=3,
            color=CMAP(k / NMBINS),
            alpha=0.5,
            label=f"{mbin}",
        )
        plt.plot(r_orb, r_orb ** (2) * xi_orb[k, :], lw=1, ls="--", color=CMAP(k/NMBINS))
        plt.plot(r_inf, r_inf ** (2) * xi_inf[k, :], lw=1, ls=":", color=CMAP(k/NMBINS))

    plt.fill_betweenx([1, 1e3], 0, 6 * RSOFT, color="k", alpha=0.25)
    plt.xlim(2e-2, 50)
    plt.xscale("log")
    plt.xlabel(r"$r [h^{-1}{\rm Mpc}]$", fontsize=SIZE_LABELS)
    plt.ylim(8, 500)
    plt.yscale("log")
    plt.ylabel(r"$r^2\xi_{\rm hm}\left(r|M\right)$", fontsize=SIZE_LABELS)
    plt.tick_params(axis="both", which="major", labelsize=SIZE_TICKS)
    plt.legend(handles=LEGEND_3, title="Particles", loc="lower right")

    plt.tight_layout()
    plt.savefig(SRC_PATH + "/data/plot/fits_xihm_profile.png", bbox_inches="tight")
    
    return


if __name__ == "__main__":
    pass
