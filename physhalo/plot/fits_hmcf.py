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
from physhalo.hmcorrfunc.model import xihm_model, power_law, error_function
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


def fits_hm():
    M_PIVOT = np.power(10.0, 14)
    # Load data
    with h5.File(SRC_PATH + "/data/mass.h5", "r") as hdf_load:
        mass = hdf_load["mass"][()]
        
    pars_orb = np.zeros((NMBINS, 4))
    pars_inf = np.zeros((NMBINS, 6))
    with h5.File(SRC_PATH + "/data/fits/mle.h5", "r") as hdf_load:
        for k, mbin in enumerate(MBINSTRS):
            pars_orb[k, :] = hdf_load[f"max_posterior/orb/{mbin}"][()]
            pars_inf[k, :] = hdf_load[f"max_posterior/inf/{mbin}"][()]
        pars_smooth = hdf_load[f"max_posterior/all/smooth_1"][()]
        
    with h5.File(join(SRC_PATH, "data/xihm.h5"), "r") as hdf_load:
        r = hdf_load["rbins"][()]
        xi = np.zeros((NMBINS, len(r)))
        xi_err = np.zeros((NMBINS, len(r)))
        xi_err_smooth = np.zeros((NMBINS, len(r)))
        for k, mbin in enumerate(MBINSTRS):
            xi[k, :] = hdf_load[f"xi/100/{mbin}"][()]
            xi_err[k, :] = np.sqrt(np.diag(hdf_load[f"cov/100/{mbin}"][()]))
            xi_err_smooth[k, :] = np.sqrt(np.diag(hdf_load[f"cov/100/{mbin}"][()]) + (10**pars_smooth[-1]*xi[k, :])**2)

    # Setup canvas
    fig = plt.figure(layout="constrained", figsize=(11, 4))
    subfigs = fig.subfigures(1, 3, wspace=0.07, width_ratios=[4, 3, 4])

    ax = subfigs[0].subplots(1, 1)
    # Plot all mass bins.
    for k, mbin in enumerate(MBINSTRS):
        r_pred = np.logspace(-2, np.log10(max(r+1)), num=100, base=10)
        xi_pred = xihm_model(r_pred, *pars_orb[k, :-1], *pars_inf[k, :-1], mass[k])
        
        # Plot data
        # Plot data
        ax.errorbar(r, r ** (2) * xi[k, :], yerr=r ** (2) * xi_err[k, :],
                    fmt=".", elinewidth=0.5, capsize=3, color=CMAP(k / NMBINS),
                    alpha=0.5, label=f"{mbin}")
        ax.plot(r_pred, r_pred ** (2) * xi_pred, ls='-', lw=0.5, 
                color=CMAP(k/NMBINS))
    
    ax.fill_betweenx([1, 1e3], 0, 6*RSOFT, color='k', alpha=0.25)
    ax.set_xlim(2e-2, 50)
    ax.set_xscale("log")
    ax.set_xlabel(r"$r [h^{-1}{\rm Mpc}]$", fontsize=SIZE_LABELS)
    ax.set_ylim(8, 500)
    ax.set_yscale("log")
    ax.set_ylabel(r"$r^2\xi_{\rm hm}\left(r|M\right)$", fontsize=SIZE_LABELS)
    ax.text(6, 350, s="Ind. Fit")
    ax.tick_params(axis="both", which="major", labelsize=SIZE_TICKS)

    ax = subfigs[-1].subplots(1, 1)
    # Plot all mass bins.
    for k, mbin in enumerate(MBINSTRS):
        r_pred = np.logspace(-2, np.log10(max(r+1)), num=100, base=10)
        xi_pred = xihm_model(r_pred, *pars_orb[k, :-1], *pars_inf[k, :-1], mass[k])
        
        # Plot data
        # Plot data
        ax.errorbar(r, r ** (2) * xi[k, :], yerr=r ** (2) * xi_err[k, :],
                    fmt=".", elinewidth=0.5, capsize=3, color=CMAP(k / NMBINS),
                    alpha=0.5, label=f"{mbin}")
        ax.plot(r_pred, r_pred ** (2) * xi_pred, ls='-', lw=0.5, 
                color=CMAP(k/NMBINS))

    ax.fill_betweenx([1, 1e3], 0, 6*RSOFT, color='k', alpha=0.25)
    ax.set_xlim(2e-2, 50)
    ax.set_xscale("log")
    ax.set_xlabel(r"$r [h^{-1}{\rm Mpc}]$", fontsize=SIZE_LABELS)
    ax.set_ylim(8, 500)
    ax.set_yscale("log")
    ax.set_ylabel(r"$r^2\xi_{\rm hm}\left(r|M\right)$", fontsize=SIZE_LABELS)
    ax.text(6, 350, s="Smooth")
    ax.tick_params(axis="both", which="major", labelsize=SIZE_TICKS)


    axes = subfigs[1].subplots(2, 1, sharex=True, sharey=True,
                            gridspec_kw={'hspace': 0, 'wspace': 0})
    ax1, ax2= axes.flatten()
    in_text = ["Ind. Fit", "Smooth"]
    for i, ax in enumerate(axes.flatten()):
        ax.tick_params(axis='both', which='major', labelsize=SIZE_TICKS)
        ax.fill_betweenx([-1, 1], 0, 6*RSOFT, color='k', alpha=0.25)
        ax.set_xlim(2e-2, 50)
        ax.set_xscale('log')
        ax.set_ylim(-0.12, 0.12)
        ax.hlines(0, 0, 50, color='k')
        ax.hlines(0.05, 0, 50, color='k', ls=':', lw=1.0)
        ax.hlines(-0.05, 0, 50, color='k', ls=':', lw=1.0)
        ax.text(0.5, -0.1, s=in_text[i])
    ax1.set_ylabel(r'fract. error', fontsize=SIZE_LABELS)
    ax2.set_ylabel(r'fract. error', fontsize=SIZE_LABELS)
    ax2.set_xlabel(r'$r [h^{-1}{\rm Mpc}]$', fontsize=SIZE_LABELS)
    
    # Plot all mass bins.
    for k, mbin in enumerate(MBINSTRS):
        # Evaluate model over grid for smooth profiles
        rh = power_law(mass[k]/M_PIVOT, *pars_smooth[:2])
        ainf = power_law(mass[k]/M_PIVOT, *pars_smooth[2:4])
        a = pars_smooth[4]
        bias = power_law(mass[k]/M_PIVOT, *pars_smooth[5:7])
        c = error_function(np.log10(mass[k]/M_PIVOT), *pars_smooth[7:10])
        g = power_law(mass[k]/M_PIVOT, *pars_smooth[10:12])
        rinf = pars_smooth[12]
        mu = pars_smooth[13]
        
        ratio = xihm_model(r, *pars_orb[k, :-1], *pars_inf[k, :-1], mass[k]) / xi[k, :] - 1
        ratio_smooth_1 = xihm_model(r, rh, ainf, a, bias, c, g, rinf, mu, mass[k]) / xi[k, :] - 1
        
        # Plot ratio and error bands
        ax1.plot(r, ratio, lw=1, color=CMAP(k/NMBINS))
        ax1.fill_between(r, xi_err[k, :]/xi[k, :], - xi_err[k, :]/xi[k, :], 
                        color=CMAP(k/NMBINS), alpha=0.1)
        
        ax2.plot(r, ratio_smooth_1, lw=1, color=CMAP(k/NMBINS))
        ax2.fill_between(r, xi_err_smooth[k, :]/xi[k, :],
                        - xi_err_smooth[k, :]/xi[k, :], 
                        color=CMAP(k/NMBINS), alpha=0.1)
    
    
    plt.savefig(SRC_PATH + "/data/plot/fits_xihm.png", bbox_inches="tight")
    return



def fit_hm_corner() -> None:
    sampler = HDFBackend(
        SRC_PATH + "/data/fits/chains.h5", name="all/smooth_1", read_only=True
    )
    flat_samples = sampler.get_chain(flat=True)
    plabs = [
        r"$r_{{\rm h}, p}$",
        r"$r_{{\rm h}, s}$",
        r"$\alpha_{\infty, p}$",
        r"$\alpha_{\infty, s}$",
        r"$a$",
        r"$b_{p}$",
        r"$b_{s}$",
        r"$c_0$",
        r"$c_\mu$",
        r"$c_\sigma$",
        r"$\gamma_{p}$",
        r"$\gamma_{s}$",
        r"$r_{\rm inf}$",
        r"$\mu$",
        r"$\log_{10} \delta$",
    ]

    c = ChainConsumer()
    c.add_chain(flat_samples, parameters=plabs)
    c.configure(
        summary=False,
        sigmas=[1, 2],
        # colors=["#6591b5"],
        colors=[COLOR_GRAY],
        tick_font_size=8,
        label_font_size=SIZE_LABELS,
        max_ticks=3,
        usetex=True,
    )
    fig = c.plotter.plot(
        figsize=(7.2, 7.2),
        parameters=plabs[:-1]
    )
    fig.align_labels()
    plt.savefig(SRC_PATH + "/data/plot/fits_xihm_corner.png", bbox_inches="tight")

    c = ChainConsumer()
    c.add_chain(flat_samples, parameters=plabs)
    c.configure(
        summary=True,
        sigmas=[1, 2],
        colors=[COLOR_GRAY],
        # tick_font_size=SIZE_TICKS,
        # label_font_size=SIZE_LABELS,
        usetex=True,
    )
    fig = c.plotter.plot_distributions(col_wrap=5)
    fig.tight_layout()
    plt.savefig(SRC_PATH + "/data/plot/fits_xihm_post.png", bbox_inches="tight")
    return None



if __name__ == "__main__":
    pass
