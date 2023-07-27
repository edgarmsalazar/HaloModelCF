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
from physhalo.cosmology import RSOFT
from physhalo.hmcorrfunc.model import error_function, power_law, xi_inf_model
from physhalo.plot.config import (CMAP, COLOR_GRAY, NORM, SIZE_LABELS,
                                  SIZE_LEGEND, SIZE_TICKS)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "savefig.facecolor": 'white',
    "figure.dpi": 150,
})

def fits_inf_profile():
    # Load data
    with h5.File(join(SRC_PATH, "data/xihm_split.h5"), 'r') as hdf:
        r = hdf['rbins_ext'][()]  # x
        xi = np.zeros((NMBINS, len(r)))
        xi_err = np.zeros((NMBINS, len(r)))
        for k, mbin in enumerate(MBINSTRS):
            xi[k, :] = hdf[f'xi_ext/inf/{mbin}'][()]
            xi_err[k, :] = np.sqrt(np.diag(hdf[f'xi_ext_cov/inf/{mbin}'][()]))
    
    # Setup canvas
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    # Create a color bar to display mass ranges.
    cax = fig.add_axes([1.01, 0.2, 0.02, 0.7])
    cbar = ColorbarBase(cax, cmap=CMAP.name, norm=NORM, orientation="vertical")
    cbar.set_label(r"$\log_{10} M_{\rm orb}$", fontsize=SIZE_LEGEND)
    cbar.set_ticklabels([f"${m:.2f}$" for m in MBINEDGES], fontsize=SIZE_TICKS)

    plt.sca(ax)
    # Plot all mass bins.
    for k in range(NMBINS):
        # Plot data
        plt.errorbar(
            r,
            r ** (3/2) * xi[k, :],
            yerr=r ** (3/2) * xi_err[k, :],
            fmt=".",
            elinewidth=0.5,
            capsize=3,
            color=CMAP(k / NMBINS),
            alpha=0.5,
            label=f"{mbin}",
        )
        
    plt.fill_betweenx([-1, 1e3], 0, 6*RSOFT, color='k', alpha=0.25)
    plt.xlim(2e-2, 50)
    plt.xscale("log")
    plt.xlabel(r"$r [h^{-1}{\rm Mpc}]$", fontsize=SIZE_LABELS)
    plt.ylim(0.8, 80)
    plt.yscale('log')
    plt.ylabel(r'$r^{3/2}\xi_{\rm inf}(r|M)$', fontsize=SIZE_LABELS)
    plt.tick_params(axis="both", which="major", labelsize=SIZE_TICKS)
    
    plt.tight_layout()
    plt.savefig(SRC_PATH + "/data/plot/fits_inf_profile.png", bbox_inches="tight")
    return


def ximm_quijote() -> None:
    
    SRC_PATH_Q = "/spiff/edgarmsc/simulations/Quijote/CF/matter/fiducial/all/"
    n_obs = 15_000
    
    # Load data
    r_obs, _ = np.loadtxt(SRC_PATH_Q + "CF_m_z=0_0.txt", unpack=True)
    xi_obs = np.zeros((n_obs, len(r_obs)))
    for i in range(n_obs):
        _, xi_obs[i, :] = np.loadtxt(SRC_PATH_Q + f"CF_m_z=0_{i}.txt", unpack=True)
    
    xi_obs_cov = np.cov(xi_obs, rowvar=False, bias=False)
    xi_obs_mean = np.mean(xi_obs, axis=0)
    
    with h5.File(SRC_PATH + f"/data/xi_zel_quijote.h5", "r") as hdf_load:
        r_zel = hdf_load["r"][()]
        xi_zel = hdf_load["xi_zel"][()]
    xi_zel_interp = interp1d(r_zel, xi_zel)
    
    r_mask_fit = (r_obs > 30) * (r_obs < 50)
    def loglike(B) -> float:
        if not B > 0:
            return -np.inf
        u = xi_obs_mean[r_mask_fit]/xi_zel_interp(r_obs[r_mask_fit]) - B
        return -0.5 * np.dot(u, np.linalg.solve(xi_obs_cov[r_mask_fit, :][:, r_mask_fit], u))
    
    B_grid = np.linspace(0, 2, 100)
    loglike_grid = [loglike(bb) for bb in B_grid]

    plt.plot(B_grid, loglike_grid)
    plt.xlabel(r'$B$')
    plt.ylabel(r'$-\ln\mathcal{L}$')
    plt.savefig(SRC_PATH + "/data/plot/ximm_quijote_0.png", bbox_inches="tight")
    
    n_walkers = 50
    walker_init = 1 + 0.1*np.random.uniform(low=-1, size=(n_walkers, 1))
    sampler = EnsembleSampler(nwalkers=n_walkers, ndim=1, log_prob_fn=loglike)
    sampler.run_mcmc(initial_state=walker_init, nsteps=5_000, progress=True, 
                     progress_kwargs={"desc": f"Chain"});
    samples = sampler.get_chain(discard=2_000)
    log_prob = sampler.get_log_prob(discard=2_000)
    flat_samples = samples.reshape((-1, samples.shape[-1]))
    c = ChainConsumer()
    c.add_chain(flat_samples, parameters=[r'$B$'], posterior=log_prob.reshape(-1))
    c.configure(sigmas=[1, 2])

    # # Plot walks
    # fig = c.plotter.plot_walks(plot_posterior=True, log_scales={'posterior': True})
    # plt.savefig(SRC_PATH + "/data/plot/ximm_quijote_1.png", bbox_inches="tight")
    # c.plotter.plot()
    # plt.savefig(SRC_PATH + "/data/plot/ximm_quijote_2.png", bbox_inches="tight")
    
    (B_maxpost,  ) = [v for (_, v) in c.analysis.get_max_posteriors().items()]
    autocorr = sampler.get_autocorr_time()
    print(B_maxpost, autocorr)

    # Setup canvas
    _, axes = plt.subplots(2, 1, figsize=(8.571428571/2, 6), sharex=True, 
                             gridspec_kw={'hspace': 0, 'wspace': 0,
                                          'height_ratios': [2, 1]})

    ax1, ax2 = axes.flatten()
    for ax in axes.flatten():
        ax.tick_params(axis='both', which='major', labelsize=SIZE_TICKS)

    r_mask = (r_obs < 50)
    for i in range(n_obs):
        ax1.plot(r_obs[r_mask], r_obs[r_mask]**2*xi_obs[i, r_mask], color='k', alpha=0.01, lw=0.1)
    ax1.errorbar(r_obs[r_mask], r_obs[r_mask]**2*xi_obs_mean[r_mask], 
                yerr=r_obs[r_mask]**2*np.sqrt(np.diag(xi_obs_cov[r_mask, :][:, r_mask])),
                color='r', fmt='.', elinewidth=0.5, capsize=3, alpha=1.0,
                label=r'$\langle\xi_{\rm mm}\rangle$ Quijote')
    ax1.plot(r_zel, r_zel**2 * xi_zel, color='b', label=r'$\xi_{\rm zel}$')

    ax2.errorbar(r_obs[r_mask], xi_obs_mean[r_mask]/xi_zel_interp(r_obs[r_mask]), 
                 yerr=np.sqrt(np.diag(xi_obs_cov[r_mask, :][:, r_mask]))/xi_zel_interp(r_obs[r_mask]),
                 color='k', fmt='.', elinewidth=0.5, capsize=3)

    ax1.set_ylabel(r'$r^{2}\xi$', fontsize=SIZE_LABELS)
    ax1.set_xlim(5, 50)
    ax1.set_ylim(16, 40)
    ax1.legend(loc='lower left', fontsize=SIZE_LEGEND)
    
    ax2.hlines(1.0, 0, 50, ls='-', lw=1.0, color='k')
    ax2.hlines(B_maxpost, 0, 50, ls='--', lw=1.0, color='r', label=f"{B_maxpost:.3f}")
    ax2.set_ylabel(r'$\langle\xi_{\rm mm}\rangle /\xi_{\rm zel} $', fontsize=SIZE_LABELS)
    ax2.set_xlabel(r'$r~[h^{-1}{\rm Mpc}]$', fontsize=SIZE_LABELS)
    ax2.set_ylim(0.8, 1.2)
    ax2.legend(loc='upper left', fontsize=SIZE_LEGEND)
    
    plt.tight_layout()
    plt.savefig(SRC_PATH + "/data/plot/ximm_quijote.png", bbox_inches="tight")
    return None


def fits_inf_xi_ratios():
    from matplotlib.lines import Line2D

    rh = np.zeros(NMBINS)
    with h5.File(SRC_PATH + "/data/fits/mle_inf_good.h5", "r") as hdf_load:
        for k, mbin in enumerate(MBINSTRS):
            rh[k] = hdf_load[f"max_posterior/orb/{mbin}"][0]
    
    with h5.File(SRC_PATH + "/data/xihm_split.h5", "r") as hdf:
        x = hdf['rbins_ext'][()]
        xihm = np.zeros((NMBINS, len(x)))
        for k, mbin in enumerate(MBINSTRS):
            xihm[k, :] = hdf[f'xi_ext/inf/{mbin}'][()]

    def xi_ratio(
        x: float,
        a: float,
        A: float,
    ) -> float:
        return A + a * x * np.exp(-x)

    def cost_xi_ratio(pars, *data):
        x, y, rh = data
        
        a, A = pars
        if A < 0 or a < 0:
            return np.inf
        
        u = y - xi_ratio(x/rh, a, A)
        return 0.5 * np.dot(u, u)
    
    legend = [Line2D([], [], marker='o', markerfacecolor='k', color='k',
                 markersize=5, label='Data', linestyle='None'), 
              Line2D([0], [0], color='k', lw=1.0, label='Fit')]
    
    # Setup canvas
    fig, ax1 = plt.subplots(1, 1, figsize=(4.583333333, 2.75))

    # Create a color bar to display mass ranges.
    cax = fig.add_axes([1.0, 0.2, 0.02, 0.75])
    cbar = ColorbarBase(cax, cmap=CMAP.name, norm=NORM,
                        orientation='vertical')
    cbar.set_label(r'$\log_{10} M_{\rm orb}$',
                   fontsize=SIZE_LEGEND)
    cbar.set_ticklabels([f'${m:.2f}$' for m in MBINEDGES], fontsize=SIZE_TICKS)

    x_plot = np.logspace(-2, 2, num=1000, base=10)
    rmin = 1e-2
    xi_ref = xihm[-1, :]
        
    for k, mbin in reversed(list(enumerate(MBINSTRS))):
        if k == 8:
            ax1.hlines(1, 0, 100, colors=CMAP(k/NMBINS))
        else:
            y = xi_ref / xihm[k, :]
            res = minimize(cost_xi_ratio, [1.0, 2.0], args=(x, y, rh[k]),
                        method='Powell')
            
            ax1.errorbar(x/rh[k], y, fmt='.', elinewidth=0, capsize=0, 
                            color=CMAP(k/NMBINS), alpha=0.5)
            ax1.plot(x_plot, xi_ratio(x_plot, *res.x), lw=1, 
                        color=CMAP(k/NMBINS))
                
    ax1.set_ylabel(r'$\xi_{\rm inf}^{\rm ref}/\xi_{\rm inf}$', fontsize=SIZE_LABELS)
    ax1.set_xlim(5e-2, 10)
    ax1.set_ylim(0.5, 7.2)
    ax1.set_xscale('log', base=10)
    ax1.tick_params(axis='both', which='major', labelsize=SIZE_TICKS)
    ax1.legend(handles=legend, loc='upper right', fontsize=SIZE_LEGEND)
    ax1.vlines(1, 0, 10, color='k', lw=1, ls=':')
    ax1.set_xlabel(r'$r/r_{\rm h}$', fontsize=SIZE_LABELS)
    ax1.set_xscale('log')
    ax1.set_xlim(5e-2, 50)

    plt.tight_layout()
    plt.savefig(SRC_PATH + "/data/plot/fits_inf_xi_ratios.png", bbox_inches="tight")
    
    return None


def fits_inf():
    M_PIVOT = np.power(10.0, 14)
    
    # Load data
    with h5.File(SRC_PATH + "/data/mass.h5", "r") as hdf_load:
        mass = hdf_load["mass"][()]
        
    pars = np.zeros((NMBINS, 6))
    errs = np.zeros((NMBINS, 6))
    with h5.File(SRC_PATH + "/data/fits/mle.h5", "r") as hdf_load:
            for k, mbin in enumerate(MBINSTRS):
                pars[k, :] = hdf_load[f"max_posterior/inf/{mbin}"][()]
                errs[k, :] = np.sqrt(np.diag(hdf_load[f"covariance/inf/{mbin}"][()]))
            pars_smooth_1 = hdf_load[f"max_posterior/inf/smooth_1"][()]
        
    with h5.File(join(SRC_PATH, "data/xihm_split.h5"), 'r') as hdf, \
        h5.File(SRC_PATH + "/data/fits/mle_orb_good.h5", "r") as hdf_load:
        r = hdf['rbins_ext'][()]  # x
        rh = np.zeros(len(r))
        xi = np.zeros((NMBINS, len(r)))
        xi_err = np.zeros((NMBINS, len(r)))
        xi_err_smooth_1 = np.zeros((NMBINS, len(r)))
        for k, mbin in enumerate(MBINSTRS):
            xi[k, :] = hdf[f'xi_ext/inf/{mbin}'][()]
            xi_err[k, :] = np.sqrt(np.diag(hdf[f'xi_ext_cov/inf/{mbin}'][()]))
            xi_err_smooth_1[k, :] = np.sqrt(np.diag(hdf[f'xi_ext_cov/inf/{mbin}'][()]) + (10**pars_smooth_1[-1]*xi[k, :])**2)
            rh[k] = hdf_load[f"max_posterior/orb/{mbin}"][0]

    # Setup canvas
    fig = plt.figure(layout="constrained", figsize=(7, 4))
    subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[4, 3])

    ax = subfigs[0].subplots(1, 1)
    # Plot all mass bins.
    for k in range(NMBINS):
        r_pred = np.logspace(-2, np.log10(max(r+1)), num=100, base=10)
        xi_pred = xi_inf_model(r_pred, *pars[k, :-1], rh[k])
        # Plot data
        ax.errorbar(r, r ** (3/2) * xi[k, :], yerr=r ** (3/2) * xi_err[k, :],
                    fmt=".", elinewidth=0.5, capsize=3, color=CMAP(k / NMBINS),
                    alpha=0.5, label=f"{mbin}")
        ax.plot(r_pred, r_pred ** (3/2) * xi_pred, ls='-', lw=0.5, 
                color=CMAP(k/NMBINS))
        
    ax.fill_betweenx([-1, 1e3], 0, 6*RSOFT, color='k', alpha=0.25)
    ax.set_xlim(2e-2, 50)
    ax.set_xscale("log")
    ax.set_xlabel(r"$r [h^{-1}{\rm Mpc}]$", fontsize=SIZE_LABELS)
    ax.set_ylim(0.8, 80)
    ax.set_yscale('log')
    ax.set_ylabel(r'$r^{3/2}\xi_{\rm inf}(r|M)$', fontsize=SIZE_LABELS)
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
        b_1 = power_law(mass[k]/M_PIVOT, *pars_smooth_1[:2])
        c_1 = error_function(np.log10(mass[k]/M_PIVOT), *pars_smooth_1[2:5])
        g_1 = power_law(mass[k]/M_PIVOT, *pars_smooth_1[5:7])
        b0 = pars_smooth_1[7]
        r0 = pars_smooth_1[8]
        
        ratio = xi_inf_model(r, *pars[k, :-1], rh[k]) / xi[k, :] - 1
        ratio_smooth_1 = xi_inf_model(r, b_1, c_1, g_1, b0, r0, rh[k]) / xi[k, :] - 1
        
        # Plot ratio and error bands
        ax1.plot(r, ratio, lw=1, color=CMAP(k/NMBINS))
        ax1.fill_between(r, xi_err[k, :]/xi[k, :], - xi_err[k, :]/xi[k, :], 
                        color=CMAP(k/NMBINS), alpha=0.1)
        
        ax2.plot(r, ratio_smooth_1, lw=1, color=CMAP(k/NMBINS))
        ax2.fill_between(r, xi_err_smooth_1[k, :]/xi[k, :],
                        - xi_err_smooth_1[k, :]/xi[k, :], 
                        color=CMAP(k/NMBINS), alpha=0.1)
        
    plt.savefig(SRC_PATH + "/data/plot/fits_inf.png", bbox_inches="tight")
    return


def fits_inf_pars():
    from matplotlib.ticker import NullFormatter, NullLocator

    plabels = [r"$b$", r"$c$", r"$\gamma$", r"$\beta_0$", r'$r_{0}~[h^{-1}{\rm Mpc}]$', r"$\log_{10}\delta$"]
    M_PIVOT = np.power(10.0, 14)
    
    mass_pred = np.logspace(13, 15, num=100, base=10)
    pars = np.zeros((NMBINS, 6))
    errs = np.zeros((NMBINS, 6))
    with h5.File(SRC_PATH + "/data/fits/mle.h5", "r") as hdf_load:
            for k, mbin in enumerate(MBINSTRS):
                pars[k, :] = hdf_load[f"max_posterior/inf/{mbin}"][()]
                errs[k, :] = np.sqrt(np.diag(hdf_load[f"covariance/inf/{mbin}"][()]))
            pars_smooth_1 = hdf_load[f"max_posterior/inf/smooth_1"][()]

    with h5.File(SRC_PATH + "/data/mass.h5", "r") as hdf_load:
        mass = hdf_load["mass"][()]
        
    _, axes = plt.subplots(2, 3, figsize=(10, 6))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    ax6.axis('off')
    
    for i, ax in enumerate(axes.flatten()[:-1]):
        ax.set_xlabel(r'$\log M_{\rm orb}$', fontsize=SIZE_LABELS)
        ax.set_xlim(1e13, 1e15)
        ax.set_xscale('log', base=10)
        ax.set_ylabel(plabels[i], fontsize=SIZE_LABELS)
        ax.tick_params(axis="both", which="major", labelsize=SIZE_TICKS)
        ax.errorbar(mass, pars[:, i], yerr=errs[:, i], fmt='.',
                    elinewidth=0.5, capsize=3, color='k', label="Individual")

    ax1.plot(mass_pred, power_law(mass_pred/M_PIVOT, *pars_smooth_1[:2]), color='k', 
             lw=1.0, label=r"Smooth")
    ax1.set_yscale('log')
    ax1.set_yticks([1.0, 1.6, 2.5, 4.0])
    ax1.set_yticklabels(['1.0', '1.6', '2.5', '4.0'])
    ax1.yaxis.set_minor_formatter(NullFormatter())
    ax1.yaxis.set_minor_locator(NullLocator())
    ax1.legend(loc="upper left", fontsize=10)

    ax2.plot(mass_pred, error_function(np.log10(mass_pred/M_PIVOT), *pars_smooth_1[2:5]), color='k', 
            lw=1.0)

    ax3.plot(mass_pred, power_law(mass_pred/M_PIVOT, *pars_smooth_1[5:7]), color='k',
            lw=1.0)

    ax4.hlines(pars_smooth_1[7], 1e13, 1e15, color='k', lw=1.0)

    ax5.hlines(pars_smooth_1[8], 1e13, 1e15, color='k', lw=1.0)

    plt.tight_layout()
    plt.savefig(SRC_PATH + "/data/plot/fits_inf_pars.png", bbox_inches="tight")
    
    return


def fit_inf_corner() -> None:
    sampler = HDFBackend(
        SRC_PATH + "/data/fits/chains.h5", name="inf/smooth_1", read_only=True
    )
    flat_samples = sampler.get_chain(flat=True)
    plabs = [
        r"$b_{p}$",
        r"$b_{s}$",
        r"$c_0$",
        r"$c_\mu$",
        r"$c_\sigma$",
        r"$\gamma_{p}$",
        r"$\gamma_{s}$",
        r"$\beta_0$",
        r"$r_0$",
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
    plt.savefig(SRC_PATH + "/data/plot/fits_inf_corner.png", bbox_inches="tight")

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
    plt.savefig(SRC_PATH + "/data/plot/fits_inf_post.png", bbox_inches="tight")
    return None


def biases():
    M_PIVOT = np.power(10.0, 14)
    
    with h5.File(SRC_PATH + "/data/mass.h5", "r") as hdf_load:
        mass = hdf_load["mass"][()]
    mass_offset_low = 0.95 * mass
    mass_offset_high = 1.05 * mass
    
    # Peak-background split bias prediction
    BIAS_PBS = [1.389282066493024, 1.5158112746528332, 1.668474653924278,
            1.8538283318415552, 2.0802206432387607, 2.3582910676248616,
            2.7016032695167382, 3.191561513786411, 4.145824540393249]
    
    # Individual fit bias.
    bias_model = np.zeros(NMBINS, dtype=float)
    bias_errs = np.zeros(NMBINS, dtype=float)
    with h5.File(SRC_PATH + "/data/fits/mle.h5", "r") as hdf:
        for k, mbin in enumerate(MBINSTRS):
            bias_model[k] = hdf[f'max_posterior/inf/{mbin}'][0]
            bias_errs[k] = np.sqrt(hdf[f'covariance/inf/{mbin}'][0, 0])
        bias_smooth = power_law(mass/M_PIVOT, *hdf["max_posterior/inf/smooth_1"][:2])
    
    # Find large scale bias from xihm/ximm   
    with h5.File(SRC_PATH + "/data/ximm.h5", "r") as hdf_mm, \
        h5.File(SRC_PATH + "/data/xihm.h5", "r") as hdf_hm:
        # ximm = hdf_mm['xi/100'][()]
        # ximmcov = hdf_mm['cov/100'][()]
        r = hdf_mm['rbins'][()]
        
        jk_samples = hdf_hm[f'xi_i/100/{MBINSTRS[0]}'].shape[0]
        bias_jk = np.zeros((NMBINS, jk_samples, len(r)))
        
        for k, mbin in enumerate(MBINSTRS):
            for j in range(jk_samples):
                bias_jk[k, j, :] = hdf_hm[f'xi_i/100/{mbin}'][j, :] / hdf_mm['xi_i/100'][j, :]

    # Compute the average bias and error
    bias_mean = np.mean(bias_jk, axis=1)
    bias_cov = np.zeros((NMBINS, len(r), len(r)))
    for k in range(NMBINS):
        bias_cov[k, :, :] = (jk_samples - 1.) * np.cov(bias_jk[k, :, :].T, bias=True)
    
    bias_ratio = np.zeros(9)
    bias_ratio_cov = np.zeros(9)
    for k, mbin in enumerate(MBINSTRS):
        mask = (r >= 10)
        bias_ratio[k], bias_ratio_cov[k] = curve_fit(lambda x, b: b, r,
                                                     bias_mean[k, mask], 
                                                     sigma=bias_cov[k, mask][:, mask], 
                                                     p0=[2.0])
        
    # Interpolate ratio bias to offset points in plot
    b_ratio_interp = interp1d(mass, bias_ratio, fill_value='extrapolate')
    
    _, axes = plt.subplots(1, 2, figsize=(6, 3),
                       gridspec_kw={'hspace': 0})

    ax1, ax2 =  axes.flatten()

    ax1.plot(mass, BIAS_PBS, color='k', alpha=0.75, label=r'$b_{\rm PB}$', lw=1.0)
    ax1.plot(mass, bias_smooth, color='gray', alpha=0.75, label=r'$b_{\rm smooth}$',
             ls="--", lw=1.0)
    ax1.errorbar(mass, bias_model, yerr=bias_errs,
                color='r', fmt='.', elinewidth=0.5, capsize=3, label=r'$b$')
    ax1.errorbar(mass_offset_high, b_ratio_interp(mass_offset_high), yerr=np.sqrt(bias_ratio_cov),
                color='b', fmt='^', ms=2.0, elinewidth=0.5, capsize=3, label=r'$\xi_{\rm hm}/\xi_{\rm mm}$')
    ax1.set_xlabel(r'$\log M_{\rm orb}$', fontsize=SIZE_LABELS)
    ax1.set_xscale('log')
    ax1.set_ylabel('Bias', fontsize=SIZE_LABELS)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.tick_params(axis="both", which="major", labelsize=SIZE_TICKS)


    ax2.plot(mass, bias_smooth/BIAS_PBS, color='gray', alpha=0.75, ls="--",
             label=r'$b_{\rm smooth}$', lw=1.0)
    ax2.errorbar(mass, bias_model/BIAS_PBS, yerr=bias_errs/BIAS_PBS,
                color='r', fmt='.', elinewidth=0.5, capsize=3, label=r'$b$')
    ax2.errorbar(mass_offset_high, bias_ratio/BIAS_PBS, yerr=np.sqrt(bias_ratio_cov),
                color='b', fmt='^', ms=2.0, elinewidth=0.5, capsize=3, label=r'$\xi_{\rm hm}/\xi_{\rm mm}$')
    ax2.hlines(1, min(mass_offset_low), max(mass_offset_high), lw=1.0, ls=":", color='k')
    ax2.set_ylabel(r'$b/b_{\rm PB}$', fontsize=SIZE_LABELS)

    ax2.set_xlabel(r'$\log M_{\rm orb}$', fontsize=SIZE_LABELS)
    ax2.set_xscale('log')
    ax2.tick_params(axis="both", which="major", labelsize=SIZE_TICKS)
    plt.tight_layout()
    plt.savefig(SRC_PATH + "/data/plot/biases.png", bbox_inches="tight")
    plt.show()
    
    return None

if __name__ == "__main__":
    pass
