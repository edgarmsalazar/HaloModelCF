from multiprocessing import Pool
from os.path import join
from typing import Callable

import h5py as h5
import numpy as np
from chainconsumer import ChainConsumer
from emcee.autocorr import integrated_time
from emcee.backends import HDFBackend
from emcee.ensemble import EnsembleSampler
from emcee.moves import DEMove, StretchMove
from scipy.optimize import curve_fit

from physhalo.config import MBINSTRS, NMBINS, SRC_PATH
from physhalo.cosmology import RHOM, RSOFT
from physhalo.hmcorrfunc.model import power_law, xi_inf_model, xi_zel_interp


class MCMC:
    def __init__(
        self,
        ndim: int,
        nwalkers: int,
        nsteps: int,
        burnin: int,
    ) -> None:
        # User input
        self._ndim = ndim
        self._nwalkers = nwalkers
        self._nsteps = nsteps
        self._nburn = burnin
        self._savepath = SRC_PATH + f"/data/fits/chains.h5"
        self._mlepath = SRC_PATH + f"/data/fits/mle.h5"

        # Subclass parameters
        self._lnpost = None
        self._lnpost_args = None
        self._chain_name = None

    @property
    def pars_init(self):
        return self._pars_init

    @pars_init.setter
    def pars_init(self, values):
        self._pars_init = values

    @staticmethod
    def walker_reinit(chain, log_prob) -> np.ndarray:
        """Apply a median absolute deviation (MAD) criteria to select 'good' and
        'bad' walkers from a chain, where 'bad' walkers get new positions from a
        previous state of any 'good' walker. Returns a list with the new
        positions for all walkers.

        Args:
            chain (_type_): _description_
            log_prob (_type_): _description_

        Returns:
            ndarray:
        """
        nsteps, nwalkers, ndim = chain.shape
        if nsteps >= 50:
            ntail = 50
        else:
            ntail = nsteps

        # Append log-likelihood to the samples in order to compute all medians
        # in a vectorized manner.
        samples = np.zeros((nsteps, nwalkers, ndim + 1))
        samples[:, :, :-1] = chain
        samples[:, :, -1] = log_prob

        # Median for parameter i and walker a (nwalkers x ndim+1)
        theta_ia = np.median(samples, axis=0)
        # Median for parameter i (median over all walkers) (ndim+1)
        theta_i = np.median(theta_ia, axis=0)
        # Median absolute deviation (MAD)
        dtheta_ia = np.abs(theta_ia - theta_i)
        sigma = 1.4826 * np.median(dtheta_ia, axis=0)

        # Select 'good' walkers.
        good_walkers = np.zeros(nwalkers, dtype=bool)
        for i in range(nwalkers):
            # If inequality is satisfied for any parameter (or log-likelihood),
            # label the walker as 'good'.
            good_walkers[i] = all(dtheta_ia[i, :] / sigma < 3)
        bad_walkers = ~good_walkers
        good_walkers = good_walkers

        # Draw n_bad_walker samples from 'good' walkers earlier in time and
        # replace the 'bad' walkers final positons per dimension.
        pos_new = chain[-1, :, :]
        for i in range(ndim):
            pos_new[bad_walkers, i] = np.random.choice(
                chain[-ntail:, good_walkers, i].flatten(), size=bad_walkers.sum()
            )

        return pos_new

    def walker_init(self) -> None:
        """Checks initial point for the chain and that all walkers are
        initialized correctly around this point.

        Raises:
            ValueError: If log-posterior returned infinity at inital point.
            ValueError: If log-posterior returned infinity for any of the
                        walkers.
        """
        # Check initial point.
        if self._lnpost is None or self._lnpost_args is None:
            raise RuntimeError("MCMC is an abstract class.")

        lnpost_init = self._lnpost(self.pars_init, *self._lnpost_args)

        if not np.isfinite(lnpost_init):
            raise ValueError("Initial point returned infinity")
        else:
            print(f"\t Initial log posterior {lnpost_init:.2f}")

        # Initialize walkers around initial point with a 10% uniform scatter.
        rand = np.random.uniform(low=-1, size=(self._nwalkers, self._ndim))
        self._walkers = self.pars_init * (1.0 + 0.1 * rand)

        # Check walkers.
        lnlike_inits = [
            self._lnpost(self._walkers[i], *self._lnpost_args)
            for i in range(self._nwalkers)
        ]

        if not all(np.isfinite(lnlike_inits)):
            raise ValueError("Some walkers are not properly initialized.")

        return None

    def run_chain(self, ncpu=4) -> None:
        if self._chain_name is None:
            raise RuntimeError("MCMC is an abstract class.")

        backend = HDFBackend(SRC_PATH + f"/data/fits/burn.h5", name="burn")
        backend.reset(self._nwalkers, self._ndim)

        # Run burnin samples
        with Pool(ncpu) as pool:
            sampler = EnsembleSampler(
                self._nwalkers,
                self._ndim,
                self._lnpost,
                pool=pool,
                backend=backend,
                args=self._lnpost_args,
            )
            sampler.run_mcmc(
                self._walkers,
                self._nburn,
                progress=True,
                progress_kwargs={"desc": f"Burnin"},
            )

        # Sample from the last 20% of burnin steps
        burn_tail = int(0.8 * self._nburn)
        self._walkers = self.walker_reinit(
            sampler.get_chain(discard=burn_tail),
            sampler.get_log_prob(discard=burn_tail),
        )
        print(f"Chain initial point {self._walkers[0, :]}")

        backend = HDFBackend(self._savepath, name=self._chain_name)
        backend.reset(self._nwalkers, self._ndim)

        # Run chain
        with Pool(ncpu) as pool:
            sampler = EnsembleSampler(
                self._nwalkers,
                self._ndim,
                self._lnpost,
                pool=pool,
                backend=backend,
                args=self._lnpost_args,
                moves=[
                    (DEMove(), 0.5),
                    (StretchMove(), 0.5),
                ],
            )
            sampler.run_mcmc(
                self._walkers,
                self._nsteps,
                progress=True,
                progress_kwargs={"desc": f"Chain {self._chain_name[4:]}"},
            )
        return None

    def summary(self) -> None:
        if self._chain_name is None:
            raise RuntimeError("MCMC is an abstract class.")

        sampler = HDFBackend(self._savepath, name=self._chain_name, read_only=True)
        flat_samples = sampler.get_chain(flat=True)
        log_prob = sampler.get_log_prob()

        tau = [
            integrated_time(flat_samples[:, i], c=150, tol=50)[0]
            for i in range(self._ndim)
        ]
        print(f"Autocorrelation length {max(tau):.2f}")

        # Setup chainconsumer for computing MLE parameters
        c = ChainConsumer()
        c.add_chain(flat_samples, posterior=log_prob.reshape(-1))
        c.configure(sigmas=[1, 2])

        # Compute max posterior and quantiles (16, 50, 84).
        quantiles = np.zeros((self._ndim, 3))
        max_posterior = np.zeros(self._ndim)
        for i, ((_, v1), (_, v2)) in enumerate(
            zip(
                c.analysis.get_summary().items(),
                c.analysis.get_max_posteriors().items(),
            )
        ):
            quantiles[i, :] = v1
            max_posterior[i] = v2
        cov = c.analysis.get_covariance()[-1]
        corr = c.analysis.get_correlations()[-1]

        # Save to file.
        with h5.File(self._mlepath, "a") as hdf:
            for ds, var in zip(
                [f"quantiles/", f"max_posterior/", f"covariance/", f"correlations/"],
                [quantiles, max_posterior, cov, corr],
            ):
                name_ = ds + self._chain_name
                # Overwirte existing values.
                if name_ in hdf.keys():
                    sp = hdf[name_]
                    sp[...] = var
                # Create dataset otherwise.
                else:
                    hdf.create_dataset(name_, data=var)
        return None


class BinMC(MCMC):
    def __init__(
        self,
        ndim: int,
        nwalkers: int,
        nsteps: int,
        burnin: int,
        log_posterior: Callable,
        mbin_i: int,
        ptype: str,
    ) -> None:
        super().__init__(ndim, nwalkers, nsteps, burnin)
        self._lnpost = log_posterior
        self._nbin = mbin_i
        self._mbin = MBINSTRS[self._nbin]
        self._ptype = ptype
        self._mpivot = np.power(10.0, 14)
        self._chain_name = f"{self._ptype}/{self._mbin}"

        self.load_data()
        self.walker_init()

    def load_data(self) -> None:
        # Load mass
        with h5.File(SRC_PATH + "/data/mass.h5", "r") as hdf_load:
            self._mass = hdf_load["mass"][self._nbin]

        if self._ptype == "orb":
            # X, Y data points
            with h5.File(SRC_PATH + "/data/xihm_split.h5", "r") as hdf_load:
                self._x = hdf_load["rbins"][()]
                self._y = hdf_load[f"rho/orb/{self._mbin}"][()]
                self._covy = RHOM**2 * hdf_load[f"xi_cov/all/{self._mbin}"][()]
                rho_ratio = self._y / hdf_load[f"rho/all/{self._mbin}"][()]
                self._mask = (rho_ratio > 1e-2) * (self._x > 6 * RSOFT)

            # Arguments passed to the log-posterior
            self._lnpost_args = (
                self._x[self._mask],
                self._y[self._mask],
                self._covy[self._mask, :][:, self._mask],
                self._mass,
            )

            # Initialize parameters to the best fit values from the mass
            # correction fit.
            if self._nbin == 0:
                with h5.File(SRC_PATH + "/data/mass.h5", "r") as hdf_load:
                    self.pars_init = [*hdf_load["best_fit"][0, 1:], -2]

            # Initialize parameters to the previous mass bin best fit values.
            # This is ok because the parameter variation with mass is smooth.
            else:
                with h5.File(SRC_PATH + "/data/fits/mle.h5", "r") as hdf_load:
                    self.pars_init = hdf_load[
                        f"max_posterior/orb/{MBINSTRS[self._nbin-1]}"
                    ][()]

        elif self._ptype == "inf":
            # X, Y data points and covariance matrix
            with h5.File(SRC_PATH + "/data/xihm_split.h5", "r") as hdf_load, h5.File(
                SRC_PATH + "/data/xihm.h5", "r"
            ) as hdf_load_2:
                self._x = hdf_load["rbins_ext"][()]
                self._y = hdf_load[f"xi_ext/inf/{self._mbin}"][()]
                self._covy = hdf_load[f"xi_ext_cov/inf/{self._mbin}"][()]
                rho_ratio = (1 + self._y) / (1 + hdf_load_2[f"xi/100/{self._mbin}"][()])
                self._mask = (rho_ratio > 1e-2) * (self._x > 6 * RSOFT)

            # Load rh from rho individual fit.
            with h5.File(SRC_PATH + "/data/fits/mle.h5", "r") as hdf_load:
                rh = hdf_load[f"max_posterior/orb/{self._mbin}"][0]

            # Arguments passed to the log-posterior
            self._lnpost_args = (
                self._x[self._mask],
                self._y[self._mask],
                self._covy[self._mask, :][:, self._mask],
                rh,
            )

            # Find best fit parameters for lowest mass bin using least-squares
            # minimization.
            if self._nbin == 0:
                # Find best fit bias
                mask_bias = self._x > 20
                (bias_init,), _ = curve_fit(
                    lambda x, bias: bias * xi_zel_interp(x),
                    self._x[mask_bias],
                    self._y[mask_bias],
                    p0=[1.2],
                    sigma=np.diag(self._covy)[mask_bias],
                )

                # Find best fit c
                with h5.File(SRC_PATH + "/data/xihm_split.h5", "r") as hdf_load:
                    xi_ref = hdf_load[f"xi_ext/inf/{MBINSTRS[-1]}"][()]

                (c_init,), _ = curve_fit(
                    lambda x, c: 1.0 + c * x * np.exp(-x),
                    self._x / rh,
                    (xi_ref / 3.65) / (self._y / bias_init),
                    p0=[1.0],
                )

                # Find beta0 and r0 with fixed gamma = 2
                (beta0_init, r0_init), _ = curve_fit(
                    lambda x, b, r: xi_inf_model(x, bias_init, c_init, 2.0, b, r, rh),
                    self._x,
                    self._y,
                    p0=[70.0, 0.4],
                    sigma=np.diag(self._covy),
                )
                # print(beta0_init, r0_init)

                # print()

                # print("".join("{:>6.3f}\t".format(i) for i in res.x))
                self.pars_init = [bias_init, c_init, 2, beta0_init, r0_init, -2]

            # Initialize parameters to the previous mass bin best fit values.
            # This is ok because the parameter variation with mass is smooth.
            else:
                with h5.File(SRC_PATH + "/data/fits/mle.h5", "r") as hdf_load:
                    self.pars_init = hdf_load[
                        f"max_posterior/inf/{MBINSTRS[self._nbin-1]}"
                    ][()]

        else:
            raise ValueError("Must select ['orb', 'inf'].")

        return None


class FullMC(MCMC):
    def __init__(
        self,
        ndim: int,
        nwalkers: int,
        nsteps: int,
        burnin: int,
        log_posterior: Callable,
        ptype: str,
        smooth_iter: int,
    ) -> None:
        super().__init__(ndim, nwalkers, nsteps, burnin)
        self._lnpost = log_posterior
        self._ptype = ptype
        self._mpivot = np.power(10.0, 14)
        self._chain_name = f"{self._ptype}/smooth_{smooth_iter}"
        self._smooth_iter = smooth_iter

        self.load_data()
        self.walker_init()

    def load_data(self) -> None:
        # Load mass
        with h5.File(SRC_PATH + "/data/mass.h5", "r") as hdf_load:
            self._mass = hdf_load["mass"][()]

        if self._ptype == "orb":
            # X, Y data points
            with h5.File(SRC_PATH + "/data/xihm_split.h5", "r") as hdf_load:
                self._x = hdf_load["rbins"][()]
                self._y = np.zeros((NMBINS, len(self._x)))
                self._mask = np.zeros_like(self._y, dtype=bool)
                self._covy = np.zeros((NMBINS, len(self._x), len(self._x)))

                for k, mbin in enumerate(MBINSTRS):
                    self._y[k, :] = hdf_load[f"rho/orb/{mbin}"][()]
                    self._covy[k, :, :] = RHOM**2 * hdf_load[f"xi_cov/all/{mbin}"][()]
                    self._mask[k, :] = (
                        self._y[k, :] / hdf_load[f"rho/all/{mbin}"][()] > 1e-2
                    ) * (self._x > 6 * RSOFT)

            # Get initialization by fitting parametric models to the individual
            # best fits as a function of mass.
            params = np.zeros((NMBINS, 4))
            errors = np.zeros((NMBINS, 4))
            with h5.File(SRC_PATH + "/data/fits/mle.h5", "r") as hdf_load:
                for k, mbin in enumerate(MBINSTRS):
                    params[k, :] = hdf_load[f"max_posterior/orb/{mbin}"][()]
                    errors[k, :] = np.sqrt(
                        np.diag(hdf_load[f"covariance/orb/{mbin}"][()])
                    )

            # Find initialization for rh
            p_init = np.mean(params[:, 0])
            s_init = np.mean(np.diff(params[:, 0])) / np.mean(
                np.diff(self._mass / self._mpivot)
            )
            (rh_p_init, rh_s_init), _ = curve_fit(
                power_law,
                self._mass / self._mpivot,
                params[:, 0],
                p0=(p_init, s_init),
                sigma=errors[:, 0],
            )

            # Find initialization for ainf
            p_init = np.mean(params[:, 1])
            s_init = np.mean(np.diff(params[:, 1])) / np.mean(
                np.diff(self._mass / self._mpivot)
            )
            (ainf_p_init, ainf_s_init), _ = curve_fit(
                power_law,
                self._mass / self._mpivot,
                params[:, 1],
                p0=(p_init, s_init),
                sigma=errors[:, 1],
            )

            # Find initialization for a
            p_init = np.mean(params[:, 2])
            s_init = np.mean(np.diff(params[:, 2])) / np.mean(
                np.diff(self._mass / self._mpivot)
            )
            (a_p_init, a_s_init), _ = curve_fit(
                power_law,
                self._mass / self._mpivot,
                params[:, 2],
                p0=(p_init, s_init),
                sigma=errors[:, 2],
            )

            # Arguments passed to the log-posterior
            self._lnpost_args = (
                self._x,
                self._y,
                self._covy,
                self._mask,
                self._mass,
                self._mpivot,
            )

            if self._smooth_iter == 1:
                self.pars_init = [
                    rh_p_init,
                    rh_s_init,
                    ainf_p_init,
                    ainf_s_init,
                    a_p_init,
                    a_s_init,
                    np.mean(params[:, -1]),
                ]

            elif self._smooth_iter == 2:
                self.pars_init = [
                    rh_p_init,
                    rh_s_init,
                    ainf_p_init,
                    ainf_s_init,
                    a_p_init,
                    np.mean(params[:, -1]),
                ]

            elif self._smooth_iter == 3:
                self.pars_init = [
                    rh_p_init,
                    rh_s_init,
                    ainf_p_init,
                    a_p_init,
                    a_s_init,
                    np.mean(params[:, -1]),
                ]

            else:
                raise NotImplementedError("Select up to smooth iteration 3")

        elif self._ptype == "inf":
            # X, Y data points
            with h5.File(SRC_PATH + "/data/xihm_split.h5", "r") as hdf_load, h5.File(
                SRC_PATH + "/data/xihm.h5", "r"
            ) as hdf_load_2:
                self._x = hdf_load["rbins_ext"][()]
                self._y = np.zeros((NMBINS, len(self._x)))
                self._mask = np.zeros_like(self._y, dtype=bool)
                self._covy = np.zeros((NMBINS, len(self._x), len(self._x)))

                for k, mbin in enumerate(MBINSTRS):
                    self._y[k, :] = hdf_load[f"xi_ext/inf/{mbin}"][()]
                    self._covy[k, :, :] = hdf_load[f"xi_ext_cov/inf/{mbin}"][()]
                    ratio = (1 + self._y[k, :]) / (1 + hdf_load_2[f"xi/100/{mbin}"][()])
                    self._mask[k, :] = (ratio > 1e-2) * (self._x > 6 * RSOFT)

            # Load rh from rho individual fit.
            self._rh = np.zeros(NMBINS)
            self._alpha = np.zeros(NMBINS)
            with h5.File(SRC_PATH + "/data/fits/mle.h5", "r") as hdf_load:
                for k, mbin in enumerate(MBINSTRS):
                    self._rh[k] = hdf_load[f"max_posterior/orb/{mbin}"][0]
                    self._alpha[k] = hdf_load[f"max_posterior/orb/{mbin}"][1]

            # Get initialization by fitting parametric models to the individual
            # best fits as a function of mass.
            params = np.zeros((NMBINS, 6))
            errors = np.zeros((NMBINS, 6))
            with h5.File(SRC_PATH + "/data/fits/mle.h5", "r") as hdf_load:
                for k, mbin in enumerate(MBINSTRS):
                    params[k, :] = hdf_load[f"max_posterior/inf/{mbin}"][()]
                    errors[k, :] = np.sqrt(
                        np.diag(hdf_load[f"covariance/inf/{mbin}"][()])
                    )

            # Find initialization for b
            p_init = np.mean(params[:, 0])
            s_init = np.mean(np.diff(params[:, 0])) / np.mean(
                np.diff(self._mass / self._mpivot)
            )
            (b_p_init, b_s_init), _ = curve_fit(
                power_law,
                self._mass / self._mpivot,
                params[:, 0],
                p0=(p_init, s_init),
                sigma=errors[:, 0],
            )

            # Find initialization for b
            c_0_init = np.max(params[:, 1])
            c_mu_init = 0.01  # Arbitrary
            c_v_init = 1.0  # Arbitrary

            p_init = np.mean(params[:, 2])
            s_init = np.mean(np.diff(params[:, 2])) / np.mean(
                np.diff(self._mass / self._mpivot)
            )
            (g_p_init, g_s_init), _ = curve_fit(
                power_law,
                self._mass / self._mpivot,
                params[:, 2],
                p0=(p_init, s_init),
                sigma=errors[:, 2],
            )

            if self._smooth_iter == 1:
                # Arguments passed to the log-posterior
                self._lnpost_args = (
                    self._x,
                    self._y,
                    self._covy,
                    self._mask,
                    self._mass,
                    self._mpivot,
                    self._rh,
                )

                self.pars_init = [
                    b_p_init,
                    b_s_init,
                    c_0_init,
                    c_mu_init,
                    c_v_init,
                    g_p_init,
                    g_s_init,
                    np.mean(params[:, 3]),
                    np.mean(params[:, 4]),
                    np.mean(params[:, -1]),
                ]
            elif self._smooth_iter == 2:
                # Arguments passed to the log-posterior
                self._lnpost_args = (
                    self._x,
                    self._y,
                    self._covy,
                    self._mask,
                    self._mass,
                    self._mpivot,
                    self._rh,
                    4.0 - self._alpha,
                )

                self.pars_init = [
                    b_p_init,
                    b_s_init,
                    c_0_init,
                    c_mu_init,
                    c_v_init,
                    np.mean(params[:, 3]),
                    np.mean(params[:, 4]),
                    np.mean(params[:, -1]),
                ]

            else:
                raise NotImplementedError("Select smooth iteration 1")
            print()
        else:
            raise ValueError("Must select ['orb', 'inf'].")

        return None


if __name__ == "__main__":
    pass
#
