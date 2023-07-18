from typing import List, Tuple, Union

import h5py as h5
import numpy as np
from numba import njit
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import erf

from physhalo.config import NMBINS, SRC_PATH
from physhalo.cosmology import RHOM

# Interpolate Zel'dovich approximation correlation function from file.
with h5.File(SRC_PATH + "/data/xi_zel.h5", "r") as hdf:
    xi_zel_interp = interp1d(hdf["r"][()], hdf["xi_zel"][()])


@njit()
def power_law(x: float, p: float, s: float) -> float:
    return p * np.power(x, s)


@njit()
def error_function(x: float, c0: float, cm: float, cv: float) -> float:
    return 0.5 * c0 * (1 - erf((x - cm) / cv))


@njit()
def rho_orb_dens_distr(x: float, ainf: float, a: float) -> float:
    alpha = ainf * x / (a + x)
    return np.power(x/a, -alpha) * np.exp(-0.5 * x**2)


# 4 parameter rho orb model
@njit()
def rho_orb_model_norm(
    r: float, logA: float, rh: float, ainf: float, a: float
) -> float:
    return 10**logA * rho_orb_dens_distr(r / rh, ainf, a)


def rho_orb_model(
        r: float,
        rh: float,
        ainf: float,
        a: float,
        mass: float,
    ) -> float:
    '''Orbiting density profile.

    Args:
        r (float): radial coordinate in Mpc/h.
        rh (float): exponential decay scale in Mpc/h.
        ainf (float): slope at large scales.
        a (float): slope shape and scaling parameter.
        mass (float): halo (corrected) mass.
        
    Returns:
        float: 
    '''
    def func(x, alpha, a): return x**2 * rho_orb_dens_distr(x, alpha, a)
    rho = mass * rho_orb_dens_distr(r/rh, ainf, a)
    rho /= 4. * np.pi * rh ** 3 * quad(func, 0, np.inf, args=(ainf, a))[0]
    return rho


def lnpost_orb(pars: Union[List[float], Tuple[float, ...]], *data) -> float:
    # Unpack data
    x, y, covy, mass = data

    # Check priors.
    rh, ainf, a, logd = pars
    if rh < 0 or ainf < 0 or a < 0 or not (-4 < logd < 0):
        return -np.inf
    delta = np.power(10., logd)

    # Compute model deviation from data
    u = y - (rho_orb_model(x, *pars[:-1], mass)/RHOM - 1)
    # Add percent error to the covariance - regulated by delta
    cov = covy + np.diag((delta * y)**2)
    # Compute chi squared
    chi2 = np.dot(u, np.linalg.solve(cov, u))

    return -chi2 - np.log(np.linalg.det(cov))


@njit()
def xi_inf_beta_model(
    r: float, b0: float, r0: float, g: float,
) -> float:
    return 1.0 + b0 / np.power(1 + r/r0, g)

@njit()
def xi_inf_den_model(
    r: float, rh: float, c: float,
) -> float:
    x = r / rh
    return 1 + c * x * np.exp(-x)


def xi_inf_model(
    r: float, bias: float, c: float, g: float, b0: float, r0: float, rh: float,
) -> float:
    xi_mod = bias * xi_inf_beta_model(r, b0, r0, g) * xi_zel_interp(r)
    xi_mod /= xi_inf_den_model(r, rh, c)
    return xi_mod


def xihm_model(
    r: float,
    rh: float,
    ainf: float,
    a: float,
    bias: float,
    c: float,
    g: float,
    b0: float,
    r0: float,
    mass: float,
) -> float:
    orb = rho_orb_model(r, rh, ainf, a, mass)
    inf = RHOM * (1.0 + xi_inf_model(r, bias, c, g, b0, r0, rh))
    return (orb + inf) / RHOM - 1.0



def lnpost_inf(pars: Union[List[float], Tuple[float, ...]], *data) -> float:
    # Unpack data
    x, y, covy, rh = data

    # Check priors.
    b, c, g, b0, r0, logd = pars
    if not (0 < b < 8) or c < 0 or g < 0 or b0 < 0 or r0 < 0 or not (-4 < logd < 0):
        return -np.inf
    delta = np.power(10., logd)

    # Compute model deviation from data
    u = y - xi_inf_model(x, b, c, g, b0, r0, rh)
    # Add percent error to the covariance - regulated by delta
    cov = covy + np.diag((delta * y)**2)
    # Compute chi squared
    chi2 = np.dot(u, np.linalg.solve(cov, u))

    return -chi2 - np.log(np.linalg.det(cov))


def lnpost_inf_1(pars: Union[List[float], Tuple[float, ...]], *data) -> float:
    # Unpack data
    x, y, covy, rh = data

    # Check priors. 5 model + 1 likelihood parameters
    bias, c, g, b0, r0 = pars
    if any([p < 0 for p in pars]):
        return -np.inf
    
    # Compute model deviation from data
    u = y - xi_inf_model(x, bias, c, g, b0, r0, rh)
    # Compute chi squared
    chi2 = np.dot(u, np.linalg.solve(covy, u))
    return -chi2


def lnpost_orb_smooth(pars: Union[List[float], Tuple[float, ...]], *data) -> float:
    # Unpack data
    x, y, covy, mask, mass, m_pivot = data

    # Check priors. 3 model + 1 likelihood parameters
    pr, sr, pa, sa, eps, logd = pars
    if logd > 0 or sa > 0 or pr < 0 or pa < 0 or sr < 0 or eps < 0:
        return -np.inf
    delta = np.power(10., logd)
    rh = power_law(mass/m_pivot, pr, sr)
    ainf = power_law(mass/m_pivot, pa, sa)

    # Aggregate likelihood for all mass bins
    lnlike = 0
    for k in range(NMBINS):
        # Compute model deviation from data
        u = y[k, mask[k, :]] - (rho_orb_model(x[mask[k, :]], rh[k], ainf[k], eps, mass[k])/RHOM - 1)
        # Add percent error to the covariance - regulated by delta
        cov = covy[k, mask[k, :], :][:, mask[k, :]] + \
            np.diag(np.power(delta * y[k, mask[k, :]], 2))
        # Compute chi squared
        chi2 = np.dot(u, np.linalg.solve(cov, u))

        lnlike -= chi2 + np.log(np.linalg.det(cov))
    return lnlike


if __name__ == "__main__":
    pass
#
