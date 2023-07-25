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
    return np.power(x/a, -alpha) * np.exp(-0.5 * x ** 2)


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
    scale = RHOM**2

    # Unpack data
    x, y, covy, mass = data

    # Check priors. 3 model + 1 likelihood parameters
    rh, ainf, a, logd = pars
    if not (-4 < logd < 0) or rh < 0 or ainf < 0 or a < 0:
        return -np.inf
    delta = np.power(10., logd)

    # Compute model deviation from data
    # u = y - rho_orb_model(x, rh, ainf, a, mass)
    u = x**2 * (y - rho_orb_model(x, rh, ainf, a, mass))
    # Add percent error to the covariance - regulated by delta
    # cov = covy + np.diag(np.power(delta * y, 2))
    cov = np.outer(x**2, x**2)*(covy + np.diag((delta * y)))
    # Compute chi squared
    chi2 = np.dot(u, np.linalg.solve(cov, u))
    # Compute ln|C| in a 'smart' way
    lndetc = len(u) * np.log(scale) + np.log(np.linalg.det(cov / scale))

    return -0.5 * (chi2 + lndetc)


def lnpost_orb_smooth_a_pl_ai_pl(pars: Union[List[float], Tuple[float, ...]], *data) -> float:
    scale = RHOM**2
    # Unpack data
    x, y, covy, mask, mass, m_pivot = data

    # Check priors. 3 model + 1 likelihood parameters
    pr, sr, painf, sainf, pa, sa, logd = pars
    if not (-4 < logd < 0) or sainf > 0 or any([p < 0 for p in [pr, sr, painf, pa, sa]]):
        return -np.inf
    delta = np.power(10., logd)
    rh = power_law(mass/m_pivot, pr, sr)
    ainf = power_law(mass/m_pivot, painf, sainf)
    a = power_law(mass/m_pivot, pa, sa)

    # Aggregate likelihood for all mass bins
    lnlike = 0
    for k in range(NMBINS):
        # Compute model deviation from data
        u = y[k, mask[k, :]] - rho_orb_model(x[mask[k, :]], rh[k], ainf[k], a[k], mass[k])
        # Add percent error to the covariance - regulated by delta
        cov = covy[k, mask[k, :], :][:, mask[k, :]] + \
            np.diag(np.power(delta * y[k, mask[k, :]], 2))
        # Compute chi squared
        chi2 = np.dot(u, np.linalg.solve(cov, u))
        lndetc = len(u) * np.log(scale) + np.log(np.linalg.det(cov / scale))
        lnlike -= chi2 + lndetc
    return lnlike


def lnpost_orb_smooth_ai_pl(pars: Union[List[float], Tuple[float, ...]], *data) -> float:
    scale = RHOM**2
    # Unpack data
    x, y, covy, mask, mass, m_pivot = data

    # Check priors. 3 model + 1 likelihood parameters
    pr, sr, painf, sainf, a, logd = pars
    if not (-4 < logd < 0) or sainf > 0 or any([p < 0 for p in [pr, sr, painf, a]]):
        return -np.inf
    delta = np.power(10., logd)
    rh = power_law(mass/m_pivot, pr, sr)
    ainf = power_law(mass/m_pivot, painf, sainf)

    # Aggregate likelihood for all mass bins
    lnlike = 0
    for k in range(NMBINS):
        # Compute model deviation from data
        u = y[k, mask[k, :]] - rho_orb_model(x[mask[k, :]], rh[k], ainf[k], a, mass[k])
        # Add percent error to the covariance - regulated by delta
        cov = covy[k, mask[k, :], :][:, mask[k, :]] + \
            np.diag(np.power(delta * y[k, mask[k, :]], 2))
        # Compute chi squared
        chi2 = np.dot(u, np.linalg.solve(cov, u))
        lndetc = len(u) * np.log(scale) + np.log(np.linalg.det(cov / scale))
        lnlike -= chi2 + lndetc
    return lnlike


def lnpost_orb_smooth_a_pl(pars: Union[List[float], Tuple[float, ...]], *data) -> float:
    scale = RHOM**2
    # Unpack data
    x, y, covy, mask, mass, m_pivot = data

    # Check priors. 3 model + 1 likelihood parameters
    pr, sr, ainf, pa, sa, logd = pars
    if not (-4 < logd < 0) or any([p < 0 for p in [pr, sr, ainf, pa, sa, ainf]]):
        return -np.inf
    delta = np.power(10., logd)
    rh = power_law(mass/m_pivot, pr, sr)
    a = power_law(mass/m_pivot, pa, sa)

    # Aggregate likelihood for all mass bins
    lnlike = 0
    for k in range(NMBINS):
        # Compute model deviation from data
        u = y[k, mask[k, :]] - rho_orb_model(x[mask[k, :]], rh[k], ainf, a[k], mass[k])
        # Add percent error to the covariance - regulated by delta
        cov = covy[k, mask[k, :], :][:, mask[k, :]] + \
            np.diag(np.power(delta * y[k, mask[k, :]], 2))
        # Compute chi squared
        chi2 = np.dot(u, np.linalg.solve(cov, u))
        lndetc = len(u) * np.log(scale) + np.log(np.linalg.det(cov / scale))
        lnlike -= chi2 + lndetc
    return lnlike


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


def lnpost_inf(pars: Union[List[float], Tuple[float, ...]], *data) -> float:
    # Unpack data
    x, y, covy, rh = data

    # Check priors.
    b, c, g, b0, r0, logd = pars
    if not (0 < b < 8) or any([p < 0 for p in [c, g, b0, r0]]) or not (-4 < logd < 0):
        return -np.inf
    delta = np.power(10., logd)

    # Compute model deviation from data
    u = x**2 * (y - xi_inf_model(x, b, c, g, b0, r0, rh))
    # Add percent error to the covariance - regulated by delta
    cov = np.outer(x**2, x**2) * (covy + np.diag((delta * y)**2))
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




if __name__ == "__main__":
    pass
#
