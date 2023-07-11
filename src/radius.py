from typing import Tuple

import numpy as np

from src.cosmology import RSOFT


def get_radial_bins(
    n_rbins: int = 20,
    n_ext: int = 7,
    rmin: float = RSOFT,
    rmid: float = 5,
    rmax: float = 50,
    full: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates radial bins.

    Args:
        n_rbins (int, optional): number of radial bins below `rmid`.
            Defaults to 20.
        n_ext (int, optional): number of radial bins above `rmid`. 
            Defaults to 7.
        rmin (int | float, optional): lower radial bound. Defaults to `RSOFT`.
        rmid (int | float, optional): generated `n_rbins` up to this radial
            scale. Defaults to 5 Mpc/h.
        rmax (int | float, optional): upper radial bound. Defaults to 50 Mpc/h. 
            If `full` is `False`, then the upper radial bound is `rmin`.
        full (bool, optional): If `True` extends the radial bins from `rmin` to
            `rmax`. Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: radial bins and bin edges.
    """

    # The first 'n_rbins' are to match the density profile radial bins
    r_edges = np.logspace(np.log10(rmin), np.log10(rmid), num=n_rbins + 1, base=10)
    if full:
        # Extend from 'rmid' up to 'rmax' for another 'n_ext' bins.
        r_edges_ext = np.logspace(
            np.log10(rmid), np.log10(rmax), num=n_ext + 1, base=10
        )
        # Stack the edges into a sigle list.
        r_edges = np.hstack([r_edges, r_edges_ext[1:]])

    # Radial bin middle point
    rbins = 0.5 * (r_edges[1:] + r_edges[:-1])
    return rbins, r_edges
