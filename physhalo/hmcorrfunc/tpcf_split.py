# NOTE: ONLY use these functions to compute the covariances!
# Do not use this functions to compute the correlation function at small
# scales

# NOTE: We cannot use the same `tpcf` module to compute correlation functions
# of tagged particles because we do not have the xyz coordinates but rather the
# radial distance to the centre of the halo.

import math
from typing import Tuple

import h5py as h5
import numpy as np
from tqdm import tqdm
from scipy.linalg import block_diag

from physhalo.config import MBINSTRS, MEMBSIZE, SRC_PATH
from physhalo.cosmology import BOXSIZE, GRIDSIZE, PARTMASS, RHOM
from physhalo.radius import get_radial_bins
from physhalo.hmcorrfunc.tpcf import partition_box


def process_DD_pairs(
    data_1_id: list,
    ohid: np.ndarray,
    pid: np.ndarray,
    ohid_split: np.ndarray,
    pid_split: np.ndarray,
    r_split: np.ndarray,
    boxsize: float,
    gridsize: float,
    rbins: np.ndarray,
) -> np.ndarray:
    """d1: haloes x, y, z coordinates"""

    Nperdim = int(math.ceil(boxsize / gridsize))

    # Create pairs list
    ddpairs_i = np.zeros((Nperdim**3, rbins.size - 1))
    zeros = np.zeros(rbins.size - 1)

    isin_pid = np.isin(pid_split, pid)

    # Pair counting
    # Loop for each minibox
    for s1 in tqdm(range(Nperdim**3), desc="DD pair counting"):
        # Get data1 box
        ohid1 = ohid[data_1_id[s1]]

        isin_ohid = np.isin(ohid_split, ohid1)

        # Data pair counting
        if np.size(ohid1) != 0 and np.size(pid) != 0:
            mask = isin_ohid & isin_pid
            dd, _ = np.histogram(r_split[mask], rbins)
            ddpairs_i[s1, :] = dd
        else:
            ddpairs_i[s1, :] = zeros

    return ddpairs_i


def DD_to_tpcf_jk(
    d1,
    d2,
    d1_id,
    boxsize,
    gridsize,
    rbins,
    ddpairs_i,
):
    # Some quantities
    # Number of cells per dimension
    Nperdim = int(math.ceil(boxsize / gridsize))
    N = Nperdim**3
    d1tot = np.size(d1, 0)  # Number of objects in d1
    d2tot = np.size(d2, 0)  # Number of objects in d2
    Vbox = boxsize**3  # Volume of box
    Vshell = 4.0 / 3.0 * np.pi * (rbins[1:] ** 3 - rbins[:-1] ** 3)
    n1 = float(d1tot) / Vbox  # Number density of d1
    n2 = float(d2tot) / Vbox  # Number density of d2

    # Some arrays
    nbins = rbins.size - 1
    xi_i = np.zeros((N, nbins))

    ddpairs = np.sum(ddpairs_i, axis=0)
    ddpairs_removei = ddpairs[None, :] - ddpairs_i

    # Compute xi_i
    for s1 in range(Nperdim**3):
        d1tot_s1 = np.size(d1, 0) - np.size(d1[d1_id[s1]], 0)
        xi_i[s1] = ddpairs_removei[s1] / (d1tot_s1 * n2 * Vshell) - 1

    # Compute meanxi_i
    meanxi_i = np.mean(xi_i, axis=0)

    # Compute covariance matrix
    cov = (float(N) - 1.0) * np.cov(xi_i.T, bias=True)

    # Compute xi
    xi = ddpairs / (n1 * n2 * Vbox * Vshell) - 1

    return xi, xi_i, meanxi_i, cov


def cross_tpcf_jk(
    data_1: np.ndarray,
    ohid: np.ndarray,
    pid: np.ndarray,
    ohid_split: np.ndarray,
    pid_split: np.ndarray,
    r_split: np.ndarray,
    box_size: float,
    grid_size: float,
    radia_edges: np.ndarray,
    jk_estimates: bool = False,
) -> Tuple[np.ndarray]:
    # Partition boxes
    data_1_id = partition_box(data_1, box_size, grid_size)

    # Pair counting
    ddpairs_i = process_DD_pairs(
        data_1_id=data_1_id,
        ohid=ohid,
        pid=pid,
        ohid_split=ohid_split,
        pid_split=pid_split,
        r_split=r_split,
        boxsize=box_size,
        gridsize=grid_size,
        rbins=radia_edges,
    )

    # Compute xi
    xi, xi_i, meanxi_i, cov = DD_to_tpcf_jk(
        data_1, pid, data_1_id, box_size, grid_size, radia_edges, ddpairs_i
    )

    # Return estimators
    if jk_estimates is True:
        return meanxi_i, cov, xi_i, xi
    else:
        return meanxi_i, cov


def compute_hmcf_cov_split() -> None:
    """Computes the correlation function for `tagged` particles: orbiting,
    infalling and total (orb + inf).

    Args:
        n_rbins (int, optional): number of radial bins (log10). Defaults to 20.
        rmax (int | float, optional): maximum radius in Mpc / h. Defaults to 5.
    """
    # Load haloes
    with h5.File(SRC_PATH + "/halo_catalogue.h5", "r") as hdf:
        # Halo ID
        ohids = hdf["OHID"][()]
        # Halo x, y, z coord
        h_coords = np.stack([hdf["x"][()], hdf["y"][()], hdf["z"][()]]).T

    # Load particles
    with h5.File(
        SRC_PATH + "/orbits/orbit_catalogue_%d.h5",
        "r",
        driver="family",
        memb_size=MEMBSIZE,
    ) as hdf_orb_cat, h5.File(SRC_PATH + "/particle_classification.h5", "r") as hdf_tag:
        opids = hdf_orb_cat["PID"][()]
        hids = hdf_orb_cat["HID"][()]
        rps = hdf_orb_cat["Rp"][:, 0]
        orb_masks = hdf_tag["CLASS"][()]

    rbins, r_edges = get_radial_bins(full=False)

    # Iterate over mass bins.
    for k, mbin in enumerate(MBINSTRS):
        # Select haloes in mass bin
        with h5.File(SRC_PATH + "/data/mass_bin_haloes.h5", "r") as hdf:
            halo_mass_mask = hdf[mbin][()]
        ohid = ohids[halo_mass_mask]
        h_coord = h_coords[halo_mass_mask]

        # Select member particles to selected haloes.
        with h5.File(SRC_PATH + "/data/mass_bin_particles.h5", "r") as hdf:
            particle_mass_mask = hdf[mbin][()]
        pid = opids[particle_mass_mask]
        hid = hids[particle_mass_mask]
        rp = rps[particle_mass_mask]
        orb_mask = orb_masks[particle_mass_mask]

        # Split orbiting and infalling particles in selection
        r_orb = rp[orb_mask]
        hid_orb = hid[orb_mask]
        pid_orb = pid[orb_mask]
        r_inf = rp[~orb_mask]
        hid_inf = hid[~orb_mask]
        pid_inf = pid[~orb_mask]

        with h5.File(SRC_PATH + "/data/xihm_split.h5", "a") as hdf:
            _, cov_orb, _, _ = cross_tpcf_jk(
                h_coord,
                ohids,
                opids,
                hid_orb,
                pid_orb,
                r_orb,
                BOXSIZE,
                GRIDSIZE,
                r_edges,
                jk_estimates=True,
            )
            hdf.create_dataset(f"xi_cov/orb/{mbin}", data=cov_orb)

            _, cov_inf, _, _ = cross_tpcf_jk(
                h_coord,
                ohid,
                opids,
                hid_inf,
                pid_inf,
                r_inf,
                BOXSIZE,
                GRIDSIZE,
                r_edges,
                jk_estimates=True,
            )

            hdf.create_dataset(f"xi_cov/inf/{mbin}", data=cov_inf)

            _, cov_all, _, _ = cross_tpcf_jk(
                h_coord,
                ohid,
                opids,
                hid,
                pid,
                rp,
                BOXSIZE,
                GRIDSIZE,
                r_edges,
                jk_estimates=True,
            )
            hdf.create_dataset(f"xi_cov/all/{mbin}", data=cov_all)

            if k == 0 and not "rbins" in hdf.keys():
                hdf.create_dataset("r_edges", data=r_edges)
                hdf.create_dataset("rbins", data=rbins)

    return


def compute_hmcf_split() -> None:
    """Compute the density profile of orbiting and infalling particles. Requires
    particle tags.

    Args:
        n_rbins (int, optional): number of radial bins (log10). Defaults to 20.
        rmax (int | float, optional): maximum radius in Mpc / h. Defaults to 5.
    """
    # Load particle radii and HID.
    with h5.File(
        SRC_PATH + "/orbits/orbit_catalogue_%d.h5",
        "r",
        driver="family",
        memb_size=MEMBSIZE,
    ) as hdf_orb_cat, h5.File(SRC_PATH + "/particle_classification.h5", "r") as hdf_tag:
        p_radii = hdf_orb_cat["Rp"][:, 0]
        orb = hdf_tag["CLASS"][()]

    rbins, r_edges = get_radial_bins(full=False)
    n_rbins = len(rbins)

    # For each mass bin, compute the density profiles.
    with h5.File(SRC_PATH + "/data/xihm_split.h5", "a") as save:
        for mbin in MBINSTRS:
            rho_orb = np.zeros(n_rbins)
            rho_inf = np.zeros(n_rbins)
            rho_all = np.zeros(n_rbins)

            # Load mass bin mask for particles
            with h5.File(SRC_PATH + "/data/mass_bin_particles.h5", "r") as hdf:
                mbin_mask = hdf[mbin][()]
            # Select radii of particles in mass bin
            orb_radii = p_radii[mbin_mask * orb]
            inf_radii = p_radii[mbin_mask * ~orb]
            all_radii = p_radii[mbin_mask]

            # Count number of haloes in mass bin
            with h5.File(SRC_PATH + "/data/mass_bin_haloes.h5", "r") as hdf:
                n_haloes = hdf[mbin][()]
            n_haloes = n_haloes.sum()

            # Compute spherical shells volume
            const = 4.0 * np.pi / 3.0
            v_shell = const * np.diff((np.power(r_edges, 3)))

            # Compute mass density per spherical shell
            for i in tqdm(range(n_rbins), desc="Compute hmcf"):
                orb_mask = (r_edges[i] < orb_radii) & (orb_radii <= r_edges[i + 1])
                inf_mask = (r_edges[i] < inf_radii) & (inf_radii <= r_edges[i + 1])
                all_mask = (r_edges[i] < all_radii) & (all_radii <= r_edges[i + 1])
                dens = PARTMASS / (n_haloes * v_shell[i])
                rho_orb[i] = dens * orb_mask.sum()
                rho_inf[i] = dens * inf_mask.sum()
                rho_all[i] = dens * all_mask.sum()

            # Save to file
            save.create_dataset(f"rho/orb/{mbin}", data=rho_orb)
            save.create_dataset(f"rho/inf/{mbin}", data=rho_inf)
            save.create_dataset(f"rho/all/{mbin}", data=rho_all)

            save.create_dataset(f"xi/orb/{mbin}", data=(rho_orb / RHOM - 1))
            save.create_dataset(f"xi/inf/{mbin}", data=(rho_inf / RHOM - 1))
            save.create_dataset(f"xi/all/{mbin}", data=(rho_all / RHOM - 1))

        if not "rbins" in save.keys():
            save.create_dataset("rbins", data=rbins)
            save.create_dataset("r_edges", data=r_edges)
    return


def extend_xi_inf(
    ds: int = 100,
) -> None:
    """Patch the correlation function computed from the density profile with the
    correlation function computed using the full particle catalogue.

    Args:
        ds (int, optional): particle catalogue downsample factor. Defaults to
                            100.
    """
    for k, mbin in enumerate(MBINSTRS):
        # Load density profile and compute correlation function.
        with h5.File(SRC_PATH + "/data/xihm_split.h5", "r") as hdf:
            r_edges = hdf["r_edges"][()]
            r = hdf["rbins"][()]
            xi = hdf[f"xi/inf/{mbin}"][()]
            cov = hdf[f"xi_cov/inf/{mbin}"][()]
        start_ = len(r)

        # Load total correlation function and append values from 'start_' up to
        # the last element.
        with h5.File(SRC_PATH + "/data/xihm.h5", "r") as hdf:
            r_edges = np.hstack([r_edges, hdf["r_edges"][start_:]])
            r = np.hstack([r, hdf["rbins"][start_:]])
            xi = np.hstack([xi, hdf[f"xi/{ds}/{mbin}"][start_:]])
            cov = block_diag(cov, hdf[f"cov/{ds}/{mbin}"][start_:, start_:])

        with h5.File(SRC_PATH + "/data/xihm_split.h5", "a") as save:
            if k == 0:
                save.create_dataset("rbins_ext", data=r)
                save.create_dataset("r_edges_ext", data=r_edges)
            save.create_dataset(f"xi_ext/inf/{mbin}", data=xi)
            save.create_dataset(f"xi_ext_cov/inf/{mbin}", data=cov)

    return


if __name__ == "__main__":
    pass
