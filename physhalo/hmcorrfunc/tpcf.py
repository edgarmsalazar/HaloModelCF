import math
from os.path import join
from typing import List, Tuple
from warnings import filterwarnings

filterwarnings("ignore")

import Corrfunc
import h5py as h5
import numpy as np
from tqdm import tqdm

from physhalo.config import MBINSTRS, SRC_PATH
from physhalo.cosmology import BOXSIZE, GRIDSIZE
from physhalo.radius import get_radial_bins


def partition_box(
    data: np.ndarray,
    box_size: float,
    grid_size: float,
) -> List[float]:
    # Number of grid cells per side.
    n_cpd = int(math.ceil(box_size / grid_size))
    # Grid ID for each data point.
    grid_id = data[:] / grid_size
    grid_id = grid_id.astype(int)
    # Correction for points on the edges.
    grid_id[np.where(grid_id == n_cpd)] = n_cpd - 1

    # This list stores all of the particles original IDs in a convenient 3D
    # list. It is kind of a pointer of size n_cpd**3
    data_id = [[] for _ in range(n_cpd**3)]
    cells = n_cpd**2 * grid_id[:, 0] + n_cpd * grid_id[:, 1] + grid_id[:, 2]
    for cell in tqdm(range(np.size(data, 0)), desc="Partitioning box"):
        data_id[cells[cell]].append(cell)

    return data_id


def process_DD_pairs(
    data_1: np.ndarray,
    data_2: np.ndarray,
    data_1_id: list,
    box_size: float,
    grid_size: float,
    radial_edges: np.ndarray,
    nthreads: int = 16,
) -> np.ndarray:
    # Number of grid cells per side
    n_cpd = int(math.ceil(box_size / grid_size))

    # Create pairs list
    ddpairs_i = np.zeros((n_cpd**3, radial_edges.size - 1))
    zeros = np.zeros(radial_edges.size - 1)

    # Pair counting
    # Loop for each minibox
    for s1 in tqdm(range(n_cpd**3), desc="Pair counting"):
        # Get data1 box
        xd1 = data_1[data_1_id[s1], 0]
        yd1 = data_1[data_1_id[s1], 1]
        zd1 = data_1[data_1_id[s1], 2]

        # Get data2 box
        xd2 = data_2[:, 0]
        yd2 = data_2[:, 1]
        zd2 = data_2[:, 2]

        # Data pair counting
        if np.size(xd1) != 0 and np.size(xd2) != 0:
            # Data pair counting
            autocorr = 0
            DD_counts = Corrfunc.theory.DD(
                autocorr=autocorr,
                nthreads=nthreads,
                binfile=radial_edges,
                X1=xd1,
                Y1=yd1,
                Z1=zd1,
                X2=xd2,
                Y2=yd2,
                Z2=zd2,
                periodic=True,
                boxsize=box_size,
                verbose=False,
            )["npairs"]
            ddpairs_i[s1] = DD_counts
        else:
            ddpairs_i[s1] = zeros

    return ddpairs_i


def tpck_with_jk_from_DD(
    data_1: np.ndarray,
    data_2: np.ndarray,
    data_1_id: list,
    box_size: float,
    grid_size: float,
    radial_edges: np.ndarray,
    dd_pairs: np.ndarray,
) -> Tuple[np.ndarray]:
    # Set up bins
    nbins = radial_edges.size - 1

    n_cpd = int(math.ceil(box_size / grid_size))  # Number of cells per dimension
    n_jk = n_cpd**3  # Number of jackknife samples
    d1tot = np.size(data_1, 0)  # Number of objects in d1
    d2tot = np.size(data_2, 0)  # Number of objects in d2
    Vbox = box_size**3  # Volume of box
    Vshell = np.zeros(nbins)  # Volume of spherical shell
    # Vjk = (N - 1) / N * Vbox
    for m in range(nbins):
        Vshell[m] = (
            4.0 / 3.0 * np.pi * (radial_edges[m + 1] ** 3 - radial_edges[m] ** 3)
        )
    n1 = float(d1tot) / Vbox  # Number density of d2
    n2 = float(d2tot) / Vbox  # Number density of d2

    # Some arrays
    ddpairs_removei = np.zeros((n_jk, nbins))
    xi = np.zeros(nbins)
    xi_i = np.zeros((n_jk, nbins))
    meanxi_i = np.zeros(nbins)
    cov = np.zeros((nbins, nbins))

    ddpairs = np.sum(dd_pairs, axis=0)

    ddpairs_removei = ddpairs[None, :] - dd_pairs
    for s1 in range(n_jk):
        d1tot_s1 = np.size(data_1, 0) - np.size(data_1[data_1_id[s1]], 0)
        # xi_i[s1] = dd_pairs_i[s1] / (n1 * n2 * Vjk * Vshell) - 1
        xi_i[s1] = ddpairs_removei[s1] / (d1tot_s1 * n2 * Vshell) - 1

    # Compute mean xi from all jk samples
    for i in range(nbins):
        meanxi_i[i] = np.mean(xi_i[:, i])

    # Compute covariance matrix
    cov = (float(n_jk) - 1.0) * np.cov(xi_i.T, bias=True)

    # Compute the total xi
    xi = ddpairs / (n1 * n2 * Vbox * Vshell) - 1

    return xi, xi_i, meanxi_i, cov


def cross_tpcf_jk(
    data_1: np.ndarray,
    data_2: np.ndarray,
    box_size: float,
    grid_size: float,
    radial_edges: np.ndarray,
    nthreads: int = 16,
    jk_estimates: bool = True,
) -> Tuple[np.ndarray]:
    # Partition boxes
    data_1_id = partition_box(data_1, box_size, grid_size)

    # Pair counting. NOTE: data_1 and data_2 must have the same dtype.
    dd_pairs = process_DD_pairs(
        data_1=data_1,
        data_2=data_2,
        data_1_id=data_1_id,
        box_size=box_size,
        grid_size=grid_size,
        radial_edges=radial_edges,
        nthreads=nthreads,
    )

    xi, xi_i, mean_xi_i, cov = tpck_with_jk_from_DD(
        data_1=data_1,
        data_2=data_2,
        data_1_id=data_1_id,
        box_size=box_size,
        grid_size=grid_size,
        radial_edges=radial_edges,
        dd_pairs=dd_pairs,
    )
    # Total correlation function
    # Correlation function per subsample (subvolume)
    # Mean correlation function
    # Covariance
    if jk_estimates:
        return xi, xi_i, mean_xi_i, cov
    # Mean correlation function
    # Covariance
    else:
        return mean_xi_i, cov


def compute_mmcf(
    ds: int = 100,
) -> None:
    # Load particles' x, y, z coordinates.
    with h5.File(join(SRC_PATH, f"particle_catalogue.h5"), "r") as hdf_load:
        p_coord = np.vstack(
            [
                hdf_load[f"snap99/{ds}/x"][()],
                hdf_load[f"snap99/{ds}/y"][()],
                hdf_load[f"snap99/{ds}/z"][()],
            ]
        ).T

    rbins, r_edges = get_radial_bins()

    with h5.File(join(SRC_PATH, f"data/ximm.h5"), "a") as hdf_save:
        xi, xi_i, mean_xi_i, cov = cross_tpcf_jk(
            p_coord, p_coord, BOXSIZE, GRIDSIZE, r_edges, 16, jk_estimates=True
        )
        hdf_save.create_dataset(f"xi/{ds}", data=xi)
        hdf_save.create_dataset(f"xi_i/{ds}", data=xi_i)
        hdf_save.create_dataset(f"meanxi_i/{ds}", data=mean_xi_i)
        hdf_save.create_dataset(f"cov/{ds}", data=cov)
        hdf_save.create_dataset("r_edges", data=r_edges)
        hdf_save.create_dataset("rbins", data=rbins)

    return


def compute_hmcf(
    ds: int = 100,
) -> None:
    # Load particles' x, y, z coordinates.
    with h5.File(join(SRC_PATH, f"particle_catalogue.h5"), "r") as hdf_load:
        p_coord = np.vstack(
            [
                hdf_load[f"snap99/{ds}/x"][()],
                hdf_load[f"snap99/{ds}/y"][()],
                hdf_load[f"snap99/{ds}/z"][()],
            ]
        ).T

    # Load all haloes's x, y, z coordinates.
    with h5.File(join(SRC_PATH, "halo_catalogue.h5"), "r") as hdf_load:
        h_coord_all = np.vstack(
            [hdf_load["x"][()], hdf_load["y"][()], hdf_load["z"][()]]
        ).T

    rbins, r_edges = get_radial_bins()

    # Iterate over mass bins to compute the correlation function.
    for k, mbin in enumerate(MBINSTRS):
        # Select haloes in mass bin
        with h5.File(join(SRC_PATH, "data/mass_bin_haloes.h5"), "r") as hdf_load:
            halo_mass_mask = hdf_load[mbin][()]
        h_coord = h_coord_all[halo_mass_mask]

        with h5.File(join(SRC_PATH, f"data/xihm.h5"), "a") as hdf_save:
            xi, xi_i, mean_xi_i, cov = cross_tpcf_jk(
                h_coord, p_coord, BOXSIZE, GRIDSIZE, r_edges, 16, jk_estimates=True
            )
            hdf_save.create_dataset(f"xi/{ds}/{mbin}", data=xi)
            hdf_save.create_dataset(f"xi_i/{ds}/{mbin}", data=xi_i)
            hdf_save.create_dataset(f"meanxi_i/{ds}/{mbin}", data=mean_xi_i)
            hdf_save.create_dataset(f"cov/{ds}/{mbin}", data=cov)

            if k == 0 and "rbins" not in hdf_save.keys():
                hdf_save.create_dataset("r_edges", data=r_edges)
                hdf_save.create_dataset("rbins", data=rbins)

    return


if __name__ == "__main__":
    pass
# 
