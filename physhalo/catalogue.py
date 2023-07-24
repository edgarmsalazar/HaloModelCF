import sys
from os.path import join

sys.path.append("/home/edgarmsc/Projects/HaloModelCF/")

import h5py as h5
import numpy as np

from physhalo.config import MBINEDGES, MBINSTRS, MEMBSIZE, NMBINS, SRC_PATH


def generate_halo_mass_bin_masks() -> None:
    """Creates a mask per mass bin of Morb to quickly select haloes from the
    catalogue.
    """
    # Load Morb
    with h5.File(join(SRC_PATH, "halo_catalogue.h5"), "r") as hdf:
        mass = hdf["Morb"][()]
    # Compute the log10(Morb). Bypass haloes without particles (Morb = 0).
    logmass = np.zeros(mass.shape)
    logmass[(mass != 0)] = np.log10(mass[(mass != 0)])
    logmass[(mass == 0)] = 0

    mean_ = np.zeros(len(MBINEDGES) - 1)
    median_ = np.zeros_like(mean_)
    with h5.File(join(SRC_PATH, "data/mass_bin_haloes.h5"), "w") as hdf:
        # For each mass bin,
        for i in range(NMBINS):
            mask = (MBINEDGES[i] < logmass) & (logmass < MBINEDGES[i + 1])
            mean_[i] = np.mean(mass[mask])
            median_[i] = np.median(mass[mask])

            # Save mask to file
            hdf.create_dataset(MBINSTRS[i], data=mask, dtype=bool)
        hdf.create_dataset("mean", data=mean_)
        hdf.create_dataset("median", data=median_)

    return


def generate_particle_mass_bin_masks() -> None:
    """Generate masks for particles in Morb to quickly select particles from the
    orbit catalogue.
    """

    with h5.File(join(SRC_PATH, "halo_catalogue.h5"), "r") as hdf1, h5.File(
        join(SRC_PATH, "orbits/orbit_catalogue_%d.h5"),
        "r",
        driver="family",
        memb_size=MEMBSIZE,
    ) as hdf2:
        hids = hdf1["OHID"][()]  # Load halo IDs
        phids = hdf2["HID"][()]  # Load particle HIDs

    # Load HID masks
    bins_file = join(SRC_PATH, "data/mass_bin_haloes.h5")
    save_file = join(SRC_PATH, "data/mass_bin_particles.h5")

    with h5.File(bins_file, "r") as hdf, h5.File(save_file, "w") as save:
        # For each bin, locate particles in haloes.
        for key in hdf.keys():
            # Ignore other datasets
            if key in ["mean", "median"]:
                continue

            mask = np.isin(phids, hids[hdf[key][()]])
            save.create_dataset(key, data=mask, dtype=bool)

    return


if __name__ == "__main__":
    pass
#