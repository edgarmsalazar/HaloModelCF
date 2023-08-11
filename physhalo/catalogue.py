import h5py as h5
import numpy as np

from physhalo.config import MBINEDGES, MBINSTRS, MEMBSIZE, NMBINS, SRC_PATH
from physhalo.utils import replace_val


def generate_halo_mass_bin_masks() -> None:
    """Creates a mask per mass bin of Morb to quickly select haloes from the
    catalogue.
    """
    # Load Morb
    with h5.File(SRC_PATH + "/halo_catalogue.h5", "r") as hdf:
        mass = hdf["Morb"][()]
    # Compute the log10(Morb). Bypass haloes without particles (Morb = 0).
    logmass = np.zeros(mass.shape)
    logmass[(mass != 0)] = np.log10(mass[(mass != 0)])
    logmass[(mass == 0)] = 0

    mean_ = np.zeros(len(MBINEDGES) - 1)
    median_ = np.zeros_like(mean_)
    with h5.File(SRC_PATH + "/data/mass_bin_haloes.h5", "w") as hdf:
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

    with h5.File(SRC_PATH + "/halo_catalogue.h5", "r") as hdf1, h5.File(
        SRC_PATH + "/orbits/orbit_catalogue_%d.h5",
        "r",
        driver="family",
        memb_size=MEMBSIZE,
    ) as hdf2:
        hids = hdf1["OHID"][()]  # Load halo IDs
        phids = hdf2["HID"][()]  # Load particle HIDs

    # Load HID masks
    bins_file = SRC_PATH + "/data/mass_bin_haloes.h5"
    save_file = SRC_PATH + "/data/mass_bin_particles.h5"

    with h5.File(bins_file, "r") as hdf, h5.File(save_file, "w") as save:
        # For each bin, locate particles in haloes.
        for key in hdf.keys():
            # Ignore other datasets
            if key in ["mean", "median"]:
                continue

            mask = np.isin(phids, hids[hdf[key][()]])
            save.create_dataset(key, data=mask, dtype=bool)

    return


def gen_particle_list(
    halo_variable,
    dtype: np.dtype,
) -> np.ndarray:
    """Generates an array of size `n particles` where each array element is
    equal to the parent halo's value of `halo_variable`. Generating arrays of
    this type is useful to speedup computations over all particles using array.

    Args:
        halo_variable (list | np.ndarray): Array of size `n_haloes` with the
                                           values to be placed in the array.
        dtype (np.dtype): data type of `variable` in save file.

    Returns:
        np.ndarray: 1-D array of type `dtype`.
    """
    # Create an empty array of shape `n particles`.
    with h5.File(
        SRC_PATH + "/orbits/orbit_catalogue_%d.h5",
        "r",
        driver="family",
        memb_size=MEMBSIZE,
    ) as hdf:
        arr = np.full(hdf["PID"].shape, -1, dtype=dtype)

    # Load haloes
    with h5.File(SRC_PATH + "/halo_catalogue.h5", "r") as hdf1, h5.File(
        SRC_PATH + "/halo_particle_dict.h5", "r"
    ) as hdf2:
        hid = hdf1["OHID"][()]
        good_hid = hid[(hdf1["Morb"][()] > 0)]  # with particles

        # Iterate over all haloes
        for key in good_hid:
            arr = replace_val(
                var_io=arr, idx=hdf2[str(key)][()], value=halo_variable[(hid == key)]
            )

    return arr


def generate_hvar_list(hvar: str, file: str, dtype) -> None:
    """Generate a list with hvar values for each particle."""
    with h5.File(SRC_PATH + "/halo_catalogue.h5", "r") as hdf:
        var = hdf[hvar][()]
    part_hvar = gen_particle_list(var, dtype)

    with h5.File(SRC_PATH + f"/orbits/hvar_lists/{file}.h5", "w") as hdf:
        hdf.create_dataset(hvar, data=part_hvar, dtype=dtype)

    return


if __name__ == "__main__":
    pass
#
