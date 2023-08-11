import os
from multiprocessing.pool import ThreadPool as Pool

import h5py as h5
import numpy as np
from colossus.cosmology import cosmology
from scipy.interpolate import interp1d

from physhalo.config import MEMBSIZE, SRC_PATH
from physhalo.cosmology import COSMO, G_GRAV
from physhalo.utils import timer

MBINEDGES = [
    [3e13, 4e13],
    [4e13, 5e13],
    [5e13, 7e13],
    [7e13, 1e14],
    [1e14, 2e14],
    [2e14, 3e14],
    [3e14, 4e14],
    [4e14, 5e14],
    [5e14, 7e14],
    [7e14, 1e15],
    [1e15, 3e15],
]
MBINSTRS = [
    "3_4e13",
    "4e13_5e13",
    "5_7e13",
    "7e13_1e14",
    "1_2e14",
    "2_3e14",
    "3_4e14",
    "4_5e14",
    "5_7e14",
    "7e14_1e15",
    "1_3e15",
]
NMBINS = len(MBINSTRS)


@timer
def vt():
    #
    # Compute Vt ===============================================================
    #
    with h5.File(SRC_PATH + "/orbits/garcia22/vt.h5", "w") as hdf_save, h5.File(
        SRC_PATH + "/orbits/garcia22/mt.h5", "r"
    ) as hdf_mass, h5.File(SRC_PATH + "/orbits/garcia22/rt.h5", "r") as hdf_rt:
        mt = hdf_mass["Mt"][()]
        rt = hdf_rt["Rt"][()]
        hdf_save.create_dataset("Vt", data=np.sqrt(np.divide(G_GRAV * mt, rt)))

    #
    # Create mass bins for haloes (Mt bins) ====================================
    #
    with h5.File(SRC_PATH + "/halo_catalogue.h5", "r") as hdf:
        mass = hdf["Mt"][()]

    mean_ = np.zeros(NMBINS)
    median_ = np.zeros(NMBINS)
    with h5.File(SRC_PATH + "/orbits/garcia22/mass_bin_haloes.h5", "w") as hdf:
        # For each mass bin,
        for i in range(NMBINS):
            mask = (MBINEDGES[i][0] < mass) & (mass < MBINEDGES[i][1])
            mean_[i] = np.mean(mass[mask])
            median_[i] = np.median(mass[mask])

            # Save mask to file
            hdf.create_dataset(MBINSTRS[i], data=mask, dtype=bool)
        hdf.create_dataset("mean", data=mean_)
        hdf.create_dataset("median", data=median_)

    #
    # Create mass bins for particles (Mt bins) =================================
    #
    with h5.File(SRC_PATH + "/halo_catalogue.h5", "r") as hdf1, h5.File(
        SRC_PATH + "/orbits/orbit_catalogue_%d.h5",
        "r",
        driver="family",
        memb_size=MEMBSIZE,
    ) as hdf2:
        hids = hdf1["OHID"][()]  # Load halo IDs
        phids = hdf2["HID"][()]  # Load particle HIDs
    
    with h5.File(SRC_PATH + "/orbits/garcia22/mass_bin_haloes.h5", "r") as hdf, h5.File(
        SRC_PATH + "/orbits/garcia22/mass_bin_particles.h5", "w"
    ) as save:
        # For each bin, locate particles in haloes.
        for key in hdf.keys():
            # Ignore other datasets
            if key in ["mean", "median"]:
                continue

            mask = np.isin(phids, hids[hdf[key][()]])
            save.create_dataset(key, data=mask, dtype=bool)
    return


@timer
def get_tinf():
    # Load scale factor
    with h5.File(SRC_PATH + "/scale_factor.h5", "r") as hdf_load:
        scales_all = hdf_load["scale_factor"][()]
    scales_mask = scales_all >= 0.4
    scales = scales_all[scales_mask]
    
    # Load Rt
    with h5.File(SRC_PATH + "/orbits/garcia22/rt.h5", "r") as hdf_load:
        rt = hdf_load["Rt"][()]
        
    # Load particles orbits as a global variable
    with h5.File(SRC_PATH + "/orbits/orbit_catalogue_%d.h5", "r",
                 driver="family", memb_size=MEMBSIZE) as hdf:
        rps = hdf["Rp"]

        def find_a_acc(i):
            rp_j = rps[i][scales_mask]
            rt_j = rt[i]
            if rp_j[-1] <= rt_j:
                a_acc_j = scales[-1]
            elif np.all(rp_j > rt_j):
                a_acc_j = scales[0]
            else:
                mask = np.ones_like(scales, dtype=bool)
                crossed_rt = (np.diff(np.sign(rp_j - rt_j)) != 0)
                if crossed_rt.sum() >= 1:
                    mask = scales <= scales[:-1][crossed_rt][-1]
                scales_interp = interp1d(rp_j[mask],
                                         scales[mask],
                                         kind="linear",
                                         bounds_error=False,
                                         fill_value=(
                                             scales[mask][0],
                                             scales[mask][-1]
                                             ),
                                         )
                a_acc_j = scales_interp(rt_j)
            return a_acc_j
        
        # a_acc = []
        # for i in range(10000):
        #     a_acc.append(find_a_acc(i))
        with Pool() as pool:
            a_acc = pool.map(find_a_acc, range(10000))
    
    with h5.File(SRC_PATH + "/orbits/garcia22/a_acc_rt.h5", "w") as hdf_save:
        hdf_save.create_dataset("a_acc", data=np.array(a_acc), dtype=np.float32)
    
    return


@timer
def get_vr():
    # Load free fall time
    with h5.File(SRC_PATH + "/orbits/garcia22/tff.h5", "r") as hdf:
        tff = hdf["tff"][()]
    
    
    
    return

def main(mbin_i=0) -> None:
    basepath = "/spiff/rgarciamar/susmita_catalog/rafael_halo_model/data/rt"
    if mbin_i > 6:
        basepath = os.path.join(basepath, "6Mpc")
    print(mbin_i, basepath)

    Mmin, Mmax = MBINEDGES[mbin_i]
    mbinstr = MBINSTRS[mbin_i]

    with h5.File(SRC_PATH + "/scale_factor.h5", "r") as hdf_load:
        scales = hdf_load["scale_factor"][()]
    scales_mask = scales >= 0.4
    scales = scales[scales_mask]

    # Instantiate Colossus cosmology
    COSMO_COLOSSUS = {'flat': True, 'H0': 100*COSMO["h"], 'Om0': COSMO["Om0"], 'Ob0': COSMO["Ob0"], 'sigma8': COSMO["sigma8"], 'ns': COSMO["ns"]}
    cosmology.addCosmology("myCosmo", COSMO_COLOSSUS)
    cosmo = cosmology.setCosmology("myCosmo")
    
    #
    # Rescale r by Rt and vr by Vt =============================================
    #
    # with h5.File(SRC_PATH + "/orbits/garcia22/mass_bin_particles.h5", "r") as hdf_load:
    #     mask_mbin = hdf_load[mbinstr][()]
    # with h5.File(SRC_PATH + "/orbits/garcia22/rt.h5", "r") as hdf_load:
    #     rt = hdf_load["Rt"][()]
    # rt = rt[mask_mbin]
    # with h5.File(SRC_PATH + "/orbits/garcia22/vt.h5", "r") as hdf_load:
    #     vt = hdf_load["Vt"][()]
    # vt = vt[mask_mbin]
    # with h5.File(SRC_PATH + "/orbits/garcia22/tff.h5", "r") as hdf_load:
    #     tff = hdf_load["tff"][()]
    # tff = tff[mask_mbin]
    
    # with h5.File(SRC_PATH + "/orbits/orbit_catalogue_%d.h5", "r",
    #              driver="family", memb_size=MEMBSIZE) as hdf_load:
        # pid = hdf_load["PID"][()]
        # pid = pid[mask_mbin]
        # hid = hdf_load["HID"][()]
        # hid = hid[mask_mbin]
        # rp = hdf_load["Rp"][:, [0, 1, 2]]
        # rp = rp[mask_mbin, :]
        # vrp = hdf_load["Vrp"][:, scales_mask_int]
        # vrp = vrp[mask_mbin] / vt
    # pid = np.load(os.path.join(basepath, "{}/pid.npy".format(mbinstr)))
    hid = np.load(os.path.join(basepath, "{}/hid.npy".format(mbinstr)))
    # mass = np.load(os.path.join(basepath, "{}/mass.npy".format(mbinstr)))

    rp = np.load(os.path.join(basepath, "{}/rp_scaled.npy".format(mbinstr)))[:, scales_mask]
    # vrp = np.load(os.path.join(basepath, "{}/vrp_scaled.npy".format(mbinstr)))[:, scales_mask]
    tff = np.load(os.path.join(basepath, "{}/tff.npy".format(mbinstr)))
    tdyn = tff / 4 * np.minimum(rp[:, 0], 1)
    tdyn[hid == -1] = 0
    # scalecut = 1 / (1 + cosmo.lookbackTime(tdyn, inverse=True))
    
    def find_tinf(j):
        alpha = 1.00
        
        if rp[j][-1] <= alpha:
            tinf_j = scales[-1]
        elif np.all(rp[j] > alpha):
            tinf_j = scales[0]
        else:
            mask = np.ones_like(scales, dtype=bool)
            crosses = (np.diff(np.sign(rp[j] - alpha)) != 0)
            if crosses.sum() >= 1:
                mask = scales <= scales[:-1][crosses][-1]
            scales_interp = interp1d(rp[j][mask], scales[mask], kind="linear",
                                     bounds_error=False,
                                     fill_value=(scales[mask][0],
                                                 scales[mask][-1]))
            tinf_j = scales_interp(alpha)
        return tinf_j
    
    def get_tinf():
        with Pool(16) as pool:
            tinf = pool.map(find_tinf, range(rp.shape[0]))
        tinf = np.array(tinf)
        return tinf

    tinf = get_tinf()
    np.save(SRC_PATH + f"/orbits/garcia22/tinf/{mbinstr}.npy", tinf)
    return None


if __name__ == "__main__":
    # vt()
    # get_tinf()
    get_vr()
    # main()
    