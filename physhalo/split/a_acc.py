from datetime import timedelta
from multiprocessing import Pool
from time import time

import h5py as h5
import numpy as np
from scipy.interpolate import interp1d

from physhalo.config import MEMBSIZE, SRC_PATH
from physhalo.utils import BULLET, COLS

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
    start = time()
    with Pool(16) as pool:
        a_acc = pool.map(find_a_acc, range(10000))
        # a_acc = pool.map(find_a_acc, range(rps.shape[0]))
    
    print(
        f"\t{BULLET}{COLS.BOLD}{COLS.WARNING} Elapsed time:{COLS.ENDC} "
        + f"{COLS.OKCYAN}{timedelta(seconds=time()-start)}{COLS.ENDC} " 
        + f"{COLS.OKGREEN}a_acc_rt{COLS.ENDC}"
        )
    
with h5.File(SRC_PATH + "/orbits/garcia22/a_acc_rt_2.h5", "w") as hdf_save:
    hdf_save.create_dataset("a_acc", data=np.array(a_acc), dtype=np.float32)
