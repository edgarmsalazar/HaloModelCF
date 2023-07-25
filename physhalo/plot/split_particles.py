from os.path import exists

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from physhalo.config import MEMBSIZE, SRC_PATH
from physhalo.plot.config import SIZE_LABELS, SIZE_LEGEND, SIZE_TICKS
from physhalo.utils import timer

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "savefig.facecolor": 'white',
    "figure.dpi": 150,
})


@timer
def split_particles_single_halo() -> None:
    fnamer = SRC_PATH + "/data/plot/data/split_particles_single_halo_xyz.npy"
    fnamev = SRC_PATH + "/data/plot/data/split_particles_single_halo_vel.npy"
    fnamet = SRC_PATH + "/data/plot/data/split_particles_single_halo_tag.npy"
    fnamei = SRC_PATH + "/data/plot/data/split_particles_single_halo_idx.npy"

    if all([exists(fn) for fn in [fnamei, fnamet, fnamer, fnamev]]):
        xyz = np.load(fnamer)
        vel = np.load(fnamev)
        tag = np.load(fnamet)
        idx = np.load(fnamei)
        with h5.File(SRC_PATH + "/halo_catalogue.h5", "r") as hdf:
            idx_max = np.argmax(hdf["Morb"])
            halo_r200 = hdf["R200m"][idx_max]
    else:
        # Load most massive halo
        with h5.File(SRC_PATH + "/halo_catalogue.h5", "r") as hdf:
            idx_max = np.argmax(hdf["Morb"])
            hid_max = hdf["OHID"][idx_max]
            halo_r200 = hdf["R200m"][idx_max]
            halo_xyz = (hdf["x"][idx_max], hdf["y"][idx_max], hdf["z"][idx_max])
            halo_vel = (hdf["vx"][idx_max], hdf["vy"][idx_max], hdf["vz"][idx_max])

        # Load particles belonging to the selected HID
        with h5.File(SRC_PATH + "/halo_particle_dict.h5", "r") as hdf:
            idx_pid = hdf[str(hid_max)][()]

        # Load members PIDs
        with h5.File(
            SRC_PATH + "/orbits/orbit_catalogue_%d.h5",
            "r",
            driver="family",
            memb_size=MEMBSIZE,
        ) as hdf:
            idx_pid = np.argwhere(hdf["HID"][()] == hid_max).reshape(-1)
            pid = hdf["PID"][idx_pid]

        # Load members tags (TRUE == orbiting)
        with h5.File(SRC_PATH + "/particle_classification.h5", "r") as hdf:
            tag = hdf["CLASS"][idx_pid]

        df = np.stack([pid, tag]).T
        df = pd.DataFrame(df, columns=["PID", "TAG"])
        df.sort_values(by=["PID"], inplace=True, ignore_index=True)
        tag = df["TAG"].values
        tag = np.array(tag, dtype=bool)
        np.save(fnamet, tag)

        # Load all particles' PIDs
        with h5.File(SRC_PATH + "/particle_catalogue.h5", "r") as hdf:
            pid_all = hdf["snap99/1/PID"][()]
            idx_member = np.isin(pid_all, pid)
            # Match to member's PID
            df2 = np.stack(
                [
                    hdf["snap99/1/PID"][idx_member],
                    hdf["snap99/1/x"][idx_member],
                    hdf["snap99/1/y"][idx_member],
                    hdf["snap99/1/z"][idx_member],
                    hdf["snap99/1/vx"][idx_member],
                    hdf["snap99/1/vy"][idx_member],
                    hdf["snap99/1/vz"][idx_member],
                ]
            ).T
        df2 = pd.DataFrame(df2, columns=["PID", "x", "y", "z", "vx", "vy", "vz"])
        df2.sort_values(by=["PID"], inplace=True, ignore_index=True)
        xyz = df2[["x", "y", "z"]].values - halo_xyz
        vel = df2[["vx", "vy", "vz"]].values - halo_vel
        # Save relative positions for each particle.
        np.save(fnamer, xyz)
        np.save(fnamev, vel)
        np.save(fnamei, idx_member)

    # Transform coordinates from cartesian to spherical
    #
    #   r = sqrt( x**2 + y**2 + z**2 )
    #
    rps = np.sqrt(np.sum(np.square(xyz), axis=1))
    #
    #   theta = arccos( z / r )
    #
    thetas = np.arccos(xyz[:, 2] / rps)
    #
    #   phi = arctan( y / x)
    #
    phis = np.arctan2(xyz[:, 1], xyz[:, 0])

    # Get radial vector in cartesian coordinates
    rp_hat = np.zeros_like(xyz)
    rp_hat[:, 0] = np.sin(thetas) * np.cos(phis)
    rp_hat[:, 1] = np.sin(thetas) * np.sin(phis)
    rp_hat[:, 2] = np.cos(thetas)

    # rt_hat = np.zeros_like(xyz)
    # rt_hat[:, 0] = np.cos(thetas) * np.cos(phis)
    # rt_hat[:, 1] = np.cos(thetas) * np.sin(phis)
    # rt_hat[:, 2] = -np.sin(thetas)

    # Compute radial velocity as v dot r_hat
    vrps = np.sum(vel * rp_hat, axis=1)
    # vtps = np.sum(vel * rt_hat, axis=1)

    _, axes = plt.subplots(
        2,
        3,
        figsize=(10, 6),
        sharey="row",
        gridspec_kw={
            "height_ratios": [2, 1],
            "wspace": 0.05,
        },
    )
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat

    legend = [
        Line2D([0], [0], color="k", lw=1.0, ls="--", label=r"$R_{\rm 200m}$"),
    ]

    cmapp = "BuPu"
    for ax in axes.flat:
        ax.tick_params(axis="both", which="major", labelsize=SIZE_TICKS)
        # ax.set_facecolor(plt.get_cmap(cmapp)(0))
        ax.set_facecolor("w")

    # Real space projection ====================================================
    bins1 = (200, 200)
    range1 = ((-6, 6), (-6, 6))
    vmax1 = 100

    for ax in axes.flat[:3]:
        ax.set_xlabel(r"$x~[h^{-1}{\rm Mpc}]$", fontsize=SIZE_LABELS)
        ax.set_aspect("equal")
        circle1 = plt.Circle((0, 0), halo_r200, fill=False, color="k", lw=1, ls="--")
        ax.add_artist(circle1)
        ax.set_yticks([-5, -3, 0, 2, 5])
        ax.set_xticks([-5, -2, 0, 2, 5])

    # All
    ax1.hist2d(
        xyz[:, 0],
        xyz[:, 1],
        bins=bins1,
        range=range1,
        cmap=cmapp,
        norm=LogNorm(vmax=vmax1),
    )
    ax1.set_ylabel(r"$y~[h^{-1}{\rm Mpc}]$", fontsize=SIZE_LABELS, labelpad=20)
    ax1.legend(handles=legend, loc="upper left", fontsize=SIZE_LEGEND)

    # Orbiting
    ax2.hist2d(
        xyz[:, 0][tag],
        xyz[:, 1][tag],
        bins=bins1,
        range=range1,
        cmap=cmapp,
        norm=LogNorm(vmax=vmax1),
    )

    # Infalling
    ax3.hist2d(
        xyz[:, 0][~tag],
        xyz[:, 1][~tag],
        bins=bins1,
        range=range1,
        cmap=cmapp,
        norm=LogNorm(vmax=vmax1),
    )

    ax1.set_title("Total", fontsize=SIZE_LABELS)
    ax2.set_title("Orbiting", fontsize=SIZE_LABELS)
    ax3.set_title("Infalling", fontsize=SIZE_LABELS)

    # Phase space projection ===================================================
    bins2 = (100, 50)
    range2 = ((0, 6), (-5000, 5000))
    vmax2 = 50

    for ax in axes.flat[3:]:
        ax.set_xlabel(r"$r~[h^{-1}{\rm Mpc}]$", fontsize=SIZE_LABELS)
        ax.vlines(halo_r200, -5000, 5000, color="k", ls="--", lw=1.0)
        # ax.vlines(halo_rt, -5000, 5000, color='w', ls='-')
        ax.hlines(0, 0, 8, color="k", ls=":", lw=1.0)

    # All
    ax4.hist2d(rps, vrps, bins=bins2, range=range2, cmap=cmapp, vmax=vmax2)
    ax4.set_ylabel(r"$v_r~[{\rm km}/{\rm s}]$", fontsize=SIZE_LABELS)

    # Orbiting
    ax5.hist2d(rps[tag], vrps[tag], bins=bins2, range=range2, cmap=cmapp, vmax=vmax2)
    # Infalling
    ax6.hist2d(rps[~tag], vrps[~tag], bins=bins2, range=range2, cmap=cmapp, vmax=vmax2)
    
    # Save
    plt.tight_layout()
    plt.savefig(
        SRC_PATH + "/data/plot/split_particles_single_halo.png", bbox_inches="tight"
    )
    plt.clf()
    return None


if __name__ == "__main__":
    pass
