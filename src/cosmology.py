# Newtonian gravitational constant
G_GRAV = 4.3e-9                     # Mpc (km/s)^2 / M_sun
TO_GYR = (3_086. / 3.1536) / 0.672  # Convert Mpc s / km / h to Gyr

# Banerjee+20 simulation parameters and cosmology
RSOFT = 0.015                       # Softening length in Mpc/h
BOXSIZE = 1_000                     # Mpc / h
PARTMASS = 7.754657e+10             # M_sun / h
RHOCRIT = 2.77536627e+11            # h^2 M_sun / Mpc^3

COSMO = {
    "flat": True,
    "H0": 70,
    "Om0": 0.3,
    "Ob0": 0.0469,
    "sigma8": 0.8355,
    "ns": 1,
}

COSMO_CLASS = {
    "output": "mPk",
    "non linear": "halofit",
    "Omega_b": 0.0469,
    "Omega_cdm": 0.3 - 0.0469,
    "h": 0.70,
    "sigma8": 0.8355,
    "n_s": 1,
    "P_k_max_1/Mpc": 1000,
    "z_max_pk": 10.0,  # Default value is 10
}

RHOM = RHOCRIT * COSMO["Om0"]       # h^2 M_sun / Mpc^3


if __name__ == "__main__":
    pass
#