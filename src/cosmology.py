# Newtonian gravitational constant
G_GRAV = 4.3e-9                     # Mpc (km/s)^2 / M_sun
TO_GYR = (3_086. / 3.1536) / 0.672  # Convert Mpc s / km / h to Gyr

# Banerjee+20 simulation parameters and cosmology
RSOFT = 0.015                       # Softening length in Mpc/h
BOXSIZE = 1_000                     # Mpc / h
GRIDSIZE = 250                      # Mpc / h
PARTMASS = 7.754657e+10             # M_sun / h
RHOCRIT = 2.77536627e+11            # h^2 M_sun / Mpc^3

COSMO = {
    "h": 0.70,
    "Om0": 0.3,
    "Ob0": 0.0469,
    "sigma8": 0.8355,
    "ns": 1,
}

RHOM = RHOCRIT * COSMO["Om0"]       # h^2 M_sun / Mpc^3

# Quijote fiducial cosmology
COSMO_QUIJOTE = {
    "flat": True,
    "h": 0.6711,
    "Ob0": 0.049,
    "Om0": 0.3175,
    "sigma8": 0.834,
    "ns": 0.9624,
}

if __name__ == "__main__":
    pass
#