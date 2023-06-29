from os.path import abspath

# Paths to data // NOTE: These are absolute paths in calvin.physics.arizona.edu
SRC_PATH = abspath('/spiff/edgarmsc/simulations/Banerjee/')

# Common constants
MEMBSIZE = int(10 * 1000**3)    # Orbits catalogue is split into 10.0 GB files

# Bins (log10)
MBINEDGES = [13.40, 13.55, 13.70, 13.85, 14.00, 14.15, 14.30, 14.45, 14.65,
             15.00]
MBINSTRS = ['13.40-13.55', '13.55-13.70', '13.70-13.85', '13.85-14.00',
            '14.00-14.15', '14.15-14.30', '14.30-14.45', '14.45-14.65',
            '14.65-15.00']
NMBINS = len(MBINSTRS)


if __name__ == "__main__":
    pass
#