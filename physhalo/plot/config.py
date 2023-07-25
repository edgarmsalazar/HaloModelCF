import numpy as np
from matplotlib.cm import get_cmap
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D

from physhalo.config import MBINEDGES, NMBINS

CMAP = get_cmap("inferno")

LINE = Line2D(
    [], [], marker="o", markerfacecolor="k", color="k", markersize=5, linestyle="None"
)
BARLINE = LineCollection(np.empty((2, 2, 2)), color="k")
ERRLINE_1 = ErrorbarContainer(
    (LINE, [LINE], [BARLINE]),
    has_xerr=False,
    has_yerr=True,
    label="Data",
)
ERRLINE_2 = ErrorbarContainer(
    (LINE, [LINE], [BARLINE]),
    has_xerr=False,
    has_yerr=True,
    label="Individual",
)
LEGEND_1 = [ERRLINE_1, Line2D([0], [0], color="k", lw=1.0, label="Fit")]
LEGEND_2 = [ERRLINE_2, Line2D([0], [0], color="k", lw=1.0, label="Smooth")]
NORM = BoundaryNorm(MBINEDGES, CMAP.N * (NMBINS - 1) / NMBINS)

# Formatting
SIZE_TICKS = 12
SIZE_LABELS = 16
SIZE_LEGEND = 14


if __name__ == "__main__":
    pass
