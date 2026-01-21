"""
==============================================================================================
Data for Fig. 11 (Section IV.C) — Information lattice of the infinite planar toric code
==============================================================================================

This module defines the Python data structure used to generate the results shown in **Fig. 11
in Section IV.C, "Topological order in the toric code"** of the accompanying paper.

Top-level structure
-------------------
data : dict[str]
    Contains the information lattice on a subsystem of the infinite toric code.

Keys for each dataset label
---------------------------
name : str
    Dataset label ("infinite_planar_toric_code").

size : list[int]
    System size in number of plaquettes along each direction, given as [Nx, Ny].

i_local : np.ndarray
    Local multiscale quantity evaluated on rectangular subsystems.
    The array has four axes:
    - axis 0, 1: subsystem scale indices (two-dimensional scale along x and y)
    - axis 2, 3: subsystem position indices (location on the Nx×Ny plaquette lattice)

"""

## Imports

import numpy as np
import matplotlib.pyplot as plt
from utils.plotting import plot_infolattice_toric_code


## Input data
data = np.load("fig11.npy", allow_pickle=True).item(0)


## Figure
fig, ax_map = plot_infolattice_toric_code(
    i_local=data["i_local"],
    colors=['w', 'r']
)
plt.savefig("fig11.pdf", bbox_inches="tight")
plt.show()
