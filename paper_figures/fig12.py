"""
==============================================================================================
Data for Fig. 12 (Section IV.C) — Information lattice of the toric code with open boundaries
==============================================================================================

This module defines the Python data structure used to generate the results shown in **Fig. 11
in Section IV.C, "Topological order in the toric code"** of the accompanying paper.

Top-level structure
-------------------
data : dict[str]
    Contains the information lattice the toric code ground state with open boundaries.

Keys for each dataset label
---------------------------
name : str
    Dataset label ("open_boundary_toric_code").

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
data = np.load("fig12.npy", allow_pickle=True).item(0)


## Figure
fig, ax_map = plot_infolattice_toric_code(
    i_local=data["i_local"],
    colors=['b', 'w', 'r', 'k']
)
plt.savefig("fig12.pdf", bbox_inches="tight")
plt.show()
