"""
==============================================================================================
Data for Fig. 10 (Section IV.B) — Edge scaling of information in a topological superconductor
==============================================================================================

This module defines the Python data structure used to generate the results shown in **Fig. 10
in Section IV.B, "Gapped bulk with a critical edge in the chiral px + ipy superconductor"**
of the accompanying paper.

Top-level structure
-------------------
data : dict[str, dict]
    Mapping from a descriptive dataset label ("trivial" or "topological")
    to a dictionary with the following keys:

Keys for each dataset label
---------------------------
size : list[int]
    System size in number of lattice sites along each direction, given as [Nx, Ny].

hamiltonian_params : dict[str, float]
    Parameters of the px+ipy tight-binding Hamiltonian:
    - delta : float
        Superconducting pairing potential.
    - t : float
        Nearest-neighbor hopping amplitude.
    - mu : float
        Chemical potential.

i_local : np.ndarray
    Local multiscale quantity evaluated on rectangular subsystems.
    The array has four axes:
    - axis 0, 1: subsystem scale indices (two-dimensional scale along x and y)
    - axis 2, 3: subsystem position indices (location on the Nx×Ny lattice)

Example usage
-------------
Information per multiscale (used in panels a,b):
    data["trivial"]["i_local"].sum(axis=(2, 3)).T

Quasi-1D information per scale (used in panels c,d):
    data["topological"]["i_local"].sum(axis=(1, 2, 3))

Bulk-vs-edge quasi-1D decomposition at distance ζ from the boundary:
    quasi1d_information_bulk_vs_edge(data["topological"]["i_local"], ζ=13)

"""

## Imports

import numpy as np
import matplotlib.pyplot as plt

## Input data

data = np.load("fig10.npy", allow_pickle=True).item(0)
sizes = data["sizes"]  # vertical system sizes

## Helpers

def average_information_per_scale_along_edge():
    """Returns the average quasi-1D local information along the edge in the y direction (left edge).
    Done along the middle portion of the edge of length `size // 2`, at a distance `dist` from the edge.
    """
    scale = np.arange(np.max(sizes))
    i_scales = np.zeros([np.max(sizes), len(sizes)])
    for i, size in enumerate(sizes):
        dist = size // 4
        for ly in range(size - 2 * dist):
            for lx in range(3):
                i_scales[ly, i] += np.mean(
                    data["system_%d" % size]["i_local"][lx, ly, :3 - lx, dist:size - ly - dist].sum(axis=0))
    return scale, i_scales

## Figure

scale, i_scales = average_information_per_scale_along_edge()

fig = plt.figure(figsize=(3, 2), constrained_layout=True)
ax = fig.add_subplot(111)
for i, size in enumerate(sizes):
    dist = size // 4
    plt.plot(scale[:size-2*dist], (i_scales[:, i] * scale**2)[:size-2*dist], lw=1.5)
plt.axhline(1 / (12 * np.log(2)), c='k', label=r'$1/12\ln 2$', dashes=(5, 2), lw=1.5)
plt.ylabel(r'$i^{\ell_x}|_{\mathcal{C}_\mathrm{edge}}\cdot\ell_x^{2}$')
ax.yaxis.set_label_coords(-0.15,.4)
plt.xlabel(r'$\ell_x$')
plt.ylim([0.05, 0.25])
plt.xlim([0, 40])
plt.yticks([.1, .2,])
plt.legend()
plt.savefig('fig10.pdf', bbox_inches='tight')
plt.show()
