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
    Mapping from a descriptive dataset label ("system_30", "system_40", ...)
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
Edge-averaged quasi-1D information for a given size (used in Fig. 10):
    average_information_per_scale_along_edge(data)[1][:, 0]

"""

## Imports

import numpy as np
import matplotlib.pyplot as plt


## Input data

data = np.load("fig10.npy", allow_pickle=True).item(0)
sizes = data["sizes"]  # vertical system sizes


## Helpers

def average_information_per_scale_along_edge(data, sizes):
    """Return edge-averaged quasi-1D local information as a function of scale.

    Average is taken along the left edge in the y direction, over the middle portion
    of length size//2 (i.e., excluding a margin dist=size//4 at top and bottom).

    Returns
    -------
    scale : np.ndarray
        Scale value.
    i_scales : np.ndarray
        Array of shape (max(sizes), len(sizes)) storing edge-averaged information.
    """
    max_size = int(np.max(sizes))
    scale = np.arange(max_size)
    i_scales = np.zeros((max_size, len(sizes)))
    for j, size in enumerate(sizes):
        dist = size // 4
        for ly in range(size - 2 * dist):
            for lx in range(3):
                vals = data[f"system_{size}"]["i_local"][lx, ly, : 3 - lx, dist : size - ly - dist].sum(axis=0)
                i_scales[ly, j] += np.mean(vals)
    return scale, i_scales


## Figure

scale, i_scales = average_information_per_scale_along_edge(data, sizes)

fig, ax = plt.subplots(figsize=(3, 2), constrained_layout=True)

for j, size in enumerate(sizes):
    dist = size // 4
    stop = size - 2 * dist
    ax.plot(scale[:stop], (i_scales[:, j] * scale**2)[:stop], lw=1.5)

ax.axhline(1 / (12 * np.log(2)), c="k", label=r"$1/12\ln 2$", dashes=(5, 2), lw=1.5)

ax.set_xlabel(r"$\ell_x$")
ax.set_ylabel(r"$i^{\ell_x}|_{\mathcal{C}_\mathrm{edge}}\cdot\ell_x^{2}$")
ax.yaxis.set_label_coords(-0.15, 0.4)

ax.set_xlim([0, 40])
ax.set_ylim([0.05, 0.25])
ax.set_yticks([0.1, 0.2])
ax.legend()

plt.savefig("fig10.pdf", bbox_inches="tight")
plt.show()
