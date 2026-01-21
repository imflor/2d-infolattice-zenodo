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


scale = np.arange(1, np.max(Nys)+1)
i_scales = np.zeros([np.max(Nys), len(Nys)])
marg = lambda Ny: Ny // 4

for i, Ny in enumerate(Nys):
    margin = marg(Ny)
    i_loc = i_locs[i]
    penetration_depth = lat.nx - 1
    penetration_depth = 3

    for ly in range(Ny-2*margin):
        #for lx in range(penetration_depth):
        for lx in range(penetration_depth):
            # i_scales[ly, i] += np.mean(lat.i_loc[lx, ly, :penetration_depth-lx, margin:lat.ny-ly-margin].sum(axis=0))
                    # + np.mean(lat.i_loc[lx, ly, lat.nx-lx-1:lat.nx-lx, margin:lat.ny - ly - margin].sum(axis=0))
            i_scales[ly, i] += np.mean(i_loc[lx, ly, :penetration_depth-lx, margin:Ny-ly-margin].sum(axis=0)) \
                    # TOP EDGE CONTRIBUTION (Discarded): + np.mean(i_loc[lx, ly, :, margin:Ny - ly - margin][::-1, :][:penetration_depth-lx, :].sum(axis=0))

col = '#4d5c80'
fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)
xs = [Ny-2*marg(Ny)-2 for Ny in Nys]
xs = [15, 19, 24, 30, 38]
ys = [0.29/2] * 5
plt.text(xs[0]-3.5, ys[0], r'$N_x\!=$', fontsize=16, c=col, ha='right', va='center')
for i, Ny in enumerate(Nys):
    plt.plot((i_scales[:, i]*(scale-1)**2)[:Ny-2*marg(Ny)], '-', lw=1.5, markersize=0, c=col, alpha=(i+1)/(len(Nys)+2))
    plt.text(xs[i], ys[i], r'$%d$' % Ny, fontsize=16, c=col, alpha=(i+2)/(len(Nys)+2), ha='right', va='center')
plt.axhline(1 / (12 * np.log(2)), c='r', label=r'$1/12\ln 2$', dashes=(5, 0), lw=1.5)
plt.ylabel(r'$i^{\ell_x}|_{\mathcal{C}_\mathrm{edge}}\cdot\ell_x^{2}$', rotation=90, fontsize=19)
ax.yaxis.set_label_coords(-0.15,.4)
plt.xlabel(r'$\ell_x$')
plt.ylim([0.05, 0.25])
plt.xlim([0, 40])
plt.yticks([.1, .2,])
plt.legend()
plt.tight_layout()
fig.text(0.1, .87, r'(b)', fontsize=20)
plt.savefig('figures/fig6.pdf', bbox_inches='tight')
plt.show()