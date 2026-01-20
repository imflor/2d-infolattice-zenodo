"""
==============================================================================================
Data for Fig. 9 (Section IV.B) — Scaling of local information in a topological superconductor
==============================================================================================

This module defines the Python data structure used to generate the results shown in **Fig. 9
in Section IV.B, "Gapped bulk with a critical edge in the chiral px + ipy superconductor"**
of the accompanying paper. Four panels compare the local information of ground states in the
topological and trivial regimes of the topological superconductor.

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
import scipy.optimize


## Input data

data = np.load("fig9.npy", allow_pickle=True).item(0)
Nx, Ny = data["trivial"]["size"]
ζ_list = [1, 4, 8, 10, 12, 13]  # Distances from the edge at which to compute bulk and edge contributions


## Helpers

def multiscale_information_bulk_vs_edge(i_local, ζ=2):
    """Compute information per multiscale in the bulk (distance ζ from the edge) and at the edge."""
    i_bulk = np.zeros((Nx, Ny))
    i_edge = np.zeros((Nx, Ny))

    for lx in range(Nx):
        for ly in range(Ny):
            if (lx + 1 <= Nx - ζ) and (ly + 1 <= Ny - ζ):
                i_bulk[lx, ly] = i_local[lx, ly, ζ : Nx - lx - ζ, ζ : Ny - ly - ζ].sum()
            i_edge[lx, ly] = i_local[lx, ly].sum() - i_bulk[lx, ly]

    return i_bulk, i_edge


def quasi1d_information_bulk_vs_edge(i_local, ζ=2, normalize=True):
    """Compute quasi-1D information per scale contributions from bulk and edge."""
    i_bulk, i_edge = multiscale_information_bulk_vs_edge(i_local, ζ=ζ)
    i_bulk_quasi1d = i_bulk.sum(axis=1)
    i_edge_quasi1d = i_edge.sum(axis=1)

    if normalize:
        i_bulk_quasi1d *= (Nx * Ny) / i_bulk_quasi1d.sum()
        i_edge_quasi1d *= (Nx * Ny) / i_edge_quasi1d.sum()

    return i_bulk_quasi1d, i_edge_quasi1d


def remove_zeros(arr):
    idx = np.where(arr > 0)[0]
    return np.arange(1, arr.shape[0])[idx], arr[idx]


def produce_power_law_fit(arr, fit_range=(5, 38)):
    """Produce a power law fit with exponent -2 of a given dataset."""
    begin, end = fit_range
    x_data = np.log(np.arange(begin, end))
    y_data = np.log(arr)[begin - 1 : end - 1]

    f = lambda x, b: -2 * x + b
    power_law_fit = scipy.optimize.curve_fit(f, x_data, y_data)
    return power_law_fit


## Figure

fig, axs = plt.subplots(2, 2, figsize=(6, 6), constrained_layout=True)
(ax1, ax2), (ax3, ax4) = axs


# (a): Information per multiscale of the ground state in the trivial regime
ax1.text(0.02, 0.98, r"(a)", transform=ax1.transAxes, va="top")
i_trivial = data["trivial"]["i_local"].sum(axis=(2, 3)).T  # Information per multiscale
im1 = ax1.imshow(i_trivial, "bwr", vmin=-0.25, vmax=0.25, origin="lower")
ax1.set_xlabel(r"$\ell_x$")
ax1.set_title(r"Trivial ($C=0$)")
ax1.set_ylabel(r"$\ell_y$")


# (b): Information per multiscale of the ground state in the topological regime
ax2.text(0.02, 0.98, r"(b)", transform=ax2.transAxes, va="top")
i_topological = data["topological"]["i_local"].sum(axis=(2, 3)).T  # Information per multiscale
im2 = ax2.imshow(i_topological, "bwr", vmin=-0.25, vmax=0.25, origin="lower")
ax2.set_xlabel(r"$\ell_x$")
ax2.set_title(r"Topological ($C=1$)")
ax2.set_yticks([])


# Colorbar for (a,b)
cbar = fig.colorbar(im1, ax=[ax1, ax2], orientation="vertical", pad=0, fraction=0.025)
cbar.set_ticks([-0.2, 0, 0.2])
cbar.set_label(r"$I(\ell_x,\ell_y)$")


# (c): Comparison of the quasi-1D information per scale in the trivial regime with that in the topological bulk
plots_3 = []
for i, ζ in enumerate(ζ_list):
    i_bulk_quasi1d_topo, _ = quasi1d_information_bulk_vs_edge(data["topological"]["i_local"], ζ=ζ)
    pl = ax3.semilogy(
        *remove_zeros(i_bulk_quasi1d_topo),
        lw=1,
        c="k",
        alpha=(i + 1) / 6,
        label=r"$C=1$: $I(\ell_x)|_{\mathcal{C}_\mathrm{bulk}}$" if i == len(ζ_list) - 1 else None,
    )
    plots_3.append(pl[0])
ax3.semilogy(np.arange(1, Nx + 1), i_trivial.sum(axis=1), lw=1.5, c="r", label=r"$C=0$: $I(\ell_x)$")
leg1 = ax3.legend()
leg2 = ax3.legend(
    plots_3,
    [rf"${z}$" for z in ζ_list],
    title=r"$\zeta$",
    loc="lower left",
    fontsize=9,
    title_fontsize=9,
    frameon=True,
)
ax3.add_artist(leg1)
ax3.set_xlabel(r"$\ell_x$")
ax3.set_ylabel(r"$I(\ell_x)$")
ax3.set_ylim([3e-10, 5e3])
ax3.set_xlim([1, 29.9])
ax3.text(0.02, 0.98, r"(c)", transform=ax3.transAxes, va="top")


# (d): Scaling of the quasi-1D information per scale in the topological regime
ax4.text(0.02, 0.98, r"(d)", transform=ax4.transAxes, va="top")
x = np.linspace(0.5, 40, 100)
fit = produce_power_law_fit(i_topological.sum(axis=1))
ax4.loglog(x, np.exp(fit[0][0]) * x ** (-2), "-.", lw=1.5, label=rf"$\propto\ell_x^{{-{2}}}$")
ax4.loglog(np.arange(1, Nx + 1), i_topological.sum(axis=1), dashes=(1, 0.5), lw=1.5, label=r"$I(\ell_x)$")
_, i_edge_quasi1d_topo = quasi1d_information_bulk_vs_edge(data["topological"]["i_local"], ζ=13)
ax4.loglog(np.arange(1, Nx + 1), i_edge_quasi1d_topo, lw=1.5, label=r"$I(\ell_x)|_\mathrm{all\ edges}$")
ax4.set_xlabel(r"$\ell_x$")
ax4.set_xlim([1, Nx])
ax4.legend()
ax4.yaxis.tick_right()


plt.savefig("fig9.pdf")
plt.show()
