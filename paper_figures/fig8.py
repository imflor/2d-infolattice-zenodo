"""
==============================================================================================
Data for Fig. 8 (Section IV.A) — Critical ground states of the anisotropic 2D Anderson model
==============================================================================================

This module defines the Python data structure used to generate the results shown in
**Fig. 8 in Section IV.A, "Localized and critical ground states of the 2D Anderson model"**
of the accompanying paper. Each panel shows the local information at fixed position (0,0)
for critical ground states with varying anisotropy, together with an inset of the Fermi surface
and an arrow indicating the averaged Fermi-velocity orientation used in the figure.

Top-level structure
-------------------
data : dict[str, dict]
    Mapping from a descriptive dataset label (e.g. "critical_1", "critical_2", ...)
    to a dictionary with the following keys:

Keys for each dataset label
---------------------------
size : list[int]
    System size in number of lattice sites along each direction, given as [Nx, Ny].

hamiltonian_params : dict[str, float]
    Parameters of the 2D Anderson tight-binding Hamiltonian:
    - tx : float
        Nearest-neighbor hopping amplitude along x.
    - ty : float
        Nearest-neighbor hopping amplitude along y.
    - disorder_strength : float
        On-site disorder strength.

i_local : np.ndarray
    Local multiscale quantity evaluated on rectangular subsystems.
    The array has four axes:
    - axis 0, 1: subsystem scale indices (two-dimensional scale along x and y)
    - axis 2, 3: subsystem position indices (location on the Nx×Ny lattice)

Example usage
-------------
Local information at position (0, 0) in panels (a–c):
    data["critical_1"]["i_local"][:, :, 0, 0].T

"""

## Imports

import numpy as np
import matplotlib.pyplot as plt


## Input data

data = np.load("fig8.npy", allow_pickle=True).item(0)


## Helpers

def add_inset_ax(ax, size=0.5):
    rect = [0.3, 0.5, size, size]
    inset_ax = ax.inset_axes(rect)
    inset_ax.set_facecolor("none")
    return inset_ax


def plot_fermi_surface(ax, tx, ty):
    L = 1.1
    kx = np.linspace(-np.pi, np.pi, 1001)
    rhs = -(tx / ty) * np.cos(kx)
    ky = np.where(np.abs(rhs) <= 1, np.arccos(rhs), np.nan)
    ky[0] = np.nan
    ky[-1] = np.nan

    ax.plot(kx / np.pi, ky / np.pi, lw=4, clip_on=True, c="#4dc947")
    ax.plot(kx / np.pi, -ky / np.pi, lw=4, clip_on=True, c="#4dc947")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_xlabel(r"$k_x/\pi$")
    ax.set_ylabel(r"$k_y/\pi$", labelpad=-8)


def average_fermi_orientation(tx, ty, n=4000, norm=10, eps=1e-12):
    """
    Average orientation of the Fermi surface in the positive quadrant.
    Corresponds to Eq.(21), simplified to a single quadrant using symmetry.
    The returned vector is scaled by `norm` for plotting as an arrow.
    """
    kx = np.linspace(0, np.pi, n)
    rhs = -(tx / ty) * np.cos(kx)
    mask = np.abs(rhs) <= 1
    kx, ky = kx[mask], np.arccos(rhs[mask])

    vx, vy = tx * np.sin(kx), ty * np.sin(ky)
    v = np.vstack([vx, vy]).T
    v /= np.linalg.norm(v, axis=1)[:, None]

    sin_ky = np.sin(ky)
    keep = np.abs(sin_ky) > eps
    kx, v, sin_ky = kx[keep], v[keep], sin_ky[keep]

    dky_dkx = -(tx / ty) * np.sin(kx) / sin_ky
    ds_dkx = np.sqrt(1.0 + dky_dkx**2)

    dx = np.diff(kx)
    ds = 0.5 * (ds_dkx[:-1] + ds_dkx[1:]) * dx

    mean = (ds[:, None] * 0.5 * (v[:-1] + v[1:])).sum(0) / ds.sum()
    return mean * norm


## Figure

fig, axs = plt.subplots(1, 3, figsize=(7, 5), constrained_layout=True)
(ax1, ax2, ax3) = axs


# (a): Hopping t_x = t_y = 1 (Isotropic case)
ax1.text(0.02, 0.98, r"(a)", transform=ax1.transAxes, va="top")
im1 = ax1.imshow(data["critical_1"]["i_local"][:, :, 0, 0].T, "bwr", vmin=-0.02, vmax=0.02, origin="lower")
ax1.set_xlabel(r"$\ell_x$")
ax1.set_ylabel(r"$\ell_y$")
ax1.set_title(r"$t_x=%.f$" % data["critical_1"]["hamiltonian_params"]["tx"])
ax1.set_yticks([0, 10, 20])
ax1.set_xticks([0, 10])

v1 = average_fermi_orientation(
    data["critical_1"]["hamiltonian_params"]["tx"],
    data["critical_1"]["hamiltonian_params"]["ty"],
)
ax1.arrow(0, 0, v1[0], v1[1], length_includes_head=True, head_width=2, head_length=2, color="k", lw=2)

ax1_inset = add_inset_ax(ax1)
plot_fermi_surface(
    ax1_inset,
    data["critical_1"]["hamiltonian_params"]["tx"],
    data["critical_1"]["hamiltonian_params"]["ty"],
)


# (b): Hopping t_x = 1.1
ax2.text(0.02, 0.98, r"(b)", transform=ax2.transAxes, va="top")
im2 = ax2.imshow(data["critical_2"]["i_local"][:, :, 0, 0].T, "bwr", vmin=-0.02, vmax=0.02, origin="lower")
ax2.set_xlabel(r"$\ell_x$")
ax2.set_title(r"$t_x=%.f$" % data["critical_2"]["hamiltonian_params"]["tx"])
ax2.set_yticks([])

v2 = average_fermi_orientation(
    data["critical_2"]["hamiltonian_params"]["tx"],
    data["critical_2"]["hamiltonian_params"]["ty"],
)
ax2.arrow(0, 0, v2[0], v2[1], length_includes_head=True, head_width=2, head_length=2, color="k", lw=2)

ax2_inset = add_inset_ax(ax2)
plot_fermi_surface(
    ax2_inset,
    data["critical_2"]["hamiltonian_params"]["tx"],
    data["critical_2"]["hamiltonian_params"]["ty"],
)


# (c): Hopping t_x = 2.5
ax3.text(0.02, 0.98, r"(c)", transform=ax3.transAxes, va="top")
im3 = ax3.imshow(data["critical_3"]["i_local"][:, :, 0, 0].T, "bwr", vmin=-0.02, vmax=0.02, origin="lower")
ax3.set_xlabel(r"$\ell_x$")
ax3.set_title(r"$t_x=%.f$" % data["critical_3"]["hamiltonian_params"]["tx"])
ax3.set_yticks([])

v3 = average_fermi_orientation(
    data["critical_3"]["hamiltonian_params"]["tx"],
    data["critical_3"]["hamiltonian_params"]["ty"],
)
ax3.arrow(0, 0, v3[0], v3[1], length_includes_head=True, head_width=2, head_length=2, color="k", lw=2)

ax3_inset = add_inset_ax(ax3)
plot_fermi_surface(
    ax3_inset,
    data["critical_3"]["hamiltonian_params"]["tx"],
    data["critical_3"]["hamiltonian_params"]["ty"],
)


# Colorbar for (a,b,c)
cbar = fig.colorbar(im1, ax=[ax1, ax2, ax3], orientation="vertical", pad=0.02, fraction=0.025)
cbar.set_ticks([-0.02, 0, 0.02])
cbar.set_label(r"$i^{(\ell_x\ \ell_y)}_{(0\ 0)}$")

plt.savefig("fig8.pdf", bbox_inches="tight")
plt.show()
