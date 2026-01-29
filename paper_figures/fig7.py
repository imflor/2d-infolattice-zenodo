"""
==============================================================================================
Data for Fig. 7 (Section IV.A) — Localized and critical ground states of the 2D Anderson model
==============================================================================================

This module defines the Python data structure used to generate the results shown in
**Fig. 7 in Section IV.A, "Localized and critical ground states of the 2D Anderson model"**
of the accompanying paper.

Top-level structure
-------------------
data : dict[str, dict]
    Mapping from a descriptive dataset label (e.g. "localized_w40", "critical_w40", ...)
    to a dictionary with the following keys:

Keys for each dataset label
---------------------------
size : list[int]
    System size in number of lattice sites along each direction, given as `[Nx, Ny]`.

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
    The array has **four axes**:
    - axis 0, 1: subsystem scale indices
        (i.e. the two-dimensional scale of the rectangular subsystem along x and y).
    - axis 2, 3: subsystem position indices
        (i.e. the location of the rectangular subsystem on the Nx×Ny lattice).

Example usage
-------------
Information per multiscale (sum over subsystem positions, axes 2 and 3):
    data["localized_w40"]["i_local"].sum(axis=(2, 3)).T

Quasi-1D information per scale along x (sum over y-scale and x/y positions axes 1, 2 and 3):
    data["localized_w40"]["i_local"].sum(axis=(1, 2, 3))
"""

## Imports

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize


## Input data

data = np.load("fig7.npy", allow_pickle=True).item(0)


## Helpers

def f_linear(x, a, b):
    return a * x + b


def produce_data_and_exponential_fit(data, name, axis=0, fit_range=(1, 38)):
    """Produce quasi-1D information per scale along the given axis and return the data and exponential fit."""
    begin, end = fit_range
    quasi1d_data = data[name]["i_local"].sum(axis=(1 - axis, 2, 3))
    x_fit = np.arange(begin, end)
    y_fit = np.log(quasi1d_data[begin - 1 : end - 1])
    exponential_fit = scipy.optimize.curve_fit(f_linear, x_fit, y_fit)
    return quasi1d_data, exponential_fit


def produce_power_law_fit(arr, name, axis=0, fit_range=(2, 14)):
    """Produce normalized quasi-1D information per scale along the given axis and return the power law fit -2."""
    begin, end = fit_range
    quasi1d_data = arr[name]["i_local"].sum(axis=(1 - axis, 2, 3)) / arr[name]["i_local"].shape[0] ** 2
    x_fit = np.log(np.arange(begin, end))
    y_fit = np.log(quasi1d_data)[begin:end]
    f = lambda x, b: -2 * x + b
    power_law_fit = scipy.optimize.curve_fit(f, x_fit, y_fit)
    return power_law_fit


## Figure

fig, axs = plt.subplots(2, 2, figsize=(6, 6), constrained_layout=True)
(ax1, ax2), (ax3, ax4) = axs


# (a): Information per multiscale of the localized Anderson model ground state with disorder strength `W=10`
ax1.text(0.02, 0.98, r"(a)", transform=ax1.transAxes, va="top")
im1 = ax1.imshow(
    data["localized_w40"]["i_local"].sum(axis=(2, 3)).T,
    cmap="bwr",
    vmin=-1,
    vmax=1,
    origin="lower",
)
ax1.set_xlabel(r"$\ell_x$")
ax1.set_ylabel(r"$\ell_y$")
ax1.set_title(r"Localized $(W\neq0)$")


# (b): Information per multiscale of the clean critical Anderson model with disorder strength `W=0`
ax2.text(0.02, 0.98, r"(b)", transform=ax2.transAxes, va="top")
im2 = ax2.imshow(
    data["critical_w40"]["i_local"].sum(axis=(2, 3)).T,
    cmap="bwr",
    vmin=-1,
    vmax=1,
    origin="lower",
)
ax2.set_xlabel(r"$\ell_x$")
ax2.set_yticks([])
ax2.set_title(r"Critical $(W=0)$")


# Colorbar for (a,b)
cbar = fig.colorbar(im1, ax=[ax1, ax2], orientation="vertical", pad=0, fraction=0.025)
cbar.set_ticks([-1, 0, 1])
cbar.set_label(r"$I(\ell_x,\ell_y)$")


# (c): Quasi-1D information per scale of the localized Anderson model ground state with disorder strength `W=10`
ax = ax3
ax.text(0.88, 0.98, r"(c)", transform=ax.transAxes, va="top")
quasi1d_x_data, fit_x = produce_data_and_exponential_fit(data, "localized_w40", axis=0)
quasi1d_y_data, fit_y = produce_data_and_exponential_fit(data, "localized_w40", axis=1)
ax.semilogy(np.arange(1, 41), quasi1d_x_data, "-", lw=1.5, markersize=3, label=r"$I(\ell_x)$")
ax.semilogy(np.arange(1, 41), quasi1d_y_data, "-", lw=1.5, markersize=3, label=r"$I(\ell_y)$")
x = np.linspace(0.5, 40, 100)
ax.semilogy(
    x,
    np.exp(f_linear(x, fit_x[0][0], fit_x[0][1])),
    dashes=(3, 1),
    lw=1.5,
    label=r"Exp. fit of $I(\ell_x)$",
)
ax.semilogy(
    x,
    np.exp(f_linear(x, fit_y[0][0], fit_y[0][1])),
    dashes=(3, 1),
    lw=1.5,
    label=r"Exp. fit of $I(\ell_y)$",
)
ax.legend()
ax.set_xlim([0, 40])
ax.set_ylim([2e-10, 1e3])
ax.set_xlabel(r"$\ell_i$")
ax.set_ylabel(r"$I(\ell_i)$")


# (d): Quasi-1D information per scale (normalized) for the critical Anderson model for different system sizes
ax = ax4
ax.text(0.88, 0.98, r"(d)", transform=ax.transAxes, va="top")
for size in [10, 20, 30, 40]:
    quasi1d_normalized = (
        data[f"critical_w{size}"]["i_local"].sum(axis=(1, 2, 3)) / data[f"critical_w{size}"]["i_local"].shape[0] ** 2
    )
    ax.loglog(np.arange(size), quasi1d_normalized, lw=1.5, label=rf"${size}\times{size}$")
fit_critical = produce_power_law_fit(data, "critical_w40")
ax.loglog(x, np.exp(fit_critical[0][0]) * x ** (-2), dashes=(3, 1), lw=1.5, label=r"Fit $\propto\ell_x^{-2}$")
ax.legend()
ax.set_xlabel(r"$\ell_x$")
ax.set_xlim([0.5, 40])
ax.set_ylim([3e-5, 1])
ax.yaxis.tick_right()

plt.savefig("fig7.pdf", bbox_inches="tight")
plt.show()
