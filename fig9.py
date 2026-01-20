trivial_color = 'r'
topo_color = '#4d5c80'

fig = plt.figure(figsize=(8, 8.2))
gs = matplotlib.gridspec.GridSpec(2, 4, width_ratios=[1, 1, 0.05, 0.04], hspace=0.3, wspace=0.07)

i = 0
ax2 = plt.subplot(gs[0, i])
im2 = ax2.imshow(IXY_trivial, soft_bwr, vmin=-.25, vmax=.25, origin='lower')
ax2.set_xlabel(r'$\ell_x$', fontsize=20)
ax2.set_title(r'Trivial ($C=0$)')
ax2.set_ylabel(r'$\ell_y$', fontsize=20)

i = 1
ax1 = plt.subplot(gs[0, i])
im1 = ax1.imshow(IXY_topological, soft_bwr, vmin=-.25, vmax=.25, origin='lower')
ax1.set_xlabel(r'$\ell_x$', fontsize=20)
ax1.set_title(r'(b)\ Topological ($C=1$)')
ax1.set_yticks([])

cbar_ax = plt.subplot(gs[0, 3])  # Allocate space for the colorbar
cbar = plt.colorbar(im1, cax=cbar_ax, orientation='vertical')  # Add colorbar for the first row's plots
cbar.set_ticks([-.2, 0, .2])
#cbar.set_ticklabels([r'$-\frac{1}{4}$', r'$0$', r'$\frac{1}{4}$'])
cbar.ax.yaxis.set_ticks_position('left')
cbar.ax.yaxis.set_label_position('left')
cbar.ax.yaxis.labelpad = -50  # Adjust this value to control the distance
cbar.set_label(r"$I(\ell_x,\ell_y)$")
cbar.ax.tick_params(axis='y', rotation=90)

i = 0
ax = plt.subplot(gs[1, i])
opacities = np.linspace(0.1, .6, len(margins))
opacities[-1] = 1
for i, ixy_bulk in enumerate(ixys_bulk):
    vals = ixy_bulk.sum(axis=0) * (lat.n / ixy_bulk.sum())
    idx = np.where(vals>0)[0]
    plt_tp = plt.semilogy(np.arange(1, lat.nx+1)[idx], vals[idx], '.-',  lw=1.5, markersize=0, alpha=opacities[i],
                 c=topo_color, label=r'$C=1$: $I(\ell_x)\big|_{\mathcal{C}_\mathrm{bulk}}$' if (i+1==len(margins)) else '')
plt_tr = plt.semilogy(np.arange(1, lat.nx+1), IXY_trivial.sum(axis=0), '-', lw=1.5, markersize=0, c=trivial_color, label=r'$C=0$: $I(\ell_x)$')
p1 = [.72, .25]
p2 = [.52, .05]
positions = [(29, 1e-8), (26, 1e-9), (22, 4e-9), (18.5, 1.5e-8), (16, .5e-7), (10, .5e-6)]
xis = margins
for i, pos in enumerate(positions):
    plt.text(pos[0], pos[1], (r'' if (i<len(margins)-1) else r'$\zeta=$')+f'{xis[i]}', ha='center', va='center', fontsize=14, alpha=0.5+opacities[i]/2, c=topo_color)
# plot_gradient_curve(p1, p2, ax, color=topo_color)
# ax.text(.77, .26, 'Excluding subsets\nnear the edge', transform=ax.transAxes, color=topo_color, ha='center', va='bottom', fontsize=12)
plt.legend([plt_tr[0], plt_tp[0]],[plt_tr[0].get_label(), plt_tp[0].get_label()], loc=0)
ax.set_xlabel(r'$\ell_x$', fontsize=20)
ax.set_ylabel(r'$I(\ell_x)$')
plt.ylim([3e-10, 50*10**2])
plt.xlim([1, 29.9])

i = 1
ax = plt.subplot(gs[1, i])
ax.loglog(x, np.exp(fit0[0][0])*x**(-2), '-.', lw=1.5, markersize=3, c=topo_color, label=r'$\propto\ell_x^{-%d}$' % (2))
ax.loglog(np.arange(1, lat.nx+1), ixy_bulk.sum(axis=0)+ixy_edge.sum(axis=0), '-', dashes=(1,.5), c=topo_color,  lw=1.5, markersize=3, label=r'$I(\ell_x)$')
ax.loglog(np.arange(1, lat.nx+1), ixy_edge.sum(axis=0), '-', c=topo_color,  lw=1.5, markersize=3, label=r'$I(\ell_x)\big|_\mathrm{all\ edges}$')
ax.set_xlabel(r'$\ell_x$', fontsize=20)
plt.xlim([1, lat.n_sites.max()])
plt.legend(loc=0)
ax.yaxis.tick_right()

fig.text(0.15, .90, r'(a)', fontsize=21, ha='center', va='center')
#fig.text(0.5, .905, r'(b)', fontsize=21, ha='center', va='center')
fig.text(0.15, .47, r'(c)', fontsize=21, ha='center', va='center')
fig.text(0.51, .47, r'(d)', fontsize=21, ha='center', va='center')

plt.savefig('figures/fig5_new.pdf')
plt.show()