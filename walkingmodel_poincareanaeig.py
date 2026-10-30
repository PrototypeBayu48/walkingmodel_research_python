import numpy as np
import matplotlib.pyplot as plt
import os

# Define Case
#case = 'openloop'
#case = 'constoff'
case = 'propcont'

# Define range for (omega_0)*(T)
deltaomega0T_array = 0.01
omega0T_array = np.arange(0.0, 2.0 + deltaomega0T_array, deltaomega0T_array)

if case == 'propcont':
    # Define reference constants
    omega0T_ref = 1.5
    k_ref = 0.1

    # Define range for k
    deltak = 0.01
    k_array = np.arange(0.0, 1.0 + deltak, deltak)

# Display results
fig_save = False

eigval_color = ['C0', 'C1', 'C2']

fig, axes = plt.subplots(figsize=(16,9))

if case == 'constoff':
    axes.plot(omega0T_array, np.abs( -np.exp(-omega0T_array) ), color=eigval_color[0], zorder=1, label=r'$\lambda_{\mathrm{c},1}$')
    axes.plot(omega0T_array, np.abs( 0.0*np.ones(len(omega0T_array)) ), color=eigval_color[1], zorder=2, label=r'$\lambda_{\mathrm{c},2}$')
elif case == 'propcont':
    axes.plot(omega0T_array, np.abs( k_ref*(np.exp(omega0T_array) - 1) - 1 ), color=eigval_color[0], zorder=1, label=r'$\lambda_{\mathrm{p},1}$')
    axes.plot(omega0T_array, np.abs( -np.exp(-omega0T_array) ), color=eigval_color[1], zorder=2, label=r'$\lambda_{\mathrm{p},2}$')
    axes.plot(omega0T_array, np.abs( 0.0*np.ones(len(omega0T_array)) ), color=eigval_color[2], zorder=3, label=r'$\lambda_{\mathrm{p},3}$')
else:
    axes.plot(omega0T_array, np.abs( -np.exp(omega0T_array) ), color=eigval_color[0], zorder=1, label=r'$\lambda_{\mathrm{o},1}$')
    axes.plot(omega0T_array, np.abs( -np.exp(-omega0T_array) ), color=eigval_color[1], zorder=2, label=r'$\lambda_{\mathrm{o},1}$')

axes.axhline(1.0, c='k', ls='--', lw=2.0, zorder=0, label=r'Stability Limit')

axes.set_xlim(omega0T_array[0], omega0T_array[-1])
axes.set_ylim(-0.01, 2.0)
axes.set_xlabel(r'$\omega_{0}T$', fontsize=20)
axes.set_ylabel(r'|$\lambda$|', fontsize=20)
axes.tick_params(axis='both', labelsize=20)
fig.legend(loc='upper center', fontsize=20, ncol=2)

if fig_save:
    if case == 'constoff':
        fig_name = f"fig_eigval_constoff.pdf"
    elif case == 'propcont':
        fig_name = f"fig_eigval_propcont_omega0T.pdf"
    else:
        fig_name = f"fig_eigval_openloop.pdf"
    plt.savefig(os.path.join("simulation_fig", fig_name), bbox_inches='tight')

plt.show()

if case == 'propcont':
    fig, axes = plt.subplots(figsize=(16,9))
    axes.plot(k_array, np.abs( k_array*(np.exp(omega0T_ref) - 1) - 1 ), color=eigval_color[0], zorder=1, label=r'$\lambda_{\mathrm{p},1}$')
    axes.plot(k_array, np.abs( -np.exp(-omega0T_ref)*np.ones(len(k_array)) ), color=eigval_color[1], zorder=2, label=r'$\lambda_{\mathrm{p},2}$')
    axes.plot(k_array, np.abs( 0.0*np.ones(len(k_array)) ), color=eigval_color[2], zorder=3, label=r'$\lambda_{\mathrm{p},3}$')

    axes.axhline(1.0, c='k', ls='--', lw=2.0, zorder=0, label=r'Stability Limit')

    axes.set_xlim(k_array[0], k_array[-1])
    axes.set_ylim(-0.01, 1.5)
    axes.set_xlabel(r'$k_{p}$', fontsize=20)
    axes.set_ylabel(r'|$\lambda$|', fontsize=20)
    axes.tick_params(axis='both', labelsize=20)
    fig.legend(loc='upper center', fontsize=20, ncol=2)

    if fig_save:
        fig_name = f"fig_eigval_propcont_k.pdf"
        plt.savefig(os.path.join("simulation_fig", fig_name), bbox_inches='tight')

    plt.show()



