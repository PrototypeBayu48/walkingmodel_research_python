import numpy as np
import scipy as sc
from scipy.integrate import solve_ivp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import optimize
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
import pickle
import os

from walkingmodel_function import model_dyn_def, U0_array_func, step_dyn, plotfunc_state_full, model_num_linear_dyn_def, model_linear_dyn_def, jacobian_num, x0_fp_func

# Basin-Switching Control Recovery Basin Results

# Load Data
data_propcont = True

if data_propcont:
    file_name = "basin_walkdyn_propcont_copadj_1.pkl"
else:
    file_name = "basin_walkdyn_copadj_1.pkl"

with open(os.path.join(os.getcwd(), "simulation_data", file_name), "rb") as file:
    basin_res = pickle.load(file)

U0_array = basin_res['U0_array']

z_test = basin_res['z_test']
dz_test = basin_res['dz_test']
basin_x0_z = basin_res['basin_x0_z']
basin_U0_z = basin_res['basin_U0_z']

if data_propcont:
    z_c_err_test = basin_res['z_c_err_test']
    propcont_dz_idx = 7
    print(f"zdot = {dz_test[propcont_dz_idx] + U0_array[3,0]:.2f} m/s")

    basin_x0_z = np.swapaxes(basin_x0_z[propcont_dz_idx, :, :, :, :], 0, 1)
    basin_U0_z = np.swapaxes(basin_U0_z[propcont_dz_idx, :, :, :, :], 0, 1)

if basin_res['x_test'] != None:
    x_test = basin_res['x_test']
    dx_test = basin_res['dx_test']
    basin_x0_x = basin_res['basin_x0_x']
    basin_U0_x = basin_res['basin_U0_x']

# Settings
if basin_res['cop_params']['c2'] > 0.0:
    copquad = 1
else:
    copquad = 0

if basin_res['sys_setting']['propcont']:
    propcont = 1
else:
    propcont = 0

if basin_res['sys_setting']['copadj']:
    copadj = 1
else:
    copadj = 0

copadj = basin_res['sys_setting']['copadj']

# Constants
n = 4
g = basin_res['sys_const']['g']
l = basin_res['sys_const']['l']
omega_0 = np.sqrt(g/l)

k = basin_res['sys_const']['k']

const_T = basin_res['const_T']
n_steps = len(U0_array[0,:])
s_c = U0_array[0,1] - U0_array[0,0]
w_c = np.abs(U0_array[1,1] - U0_array[1,0])
z_c = (U0_array[1,1] + U0_array[1,0])/2

footl = basin_res['sys_const']['footl']
footw = basin_res['sys_const']['footw']

dt = basin_res['dt']

alpha = (basin_res['cop_params']['b1'])*(basin_res['const_T'])
beta = (basin_res['cop_params']['c2'])/(basin_res['cop_params']['b1']**2)

deltaz = len(z_test)

if basin_res['x_test'] != None:
    deltax = len(x_test)

# Variables
x0 = basin_res['x0']
copparlim = np.array([[footl, 0.0], [(4*footw)/(footl**2), 0.0]])
cop_params = basin_res['cop_params'].copy()
cop_params_test = basin_res['cop_params'].copy()

U0_array_fp = U0_array[:,:2]

# Functions
def ddU_x(t,x):
    return 0.0

def ddU_z_quad(t,x):
    return 2*cop_params_test['c2']

def ddU_z_const(t,x):
    return 0.0

def dot_U_func(t,x):
    dot_U = np.array([x[2], x[3], ddU_x(t,x), ddU_z_quad(t,x)])
    return dot_U

# Define model
if basin_res['sys_setting']['linear']:
    model_dyn = model_linear_dyn_def([g, l], ddU_x, ddU_z_quad)
    model_dyn_const = model_linear_dyn_def([g, l], ddU_x, ddU_z_const)
else:
    model_dyn = model_dyn_def([g, l], ddU_x, ddU_z_quad)
    model_dyn_const = model_dyn_def([g, l], ddU_x, ddU_z_const)

def simulate_twostep(x1):
    if copquad == 1:
        [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x1, dt, U0_array_fp, dot_U=dot_U_func, cop_params=cop_params_test, copparlim = copparlim, const_T=const_T, propcont=propcont, constoff=1, stepprint=0)
    else:
        [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn_const, x1, dt, U0_array_fp, const_T=const_T, propcont=propcont, constoff=1, stepprint=0)
    x2 = x0_full.step[:,1]
    return x2

# Data Processing
crit = 'viability'

testlen = 1
x_fp = x0_fp_func(np.array([0.25, 0.02, 0.0, 0.0]), simulate_twostep)

if data_propcont:
    def simulate_twostep_propcont(xz1):
        x1 = np.array([0.25, xz1[0], 0.0, xz1[1]])
        sys_par_propcont = (g, l, k, U0_array_fp[1,0] + xz1[2], 0.8*l)
        if copquad == 1:
            [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x1, dt, U0_array_fp, dot_U=dot_U_func, cop_params=cop_params, sys_par=sys_par_propcont, const_T=const_T, propcont=1, constoff=1, stepprint=0)
        else:
            [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn_const, x1, dt, U0_array_fp, sys_par=sys_par_propcont, const_T=const_T, propcont=1, constoff=1, stepprint=0)
        x2 = x0_full.step[:,1]
        return np.array([x2[1], x2[3], -(U0_array_fp[1,0] + xz1[2] - U0_full.res[1,1])])
    x_fp_propcont = x0_fp_func(np.array([0.02, 0.0, 0.0]), simulate_twostep_propcont)

w_max = 0.8*l
tol_conv = 1e-4

def dataprocess_func(n1, n2, x0_res, U0_res, crit):
    point_score = 0.0

    for p1 in range(testlen):
        leftright = 1
        trial_result = 1.0

        for m1 in range(n_steps):
            if m1 > 0:
                if (np.abs(U0_res[1,m1] - U0_res[1,m1-1]) > (w_max)) or (np.abs(U0_res[0,m1] - U0_res[0,m1-1]) > (w_max)):
                    trial_result = -1.0
                    break
                
                if leftright == 1:
                    if (U0_res[1,m1] >= U0_res[1,m1-1]) or (U0_res[0,m1] <= U0_res[0,m1-1]):
                        trial_result = -1.0
                        break
                else:
                    if (U0_res[1,m1] <= U0_res[1,m1-1]) or (U0_res[0,m1] <= U0_res[0,m1-1]):
                        trial_result = -1.0
                        break

            leftright = 0 if leftright == 1 else 1

        point_score += trial_result
    
    return n1, n2, (point_score / testlen)

print('Simulating Lateral Motion')
if data_propcont:
    basin_z = np.zeros((len(z_c_err_test), len(z_test)))
    results_z = Parallel(n_jobs=-1, verbose=5)(
        delayed(dataprocess_func)(n1, n2, basin_x0_z[n1,n2,:,:], basin_U0_z[n1,n2,:,:], crit)
        for n1 in range(len(z_c_err_test))
        for n2 in range(len(z_test))
    )
    for n1, n2, avg_val in results_z:
        basin_z[n1, n2] = avg_val
else:
    basin_z = np.zeros((len(dz_test), len(z_test)))
    results_z = Parallel(n_jobs=-1, verbose=5)(
        delayed(dataprocess_func)(n1, n2, basin_x0_z[n1,n2,:,:], basin_U0_z[n1,n2,:,:], crit)
        for n1 in range(len(dz_test))
        for n2 in range(len(z_test))
    )
    for n1, n2, avg_val in results_z:
        basin_z[n1, n2] = avg_val

print('Average Basin Value (z): ' + str(np.mean(basin_z)))

# Sum the amount of viable initial conditions
if crit == 'viability':
    viable_sum = np.sum(basin_z > 0.0)
    print('Total Viable Basin Initial Condition: ' + str(viable_sum))

# Define recovery basin boundaries
if data_propcont:
    def propcont_lim(zeta_test, beta_test):
        c1 = -alpha*beta_test*cop_params_test['b1']
        c2 = beta_test*(cop_params_test['b1']**2)
        b_z = ( (1)/(1 + np.exp(omega_0*const_T)) )*( (w_c) - (c1)*( (const_T) + ((1 - np.exp(omega_0*const_T))/omega_0) ) - (c2)*( (const_T**2) + ((2*const_T)/(omega_0)) + ((2*(1 - np.exp(omega_0*const_T)))/(omega_0**2)) ) )

        quadcop_term = ( (c2/(omega_0**2))*(2*np.exp(omega_0*const_T) - (omega_0*const_T)**2 - (2*omega_0*const_T) - 2) + (c1/omega_0)*(np.exp(omega_0*const_T) - (omega_0*const_T) - 1) )

        basin_boundary_ana = np.zeros((4,len(zeta_test)))
        propcont_boundary_coeff = np.zeros((4,4))
        basin_top_check = np.inf
        basin_bottom_check = -np.inf
        propcont_boundary_coeff[0,0] = np.exp(-omega_0*const_T)*( (k)/(k + 1) )
        propcont_boundary_coeff[1,0] = np.exp(-omega_0*const_T)*( (-b_z)/(k + 1) ) + np.exp(-omega_0*const_T)*quadcop_term
        propcont_boundary_coeff[2,0] = np.exp(-omega_0*const_T)*( (w_max - b_z)/(k + 1) ) + np.exp(-omega_0*const_T)*quadcop_term
        propcont_boundary_coeff[3,0] = np.exp(omega_0*const_T)*(k - propcont_boundary_coeff[0,0]*(k + 1))
        for n1 in np.arange(1,4):
            propcont_boundary_coeff[0,n1] = np.exp(-omega_0*const_T)
            propcont_boundary_coeff[1,n1] = (b_z*(propcont_boundary_coeff[0,n1-1] - 1) + propcont_boundary_coeff[1,n1-1] + quadcop_term*(-propcont_boundary_coeff[0,n1-1]*(k+1) + k))/propcont_boundary_coeff[3,n1-1]
            propcont_boundary_coeff[2,n1] = (b_z*(propcont_boundary_coeff[0,n1-1] - 1) + propcont_boundary_coeff[2,n1-1] + quadcop_term*(-propcont_boundary_coeff[0,n1-1]*(k+1) + k))/propcont_boundary_coeff[3,n1-1]
            propcont_boundary_coeff[3,n1] = np.exp(omega_0*const_T)*(k - propcont_boundary_coeff[0,n1]*(k + 1))

            bound_1 = -( ( zeta_test - propcont_boundary_coeff[1,n1] )/propcont_boundary_coeff[0,n1] ) + zeta_test
            bound_2 = -( ( zeta_test - propcont_boundary_coeff[2,n1] )/propcont_boundary_coeff[0,n1] ) + zeta_test

            basin_top = np.maximum(bound_1, bound_2)
            basin_bottom = np.minimum(bound_1, bound_2)

            if basin_top[0] < basin_top_check:
                basin_boundary_ana[2,:] = basin_top
                basin_top_check = basin_top[0].copy()
            if basin_bottom[0] > basin_bottom_check:
                basin_boundary_ana[3,:] = basin_bottom
                basin_bottom_check = basin_bottom[0].copy()

        bound_1_0 = -( ( zeta_test - propcont_boundary_coeff[1,0] )/propcont_boundary_coeff[0,0] ) + zeta_test
        bound_2_0 = -( ( zeta_test - propcont_boundary_coeff[2,0] )/propcont_boundary_coeff[0,0] ) + zeta_test
        basin_top_0 = np.maximum(bound_1_0, bound_2_0)
        basin_bottom_0 = np.minimum(bound_1_0, bound_2_0)
        basin_boundary_ana[0,:] = basin_top_0
        basin_boundary_ana[1,:] = basin_bottom_0

        basin_top_total = np.minimum(basin_boundary_ana[0,:], basin_boundary_ana[2,:])
        basin_bottom_total = np.maximum(basin_boundary_ana[1,:], basin_boundary_ana[3,:])
        basin_region = basin_top_total >= basin_bottom_total
        #basin_boundary_ana[:, ~basin_region] = np.nan
        for n1 in range(4):
            if n1 == 0 or n1 == 2:
                basin_boundary_ana[n1, ~(basin_region & (np.abs(basin_boundary_ana[n1,:] - basin_top_total) < 1e-9))] = np.nan
            else:
                basin_boundary_ana[n1, ~(basin_region & (np.abs(basin_boundary_ana[n1,:] - basin_bottom_total) < 1e-9))] = np.nan

        return basin_boundary_ana

    deltaz_dense = 10000
    z_test_dense = np.linspace(z_test[0], z_test[-1], deltaz_dense)
    zeta_test_dense = z_test_dense + (dz_test[propcont_dz_idx] + U0_array[3,0])/omega_0
    basin_boundary_ana = propcont_lim(zeta_test_dense, beta)
    if copadj == 'basin':
        basin_boundary_ana_maxbeta = propcont_lim(zeta_test_dense, copparlim[1,0])
        basin_boundary_ana_minbeta = propcont_lim(zeta_test_dense, copparlim[1,1])

        nombeta_top = np.fmin(basin_boundary_ana[0,:], basin_boundary_ana[2,:])
        nombeta_bottom = np.fmax(basin_boundary_ana[1,:], basin_boundary_ana[3,:])

        for bounds in (basin_boundary_ana_maxbeta, basin_boundary_ana_minbeta):
            bounds[(bounds >= nombeta_bottom) & (bounds <= nombeta_top)] = np.nan
else:
    b_z = ( (1)/(1 + np.exp(omega_0*const_T)) )*( (w_c) - (cop_params_test['c1'])*( (const_T) + ((1 - np.exp(omega_0*const_T))/omega_0) ) - cop_params_test['c2']*( (const_T**2) + ((2*const_T)/(omega_0)) + ((2*(1 - np.exp(omega_0*const_T)))/(omega_0**2)) ) )
    def dz_lim_low_func(z):
        return omega_0*((np.exp(-omega_0*const_T)*(-b_z + ( (cop_params_test['c2']/(omega_0**2))*(2*np.exp(omega_0*const_T) - (omega_0*const_T)**2 - (2*omega_0*const_T) - 2) + (cop_params_test['c1']/omega_0)*(np.exp(omega_0*const_T) - (omega_0*const_T) - 1) ))) - z)
    def dz_lim_high_func(z):
        return omega_0*((np.exp(-omega_0*const_T)*(w_max - b_z + ( (cop_params_test['c2']/(omega_0**2))*(2*np.exp(omega_0*const_T) - (omega_0*const_T)**2 - (2*omega_0*const_T) - 2) + (cop_params_test['c1']/omega_0)*(np.exp(omega_0*const_T) - (omega_0*const_T) - 1) ))) - z)
    if copadj == 'basin':
        c2_max = ((4*footw)/(footl**2))*((footl/const_T)**2)
        c1_max = -footl*((4*footw)/(footl**2))*(footl/const_T)
        b_z_min = ( (1)/(1 + np.exp(omega_0*const_T)) )*( (w_c) - (c1_max)*( (const_T) + ((1 - np.exp(omega_0*const_T))/omega_0) ) - (c2_max)*( (const_T**2) + ((2*const_T)/(omega_0)) + ((2*(1 - np.exp(omega_0*const_T)))/(omega_0**2)) ) )
        b_z_max = ( (1)/(1 + np.exp(omega_0*const_T)) )*(w_c)
        def dz_lim_min_func(z):
            return omega_0*((np.exp(-omega_0*const_T)*(-b_z_min + ( (c2_max/(omega_0**2))*(2*np.exp(omega_0*const_T) - (omega_0*const_T)**2 - (2*omega_0*const_T) - 2) + (c1_max/omega_0)*(np.exp(omega_0*const_T) - (omega_0*const_T) - 1) ))) - z)
        def dz_lim_max_func(z):
            return omega_0*((np.exp(-omega_0*const_T)*(w_max - b_z_max)) - z)
    basin_boundary_ana = np.zeros((2,deltaz))
    basin_boundary_ana[0,:] = dz_lim_low_func(z_test)
    basin_boundary_ana[1,:] = dz_lim_high_func(z_test)

# Display results
eigvec_color = ['darkgoldenrod','lawngreen']
eig_color_count = 0
if crit == 'viability':
    fig_type = 'Recovery Basin'
    fig_cmap = 'RdBu'
    fig_norm = 'linear'
elif crit == 'convergence':
    fig_type = 'Convergence Basin'
    fig_cmap_bounds = np.arange(0.0, 13.0)
    fig_cmap = plt.get_cmap('bwr', len(fig_cmap_bounds) - 1)
    fig_norm = matplotlib.colors.BoundaryNorm(boundaries=fig_cmap_bounds, ncolors=fig_cmap.N)

if data_propcont:
    xlim = (z_test[0] + (dz_test[propcont_dz_idx] + U0_array[3,0])/omega_0, z_test[-1] + (dz_test[propcont_dz_idx] + U0_array[3,0])/omega_0)
    ylim = (z_c_err_test[0], z_c_err_test[-1])
    xlabel = r'$\zeta$ (m)'
    ylabel = r'$\zeta - z_{c}$ (m)'
else:
    xlim = (z_test[0], z_test[-1])
    ylim = (dz_test[0] + U0_array[3,0], dz_test[-1] + U0_array[3,0])
    xlabel = r'$z$ (m)'
    ylabel = r'$\dot{z}$ (m/s)'

fig, axes = plt.subplots(figsize=(8,8))
pos_z = axes.imshow(basin_z, cmap=fig_cmap, norm=fig_norm, 
                    origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect='auto')
axes.tick_params(axis='both', which='major', labelsize=12)

if not data_propcont:
    axes.plot(x_fp[1], x_fp[3] + U0_array[3,0], 'o', mfc='w', mec='k', mew=1.5, markersize=10, label='Fixed Point', zorder=10)
    axes.plot(z_test, basin_boundary_ana[0,:], 
            color='yellow', path_effects=[pe.Stroke(linewidth=5.0, foreground='black'), pe.Normal()], lw=3.0, label=r'Nominal $\beta$ Analytical Boundary')
    axes.plot(z_test, basin_boundary_ana[1,:], 
            color='yellow', path_effects=[pe.Stroke(linewidth=5.0, foreground='black'), pe.Normal()], lw=3.0)
    if copadj == 'basin':
        axes.plot(z_test, dz_lim_min_func(z_test), 
                color='cyan', path_effects=[pe.Stroke(linewidth=5.0, foreground='black'), pe.Normal()], lw=3.0, label=r'Max $\beta$ Analytical Boundary')
        axes.plot(z_test, dz_lim_max_func(z_test), 
                color='lime', path_effects=[pe.Stroke(linewidth=5.0, foreground='black'), pe.Normal()], lw=3.0, label=r'Min $\beta$ Analytical Boundary')
else:
    axes.plot(x_fp_propcont[0] + (x_fp_propcont[1] + U0_array[3,0])/omega_0, x_fp_propcont[2] - ( x_fp_propcont[0] + (x_fp_propcont[1] + U0_array[3,0])/omega_0 ), 'o', mfc='w', mec='k', mew=1.5, markersize=10, label='Fixed Point', zorder=10)
    for n1 in range(4):
        if n1 == 0:
            axes.plot(zeta_test_dense, basin_boundary_ana[n1,:], 
                    color='yellow', lw=3.0, zorder=2, label=r'Nominal $\beta$ Analytical Boundary')
        else:
            axes.plot(zeta_test_dense, basin_boundary_ana[n1,:], 
                    color='yellow', lw=3.0, zorder=2)
    if copadj == 'basin':
        for n1 in range(4):
            if n1 == 0:
                axes.plot(zeta_test_dense, basin_boundary_ana_maxbeta[n1,:], 
                        color='cyan', lw=3.0, zorder=1, label=r'Max $\beta$ Analytical Boundary')
                axes.plot(zeta_test_dense, basin_boundary_ana_minbeta[n1,:], 
                        color='lime', lw=3.0, zorder=1, label=r'Min $\beta$ Analytical Boundary')
            else:
                axes.plot(zeta_test_dense, basin_boundary_ana_maxbeta[n1,:], 
                        color='cyan', lw=3.0, zorder=1)
                axes.plot(zeta_test_dense, basin_boundary_ana_minbeta[n1,:], 
                        color='lime', lw=3.0, zorder=1)

axes.set_xlim(xlim[0], xlim[1])
axes.set_ylim(ylim[0], ylim[1])
axes.set_xlabel(xlabel, fontsize=14)
axes.set_ylabel(ylabel, fontsize=14)

if crit == 'viability':
    fig.legend(loc='upper right', fontsize='large')
elif crit == 'convergence':
    fig.legend(loc='upper right', fontsize='large', ncol=3, bbox_to_anchor=(0.95, 1.0))
    cbar = fig.colorbar(pos_z, ax=axes, ticks=fig_cmap_bounds)
    cbar.set_label(rf'Steps to Convergence ($\epsilon = {tol_conv:.2g}$)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
plt.show()



