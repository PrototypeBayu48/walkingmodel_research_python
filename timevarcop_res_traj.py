import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import optimize
import os

from walkingmodel_function import model_dyn_def, U0_array_func, step_dyn, plotfunc_state_full, model_num_linear_dyn_def, model_linear_dyn_def, step_analysis_func, plotanim_func

# Time-Variant CoP Trajectory Testing Results

linear_dyn = 1
propcont = 1

x0_fp = 1
footgeom = 1

def x0_fp_func(x0, model_dyn, U0_array, dot_U_func, cop_params):
    def simulate_twostep(x1):
        [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x1, dt, U0_array, dot_U=dot_U_func, cop_params=cop_params, const_T=const_T, propcont=propcont, constoff=1, stepprint=0)
        x2 = x0_full.step[:,1]
        return x2
    constoff_fp_func = lambda x: x - simulate_twostep(x)
    constoff_fp_full = optimize.root(constoff_fp_func, x0)
    x0 = constoff_fp_full.x
    return x0

def coptorque_func(U_step, xstep, xdot_func, anklept):
    U_torque = np.zeros((2,len(U_step[0,:])))
    xdot = xdot_func(0.0, xstep)
    for n1 in range(len(U_step[0,:])):
        if linear_dyn == 0:
            R_y = mass*( -( ((xstep[0,n1]*xstep[2,n1] + xstep[1,n1]*xstep[3,n1])**2)/(l**2 - (xstep[0,n1]**2 + xstep[1,n1]**2)) 
                            + (xstep[2,n1]**2 + xstep[3,n1]**2 + xstep[0,n1]*xdot[2,n1] + xstep[1,n1]*xdot[3,n1]) )/np.sqrt(l**2 - (xstep[0,n1]**2 + xstep[1,n1]**2)) + g )
        else:
            R_y = mass*g
        U_torque[0,n1] = R_y*(U_step[1,n1] - anklept[1])
        U_torque[1,n1] = -R_y*(U_step[0,n1] - anklept[0])
    return U_torque

# Constants
n = 4
g = 9.81
l = 1.09
omega_0 = np.sqrt(g/l)

mass = 80

const_T = 0.5
n_steps = 14
s_c = 0.75
w_c = 0.1
z_c = 0.0

# Define CoP Variables
footl = 0.30
footw = 0.12

copparlim = np.array([[footl, 0.0], [(4*footw)/(footl**2), 0.0]])

if footgeom == 1:
    alpha = copparlim[0,0]
    beta = copparlim[1,0]
else:
    alpha = 0.8*copparlim[0,0]
    beta = copparlim[1,0]/3

b1 = alpha/const_T
c1 = -alpha*beta*b1
c2 = beta*(b1**2)

cop_params = {'b1': b1, 'c1': c1, 'c2': c2}

print('U_x Coefficient:')
print(np.array([cop_params['b1']]))
print('U_z Coefficients:')
print(np.array([cop_params['c1'], cop_params['c2']]))

# Define CoP functions
def cop_func_x(t):
    return cop_params['b1']*(t)

def cop_func_z(t):
    return cop_params['c2']*(t**2) + cop_params['c1']*(t)

def ddU_x(t,x):
    return 0.0

def ddU_z_quad(t,x):
    return 2*cop_params['c2']

def ddU_z_const(t,x):
    return 0.0

def dot_U_func(t,x):
    dot_U = np.array([x[2], x[3], ddU_x(t,x), ddU_z_quad(t,x)])
    return dot_U

# Define model
if linear_dyn == 1:
    model_dyn = model_linear_dyn_def([g, l], ddU_x, ddU_z_quad)
    model_dyn_const = model_linear_dyn_def([g, l], ddU_x, ddU_z_const)
else:
    model_dyn = model_dyn_def([g, l], ddU_x, ddU_z_quad)
    model_dyn_const = model_dyn_def([g, l], ddU_x, ddU_z_const)

# Simulation
dt = 0.01
t_span = (0.0, const_T)
t_eval = np.arange(t_span[0], t_span[1] + dt, dt)

U0_array = U0_array_func(n_steps, s_c, w_c, z_c, dU_x_0=cop_params['b1'], dU_z_0=cop_params['c1'])
U0_array_const = U0_array_func(n_steps, s_c, w_c, z_c)

# Define x0
if x0_fp == 1:
    U0_array_fp = U0_array[:,:2]
    U0_array_const_fp = U0_array_const[:,:2]

    x0 = x0_fp_func(np.array([0.25, 0.02, 0.0, 0.0]), model_dyn, U0_array_fp, dot_U_func, cop_params)
    x0_const = x0_fp_func(np.array([0.25, 0.02, 0.0, 0.0]), model_dyn_const, U0_array_const_fp, False, None)
    print('Quadratic CoP:')
    print(x0)
    print('Constant CoP:')
    print(x0_const)
else:
    x0 = np.array([0.25, 0.02, 0.0, 0.0])
    x0_const = np.array([0.25, 0.02, 0.0, 0.0])

x0 = x0 + np.array([0.0, 0.01, 0.0, 0.0])
x0_const = x0_const + np.array([0.0, 0.01, 0.0, 0.0])

[x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x0, dt, U0_array, dot_U=dot_U_func, cop_params=cop_params, copparlim = copparlim, const_T=const_T, propcont=propcont, constoff=1, stepprint=0)

t_step = x_full.t
x_step_3D = x_full.inertial_3D
x0_step_3D = x0_full.inertial_3D
U_array = U0_full.inertial
cop_params_res = U0_full.params_res
U0_noise_res = U0_full.noise_res
deltat_con = U0_full.deltat

xcom = x_full.inertial[:2,:] + (x_full.inertial[2:,:]/omega_0)
xcom_x0 = x0_full.inertial[:2,:] + (x0_full.inertial[2:,:]/omega_0)

alphabeta_res = np.zeros((2,n_steps))
alphabeta_res[0,:] = cop_params_res[0,:]*(T_array)
alphabeta_res[1,:] = cop_params_res[2,:]/(cop_params_res[0,:]**2)
print('Alpha and Beta Results:')
print(alphabeta_res)

step_ana_res = step_analysis_func(U0_full.res)

U_torque_res = coptorque_func(U0_full.step, x_full.step, model_dyn, np.array([copparlim[0,0]/4, 0.0]))

KE_res = (1/2)*(mass)*(np.linalg.norm(x_step_3D[3:,:], axis=0)**2)
KE0_res = (1/2)*(mass)*(np.linalg.norm(x0_step_3D[3:,:], axis=0)**2)
PE_res = (mass)*(g)*(x_step_3D[1,:])
PE0_res = (mass)*(g)*(x0_step_3D[1,:])

print('Average Control Frequency:')
print(str(1/np.mean(deltat_con)) + ' Hz')

print('Final Step Length:')
print(step_ana_res[0,-1])
print('Final Step Width:')
print(step_ana_res[1,-1])

cop_params = {'b1': b1, 'c1': c1, 'c2': c2}

[x_full_ref, x0_full_ref, U0_full_ref, T_array_ref] = step_dyn(model_dyn_const, x0_const, dt, U0_array_const, const_T=const_T, propcont=propcont, constoff=1, stepprint=0)

t_step_ref = x_full_ref.t
x_step_3D_ref = x_full_ref.inertial_3D
x0_step_3D_ref = x0_full_ref.inertial_3D
U_array_ref = U0_full_ref.inertial
deltat_con_ref = U0_full_ref.deltat

xcom_ref = x_full_ref.inertial[:2,:] + (x_full_ref.inertial[2:,:]/omega_0)
xcom_x0_ref = x0_full_ref.inertial[:2,:] + (x0_full_ref.inertial[2:,:]/omega_0)

step_ana_ref = step_analysis_func(U0_full_ref.res)

U_torque_ref = coptorque_func(U0_full_ref.step, x_full_ref.step, model_dyn, np.array([copparlim[0,0]/4, 0.0]))

KE_ref = (1/2)*(mass)*(np.linalg.norm(x_step_3D_ref[3:,:], axis=0)**2)
KE0_ref = (1/2)*(mass)*(np.linalg.norm(x0_step_3D_ref[3:,:], axis=0)**2)
PE_ref = (mass)*(g)*(x_step_3D_ref[1,:])
PE0_ref = (mass)*(g)*(x0_step_3D_ref[1,:])

print('Average Reference Control Frequency:')
print(str(1/np.mean(deltat_con_ref)) + ' Hz')

print('Final Reference Step Length:')
print(step_ana_ref[0,-1])
print('Final Reference Step Width:')
print(step_ana_ref[1,-1])

# Display results
fig_inertial_hor = True
fig_state_zoom = True
fig_anim = False

fig_save = False

fig, axes = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True, figsize=(16,14))

axes[0].plot(x_step_3D[2,:], x_step_3D[0,:], 'C0-o', zorder=1, label='CoM Trajectory (Time-Varying CoP)')
axes[0].plot(xcom[1,:], xcom[0,:], 'C0--', zorder=1, label='XcoM Trajectory (Time-Varying CoP)')
axes[0].plot(U_array[1,:], U_array[0,:], 'c:x', zorder=0, label='Time-Varying CoP Trajectory')
axes[1].plot(x_step_3D_ref[2,:], x_step_3D_ref[0,:], 'C1-o', zorder=1, label='CoM Trajectory (Constant CoP)')
axes[1].plot(xcom_ref[1,:], xcom_ref[0,:], 'C1--', zorder=1, label='XcoM Trajectory (Constant CoP)')
axes[1].plot(U_array_ref[1,:], U_array_ref[0,:], 'y:x', zorder=0, label='Constant CoP Trajectory')

axes[0].plot(x_step_3D[2,0], x_step_3D[0,0], 'go', zorder=2)
axes[1].plot(x_step_3D_ref[2,0], x_step_3D_ref[0,0], 'go', zorder=2, label='CoM Trajectory Start')
axes[0].plot(x_step_3D[2,-1], x_step_3D[0,-1], 'ro', zorder=2)
axes[1].plot(x_step_3D_ref[2,-1], x_step_3D_ref[0,-1], 'ro', zorder=2, label='CoM Trajectory End')
axes[0].plot(x0_step_3D[2,1:], x0_step_3D[0,1:], 'm+', zorder=2)
axes[1].plot(x0_step_3D_ref[2,1:], x0_step_3D_ref[0,1:], 'm+', zorder=2, label='CoM Step Transition')
axes[0].plot(xcom_x0[1,1:], xcom_x0[0,1:], 'C4+', zorder=2)
axes[1].plot(xcom_x0_ref[1,1:], xcom_x0_ref[0,1:], 'C4+', zorder=2, label='XcoM Step Transition')
axes[0].plot(U_array[1,0], U_array[0,0], 'gX', zorder=2)
axes[1].plot(U_array_ref[1,0], U_array_ref[0,0], 'gX', zorder=2, label='CoP Start')
axes[0].plot(U_array[1,-1], U_array[0,-1], 'rX', zorder=2)
axes[1].plot(U_array_ref[1,-1], U_array_ref[0,-1], 'rX', zorder=2, label='CoP End')

if propcont == 1:
    axes[0].axvline(z_c, ls='--', color='k', zorder=0)
    axes[1].axvline(z_c, ls='--', color='k', zorder=0, label='Desired Lateral Position')

axes[0].set_ylabel('X (m)', fontsize=16)
axes[0].set_xlabel('Z (m)', fontsize=16)
axes[1].set_xlabel('Z (m)', fontsize=16)

axes[0].tick_params(axis='both', labelsize=16)
axes[1].tick_params(axis='both', labelsize=16)

fig.legend(fontsize=16, ncol=3, loc='upper center')

if fig_save:
    if propcont == 1:
        fig_name = f"fig_trajsim_inertial_propcont.pdf"
    else:
        fig_name = f"fig_trajsim_inertial.pdf"
    plt.savefig(os.path.join("simulation_fig", fig_name), bbox_inches='tight')

plt.show()

state_step_labels = [r'x (m)', r'z (m)', r'$\dot{x}$ (m/s)', r'$\dot{z}$ (m/s)']
fig, axes = plt.subplots(ncols=1, nrows=4, sharex=True, figsize=(16,9))
for n1 in range(4):
    if n1 == 0:
        axes[n1].plot(t_step, x_full.step[n1,:] + U0_full.step[n1,:], 'C0', label='Time-Varying CoP')
        axes[n1].plot(t_step, x_full_ref.step[n1,:] + U0_full_ref.step[n1,:], 'C1', label='Constant CoP')
        axes[n1].plot(const_T*np.arange(n_steps)[1:], x0_full.step[n1,1:], 'm+', label='Step Transition')
        axes[n1].plot(const_T*np.arange(n_steps)[1:], x0_full_ref.step[n1,1:], 'm+')
    elif n1 > 1:
        axes[n1].plot(t_step, x_full.step[n1,:] + U0_full.step[n1,:], 'C0')
        axes[n1].plot(t_step, x_full_ref.step[n1,:] + U0_full_ref.step[n1,:], 'C1')
        axes[n1].plot(const_T*np.arange(n_steps)[1:], x0_full.step[n1,1:] + U0_full.res[n1,1:], 'm+')
        axes[n1].plot(const_T*np.arange(n_steps)[1:], x0_full_ref.step[n1,1:] + U0_full_ref.res[n1,1:], 'm+')
    else:
        axes[n1].plot(t_step, x_full.step[n1,:] + U0_full.step[n1,:], 'C0')
        axes[n1].plot(t_step, x_full_ref.step[n1,:] + U0_full_ref.step[n1,:], 'C1')
        axes[n1].plot(const_T*np.arange(n_steps)[1:], x0_full.step[n1,1:], 'm+')
        axes[n1].plot(const_T*np.arange(n_steps)[1:], x0_full_ref.step[n1,1:], 'm+')
    axes[n1].set_ylabel(state_step_labels[n1], fontsize=16)
    axes[n1].tick_params(axis='both', labelsize=16)
axes[3].set_xlabel('Time (s)', fontsize=16)

fig.legend(fontsize=16, loc='upper center')

if fig_save:
    if propcont == 1:
        fig_name = f"fig_trajsim_step_propcont.pdf"
    else:
        fig_name = f"fig_trajsim_step.pdf"
    plt.savefig(os.path.join("simulation_fig", fig_name), bbox_inches='tight')

plt.show()

if fig_inertial_hor:
    fig, axes = plt.subplots(ncols=1, nrows=2, sharex=True, sharey=True, figsize=(16,10))

    axes[0].plot(x_step_3D[0,:], x_step_3D[2,:], 'C0-o', zorder=1, label='CoM Trajectory (Time-Varying CoP)')
    axes[0].plot(xcom[0,:], xcom[1,:], 'C0--', zorder=1, label='XcoM Trajectory (Time-Varying CoP)')
    axes[0].plot(U_array[0,:], U_array[1,:], 'c:x', zorder=0, label='Time-Varying CoP Trajectory')
    axes[1].plot(x_step_3D_ref[0,:], x_step_3D_ref[2,:], 'C1-o', zorder=1, label='CoM Trajectory (Constant CoP)')
    axes[1].plot(xcom_ref[0,:], xcom_ref[1,:], 'C1--', zorder=1, label='XcoM Trajectory (Constant CoP)')
    axes[1].plot(U_array_ref[0,:], U_array_ref[1,:], 'y:x', zorder=0, label='Constant CoP Trajectory')

    axes[0].plot(x_step_3D[0,0], x_step_3D[2,0], 'go', zorder=2)
    axes[1].plot(x_step_3D_ref[0,0], x_step_3D_ref[2,0], 'go', zorder=2, label='CoM Trajectory Start')
    axes[0].plot(x_step_3D[0,-1], x_step_3D[2,-1], 'ro', zorder=2)
    axes[1].plot(x_step_3D_ref[0,-1], x_step_3D_ref[2,-1], 'ro', zorder=2, label='CoM Trajectory End')
    axes[0].plot(x0_step_3D[0,1:], x0_step_3D[2,1:], 'm+', zorder=2)
    axes[1].plot(x0_step_3D_ref[0,1:], x0_step_3D_ref[2,1:], 'm+', zorder=2, label='CoM Step Transition')
    axes[0].plot(xcom_x0[0,1:], xcom_x0[1,1:], 'C4+', zorder=2)
    axes[1].plot(xcom_x0_ref[0,1:], xcom_x0_ref[1,1:], 'C4+', zorder=2, label='XcoM Step Transition')
    axes[0].plot(U_array[0,0], U_array[1,0], 'gX', zorder=2)
    axes[1].plot(U_array_ref[0,0], U_array_ref[1,0], 'gX', zorder=2, label='CoP Start')
    axes[0].plot(U_array[0,-1], U_array[1,-1], 'rX', zorder=2)
    axes[1].plot(U_array_ref[0,-1], U_array_ref[1,-1], 'rX', zorder=2, label='CoP End')

    if propcont == 1:
        axes[0].axhline(z_c, ls='--', color='k', zorder=0)
        axes[1].axhline(z_c, ls='--', color='k', zorder=0, label='Desired Lateral Position')

    axes[0].set_ylabel('Z (m)', fontsize=15)
    axes[1].set_ylabel('Z (m)', fontsize=15)
    axes[1].set_xlabel('X (m)', fontsize=15)

    axes[0].yaxis.set_inverted(True)
    axes[1].yaxis.set_inverted(True)

    axes[0].tick_params(axis='both', labelsize=15)
    axes[1].tick_params(axis='both', labelsize=15)

    fig.legend(fontsize=15, ncol=4, loc='upper center')

    if fig_save:
        if propcont == 1:
            fig_name = f"fig_trajsim_inertial_hor_propcont.pdf"
        else:
            fig_name = f"fig_trajsim_inertial_hor.pdf"
        plt.savefig(os.path.join("simulation_fig", fig_name), bbox_inches='tight')

    plt.show()

if fig_state_zoom:
    fig, axes = plt.subplots(ncols=1, nrows=2, sharex=True, figsize=(12,9))

    fig_t_min = 0.25
    fig_t_max = 2.75

    axes[0].plot(t_step[(t_step >= fig_t_min) & (t_step <= fig_t_max)], x_full.step[1, (t_step >= fig_t_min) & (t_step <= fig_t_max)] + U0_full.step[1, (t_step >= fig_t_min) & (t_step <= fig_t_max)], 'C0', zorder=0, label='CoM Trajectory')
    axes[0].plot(const_T*np.arange(n_steps)[(const_T*np.arange(n_steps) >= fig_t_min) & (const_T*np.arange(n_steps) <= fig_t_max)], x0_full.step[1, (const_T*np.arange(n_steps) >= fig_t_min) & (const_T*np.arange(n_steps) <= fig_t_max)], 'm+', markersize=8, zorder=2, label='Step Transition')
    axes[0].axhline(x0_full.step[1,-1], ls=':', color='k', zorder=1, label='Fixed Point')
    axes[0].set_ylabel(state_step_labels[1], fontsize=18)

    axes[1].plot(t_step[(t_step >= fig_t_min) & (t_step <= fig_t_max)], x_full.step[3, (t_step >= fig_t_min) & (t_step <= fig_t_max)] + U0_full.step[3, (t_step >= fig_t_min) & (t_step <= fig_t_max)], 'C0', zorder=0)
    axes[1].plot(const_T*np.arange(n_steps)[(const_T*np.arange(n_steps) >= fig_t_min) & (const_T*np.arange(n_steps) <= fig_t_max)], x0_full.step[3, (const_T*np.arange(n_steps) >= fig_t_min) & (const_T*np.arange(n_steps) <= fig_t_max)] + U0_full.res[3, (const_T*np.arange(n_steps) >= fig_t_min) & (const_T*np.arange(n_steps) <= fig_t_max)], 'm+', markersize=8, zorder=2)
    axes[1].axhline(x0_full.step[3,-1] + U0_full.res[3,-1], ls=':', color='k', zorder=1)
    axes[1].set_ylabel(state_step_labels[3], fontsize=18)

    axes[1].set_xlabel('Time (s)', fontsize=18)

    axes[0].tick_params(axis='both', labelsize=18)
    axes[1].tick_params(axis='both', labelsize=18)

    fig.legend(fontsize=18, ncol=2, loc='upper center')

    if fig_save:
        if propcont == 1:
            fig_name = f"fig_trajsim_step_zoom_propcont.pdf"
        else:
            fig_name = f"fig_trajsim_step_zoom.pdf"
        plt.savefig(os.path.join("simulation_fig", fig_name), bbox_inches='tight')

    plt.show()

if fig_anim: 
    if fig_save:
        if propcont == 1:
            fig_name_timevar = os.path.join("simulation_fig", f"fig_anim_timevar_propcont.gif")
            fig_name_const = os.path.join("simulation_fig", f"fig_anim_const_propcont.gif")
        else:
            fig_name_timevar = os.path.join("simulation_fig", f"fig_anim_timevar.gif")
            fig_name_const = os.path.join("simulation_fig", f"fig_anim_const.gif")
    else:
        fig_name_timevar = None
        fig_name_const = None

    plotanim_func([dt, l], x_full, x0_full, U0_full, savename=fig_name_timevar)
    plotanim_func([dt, l], x_full_ref, x0_full_ref, U0_full_ref, figcolor=['C1','y','darkorange'], savename=fig_name_const)



