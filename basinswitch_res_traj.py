import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy import optimize

from walkingmodel_function import model_dyn_def, U0_array_func, step_dyn, plotfunc_state_full, model_num_linear_dyn_def, model_linear_dyn_def, step_analysis_func

# Time-Variant CoP Trajectory Testing Results

linear_dyn = 1
propcont = 1

x0_fp = 1
footgeom = 0

copadj = False
#copadj = 'basin'

def x0_fp_func(x0, model_dyn, U0_array, dot_U_func, cop_params):
    def simulate_twostep(x1):
        [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x1, dt, U0_array, dot_U=dot_U_func, cop_params=cop_params, const_T=const_T, propcont=propcont, constoff=1, stepprint=0)
        x2 = x0_full.step[:,1]
        return x2
    constoff_fp_func = lambda x: x - simulate_twostep(x)
    constoff_fp_full = optimize.root(constoff_fp_func, x0)
    x0 = constoff_fp_full.x
    return x0

def x0_propcont_fp_func(xz0, x0_ref, model_dyn, U0_array, dot_U_func, cop_params):
    def simulate_twostep_propcont(xz1):
        x1 = x0_ref
        x1[1] = xz1[0]
        x1[3] = xz1[1]
        sys_par_propcont = (g, l, k, U0_array[1,0] + xz1[2], 0.8*l)
        [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x1, dt, U0_array, dot_U=dot_U_func, cop_params=cop_params, sys_par=sys_par_propcont, const_T=const_T, propcont=1, constoff=1, stepprint=0)
        x2 = x0_full.step[:,1]
        return np.array([x2[1], x2[3], -(U0_array[1,0] + xz1[2] - U0_full.res[1,1])])
    constoff_fp_func = lambda xz: xz - simulate_twostep_propcont(xz)
    constoff_fp_full = optimize.root(constoff_fp_func, xz0)
    xz0 = constoff_fp_full.x
    return xz0

# Constants
n = 4
g = 9.81
l = 1.09
omega_0 = np.sqrt(g/l)

#k = 0.1
k = 0.3

mass = 80

const_T = 0.5
n_steps = 4
s_c = 0.75
w_c = 0.1
z_c = 0.0

sys_par = (g, l, k, 0.0, 0.8*l)

# Define CoP Variables
footl = 0.30
#footw = 0.12
footw = 0.02

copparlim = np.array([[footl, 0.0], [(4*footw)/(footl**2), 0.0]])

if footgeom == 1:
    alpha = copparlim[0,0]
    beta = copparlim[1,0]
else:
    alpha = copparlim[0,0]
    beta = 0.0
    #beta = copparlim[1,0]/2
    beta = copparlim[1,0]/3
    #beta = ((-6)*(-0.01))/(alpha**2)

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
else:
    model_dyn = model_dyn_def([g, l], ddU_x, ddU_z_quad)

# Simulation
dt = 0.01
t_span = (0.0, const_T)
t_eval = np.arange(t_span[0], t_span[1] + dt, dt)

U0_array = U0_array_func(n_steps, s_c, w_c, z_c, dU_x_0=cop_params['b1'], dU_z_0=cop_params['c1'])

# Define x0
if x0_fp == 1:
    U0_array_fp = U0_array[:,:2]

    x0 = x0_fp_func(np.array([0.25, 0.02, 0.0, 0.0]), model_dyn, U0_array_fp, dot_U_func, cop_params)

    if propcont == 1:
        xz0 = x0_propcont_fp_func(np.array([x0[1], x0[3], 0.0]), x0, model_dyn, U0_array_fp, dot_U_func, cop_params)
        x0[1] = xz0[0]
        x0[3] = xz0[1]
        sys_par = (g, l, k, U0_array_fp[1,0] + xz0[2], 0.8*l)

    print('Quadratic CoP:')
    print(x0)
else:
    x0 = np.array([0.25, 0.02, 0.0, 0.0])

F_z = -60
t_perturb = 0.1
x0_perturb = x0.copy()
x0_perturb[1] = x0[1] + x0[3]*(t_perturb) + (F_z/mass)*((t_perturb**2)/2)
x0_perturb[3] = x0[3] + (F_z/mass)*(t_perturb)

[x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x0_perturb, dt, U0_array, dot_U=dot_U_func, cop_params=cop_params, sys_par=sys_par, copparlim = copparlim, const_T=const_T, propcont=propcont, constoff=1, copadj=copadj, stepprint=0)

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

deltacop_z_res = (-alphabeta_res[1,:]*(alphabeta_res[0,:]**2))/6

step_ana_res = step_analysis_func(U0_full.res)

KE_res = (1/2)*(mass)*(np.linalg.norm(x_step_3D[3:,:], axis=0)**2)
KE0_res = (1/2)*(mass)*(np.linalg.norm(x0_step_3D[3:,:], axis=0)**2)
PE_res = (mass)*(g)*(x_step_3D[1,:])
PE0_res = (mass)*(g)*(x0_step_3D[1,:])

print('Average Control Frequency:')
print(str(1/np.mean(deltat_con)) + ' Hz')

print('Average Step Length:')
print(np.mean(step_ana_res[0,:]))
print('Average Step Width:')
print(np.mean(step_ana_res[1,:]))

cop_params = {'b1': b1, 'c1': c1, 'c2': c2}

[x_full_ref, x0_full_ref, U0_full_ref, T_array_ref] = step_dyn(model_dyn, x0, dt, U0_array, dot_U=dot_U_func, cop_params=cop_params, sys_par=sys_par, copparlim = copparlim, const_T=const_T, propcont=propcont, constoff=1, copadj=copadj, stepprint=0)

t_step_ref = x_full_ref.t
x_step_3D_ref = x_full_ref.inertial_3D
x0_step_3D_ref = x0_full_ref.inertial_3D
U_array_ref = U0_full_ref.inertial
cop_params_ref = U0_full_ref.params_res
deltat_con_ref = U0_full_ref.deltat

xcom_ref = x_full_ref.inertial[:2,:] + (x_full_ref.inertial[2:,:]/omega_0)
xcom_x0_ref = x0_full_ref.inertial[:2,:] + (x0_full_ref.inertial[2:,:]/omega_0)

alphabeta_ref = np.zeros((2,n_steps))
alphabeta_ref[0,:] = cop_params_ref[0,:]*(T_array)
alphabeta_ref[1,:] = cop_params_ref[2,:]/(cop_params_res[0,:]**2)
print('Alpha and Beta Results:')
print(alphabeta_ref)

deltacop_z_ref = (-alphabeta_ref[1,:]*(alphabeta_ref[0,:]**2))/6

step_ana_ref = step_analysis_func(U0_full_ref.res)

KE_ref = (1/2)*(mass)*(np.linalg.norm(x_step_3D_ref[3:,:], axis=0)**2)
KE0_ref = (1/2)*(mass)*(np.linalg.norm(x0_step_3D_ref[3:,:], axis=0)**2)
PE_ref = (mass)*(g)*(x_step_3D_ref[1,:])
PE0_ref = (mass)*(g)*(x0_step_3D_ref[1,:])

print('Average Reference Control Frequency:')
print(str(1/np.mean(deltat_con_ref)) + ' Hz')

print('Average Reference Step Length:')
print(np.mean(step_ana_ref[0,:]))
print('Average Reference Step Width:')
print(np.mean(step_ana_ref[1,:]))

# Display results

fig, axes = plt.subplots()

axes.plot(x_step_3D[2,:], x_step_3D[0,:], 'C0-o', zorder=1, label='CoM Trajectory (Perturbed)')
axes.plot(U_array[1,:], U_array[0,:], 'c:x', zorder=0, label='CoP Trajectory (Perturbed)')
axes.plot(x_step_3D_ref[2,:], x_step_3D_ref[0,:], 'C1-o', zorder=1, label='CoM Trajectory (Steady-State)')
axes.plot(U_array_ref[1,:], U_array_ref[0,:], 'y:x', zorder=0, label='CoP Trajectory (Steady-State)')

axes.plot(x_step_3D[2,0], x_step_3D[0,0], 'go', zorder=2)
axes.plot(x_step_3D_ref[2,0], x_step_3D_ref[0,0], 'go', zorder=2, label='CoM Trajectory Start')
axes.plot(x_step_3D[2,-1], x_step_3D[0,-1], 'ro', zorder=2)
axes.plot(x_step_3D_ref[2,-1], x_step_3D_ref[0,-1], 'ro', zorder=2, label='CoM Trajectory End')
axes.plot(U_array[1,0], U_array[0,0], 'gX', zorder=2)
axes.plot(U_array_ref[1,0], U_array_ref[0,0], 'gX', zorder=2, label='CoP Trajectory Start')
axes.plot(U_array[1,-1], U_array[0,-1], 'rX', zorder=2)
axes.plot(U_array_ref[1,-1], U_array_ref[0,-1], 'rX', zorder=2, label='CoP Trajectory End')
if propcont == 1:
    axes.axvline(sys_par[3], 0, 1, c='k', ls='--', zorder=0, label='Desired Position')

axes.set_xlabel('Z (m)')
axes.set_ylabel('X (m)')

if propcont == 1:
    fig.legend(fontsize='large', ncol=5)
else:
    fig.legend(fontsize='large', ncol=4)
plt.show()

fig, axes = plt.subplots()

#axes.plot(deltacop_z_res, 'C0-o', zorder=1, label='Perturbed')
#axes.plot(deltacop_z_ref, 'C1-o', zorder=1, label='Steady-State')
axes.plot(deltacop_z_res - deltacop_z_ref, 'C0-o', zorder=1)

axes.set_xlabel('Steps')
axes.set_ylabel('CoP Shift (m)')

#fig.legend(fontsize='large', ncol=1)
plt.show()



