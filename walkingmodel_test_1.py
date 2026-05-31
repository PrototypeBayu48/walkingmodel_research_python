import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy import optimize

from walkingmodel_function import model_dyn_def, U0_array_func, step_dyn, plotfunc_state_full, model_num_linear_dyn_def, model_linear_dyn_def

# Quadratic CoP Test

linear_dyn = 1
propcont = 0

x0_fp = 1
footgeom = 0

def disp_U_func(t_step, U0_full, inertial=0):
    if inertial == 1:
        U_res = U0_full.inertial
    else:
        U_res = U0_full.step

    fig, axes = plt.subplots(4, sharex=True)
    axes[0].plot(t_step, U_res[0,:])
    axes[1].plot(t_step, U_res[1,:])
    axes[2].plot(t_step, U_res[2,:])
    axes[3].plot(t_step, U_res[3,:])
    if inertial == 1:
        axes[0].set_ylabel('$ X $ (m)')
        axes[1].set_ylabel('$ Z $ (m)')
        axes[2].set_ylabel('$ \dot{X} $ (m/s)')
        axes[3].set_xlabel('t (s)')
        axes[3].set_ylabel('$ \dot{Z} $ (m/s)')
    else:
        axes[0].set_ylabel('$ x $ (m)')
        axes[1].set_ylabel('$ z $ (m)')
        axes[2].set_ylabel('$ \dot{x} $ (m/s)')
        axes[3].set_xlabel('t (s)')
        axes[3].set_ylabel('$ \dot{z} $ (m/s)')
    #fig.legend()
    plt.show()

def x0_fp_func(x0, model_dyn, U0_array, dot_U_func, cop_params):
    def simulate_twostep(x1):
        [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x1, dt, U0_array, dot_U=dot_U_func, cop_params=cop_params, const_T=const_T, propcont=propcont, constoff=1, stepprint=0)
        x2 = x0_full.step[:,1]
        return x2
    #x0 = optimize.fixed_point(simulate_twostep, x0)
    constoff_fp_func = lambda x: x - simulate_twostep(x)
    constoff_fp_full = optimize.root(constoff_fp_func, x0)
    x0 = constoff_fp_full.x
    return x0

# Constants
n = 4
g = 9.81
l = 1.09
omega_0 = np.sqrt(g/l)

const_T = 0.5
#const_T = 1.0
n_steps = 10
s_c = 0.75
w_c = 0.1
z_c = 0.0

# Define CoP Variables

if footgeom == 1:
    footl = 0.30
    footw = 0.12
    #footw = 0.04

    alpha = footl/s_c
    beta = (4*footw)/((alpha**2)*(s_c**2))
else:
    #alpha = 0.04
    #alpha = 0.1
    alpha = 0.4
    #alpha = 0.5
    #alpha = 0.6

    #beta = 0.1
    beta = 0.2
    #beta = 0.4
    #beta = 0.6
    #beta = 0.8
    #beta = 1.0

#b1 = 0.0
#b1 = 0.01
#b1 = 0.1
#b1 = 0.7

b1 = (alpha*s_c)/const_T

#c1 = 0.0
#c1 = -0.01
#c1 = -0.02
#c1 = -0.18

c1 = -alpha*beta*b1*s_c

#c2 = 0.0
#c2 = 0.01
#c2 = 0.05
#c2 = 0.3

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

    '''
    nonlin_model_dyn = model_dyn_def([g, l], ddU_x, ddU_z_quad)
    nonlin_model_dyn_const = model_dyn_def([g, l], ddU_x, ddU_z_const)
    xinit_lin_test = np.zeros(n)
    model_dyn_num = model_num_linear_dyn_def(xinit_lin_test, nonlin_model_dyn)
    model_dyn_num_const = model_num_linear_dyn_def(xinit_lin_test, nonlin_model_dyn_const)
    '''
else:
    model_dyn = model_dyn_def([g, l], ddU_x, ddU_z_quad)
    model_dyn_const = model_dyn_def([g, l], ddU_x, ddU_z_const)

# Simulation
dt = 0.01
t_span = (0.0, const_T)
t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
#x0_const = np.array([0.25, 0.08, 0.0, 0.0])

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

[x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x0, dt, U0_array, dot_U=dot_U_func, cop_params=cop_params, const_T=const_T, propcont=propcont, constoff=1, stepprint=0)

t_step = x_full.t
x_step_3D = x_full.inertial_3D
x0_step_3D = x0_full.inertial_3D
U_array = U0_full.inertial

xcom = x_full.inertial[:2,:] + (x_full.inertial[2:,:]/omega_0)
xcom_x0 = x0_full.inertial[:2,:] + (x0_full.inertial[2:,:]/omega_0)

[x_full_const, x0_full_const, U0_full_const, T_array_const] = step_dyn(model_dyn_const, x0_const, dt, U0_array_const, const_T=const_T, propcont=propcont, constoff=1, stepprint=0)

t_step_const = x_full_const.t
x_step_3D_const = x_full_const.inertial_3D
x0_step_3D_const = x0_full_const.inertial_3D
U_array_const = U0_full_const.inertial

xcom_const = x_full_const.inertial[:2,:] + (x_full_const.inertial[2:,:]/omega_0)
xcom_x0_const = x0_full_const.inertial[:2,:] + (x0_full_const.inertial[2:,:]/omega_0)

# Display results
'''
ax = plt.figure().add_subplot(projection='3d')
ax.plot(x_step_3D[2,:], x_step_3D[0,:], x_step_3D[1,:], label='Trajectory')
ax.plot(x_step_3D[2,0], x_step_3D[0,0], x_step_3D[1,0], 'o', label='Trajectory Start')
ax.plot(x_step_3D[2,-1], x_step_3D[0,-1], x_step_3D[1,-1], 'o', label='Trajectory End')
ax.plot(x0_step_3D[2,:], x0_step_3D[0,:], x0_step_3D[1,:], '+', label='Step Transition')
ax.plot(U_array[1,:], U_array[0,:], np.zeros(len(t_step)), ':x', label='CoP')
ax.plot(U_array[1,0], U_array[0,0], 0.0, 'X', label='CoP Start')
ax.plot(U_array[1,-1], U_array[0,-1], 0.0, 'X', label='CoP End')
ax.plot((U_array[1,0], x_step_3D[2,0]), (U_array[0,0], x_step_3D[0,0]), (0.0, x_step_3D[1,0]), 'k-')
ax.plot((U_array[1,-1], x_step_3D[2,-1]), (U_array[0,-1], x_step_3D[0,-1]), (0.0, x_step_3D[1,-1]), 'k-')
ax.set_xlabel('Z (m)')
ax.set_ylabel('X (m)')
ax.set_zlabel('Y (m)')
ax.legend()
plt.show()
'''

plt.figure()
plt.plot(x_step_3D[2,:], x_step_3D[0,:], 'C0-o', label='Quadratic CoP Trajectory')
plt.plot(x_step_3D_const[2,:], x_step_3D_const[0,:], 'C1-o', label='Constant CoP Trajectory')
plt.plot(x_step_3D[2,0], x_step_3D[0,0], 'go', label='Trajectory Start')
plt.plot(x_step_3D_const[2,0], x_step_3D_const[0,0], 'go')
plt.plot(x_step_3D[2,-1], x_step_3D[0,-1], 'ro', label='Trajectory End')
plt.plot(x_step_3D_const[2,-1], x_step_3D_const[0,-1], 'ro')
plt.plot(x0_step_3D[2,:], x0_step_3D[0,:], 'm+', label='Step Transition')
plt.plot(x0_step_3D_const[2,:], x0_step_3D_const[0,:], 'm+')
plt.plot(U_array[1,:], U_array[0,:], 'c:x', label='Quadratic CoP')
plt.plot(U_array_const[1,:], U_array_const[0,:], 'yx', label='Constant CoP')
plt.plot(U_array[1,0], U_array[0,0], 'gX', label='CoP Start')
plt.plot(U_array_const[1,0], U_array_const[0,0], 'gX')
plt.plot(U_array[1,-1], U_array[0,-1], 'rX', label='CoP End')
plt.plot(U_array_const[1,-1], U_array_const[0,-1], 'rX')
plt.xlabel('Z (m)')
plt.ylabel('X (m)')
plt.legend()
plt.show()

fig, axes = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True)
axes[0].plot(x_step_3D[2,:], x_step_3D[0,:], 'C0-o', label='Quadratic CoP Trajectory')
axes[0].plot(x_step_3D[2,0], x_step_3D[0,0], 'go', label='Trajectory Start')
axes[0].plot(x_step_3D[2,-1], x_step_3D[0,-1], 'ro', label='Trajectory End')
axes[0].plot(xcom[1,:], xcom[0,:], 'C0--', label='Quadratic CoP XcoM')
axes[0].plot(x0_step_3D[2,:], x0_step_3D[0,:], 'm+', label='Step Transition')
axes[0].plot(xcom_x0[1,:], xcom_x0[0,:], 'C4+', label='Step Transition XcoM')
axes[0].plot(U_array[1,:], U_array[0,:], 'c:x', label='Quadratic CoP')
axes[0].plot(U_array[1,0], U_array[0,0], 'gX', label='CoP Start')
axes[0].plot(U_array[1,-1], U_array[0,-1], 'rX', label='CoP End')
axes[1].plot(x_step_3D_const[2,:], x_step_3D_const[0,:], 'C1-o', label='Constant CoP Trajectory')
axes[1].plot(x_step_3D_const[2,0], x_step_3D_const[0,0], 'go')
axes[1].plot(x_step_3D_const[2,-1], x_step_3D_const[0,-1], 'ro')
axes[1].plot(xcom_const[1,:], xcom_const[0,:], 'C1--', label='Constant CoP XcoM')
axes[1].plot(x0_step_3D_const[2,:], x0_step_3D_const[0,:], 'm+')
axes[1].plot(xcom_x0_const[1,:], xcom_x0_const[0,:], 'C4+')
axes[1].plot(U_array_const[1,:], U_array_const[0,:], 'yx', label='Constant CoP')
axes[1].plot(U_array_const[1,0], U_array_const[0,0], 'gX')
axes[1].plot(U_array_const[1,-1], U_array_const[0,-1], 'rX')
axes[0].set_ylabel('X (m)')
axes[0].set_xlabel('Z (m)')
axes[1].set_xlabel('Z (m)')
fig.legend()
plt.show()

#disp_U_func(t_step, U0_full, inertial=1)

#plotfunc_state_full(dt, omega_0, x_full, x0_full, U0_full, U0_array, xzval_inertial=0)



