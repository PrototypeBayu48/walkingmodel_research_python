import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from walkingmodel_function import model_dyn_def, U0_array_func, step_dyn, plotfunc_state_full, model_num_linear_dyn_def, model_linear_dyn_def

# Model Comparison Test

propcont = 0

def disp_U_func(t_step_1, U0_full_1, inertial=0):
    if inertial == 1:
        U_res = U0_full_1.inertial
    else:
        U_res = U0_full_1.step

    fig, axes = plt.subplots(4, sharex=True)
    axes[0].plot(t_step_1, U_res[0,:])
    axes[1].plot(t_step_1, U_res[1,:])
    axes[2].plot(t_step_1, U_res[2,:])
    axes[3].plot(t_step_1, U_res[3,:])
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

#alpha = 0.04
#alpha = 0.1
alpha = 0.4
#alpha = 0.5
#alpha = 0.6

#beta = 0.1
beta = 0.2
#beta = 1.0
#beta = 2.0
#beta = 8.0

# Define variables
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

# Define CoP functions
def cop_func_x(t):
    return b1*(t)

def cop_func_z(t):
    return c2*(t**2) + c1*(t)

def ddU_x(t,x):
    return 0.0

def ddU_z_quad(t,x):
    return 2*c2

def ddU_z_const(t,x):
    return 0.0

def dot_U_func_quad(t,x):
    dot_U = np.array([x[2], x[3], ddU_x(t,x), ddU_z_quad(t,x)])
    return dot_U

# Simulation
dt = 0.01
t_span = (0.0, const_T)
t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
x0 = np.array([0.25, 0.02, 0.0, 0.0])

# Define model dynamics

xinit_lin_test = np.zeros(n)

model_dyn_nonlin_quad = model_dyn_def([g, l], ddU_x, ddU_z_quad)
model_dyn_nonlin_const = model_dyn_def([g, l], ddU_x, ddU_z_const)

model_dyn_lin_quad = model_linear_dyn_def([g, l], ddU_x, ddU_z_quad)
model_dyn_lin_const = model_linear_dyn_def([g, l], ddU_x, ddU_z_const)

model_dyn_1 = model_dyn_lin_quad
model_dyn_2 = model_num_linear_dyn_def(xinit_lin_test, model_dyn_nonlin_quad)

# Define CoP dynamics

U0_array_quad = U0_array_func(n_steps, s_c, w_c, z_c, dU_x_0=b1, dU_z_0=c1)
U0_array_const = U0_array_func(n_steps, s_c, w_c, z_c)

U0_array_1 = U0_array_quad
U0_array_2 = U0_array_quad

dot_U_1 = dot_U_func_quad
dot_U_2 = dot_U_func_quad

# Simulation results

[x_full_1, x0_full_1, U0_full_1, T_array_1] = step_dyn(model_dyn_1, x0, dt, U0_array_1, dot_U=dot_U_1, const_T=const_T, propcont=propcont, constoff=1, stepprint=0)

t_step_1 = x_full_1.t
x_step_3D_1 = x_full_1.inertial_3D
x0_step_3D_1 = x0_full_1.inertial_3D
U_array_1 = U0_full_1.inertial

xcom_1 = x_full_1.inertial[:2,:] + (x_full_1.inertial[2:,:]/omega_0)
xcom_x0_1 = x0_full_1.inertial[:2,:] + (x0_full_1.inertial[2:,:]/omega_0)

[x_full_2, x0_full_2, U0_full_2, T_array_2] = step_dyn(model_dyn_2, x0, dt, U0_array_2, dot_U=dot_U_2, const_T=const_T, propcont=propcont, constoff=1, stepprint=0)

t_step_2 = x_full_2.t
x_step_3D_2 = x_full_2.inertial_3D
x0_step_3D_2 = x0_full_2.inertial_3D
U_array_2 = U0_full_2.inertial

xcom_2 = x_full_2.inertial[:2,:] + (x_full_2.inertial[2:,:]/omega_0)
xcom_x0_2 = x0_full_2.inertial[:2,:] + (x0_full_2.inertial[2:,:]/omega_0)

# Display results
plt.figure()
plt.plot(x_step_3D_1[2,:], x_step_3D_1[0,:], 'C0-o', label='Model 1 Trajectory')
plt.plot(x_step_3D_2[2,:], x_step_3D_2[0,:], 'C1-o', label='Model 2 Trajectory')
plt.plot(x_step_3D_1[2,0], x_step_3D_1[0,0], 'go', label='Trajectory Start')
plt.plot(x_step_3D_2[2,0], x_step_3D_2[0,0], 'go')
plt.plot(x_step_3D_1[2,-1], x_step_3D_1[0,-1], 'ro', label='Trajectory End')
plt.plot(x_step_3D_2[2,-1], x_step_3D_2[0,-1], 'ro')
plt.plot(x0_step_3D_1[2,:], x0_step_3D_1[0,:], 'm+', label='Step Transition')
plt.plot(x0_step_3D_2[2,:], x0_step_3D_2[0,:], 'm+')
plt.plot(U_array_1[1,:], U_array_1[0,:], 'c:x', label='Model 1 CoP')
plt.plot(U_array_2[1,:], U_array_2[0,:], 'y:x', label='Model 2 CoP')
plt.plot(U_array_1[1,0], U_array_1[0,0], 'gX', label='CoP Start')
plt.plot(U_array_2[1,0], U_array_2[0,0], 'gX')
plt.plot(U_array_1[1,-1], U_array_1[0,-1], 'rX', label='CoP End')
plt.plot(U_array_2[1,-1], U_array_2[0,-1], 'rX')
plt.xlabel('Z (m)')
plt.ylabel('X (m)')
plt.legend()
plt.show()

fig, axes = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True)
axes[0].plot(x_step_3D_1[2,:], x_step_3D_1[0,:], 'C0-o', label='Model 1 Trajectory')
axes[0].plot(x_step_3D_1[2,0], x_step_3D_1[0,0], 'go', label='Trajectory Start')
axes[0].plot(x_step_3D_1[2,-1], x_step_3D_1[0,-1], 'ro', label='Trajectory End')
axes[0].plot(xcom_1[1,:], xcom_1[0,:], 'C0--', label='Model 1 xcom_1')
axes[0].plot(x0_step_3D_1[2,:], x0_step_3D_1[0,:], 'm+', label='Step Transition')
axes[0].plot(xcom_x0_1[1,:], xcom_x0_1[0,:], 'C4+', label='Step Transition xcom_1')
axes[0].plot(U_array_1[1,:], U_array_1[0,:], 'c:x', label='Model 1 CoP')
axes[0].plot(U_array_1[1,0], U_array_1[0,0], 'gX', label='CoP Start')
axes[0].plot(U_array_1[1,-1], U_array_1[0,-1], 'rX', label='CoP End')
axes[1].plot(x_step_3D_2[2,:], x_step_3D_2[0,:], 'C1-o', label='Model 2 Trajectory')
axes[1].plot(x_step_3D_2[2,0], x_step_3D_2[0,0], 'go')
axes[1].plot(x_step_3D_2[2,-1], x_step_3D_2[0,-1], 'ro')
axes[1].plot(xcom_2[1,:], xcom_2[0,:], 'C1--', label='Model 2 xcom_1')
axes[1].plot(x0_step_3D_2[2,:], x0_step_3D_2[0,:], 'm+')
axes[1].plot(xcom_x0_2[1,:], xcom_x0_2[0,:], 'C4+')
axes[1].plot(U_array_2[1,:], U_array_2[0,:], 'y:x', label='Model 2 CoP')
axes[1].plot(U_array_2[1,0], U_array_2[0,0], 'gX')
axes[1].plot(U_array_2[1,-1], U_array_2[0,-1], 'rX')
axes[0].set_ylabel('X (m)')
axes[0].set_xlabel('Z (m)')
axes[1].set_xlabel('Z (m)')
fig.legend()
plt.show()



