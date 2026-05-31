import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from walkingmodel_function import model_dyn_def, U0_array_func, step_dyn, plotfunc_state_full, model_num_linear_dyn_def
from walkingmodel_function import ddU_x_0 as ddU_x
from walkingmodel_function import ddU_z_0 as ddU_z

linear_dyn = 0
propcont = 0

# Constants
n = 4
g = 9.81
l = 1.09
omega_0 = np.sqrt(g/l)

const_T = 0.5
n_steps = 10
s_c = 0.75
w_c = 0.1
z_c = 0.0

# Simulation
dt = 0.01
t_span = (0.0, const_T)
t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
x0 = 1.1*np.array([0.25, 0.02, 0.0, 0.0])

U0_array = U0_array_func(n_steps, s_c, w_c, z_c)

# Define model
if linear_dyn == 1:
    nonlin_model_dyn = model_dyn_def([g, l], ddU_x, ddU_z)
    xinit_lin_test = np.zeros(n)
    model_dyn = model_num_linear_dyn_def(xinit_lin_test, nonlin_model_dyn)
else:
    model_dyn = model_dyn_def([g, l], ddU_x, ddU_z)

[x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x0, dt, U0_array, const_T=const_T, propcont=propcont, constoff=1, stepprint=0)

t_step = x_full.t
x_step_3D = x_full.inertial_3D
x0_step_3D = x0_full.inertial_3D
U_array = U0_full.inertial

xcom = x_full.inertial[:2,:] + (x_full.inertial[2:,:]/omega_0)

# Display Results
ax = plt.figure().add_subplot(projection='3d')
ax.plot(x_step_3D[2,:], x_step_3D[0,:], x_step_3D[1,:], label='Trajectory')
ax.plot(x_step_3D[2,0], x_step_3D[0,0], x_step_3D[1,0], 'o', label='Trajectory Start')
ax.plot(x_step_3D[2,-1], x_step_3D[0,-1], x_step_3D[1,-1], 'o', label='Trajectory End')
ax.plot(x0_step_3D[2,:], x0_step_3D[0,:], x0_step_3D[1,:], '+', label='Step Transition')
ax.plot(U_array[1,:], U_array[0,:], np.zeros(len(t_step)), 'x', label='CoP')
ax.plot(U_array[1,0], U_array[0,0], 0.0, 'X', label='CoP Start')
ax.plot(U_array[1,-1], U_array[0,-1], 0.0, 'X', label='CoP End')
ax.plot((U_array[1,0], x_step_3D[2,0]), (U_array[0,0], x_step_3D[0,0]), (0.0, x_step_3D[1,0]), 'k-')
ax.plot((U_array[1,-1], x_step_3D[2,-1]), (U_array[0,-1], x_step_3D[0,-1]), (0.0, x_step_3D[1,-1]), 'k-')
ax.set_xlabel('Z (m)')
ax.set_ylabel('X (m)')
ax.set_zlabel('Y (m)')
ax.legend()
plt.show()

plt.figure()
plt.plot(x_step_3D[2,:], x_step_3D[0,:], 'C0-o', label='Trajectory')
plt.plot(x_step_3D[2,0], x_step_3D[0,0], 'go', label='Trajectory Start')
plt.plot(x_step_3D[2,-1], x_step_3D[0,-1], 'ro', label='Trajectory End')
plt.plot(xcom[1,:], xcom[0,:], 'C0--', label='XcoM')
plt.plot(x0_step_3D[2,:], x0_step_3D[0,:], 'm+', label='Step Transition')
plt.plot(U_array[1,:], U_array[0,:], 'cx', label='CoP')
plt.plot(U_array[1,0], U_array[0,0], 'gX', label='CoP Start')
plt.plot(U_array[1,-1], U_array[0,-1], 'rX', label='CoP End')
plt.xlabel('Z (m)')
plt.ylabel('X (m)')
plt.legend()
plt.show()

plotfunc_state_full(dt, omega_0, x_full, x0_full, U0_full, U0_array, xzval_inertial=0)



