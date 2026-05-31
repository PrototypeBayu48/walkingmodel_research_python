import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from walkingmodel_function import model_dyn_def

from walkingmodel_function import ddU_x_0 as ddU_x
from walkingmodel_function import ddU_z_0 as ddU_z

# Constant T

# Constants
n = 4
g = 9.81
l = 1.0

# Define model
model_dyn = model_dyn_def([g, l], ddU_x, ddU_z)

def step_dyn(model_dyn, x0, t_span, t_eval, U0_array):
    def poincare(x1, U_step, leftright=1):
        if leftright == 1:
            L = np.abs(U_step[0,-1] - U_step[0,0])
            W = np.abs(U_step[1,-1] - U_step[1,0])
            x2 = np.array([-L, W, 0.0, 0.0]) + np.squeeze(np.diag([1.0, -1.0, 1.0, -1.0])@x1)
        else:
            L = np.abs(U_step[0,-1] - U_step[0,0])
            W = np.abs(U_step[1,0] - U_step[1,-1])
            x2 = np.array([-L, W, 0.0, 0.0]) + np.squeeze(np.diag([1.0, -1.0, 1.0, -1.0])@x1)
        return x2
    
    def steploc(xloc, U_init, U_targ, leftright=1):
        r = np.sqrt((xloc[0]**2) + (xloc[1]**2))
        angle = np.linspace(0.0, 2*np.pi, 5001)

        U_viable = np.zeros((2,angle.shape[0]))
        if leftright == 1:
            U_viable[0,:] = np.repeat(xloc[0], angle.shape[0]) + (r*np.sin(angle)) + np.repeat(U_init[0], angle.shape[0])
            U_viable[1,:] = np.repeat(xloc[1], angle.shape[0]) + (r*np.cos(angle)) + np.repeat(U_init[1], angle.shape[0])
        else:
            U_viable[0,:] = np.repeat(xloc[0], angle.shape[0]) + (r*np.sin(angle)) + np.repeat(U_init[0], angle.shape[0])
            U_viable[1,:] = -np.repeat(xloc[1], angle.shape[0]) + (r*np.cos(angle)) + np.repeat(U_init[1], angle.shape[0])

        dist = np.linalg.norm( np.repeat(np.reshape(U_targ, (2,1)), angle.shape[0], axis=1) - U_viable, axis=0 )

        if len(U_viable.shape) > 1:
            U_step = (U_viable[:, dist==np.min(dist)])[:,0].squeeze()
        else:
            U_step = U_viable[:, dist==np.min(dist)].squeeze()

        return U_step, U_viable
    
    n_steps = U0_array.shape[1]
    leftright = 1
    x_res = np.zeros((n_steps, n, t_eval.shape[0]))
    U0_res = np.zeros(np.shape(U0_array))
    U0_res[:,0] = U0_array[:,0]
    U0_viable = np.zeros((n_steps, 2, 5001))

    for n1 in range(n_steps):
        if n1 == 0:
            soln = solve_ivp(model_dyn, t_span, x0, t_eval=t_eval[:(len(t_eval)//2)])
        else:
            soln = solve_ivp(model_dyn, t_span, x0, t_eval=t_eval)
        x_step = soln.y
        t_step = soln.t

        if n1 != n_steps-1:
            [U0_res[:2, n1+1], U0_viable[n1, :, :]] = steploc(x_step[:,-1], U0_array[:2, n1], U0_array[:2, n1+1], leftright=leftright)
            x0 = poincare(x_step[:, -1], U0_res[:, n1:n1+2], leftright=leftright)

        if leftright == 1:
            x_res[n1,:,:t_step.shape[0]] = np.repeat(np.reshape(U0_res[:, n1], (U0_res[:, n1].shape[0],1)), t_step.shape[0], axis=1) + x_step
            leftright = 0
        else:
            x_res[n1,:,:t_step.shape[0]] = np.repeat(np.reshape(U0_res[:, n1], (U0_res[:, n1].shape[0],1)), t_step.shape[0], axis=1) + np.diag([1.0, -1.0, 1.0, -1.0])@x_step
            leftright = 1
    return x_res, U0_res, U0_viable


# Simulation parameters
dt = 0.01
#T = 0.1
T = 0.8
n_steps = 3
t_span = (0.0, T)
t_eval = np.arange(t_span[0], t_span[1], dt)
#x0 = np.array([0.05, 0.05, 2.0, 0.01])
#x0 = np.array([0.0, 0.02, 0.0, 0.0])
x0 = np.array([0.0, 0.02, 0.02, 0.0])

U0_array = np.zeros((n,n_steps))
#U0_array[0,:] = np.array([0.0, 0.4])
#U0_array[0,:] = np.array([0.0, 0.0, 0.0])
U0_array[0,:] = np.array([0.0, 0.02, 0.04])
#U0_array[0,:] = np.array([0.0, 0.02, 0.04, 0.06])
#U0_array[1,:] = np.array([-0.05, 0.05])
U0_array[1,:] = np.array([-0.04, 0.04, -0.04])
#U0_array[1,:] = np.array([-0.04, 0.04, -0.04, 0.04])

# Simulation results
[x_res_step, U0_res, U0_viable] = step_dyn(model_dyn, x0, t_span, t_eval, U0_array)
x_res_step_U0 = x_res_step - np.repeat(np.reshape(U0_res.T, (n_steps, n, 1)), t_eval.shape[0], axis=2)
x_res_inertial = x_res_step[0,:,:(len(t_eval)//2)]
x_res = x_res_step_U0[0,:,:(len(t_eval)//2)]
for n1 in range(n_steps-1):
    x_res_inertial = np.concatenate((x_res_inertial[:,:-1], x_res_step[n1 + 1,:,:]), axis=1)
    x_res = np.concatenate((x_res[:,:-1], x_res_step_U0[n1 + 1,:,:]), axis=1)

x0_step = x_res_step[:,:,0].T

#x_res_3D = np.zeros((3, int( (n_steps*t_eval.shape[0]) - (n_steps-1) )))
x_res_3D = np.zeros((3, x_res_inertial.shape[1]))
x_res_3D[0,:] = x_res_inertial[0,:]
#x_res_3D[1,:] = np.sqrt((l**2) - ((x_res_inertial[0,:]**2) + (x_res_inertial[1,:]**2)))
x_res_3D[1,:] = np.sqrt((l**2) - ((x_res[0,:]**2) + (x_res[1,:]**2)))
x_res_3D[2,:] = x_res_inertial[1,:]

# Plot results
plt.figure()
plt.subplot(3,1,1)
plt.plot(dt*np.arange(x_res_3D.shape[1]), x_res_3D[0,:])
plt.xlabel('t (s)')
plt.ylabel('x (m)')
plt.subplot(3,1,2)
plt.plot(dt*np.arange(x_res_3D.shape[1]), x_res_3D[1,:])
plt.xlabel('t (s)')
plt.ylabel('y (m)')
plt.subplot(3,1,3)
plt.plot(dt*np.arange(x_res_3D.shape[1]), x_res_3D[2,:])
plt.xlabel('t (s)')
plt.ylabel('z (m)')
plt.legend()
plt.show()

plt.figure()
plt.plot(x_res_inertial[1,:], x_res_inertial[0,:], '-o', label='Trajectory')
plt.plot(x0_step[1,:], x0_step[0,:], 'o', label='Step Transition')
plt.plot(U0_array[1,:], U0_array[0,:], '+', label='U0 Target')
plt.plot(U0_res[1,:], U0_res[0,:], '+', label='U0 Result')
#plt.plot(U0_viable[0,1,:], U0_viable[0,0,:], ':', label='U0 Viable')
plt.xlabel('z (m)')
plt.ylabel('x (m)')
plt.legend()
plt.show()

ax = plt.figure().add_subplot(projection='3d')
ax.plot(x_res_3D[2,:], x_res_3D[0,:], x_res_3D[1,:], label='Trajectory')
ax.plot(x_res_3D[2,0], x_res_3D[0,0], x_res_3D[1,0], 'o', label='Start')
ax.plot(x_res_3D[2,-1], x_res_3D[0,-1], x_res_3D[1,-1], 'x', label='End')
ax.set_xlabel('z (m)')
ax.set_ylabel('x (m)')
ax.set_zlabel('y (m)')
ax.legend()
plt.show()



