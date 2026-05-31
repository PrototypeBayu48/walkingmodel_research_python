import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from walkingmodel_function import model_dyn_def, model_linear_dyn_def

from walkingmodel_function import ddU_x_0 as ddU_x
from walkingmodel_function import ddU_z_0 as ddU_z

# Variable T

animation = 0

linear_dyn = 0
const_T = 0
propcont = 0

# Constants
n = 4
g = 9.81
l = 1.09

omega_0 = np.sqrt(g/l)

# Define model
if linear_dyn == 1:
    model_dyn = model_linear_dyn_def([g, l], ddU_x, ddU_z)
else:
    model_dyn = model_dyn_def([g, l], ddU_x, ddU_z)

def step_dyn(model_dyn, x0, dt, U0_array, const_T=0, propcont=0):
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
    
    def steploc(xloc, U_init, U_targ, leftright=1, propcont=1):
        r = np.sqrt((xloc[0]**2) + (xloc[1]**2))
        angle = np.linspace(0.0, 2*np.pi, 5001)

        if leftright == 1:
            zeta_n = (xloc[1] + U_init[1]) + (xloc[3]/omega_0)
        else:
            zeta_n = ((-xloc[1]) + U_init[1]) + ((-xloc[3])/omega_0)
        xi_n = (xloc[0] + U_init[0]) + (xloc[2]/omega_0)

        U_viable = np.zeros((2,angle.shape[0]))
        if leftright == 1:
            U_viable[0,:] = np.repeat(xloc[0], angle.shape[0]) + (r*np.sin(angle)) + np.repeat(U_init[0], angle.shape[0])
            U_viable[1,:] = np.repeat(xloc[1], angle.shape[0]) + (r*np.cos(angle)) + np.repeat(U_init[1], angle.shape[0])
        else:
            U_viable[0,:] = np.repeat(xloc[0], angle.shape[0]) + (r*np.sin(angle)) + np.repeat(U_init[0], angle.shape[0])
            U_viable[1,:] = -np.repeat(xloc[1], angle.shape[0]) + (r*np.cos(angle)) + np.repeat(U_init[1], angle.shape[0])

        #U_viable_eval = U_viable[:, U_viable[0,:] < xi_n]
        U_viable_eval = U_viable

        dist = np.linalg.norm( np.repeat(np.reshape(U_targ, (2,1)), U_viable_eval.shape[1], axis=1) - U_viable_eval, axis=0 )

        '''
        if len(U_viable.shape) > 1:
            U_step = (U_viable_eval[:, dist==np.min(dist)])[:,0].squeeze()
        else:
            U_step = U_viable_eval[:, dist==np.min(dist)].squeeze()
        '''

        U_step = np.zeros(2)
        U_step[0] = xi_n - b_x
        if propcont == 1:
            k1 = 0.1
            z_c = 0.0
            if leftright == 1:
                U_step[1] = zeta_n + b_z + k1*(zeta_n - z_c)
            else:
                U_step[1] = zeta_n - b_z + k1*(zeta_n - z_c)
        else:
            if leftright == 1:
                U_step[1] = zeta_n + b_z
            else:
                U_step[1] = zeta_n - b_z

        return U_step, U_viable
    
    steps_len = U0_array.shape[1]
    leftright = 1

    T_c = 0.5
    s_c = U0_array[0,1] - U0_array[0,0]
    w_c = U0_array[1,1] - U0_array[1,0]
    omega_0 = np.sqrt(g/l)
    b_x = (s_c)/(np.exp(omega_0*T_c) - 1)
    b_z = (w_c)/(np.exp(omega_0*T_c) + 1)

    U0_res = np.zeros(np.shape(U0_array))
    U0_res[:,0] = U0_array[:,0]
    U0_viable = np.zeros((steps_len, 2, 5001))

    x0_array = np.zeros((n, steps_len))
    x0_array_t = np.zeros(steps_len)
    x0_array[:,0] = x0

    for n1 in range(steps_len):
        print('Step: ' + str(n1 + 1))

        if n1 == 0:
            s = s_c
            xi_n = ( ((x0_array[0, n1] + U0_res[0, n1]) + (x0_array[2, n1]/omega_0)) )
        else:
            s = U0_res[0, n1] - U0_res[0, n1-1]
        if const_T == 0:
            T = (1/omega_0)*(np.log((s/(xi_n - U0_res[0, n1])) + 1))
        else:
            T = const_T
        t_span = (0.0, T)
        t_eval = np.arange(t_span[0], t_span[1], dt)

        print('Period (s): ' + str(T))

        soln = solve_ivp(model_dyn, t_span, x0_array[:, n1], t_eval=t_eval)
        x_step = soln.y
        t_step = soln.t

        xi_n = ( ((x_step[0, -1] + U0_res[0, n1]) + (x_step[2, -1]/omega_0)) )

        if n1 != steps_len-1:
            '''
            U0_array[0, n1+1] = xi_n - b_x
            if leftright == 0:
                U0_array[1, n1+1] = zeta_n - b_z
            else:
                U0_array[1, n1+1] = zeta_n + b_z
            '''
            [U0_res[:2, n1+1], U0_viable[n1, :, :]] = steploc(x_step[:,-1], U0_res[:2, n1], U0_array[:2, n1+1], leftright=leftright, propcont=1)
            x0_array[:, n1+1] = poincare(x_step[:, -1], U0_res[:, n1:n1+2], leftright=leftright)
            x0_array_t[n1+1] = x0_array_t[n1] + t_step[-1]

        if leftright == 1:
            if n1 == 0:
                x_res_inertial = np.repeat(np.reshape(U0_res[:, n1], (U0_res[:, n1].shape[0],1)), t_step.shape[0], axis=1) + x_step
                x_res = x_step
            else:
                x_step_inertial = np.repeat(np.reshape(U0_res[:, n1], (U0_res[:, n1].shape[0],1)), t_step.shape[0], axis=1) + x_step
                x_res_inertial = np.concatenate((x_res_inertial[:,:-1], x_step_inertial), axis=1)
                x_res = np.concatenate((x_res[:,:-1], x_step), axis=1)
            leftright = 0
        else:
            if n1 == 0:
                x_res_inertial = np.repeat(np.reshape(U0_res[:, n1], (U0_res[:, n1].shape[0],1)), t_step.shape[0], axis=1) + (np.diag([1.0, -1.0, 1.0, -1.0])@x_step)
                x_res = (np.diag([1.0, -1.0, 1.0, -1.0])@x_step)
            else:
                x_step_inertial = np.repeat(np.reshape(U0_res[:, n1], (U0_res[:, n1].shape[0],1)), t_step.shape[0], axis=1) + (np.diag([1.0, -1.0, 1.0, -1.0])@x_step)
                x_res_inertial = np.concatenate((x_res_inertial[:,:-1], x_step_inertial), axis=1)
                x_res = np.concatenate((x_res[:,:-1], (np.diag([1.0, -1.0, 1.0, -1.0])@x_step)), axis=1)
            leftright = 1
    
    x_res_3D = np.zeros((3, x_res_inertial.shape[1]))
    x_res_3D[0,:] = x_res_inertial[0,:]
    x_res_3D[1,:] = np.sqrt((l**2) - ((x_res[0,:]**2) + (x_res[1,:]**2)))
    x_res_3D[2,:] = x_res_inertial[1,:]

    even = 1
    for n1 in range(steps_len):
        if even == 1:
            x0_array[:,n1] = U0_res[:,n1] + x0_array[:,n1]
            even = 0
        else:
            x0_array[:,n1] = U0_res[:,n1] + (np.diag([1.0, -1.0, 1.0, -1.0])@x0_array[:,n1])
            even = 1

    x0_array_3D = np.zeros((3, steps_len))
    x0_array_3D[0,:] = x0_array[0,:]
    x0_array_3D[1,:] = np.sqrt((l**2) - (((x0_array[0,:] - U0_res[0,:])**2) + ((x0_array[1,:] - U0_res[1,:])**2)))
    x0_array_3D[2,:] = x0_array[1,:]

    return x_res_inertial, x_res_3D, x0_array_t, x0_array, x0_array_3D, U0_res, U0_viable

# Simulation parameters
dt = 0.01
n_steps = 8
s_c = 0.75
w_c = 0.1
z_c = 0.0
x0 = np.array([0.25, 0.02, 0.0, 0.0])

U0_array = np.zeros((n,n_steps))
U0_array[0,:] = np.linspace(0.0, (n_steps-1)*(s_c), n_steps)
U0_array[1,:] = z_c + (w_c/2)*((-1)**(np.arange(n_steps) + 1))

# Simulation results
[x_res_inertial, x_res_3D, x0_array_t, x0_array, x0_array_3D, U0_res, U0_viable] = step_dyn(model_dyn, x0, dt, U0_array, const_T=const_T, propcont=propcont)

xcom = np.zeros((2, x_res_inertial.shape[1]))
xcom[0,:] = x_res_inertial[0,:] + (x_res_inertial[2,:]/omega_0)
xcom[1,:] = x_res_inertial[1,:] + (x_res_inertial[3,:]/omega_0)

U0_res_lin_t = np.zeros(2*len(x0_array_t))
U0_res_lin = np.zeros((2, 2*len(x0_array_t)))
U0_array_lin = np.zeros((2, 2*len(x0_array_t)))
for n1 in np.arange(len(x0_array_t)):
    if n1 == (len(x0_array_t) - 1):
        U0_res_lin_t[2*n1] = x0_array_t[n1]
        U0_res_lin_t[(2*n1)+1] = dt*(np.arange(x_res_3D.shape[1])[-1])
    else:
        U0_res_lin_t[(2*n1):((2*n1)+2)] = x0_array_t[n1:n1+2]
    U0_res_lin[:, (2*n1):((2*n1)+2)] = np.repeat(U0_res[:2, n1].reshape(2,1), 2, axis=1)
    U0_array_lin[:, (2*n1):((2*n1)+2)] = np.repeat(U0_array[:2, n1].reshape(2,1), 2, axis=1)

U0_res_lin_interp = np.zeros((2, x_res_inertial.shape[1]))
U0_res_lin_interp[0,:] = np.interp(dt*np.arange(x_res_3D.shape[1]), U0_res_lin_t, U0_res_lin[0,:])
U0_res_lin_interp[1,:] = np.interp(dt*np.arange(x_res_3D.shape[1]), U0_res_lin_t, U0_res_lin[1,:])

U0_array_lin_interp = np.zeros((2, x_res_inertial.shape[1]))
U0_array_lin_interp[0,:] = np.interp(dt*np.arange(x_res_3D.shape[1]), U0_res_lin_t, U0_array_lin[0,:])
U0_array_lin_interp[1,:] = np.interp(dt*np.arange(x_res_3D.shape[1]), U0_res_lin_t, U0_array_lin[1,:])

# Plot results
fig, axes = plt.subplots(3, sharex=True)
axes[0].plot(dt*np.arange(x_res_3D.shape[1]), x_res_3D[0,:])
axes[0].plot(dt*np.arange(x_res_3D.shape[1]), xcom[0,:], ':')
axes[0].plot(x0_array_t, x0_array_3D[0,:], 'co')
axes[0].plot(U0_res_lin_t, U0_array_lin[0,:], 'g-')
axes[0].plot(U0_res_lin_t, U0_res_lin[0,:], 'r-')
axes[0].set_ylabel('x (m)')
axes[1].plot(dt*np.arange(x_res_3D.shape[1]), x_res_3D[1,:])
axes[1].plot(x0_array_t, x0_array_3D[1,:], 'co')
axes[1].set_ylabel('y (m)')
axes[2].plot(dt*np.arange(x_res_3D.shape[1]), x_res_3D[2,:], label='Trajectory')
axes[2].plot(dt*np.arange(x_res_3D.shape[1]), xcom[1,:], ':', label='XcoM')
axes[2].plot(x0_array_t, x0_array_3D[2,:], 'co', label='Step Transition')
axes[2].plot(U0_res_lin_t, U0_array_lin[1,:], 'g-', label='U_0 Target')
axes[2].plot(U0_res_lin_t, U0_res_lin[1,:], 'r-', label='U_0 Result')
axes[2].set_xlabel('t (s)')
axes[2].set_ylabel('z (m)')
fig.legend()
plt.show()

plt.figure()
plt.plot(x_res_inertial[1,:], x_res_inertial[0,:], '-o', label='Trajectory')
plt.plot(xcom[1,:], xcom[0,:], ':', label='XcoM')
plt.plot(x0_array[1,:], x0_array[0,:], 'co', label='Step Transition')
plt.plot(U0_array[1,:], U0_array[0,:], 'g+', label='U_0 Target')
plt.plot(U0_res[1,:], U0_res[0,:], 'r+', label='U_0 Result')
#plt.plot(U0_viable[0,1,:], U0_viable[0,0,:], ':', label='U_0 Viable')
plt.xlabel('z (m)')
plt.ylabel('x (m)')
plt.legend()
plt.show()

ax = plt.figure().add_subplot(projection='3d')
ax.plot(x_res_3D[2,:], x_res_3D[0,:], x_res_3D[1,:], label='Trajectory')
ax.plot(x0_array_3D[2,:], x0_array_3D[0,:], x0_array_3D[1,:], 'co', label='Step Transition')
ax.plot(x_res_3D[2,0], x_res_3D[0,0], x_res_3D[1,0], 'mo', label='Start')
ax.plot(x_res_3D[2,-1], x_res_3D[0,-1], x_res_3D[1,-1], 'yx', label='End')
ax.set_xlabel('z (m)')
ax.set_ylabel('x (m)')
ax.set_zlabel('y (m)')
ax.legend()
plt.show()

# Animation

if animation == 1:
    fig = plt.figure(figsize=(14,6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

    gs_stacked = gs[0].subgridspec(3, 1, hspace=0.3)
    ax1 = fig.add_subplot(gs_stacked[0])
    ax2 = fig.add_subplot(gs_stacked[1], sharex=ax1)
    ax3 = fig.add_subplot(gs_stacked[2], sharex=ax1)

    ax3d = fig.add_subplot(gs[1], projection='3d')

    ax1.set_ylim(0.0, 5.0)
    ax2.set_ylim(0.95, 1.1)
    ax3.set_ylim(-0.06, 0.06)
    ax3d.set_xlim3d(-0.06, 0.06)
    ax3d.set_zlim3d(-0.01, l)

    ax1.set_ylabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax3.set_xlabel('t (s)')
    ax3.set_ylabel('z (m)')
    ax3d.set_xlabel('z (m)')
    ax3d.set_ylabel('x (m)')
    ax3d.set_zlabel('y (m)')

    ax1.plot(dt*np.arange(x_res_3D.shape[1]), x_res_3D[0,:], 'C0')
    ax1.plot(dt*np.arange(x_res_3D.shape[1]), xcom[0,:], 'C1:')
    ax1.plot(x0_array_t, x0_array_3D[0,:], 'co')
    ax1.plot(U0_res_lin_t, U0_array_lin[0,:], 'g-')
    ax1.plot(U0_res_lin_t, U0_res_lin[0,:], 'r-')

    ax2.plot(dt*np.arange(x_res_3D.shape[1]), x_res_3D[1,:], 'C0')
    ax2.plot(x0_array_t, x0_array_3D[1,:], 'co')

    ax3.plot(dt*np.arange(x_res_3D.shape[1]), x_res_3D[2,:], 'C0', label='Trajectory')
    ax3.plot(dt*np.arange(x_res_3D.shape[1]), xcom[1,:], 'C1:', label='XcoM')
    ax3.plot(x0_array_t, x0_array_3D[2,:], 'co', label='Step Transition')
    ax3.plot(U0_res_lin_t, U0_array_lin[1,:], 'g-', label='U_0 Target')
    ax3.plot(U0_res_lin_t, U0_res_lin[1,:], 'r-', label='U_0 Result')

    ax3d.plot(x_res_3D[2,:], x_res_3D[0,:], x_res_3D[1,:], 'C0:')
    ax3d.plot(x0_array_3D[2,:], x0_array_3D[0,:], x0_array_3D[1,:], 'co')
    ax3d.plot(x_res_3D[2,0], x_res_3D[0,0], x_res_3D[1,0], 'mo', label='Start')
    ax3d.plot(x_res_3D[2,-1], x_res_3D[0,-1], x_res_3D[1,-1], 'yx', label='End')

    ax1_timeindic, = ax1.plot([], [], 'k')
    ax2_timeindic, = ax2.plot([], [], 'k')
    ax3_timeindic, = ax3.plot([], [], 'k')

    ax3d_masstraj, = ax3d.plot([], [], [], 'C0')
    ax3d_leg, = ax3d.plot([], [], [], 'k')
    ax3d_mass, = ax3d.plot([], [], [], 'bo', label='Mass')
    ax3d_U0res, = ax3d.plot([], [], [], 'rx')
    ax3d_U0targ, = ax3d.plot([], [], [], 'gx')

    fig.legend(loc='lower right')

    def anim_func(n1):
        ax1_timeindic.set_data(dt*np.array([(np.arange(x_res_3D.shape[1]))[n1], (np.arange(x_res_3D.shape[1]))[n1]]), np.array([-1.0, 6.0]))
        ax2_timeindic.set_data(dt*np.array([(np.arange(x_res_3D.shape[1]))[n1], (np.arange(x_res_3D.shape[1]))[n1]]), np.array([0.9, 1.2]))
        ax3_timeindic.set_data(dt*np.array([(np.arange(x_res_3D.shape[1]))[n1], (np.arange(x_res_3D.shape[1]))[n1]]), np.array([-0.1, 0.1]))

        ax3d_masstraj.set_data(x_res_3D[2,:n1+1], x_res_3D[0,:n1+1])
        ax3d_masstraj.set_3d_properties(x_res_3D[1,:n1+1])

        ax3d_leg.set_data(np.array([U0_res_lin_interp[1,n1], x_res_3D[2,n1]]), np.array([U0_res_lin_interp[0,n1], x_res_3D[0,n1]]))
        ax3d_leg.set_3d_properties(np.array([0.0, x_res_3D[1,n1]]))

        ax3d_mass.set_data(np.array([x_res_3D[2,n1]]), np.array([x_res_3D[0,n1]]))
        ax3d_mass.set_3d_properties(np.array([x_res_3D[1,n1]]))

        ax3d_U0res.set_data(U0_res_lin[1, U0_res_lin_t <= dt*(np.arange(x_res_3D.shape[1]))[n1]], U0_res_lin[0, U0_res_lin_t <= dt*(np.arange(x_res_3D.shape[1]))[n1]])
        ax3d_U0res.set_3d_properties(np.zeros(len(U0_res_lin[0, U0_res_lin_t <= dt*(np.arange(x_res_3D.shape[1]))[n1]])))

        ax3d_U0targ.set_data(U0_array_lin[1, U0_res_lin_t <= dt*(np.arange(x_res_3D.shape[1]))[n1]], U0_array_lin[0, U0_res_lin_t <= dt*(np.arange(x_res_3D.shape[1]))[n1]])
        ax3d_U0targ.set_3d_properties(np.zeros(len(U0_res_lin[0, U0_res_lin_t <= dt*(np.arange(x_res_3D.shape[1]))[n1]])))

        return ax1_timeindic, ax2_timeindic, ax3_timeindic, ax3d_masstraj, ax3d_leg, ax3d_U0res, ax3d_U0targ

    anim = FuncAnimation(fig, anim_func, frames=np.arange(x_res_3D.shape[1]), interval=50)

    plt.tight_layout()
    plt.show()

    anim.save("walkingmodel_varperiod_animation_1.gif", writer="pillow", fps=20)



