import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Lorenz system parameters
sigma = 10
rho = 28
beta = 8 / 3

def lorenz(t, state):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Time setup
t_span = (0, 100)
t_eval = np.linspace(t_span[0], t_span[1], 100000)

# Initial condition
initial_state = [1.0, 1.0, 1.0]

# Solve ODE
sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval, rtol=1e-8)
x, y, z = sol.y
t = sol.t

# Poincaré section: z = 25, upward crossing
z_section = 25
x_poincare = []
y_poincare = []

for i in range(1, len(z) - 1):
    if z[i-1] < z_section <= z[i] and (z[i+1] - z[i]) > 0:
        # Linear interpolation for accuracy
        dz = z[i] - z[i-1]
        alpha = (z_section - z[i-1]) / dz if dz != 0 else 0
        x_cross = x[i-1] + alpha * (x[i] - x[i-1])
        y_cross = y[i-1] + alpha * (y[i] - y[i-1])
        x_poincare.append(x_cross)
        y_poincare.append(y_cross)

# ---- Plotting ----

fig = plt.figure(figsize=(14, 6))

# Plot 1: 3D trajectory
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot(x, y, z, lw=0.5, color='navy')
ax1.set_title("Lorenz Attractor (3D Trajectory)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
ax1.view_init(elev=30, azim=120)

# Plot 2: Poincaré map
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(x_poincare, y_poincare, 'r.', markersize=2)
ax2.set_title("Poincaré Map (z = 25, upward crossing)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.grid(True)
ax2.axis("equal")

plt.tight_layout()
plt.show()



