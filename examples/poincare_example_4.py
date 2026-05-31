import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Duffing system parameters
delta = 0.2     # damping
alpha = -1.0    # linear stiffness
beta = 1.0      # nonlinear stiffness
#gamma = 0.3     # forcing amplitude
#gamma = 0.5     # forcing amplitude
gamma = 0.8     # forcing amplitude
omega = 1.2     # forcing frequency

# Time setup
T = 2 * np.pi / omega  # forcing period
num_periods = 500
t_span = (0, num_periods * T)
t_eval = np.linspace(t_span[0], t_span[1], 50000)

# Duffing system as first-order ODE
def duffing(t, state):
    x, v = state
    dx = v
    dv = -delta * v - alpha * x - beta * x**3 + gamma * np.cos(omega * t)
    return [dx, dv]

# Initial condition
initial_state = [0.1, 0.0]

# Solve system
sol = solve_ivp(duffing, t_span, initial_state, t_eval=t_eval, rtol=1e-8)
x, v = sol.y
t = sol.t

# Sample at stroboscopic times (Poincaré section)
poincare_x = []
poincare_v = []

for n in range(1, num_periods):
    t_sample = n * T
    idx = np.searchsorted(t, t_sample)
    if idx < len(t):
        poincare_x.append(x[idx])
        poincare_v.append(v[idx])

# ---- Plotting ----

plt.figure(figsize=(10, 6))

# Plot full trajectory
plt.plot(x, v, lw=0.5, color='steelblue', label='Trajectory')

# Overlay Poincaré points
plt.plot(poincare_x, poincare_v, 'ro', markersize=3, label='Poincaré Points')

plt.title("Duffing Oscillator Phase Space with Poincaré Section")
plt.xlabel("x")
plt.ylabel("v = dx/dt")
plt.grid(True)
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()



