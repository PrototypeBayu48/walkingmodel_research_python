import numpy as np
import matplotlib.pyplot as plt

# Parameters
g = 9.81        # gravity (m/s^2)
r = 0.9         # restitution coefficient (0 < r < 1)
k = 0.05        # air resistance coefficient
v0 = 10.0       # initial velocity (m/s)
y0 = 0.0        # initial height
num_bounces = 20
dt = 0.001      # time step

# Lists to store results
time_list = []
height_list = []
v_impact_list = []

# Initial conditions
t = 0.0
v = v0
y = y0

for i in range(num_bounces):
    t_bounce = []
    y_bounce = []

    while y >= 0:
        # Save data
        t_bounce.append(t)
        y_bounce.append(y)

        # Update using Euler method
        a = -g - k * v * abs(v)  # drag force depends on velocity direction
        v += a * dt
        y += v * dt
        t += dt

    # Append bounce data
    time_list.extend(t_bounce)
    height_list.extend(y_bounce)

    # Store post-impact velocity
    v_impact_list.append(abs(v))  # post-impact speed before applying r

    # Apply impact rule
    v = -r * v
    y = 0.0

# ---- Plotting ----

plt.figure(figsize=(12, 5))

# Plot 1: Height vs Time
plt.subplot(1, 2, 1)
plt.plot(time_list, height_list)
plt.title("Bouncing Ball with Air Resistance")
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.grid(True)

# Plot 2: Poincaré Map (velocity after each bounce)
plt.subplot(1, 2, 2)
plt.plot(range(1, len(v_impact_list)+1), v_impact_list, 'o-', color='red')
plt.title("Poincaré Map: Post-Impact Velocities with Drag")
plt.xlabel("Bounce Number (n)")
plt.ylabel("Velocity After Impact (vₙ) [m/s]")
plt.grid(True)

plt.tight_layout()
plt.show()



