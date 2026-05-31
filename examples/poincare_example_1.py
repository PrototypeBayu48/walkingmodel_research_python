import numpy as np
import matplotlib.pyplot as plt

# Parameters
g = 9.81        # gravity (m/s^2)
r = 0.9         # restitution coefficient (0 < r < 1)
v0 = 10.0       # initial velocity (m/s)
num_bounces = 20

# Time resolution for smooth plotting
dt = 0.001

# Lists to store results
time_list = []
height_list = []
v_impact_list = []

# Initial conditions
t = 0.0
v = v0
y = 0.0

for i in range(num_bounces):
    # Time until ball reaches max height and falls back to y=0
    t_flight = 2 * v / g
    t_vals = np.arange(t, t + t_flight, dt)
    y_vals = v * (t_vals - t) - 0.5 * g * (t_vals - t)**2

    # Save data for plotting
    time_list.extend(t_vals)
    height_list.extend(y_vals)

    # Save post-impact velocity for Poincaré map
    v_impact_list.append(v)

    # Update for next bounce
    v = r * v  # reduce velocity due to impact
    t += t_flight
    y = 0.0

# ---- Plotting ----

plt.figure(figsize=(12, 5))

# Plot 1: Height vs Time
plt.subplot(1, 2, 1)
plt.plot(time_list, height_list)
plt.title("Bouncing Ball Height Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.grid(True)

# Plot 2: Poincaré Map (velocity after each bounce)
plt.subplot(1, 2, 2)
plt.plot(range(1, len(v_impact_list)+1), v_impact_list, 'o-', color='orange')
plt.title("Poincaré Map: Post-Impact Velocities")
plt.xlabel("Bounce Number (n)")
plt.ylabel("Velocity After Impact (vₙ) [m/s]")
plt.grid(True)

plt.tight_layout()
plt.show()



