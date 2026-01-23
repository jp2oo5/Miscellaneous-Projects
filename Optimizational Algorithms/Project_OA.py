# Imports
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D

# ----------------------HYPERPARAMETERS----------------------------------------

# Initial and Final Positions
initial_position = np.array([0, 0, 0])
final_position = np.array([20, 20, 0])

# Number of Robots
n_robots = 5
# Time Step
time_step = 1
# Number of Time Steps
n_steps = 50
# Robot Weight
mass = 1  # (assumed equal for all robots)
# Drag Coefficient
drag = 0.1
# Gravitational Acceleration
gravity = np.array([0, 0, -9.81])
# Maximum Altitude
max_altitude = 10
# Communication Radius
coms_radius = 3
# Final Time
final_time = time_step * n_steps

# Making sure our randomization stays constant
random.seed(23)

# Waypoints
waypoint_list = []
for robot in range(n_robots):
    robot_waypoints = []
    waypoint = [random.randrange(0, 20), random.randrange(0, 20), random.randrange(0, max_altitude)]
    robot_waypoints.append(waypoint)
    waypoint_list.append(robot_waypoints)
waypoints = np.array(waypoint_list)

# Waypoints Times (hardcoded)
waypoint_times = np.array([
    [[7]],
    [[10]],
    [[2]],
    [[4]],
    [[16]]
])

# Interactive Bias
bias = float(input("Choose an float between 0 and 1 (0 fully optimizes distance, 1 fuel consumption):\t"))

# ---------------------------OPTIMIZATION VARIABLES----------------------------
positions = cp.Variable((n_robots * n_steps, 3))
velocities = cp.Variable((n_robots * n_steps, 3))
thrusts = cp.Variable((n_robots * n_steps, 3))
# positions[x * n_steps + t, axis] is robot x, at time step t, on axis (x,y,z)/(0,1,2)
# Same for velocities and thrusts

# ---------------------------------CONSTRAINTS---------------------------------

constraints = []

# Initial and final positions
for robot in range(n_robots):
    constraints += [positions[robot * n_steps + 0] == initial_position]
    constraints += [positions[robot * n_steps + n_steps - 1] == final_position]

# Newton's Law
for step in range(1, n_steps):
    previous = step - 1
    for robot in range(n_robots):
        #
        # Velocity
        constraints += [velocities[robot * n_steps + step] == (time_step / mass) * (thrusts[robot * n_steps + previous]
              - drag*velocities[robot * n_steps + previous] + mass * gravity) + velocities[robot * n_steps + previous]]
        # Position
        constraints += [positions[robot * n_steps + step] == time_step * velocities[robot * n_steps + previous] +
                        positions[robot * n_steps + previous]]

# Wireless Communication (distance between robots)
for step in range(n_steps):
    for robot_i in range(n_robots):
        for robot_j in range(robot_i + 1, n_robots):
            constraints += [cp.norm(positions[robot_i * n_steps + step] - positions[robot_j * n_steps + step]) <= coms_radius]


# Maximum altitude
for step in range(n_steps):
    for robot in range(n_robots):
        constraints += [positions[robot * n_steps + step][2] <= max_altitude]

# Waypoints
for robot in range(n_robots):
    for j in range(len(waypoints[robot])):
        constraints += [positions[robot * n_steps + waypoint_times[robot]] == waypoints[robot][j]]


# ---------------------------------Cost Functions---------------------------------

# Fuel Consumption
fuel_cost = cp.sum(cp.norm(thrusts, axis=1))

# Distance Traveled
total_distance = 0
for step in range(1, n_steps):
    previous = step - 1
    for robot in range(n_robots):
        total_distance += cp.norm(positions[robot * n_steps + step] - positions[robot * n_steps + previous])

# Choose weight of each cost function
objective = cp.Minimize(bias * fuel_cost + (1 - bias) * total_distance)

# Solver
problem = cp.Problem(objective, constraints)
result = problem.solve(solver=cp.SCS)
print(f"Solver Status: {problem.status}")

# Results
p_values = positions.value
v_values = velocities.value
t_values = thrusts.value

# Plot Travel
colours = ["purple", "blue", "red", "green", "orange"]
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(xs=initial_position[0], ys=initial_position[1], zs=initial_position[2], c='pink', marker='^', s=50.0,
           label="Initial Position")
ax.scatter(xs=final_position[0], ys=final_position[1], zs=final_position[2], c='black', marker='^', s=50.0,
           label="Final Position")
for robot in range(n_robots):
    ax.plot(p_values[robot*50:robot*50+50, 0], p_values[robot*50:robot*50+50, 1], p_values[robot*50:robot*50+50, 2],
            c=colours[robot],linestyle="dashed", label=f"Trajectory of Robot {robot}")
    ax.scatter(xs=waypoints[robot][0][0], ys=waypoints[robot][0][1], zs=waypoints[robot][0][2], c=colours[robot],
               marker='*', s=50, label=f"Waypoint Robot {robot}")
    ax.legend()
    ax.grid()
fig.tight_layout()
plt.title(f"Robot Trajectories (bias:{bias})")
plt.show()

# Plot Thrust
fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(10, 10))
for robot in range(n_robots):
    ax[robot].plot([time for time in range(n_steps)],
                   [(t_values[robot*n_steps+time][0]**2+t_values[robot*n_steps+time][1]**2+t_values[robot*n_steps+time][2]**2)**0.5 for time in range(n_steps)])
    ax[robot].set_title(f"Thrust Robot {robot} (bias:{bias})")
    ax[robot].set_xlabel("Time")
    ax[robot].set_ylabel("Thrust")
fig.tight_layout()
plt.show()

# Calculate and Plot Cumulative Fuel Consumption
fuel = []
for robot in range(n_robots):
    for time in range(n_steps):
        if time == 0:
            fuel += [(t_values[robot * n_steps + time][0] ** 2 + t_values[robot * n_steps + time][1] ** 2
                      + t_values[robot * n_steps + time][2] ** 2) ** 0.5]
        else:
            fuel += [fuel[-1] + (t_values[robot*n_steps+time][0]**2 + t_values[robot*n_steps+time][1]**2
                                 + t_values[robot*n_steps+time][2]**2)**0.5]

fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(10, 10))
for robot in range(n_robots):
    ax[robot].plot([time for time in range(n_steps)],
                   fuel[robot*n_steps:robot*n_steps+n_steps])
    ax[robot].set_title(f"Cumulative Fuel Consumption Robot {robot} (bias:{bias})")
    ax[robot].set_xlabel("Time")
    ax[robot].set_ylabel("Fuel Consumption")
fig.tight_layout()
plt.show()

# Pairwise Distance Calculation and Plot
pair_distance = []
x = 1
while n_robots-x != 0:
    for robot in range(x, n_robots):
        pair_distance += [sum((p_values[(x-1)*n_steps+time][dimension]-p_values[robot*n_steps+time][dimension])**2
                              for dimension in range(3))**0.5 for time in range(n_steps)]
        print(f"{x,robot}")
    x += 1

# pair_distance order, we have n_steps values for each pair:
# 01,02,03,04,12,13,14,23,24,34

fig, ax = plt.subplots(nrows=10, ncols=1, figsize=(10, 10))
x = 1
idx = 0
while n_robots-x != 0:
    for robot in range(x, n_robots):
        ax[idx].plot([time for time in range(n_steps)], pair_distance[idx*n_steps:idx*n_steps+n_steps])
        ax[idx].set_title(f"Pairwise Distance Robots [{x-1}, {robot}] (bias:{bias})")
        ax[idx].set_xlabel("Time")
        ax[idx].set_ylabel("Distance")
        idx += 1
    x += 1
fig.tight_layout()
plt.show()
