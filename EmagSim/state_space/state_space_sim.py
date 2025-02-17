import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import time

"""
This is an itial attempt at simulating
different integration techniques can be tried but many produce some oscilations
"""

# Transmission line parameters
R_m = 0  # Resistance per unit length (ohms/m)
L_m = 2.5e-7  # Inductance per unit length (H/m)
C_m = 1e-10  # Capacitance per unit length (F/m)
G_m = 0  # Conductance per unit length (S/m)

R_load = 100  # Load resistance (ohms)

# dt = 1e-6  # Time step (s)
Length = 100 # Line length (m)
N = 400  # Number of sections
dz = Length / N  # Section length (m)

# Calculate the maximum time step based on CFL condition
wave_speed = 1 / np.sqrt(L_m * C_m)
dt_max = 1 * dz / wave_speed

# num_steps = 100
end_time = 2e-6
frames = 300

# TODO make real documentation on the state space derivation
# For now it is in my notebook

# State Space Repeated values
a = -1 * (R_m * dz) / (L_m * dz)
b = -1 / (L_m * dz)
c = 1 / (C_m * dz)
d = -1 * (G_m * dz) / (C_m * dz)
e = 1 / (L_m * dz)
f = -1 / (C_m * dz)

# State Space Matrices
blocks = np.array([[a, b], [c, d]])
A = np.kron(np.eye(N), blocks)

# Place f and e matrices in the upper corner between the 2 blocks
for i in range(N - 1):
    A[i * 2 + 1, i * 2 + 2] = f
    A[i * 2 + 2, i * 2 + 1] = e

# Add resistive load at the end
A[-1, -1] += -1 / (R_load * C_m * dz)

B = np.zeros(N * 2,)
B[0] = 1 / (L_m * dz)

print("A", A)
print("B", B)

# Code the input signal
U = 5 # 5 Volt voltage source

# Initialize state
# x = np.full((N * 2, num_steps + 1), np.nan, dtype=np.float64)
# x[:, 0] = np.zeros(N * 2, dtype=np.float64)
x_0 = np.zeros(N * 2, dtype=np.float64)
t_span = (0, end_time)

# for i in range(num_steps):
#     x[:, i+1] = x[:, i] + dt* (np.dot(A, x[:, i]) + B * U)

def f(t, x):
    return np.dot(A, x) + B * U


sol = solve_ivp(f, t_span, x_0, method='Radau', t_eval=np.linspace(0, end_time, frames), max_step=dt_max)

np.save("integraion.npy", sol)

print(sol)

y = sol.y
t = sol.t

x = y[1::2, 0]

fig, ax = plt.subplots()
plt.tight_layout()
line, = ax.plot(x)

def update(frame):
    # Update the state
    x = y[1::2, frame]
    # Update the plot
    line.set_ydata(x)
    # Update the y-axis limits
    ax.set_ylim(min(x) - 0.1 * abs(min(x)), max(x) + 0.1 * abs(max(x)))
    ax.figure.canvas.draw()
    return line,

ani = FuncAnimation(fig, update, frames=range(frames), blit=True)

plt.show()
