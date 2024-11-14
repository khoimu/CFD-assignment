import numpy as np
import math
import matplotlib.pyplot as plt

# Parameters
m = 81  # number of points along x direction
n = 81  # number of points along y direction
p = (m - 2) * (n - 2)  # number of interior points
dx = 1.0 / (m - 1)
dy = 1.0 / (n - 1)
beta = dx / dy
Re = 100.0  # Reynolds number

# Initialize arrays
psi = np.zeros((m, n))
omega = np.zeros((m, n))
u = np.zeros((m, n))
v = np.zeros((m, n))
psi_prev = np.zeros((m, n))
omega_prev = np.zeros((m, n))

# Initialization and Boundary conditions
for j in range(n):
    for i in range(m):
        if j == 0:
            u[i, j] = 0.0
            v[i, j] = 0.0
            psi[i, j] = 0.0
        elif j == n - 1:
            u[i, j] = 1.0
            v[i, j] = 0.0
            psi[i, j] = 0.0
        elif i == 0:
            u[i, j] = 0.0
            v[i, j] = 0.0
            psi[i, j] = 0.0
        elif i == m - 1:
            u[i, j] = 0.0
            v[i, j] = 0.0
            psi[i, j] = 0.0
        else:
            u[i, j] = 0.0
            v[i, j] = 0.0
            psi[i, j] = 0.0

# Gauss-Siedel Method
iteration = 0
while True:
    for j in range(n):
        for i in range(m):
            psi_prev[i, j] = psi[i, j]
            omega_prev[i, j] = omega[i, j]

    # Solving for stream function
    for j in range(1, n - 1):
        for i in range(1, m - 1):
            psi[i, j] = (0.5 / (1.0 + beta ** 2)) * (
                    psi[i + 1, j] + psi[i - 1, j] +
                    beta ** 2 * (psi[i, j + 1] + psi[i, j - 1]) +
                    dx ** 2 * omega[i, j])

    # Solving for vorticity
    for j in range(1, n - 1):
        for i in range(1, m - 1):
            omega[i, j] = (0.5 / (1.0 + beta ** 2)) * (
                    (1.0 - (psi[i, j + 1] - psi[i, j - 1]) * (beta * Re / 4.0)) * omega[i + 1, j] +
                    (1.0 + (psi[i, j + 1] - psi[i, j - 1]) * (beta * Re / 4.0)) * omega[i - 1, j] +
                    (1.0 + (psi[i + 1, j] - psi[i - 1, j]) * (Re / (4.0 * beta))) * beta ** 2 * omega[i, j + 1] +
                    (1.0 - (psi[i + 1, j] - psi[i - 1, j]) * (Re / (4.0 * beta))) * beta ** 2 * omega[i, j - 1])

    # Update vorticity at boundaries
    for j in range(n):
        for i in range(m):
            if j == 0:
                omega[i, j] = (2.0 / dy ** 2) * (psi[i, j] - psi[i, j + 1])
            elif j == n - 1:
                omega[i, j] = (2.0 / dy ** 2) * (psi[i, j] - psi[i, j - 1]) - (2.0 / dy) * u[i, j]
            elif i == 0:
                omega[i, j] = (2.0 / dx ** 2) * (psi[i, j] - psi[i + 1, j])
            elif i == m - 1:
                omega[i, j] = (2.0 / dx ** 2) * (psi[i, j] - psi[i - 1, j])

    # Error calculation
    error_psi = np.sum((psi - psi_prev) ** 2) / p
    error_omega = np.sum((omega - omega_prev) ** 2) / p
    error_psi = math.sqrt(error_psi)
    error_omega = math.sqrt(error_omega)

    print("iteration=%d\t" % iteration, end='')
    print("error_psi=%.101f\terror_omega=%.101f\n" % (error_psi, error_omega), end='')
    iteration += 1

    if error_psi <= 1.0e-5 and error_omega <= 1.0e-5:
        break

# Update velocities u and v
for j in range(1, n - 1):
    for i in range(1, m - 1):
        u[i, j] = 0.5 / dy * (psi[i, j + 1] - psi[i, j - 1])
        v[i, j] = -0.5 / dx * (psi[i + 1, j] - psi[i - 1, j])

def plot_psi_contours():
    plt.figure(figsize=(10, 8))
    X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, m)) 
    plt.contour(X, Y, psi.T, levels=20, cmap='viridis')  
    plt.colorbar(label='Psi') #khoimu singh
    plt.title(f'Stream Function (Psi) Contours (Re={Re}, Grid Size: {m}x{n})')
    plt.xlabel('X') 
    plt.ylabel('Y')  
    plt.show()

# Function to plot omega contours
def plot_omega_contours():
    plt.figure(figsize=(10, 8))
    X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, m)) 
    plt.contour(X, Y, omega.T, levels=20, cmap='viridis')  
    plt.colorbar(label='Omega')
    plt.title(f'Vorticity (Omega) Contours (Re={Re}, Grid Size: {m}x{n})')
    plt.xlabel('X') 
    plt.ylabel('Y')  
    plt.show()

# Function to plot velocity vectors
def plot_velocity_vectors():
    plt.figure(figsize=(10, 8))
    X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, m))  
    plt.quiver(Y, X, v.T, u.T, scale=20)  
    plt.title(f'Velocity Vector Field (Re={Re}, Grid Size: {m}x{n}) ')
    plt.xlabel('X')  
    plt.ylabel('Y')  
    plt.show()

# Plot contours for psi, omega, and velocity vectors
plot_psi_contours()
plot_omega_contours()
plot_velocity_vectors()
