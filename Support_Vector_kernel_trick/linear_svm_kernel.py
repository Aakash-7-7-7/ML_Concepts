import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Gaussian Kernel Function
def gaussian_kernel(x, l, sigma):
    return np.exp(-np.linalg.norm(x - l) ** 2 / (2 * sigma ** 2))

# Create a grid of points
x1, x2 = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
X = np.array([x1.ravel(), x2.ravel()]).T

# Landmark at the origin
landmark = np.array([0, 0])
sigma = 1.0

# Compute kernel values for each point on the grid
Z = np.array([gaussian_kernel(x, landmark, sigma) for x in X]).reshape(x1.shape)

# Plot the 3D surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1, x2, Z, cmap='viridis')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Kernel Value')

plt.title("Gaussian (RBF) Kernel Visualization")
plt.show()

