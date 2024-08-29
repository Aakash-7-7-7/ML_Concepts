from sklearn.datasets import make_classification
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Generate linearly separable data
X, y = make_classification(n_features=3, n_classes=2, n_samples=100, n_clusters_per_class=1, n_informative=2, n_redundant=0, n_repeated=0, random_state=42)

# Fit polynomial SVM
svc = SVC(kernel='poly', degree=3, coef0=1, decision_function_shape='ovo')
svc.fit(X, y)

# Plotting decision boundary in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of data points
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='plasma')

# Create a grid of points in the feature space
x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)
xx, yy = np.meshgrid(x_range, y_range)

# Choose a specific z value for the decision boundary
z_value = np.mean(X[:, 2])  # Use the mean of z values for a slice

# Calculate decision function over the grid
grid_points = np.c_[xx.ravel(), yy.ravel(), np.full_like(xx.ravel(), z_value, dtype=float)]
decision_values = svc.decision_function(grid_points)
zz = decision_values.reshape(xx.shape)

# Plot the decision boundary surface
ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')

# Labels and title
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
plt.title('Polynomial Kernel SVM Decision Boundary (Single Slice)')
plt.show()

