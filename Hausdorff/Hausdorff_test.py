import numpy as np
from scipy.spatial.distance import directed_hausdorff, cdist
import matplotlib.pyplot as plt

######################### Standard Hausdorff ###############################
# Understanding test
A = np.array([[1.9, 1], [1.5, 0]])  # Points from Image A
B = np.array([[2, 1], [2, 0]])  # Points from Image B

# Compute Hausdorff distance (Standard)
hausdorff_distance = max(directed_hausdorff(A, B)[0], directed_hausdorff(B, A)[0])
print("Hausdorff Distance:", hausdorff_distance)

######################### Fractional Hausdorff ############################
# Fractional power for the metric (alpha between 0 and 1)
alpha = 0.5

# Compute pairwise distances between all points in A and B
pairwise_distances = cdist(A, B)

# Apply fractional power to distances
fractional_distances = pairwise_distances ** alpha

# Compute the directed fractional Hausdorff distance
directed_A_to_B = np.max(np.min(fractional_distances, axis=1))
directed_B_to_A = np.max(np.min(fractional_distances, axis=0))

# Fractional Hausdorff distance (symmetric)
fractional_hausdorff_distance = max(directed_A_to_B, directed_B_to_A)
print(f"Fractional Hausdorff Distance (alpha={alpha}):", fractional_hausdorff_distance)

######################### Plotting ###############################
plt.figure(figsize=(6, 6))

# Plot set A in blue
plt.scatter(A[:, 0], A[:, 1], color='blue', label='Set A', s=100, marker='o')

# Plot set B in red
plt.scatter(B[:, 0], B[:, 1], color='red', label='Set B', s=100, marker='x')

# Add labels and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Plot of Sets A and B')
plt.legend()

# Set equal scaling for axes
plt.gca().set_aspect('equal', adjustable='box')

# Add labels for Hausdorff distances at the bottom
plt.figtext(0.9, 0.02, f"Standard Hausdorff Distance: {hausdorff_distance:.2f}", wrap=True, horizontalalignment='center', fontsize=12)
plt.figtext(0.1, 0.01, f"Fractional Hausdorff Distance (α={alpha}): {fractional_hausdorff_distance:.2f}", wrap=True, horizontalalignment='center', fontsize=12)

# Show the plot
plt.grid(True)
plt.show()
