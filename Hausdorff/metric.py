import cv2  # OpenCV for image processing
import numpy as np
from scipy.spatial.distance import directed_hausdorff, cdist
import matplotlib.pyplot as plt

# Load the images
image_1 = cv2.imread("first_test_copy/5.png")
image_2 = cv2.imread("first_test_copy/15.png")


# Convert to grayscale
gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

# Threshold the image to get the contour (binary mask)
_, binary_1 = cv2.threshold(gray_1, 1, 255, cv2.THRESH_BINARY)
_, binary_2 = cv2.threshold(gray_2, 1, 255, cv2.THRESH_BINARY)

# Find coordinates of non-zero pixels (contour points)
points_1 = np.column_stack(np.where(binary_1 > 0))
points_2 = np.column_stack(np.where(binary_2 > 0))

# Convert to (x, y) format
points_1 = np.flip(points_1, axis=1)  # Flip columns to get (x, y)
points_2 = np.flip(points_2, axis=1)  # Flip columns to get (x, y)

#np.set_printoptions(threshold=np.inf)
# Print the resulting set of points_1
print(f"Size of the array 1: {points_1.size}, Size of the array 2: {points_2.size}")
print(f"Total points_1:{points_1}, Total points_2:{points_2}\n")
print(f"Set of points_1 (first 10):{points_1[:10]}, Set of points_2 (first 10):{points_2[:10]}\n")

# Plot the points_1 for verification
# plt.figure(figsize=(6, 6))
# plt.scatter(points_1[:, 0], points_1[:, 1], color='magenta', s=5, label='Contour Points')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("Contour Points from Image")
# plt.legend()
# plt.grid(True)
# plt.show()


######################### Standard Hausdorff ###############################
# Understanding test
A = points_1  # Points from Image A
B = points_2   # Points from Image B

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
plt.scatter(B[:, 0], B[:, 1], color='red', label='Set B', s=100, marker='o')

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
