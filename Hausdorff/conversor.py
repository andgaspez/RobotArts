import cv2  # OpenCV for image processing
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("first_test_copy/1.png")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the image to get the contour (binary mask)
_, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# Find coordinates of non-zero pixels (contour points)
points = np.column_stack(np.where(binary > 0))

# Convert to (x, y) format
points = np.flip(points, axis=1)  # Flip columns to get (x, y)

#np.set_printoptions(threshold=np.inf)
# Print the resulting set of points
print(f"Size of the array: {points.size}, and type: {type(points)}")
print(f"Total points:{points}\n")
print(f"Set of points (first 10):{points[:10]}.\n")

# Plot the points for verification
plt.figure(figsize=(6, 6))
plt.scatter(points[:, 0], points[:, 1], color='magenta', s=5, label='Contour Points')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Contour Points from Image")
plt.legend()
plt.grid(True)
plt.show()


