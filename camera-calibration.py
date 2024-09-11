import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# Define the dimensions of the chessboard (number of corners per row and column)
chessboard_size = (9, 6)

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Load all calibration images
images = glob.glob('calibration_images/*.png')

for image_file in images:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If found, add object points and image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw the chessboard corners on the image
        img_with_corners = cv2.drawChessboardCorners(img, chessboard_size, corners, ret)

        # Display the image with corners using matplotlib
        plt.imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
        plt.title('Chessboard Corners')
        plt.show()

# Perform camera calibration to get the camera matrix, distortion coefficients, etc.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print the camera matrix and distortion coefficients
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

# Calculate the reprojection error to assess the accuracy of the calibration
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print("Total reprojection error: ", mean_error / len(objpoints))

# Load an example image to undistort
example_image = cv2.imread('calibration_images/Im_L_1.png')

# Undistort the image using the obtained camera matrix and distortion coefficients
undistorted_img = cv2.undistort(example_image, mtx, dist, None, mtx)

# Display the original and undistorted image side by side using matplotlib
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title("Undistorted Image")
plt.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))

plt.show()