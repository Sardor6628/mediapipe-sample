import cv2
import time
import mediapipe as mp
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Choose camera index (0 for default camera, 1 for the next camera, etc.)
camera_index = 0

# Initialize MediaPipe's holistic model for pose, face, and hand tracking
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,  # Minimum confidence value for detection to be considered successful
    min_tracking_confidence=0.5    # Minimum confidence value for tracking to be considered successful
)

# Initialize video capture with the specified camera index
capture = cv2.VideoCapture(camera_index)

# Variables to calculate frames per second (FPS)
previousTime = 0  # Previous time to calculate FPS
currentTime = 0   # Current time to calculate FPS

# Set up the matplotlib figure and 3D axis for real-time updating
plt.ion()  # Turn on interactive mode for real-time updates
fig = plt.figure()  # Create a new figure for plotting
ax = fig.add_subplot(111, projection='3d')  # Add a 3D subplot

# Initialize scatter plot for landmarks
scatter = ax.scatter([], [], [], c='b', marker='o')  # Create an empty scatter plot
lines = []  # List to store line objects connecting landmarks

# Set axis labels
ax.set_xlabel('X Label')  # Label for X-axis
ax.set_ylabel('Y Label')  # Label for Y-axis
ax.set_zlabel('Z Label')  # Label for Z-axis

# Function to update the 3D plot with new landmarks
def update_3d_plot(landmarks):
    if landmarks:  # Check if landmarks are available
        # Extract X, Y, Z coordinates of each landmark
        x_vals = [landmark.x for landmark in landmarks.landmark]
        y_vals = [landmark.y for landmark in landmarks.landmark]
        z_vals = [landmark.z for landmark in landmarks.landmark]

        # Update the scatter plot with new coordinates
        scatter._offsets3d = (x_vals, y_vals, z_vals)

        # Remove old lines connecting landmarks
        for line in lines:
            line.remove()
        lines.clear()  # Clear the list of lines

        # Draw new lines connecting landmarks as per MediaPipe connections
        for connection in mp_holistic.POSE_CONNECTIONS:
            start_idx, end_idx = connection  # Indices of the start and end points of the line
            start = landmarks.landmark[start_idx]  # Start landmark
            end = landmarks.landmark[end_idx]      # End landmark
            line = ax.plot(
                [start.x, end.x],  # X coordinates
                [start.y, end.y],  # Y coordinates
                [start.z, end.z],  # Z coordinates
                c='g'  # Color of the line
            )[0]
            lines.append(line)  # Add line to the list

        plt.draw()  # Update the plot with new data
        plt.pause(0.001)  # Pause briefly to allow for the plot to update

# Main loop to capture video frames and process them
while capture.isOpened():
    # Capture frame by frame from the video feed
    ret, frame = capture.read()
    if not ret:  # If frame capturing fails, exit the loop
        print("Failed to grab frame")
        break

    # Resize the frame for better visualization
    frame = cv2.resize(frame, (800, 600))

    # Convert the frame from BGR (OpenCV format) to RGB (MediaPipe format)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Make predictions using the holistic model (pose, face, hand landmarks)
    image.flags.writeable = False  # Mark image as not writeable to improve performance
    results = holistic_model.process(image)  # Process the image and extract landmarks
    image.flags.writeable = True   # Re-enable write access to the image

    # Convert the processed image back from RGB to BGR (for OpenCV display)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # If pose landmarks are detected, update the 3D plot with these landmarks
    if results.pose_landmarks:
        update_3d_plot(results.pose_landmarks)

    # Calculate and display the frames per second (FPS) on the image
    currentTime = time.time()  # Get current time
    fps = 1 / (currentTime - previousTime)  # Calculate FPS
    previousTime = currentTime  # Update previous time to current time
    cv2.putText(image, str(int(fps)) + " FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)  # Display FPS

    # Display the resulting image with landmarks and FPS
    cv2.imshow("Body and Hand Landmarks", image)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the capture object and close all OpenCV windows
capture.release()
cv2.destroyAllWindows()