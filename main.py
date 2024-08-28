import cv2
import time
import mediapipe as mp
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider  # Import Button and Slider widgets

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

# Set up the matplotlib figure and 3D axis for real-time updating
plt.ion()  # Turn on interactive mode for real-time updates
fig = plt.figure(figsize=(8, 6))  # Create a new figure for plotting
ax = fig.add_subplot(111, projection='3d')  # Add a 3D subplot

# Initialize scatter plot for landmarks
scatter = ax.scatter([], [], [], c='b', marker='o')  # Create an empty scatter plot
lines = []  # List to store line objects connecting landmarks

# Set axis labels
ax.set_xlabel('X Label')  # Label for X-axis
ax.set_ylabel('Y Label')  # Label for Y-axis
ax.set_zlabel('Z Label')  # Label for Z-axis

# Set axis limits to maintain a consistent view
ax.set_xlim(0, 1)   # Limit X axis
ax.set_ylim(-1, 1)  # Inverted Y-axis: head at the top, legs at the bottom
ax.set_zlim(0, 1)   # Limit Z axis (depth)

# Remove grid lines
ax.grid(False)

# Default view settings for the 3D plot
default_elevation = 270  # Elevation angle
default_azimuth = 90     # Azimuth angle
default_roll = 180       # Roll angle
ax.view_init(elev=default_elevation, azim=default_azimuth, roll=default_roll)  # Apply default view

# Function to update the 3D plot with new landmarks
def update_3d_plot(landmarks):
    if landmarks:  # Check if landmarks are available
        # Extract X, Y, Z coordinates of each landmark
        x_vals = [landmark.x for landmark in landmarks.landmark]
        y_vals = [-landmark.y for landmark in landmarks.landmark]  # Invert Y for natural orientation
        z_vals = [landmark.z for landmark in landmarks.landmark]  # Use Z values as is

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
                [start.x_, end.x],  # X coordinates
                [-start.y, -end.y],  # Invert Y coordinates for correct orientation
                [start.z, end.z],  # Z coordinates
                c='g'  # Color of the line
            )[0]
            print(line,[start.x, end.x],  # X coordinates
                [-start.y, -end.y],  # Invert Y coordinates for correct orientation
                [start.z, end.z])
            lines.append(line)  # Add line to the list

        plt.draw()  # Update the plot with new data
        plt.pause(0.001)  # Pause briefly to allow for the plot to update

# Function to update the view using sliders
# def update_view(val):
#     elev = slider_elev.val
#     azim = slider_azim.val
#     roll = slider_roll.val
#     ax.view_init(elev=elev, azim=azim)
#     ax.roll = roll
#     plt.draw()

# Function to exit the application
def exit_app(event):
    global running
    running = False
    plt.close(fig)

# Add sliders to control the view angles
axcolor = 'lightgoldenrodyellow'
# ax_elev = plt.axes([0.15, 0.02, 0.65, 0.03], facecolor=axcolor)
# ax_azim = plt.axes([0.15, 0.06, 0.65, 0.03], facecolor=axcolor)
# ax_roll = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)

# slider_elev = Slider(ax_elev, 'Elev', 0, 360, valinit=default_elevation)
# slider_azim = Slider(ax_azim, 'Azim', 0, 360, valinit=default_azimuth)
# slider_roll = Slider(ax_roll, 'Roll', 0, 360, valinit=default_roll)
#
# # Connect the sliders to the update_view function
# slider_elev.on_changed(update_view)
# slider_azim.on_changed(update_view)
# slider_roll.on_changed(update_view)

# Add an Exit button to terminate the application
button_ax = plt.axes([0.8, 0.9, 0.1, 0.075])
exit_button = Button(button_ax, 'Exit', color='red', hovercolor='darkred')
exit_button.on_clicked(exit_app)

# Variable to control the main loop
running = True

# Main loop to capture video frames and process them
while running and capture.isOpened():
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

    # If pose landmarks are detected, update the 3D plot with these landmarks
    if results.pose_landmarks:
        update_3d_plot(results.pose_landmarks)

    # Optionally, display the video feed (if needed for reference)
    # Position the video window at (100, 100) on the screen
    # cv2.imshow("Video Feed", frame)
    # cv2.moveWindow("Video Feed", 100, 100)  # Move window to specified location

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the capture object and close all OpenCV windows
capture.release()
cv2.destroyAllWindows()