import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
import csv

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Set up matplotlib for 3D visualization
plt.ion()  # Turn on interactive mode
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Set axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Set axis limits
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(-1, 1)

# Initialize scatter plot for landmarks and list for lines
scatter = ax.scatter([], [], [], c='b', marker='o')
lines = []  # To store line objects

# Default view settings for the 3D plot
default_elevation = 90  # Elevation angle
default_azimuth = 90   # Azimuth angle
default_roll = 180     # Roll angle
current_elevation = default_elevation
current_azimuth = default_azimuth
current_roll = default_roll

ax.view_init(elev=current_elevation, azim=current_azimuth, roll=current_roll)

# Display text for current elevation, azimuth, and roll
text_elev = ax.text2D(0.05, 0.95, f"Elev: {current_elevation}", transform=ax.transAxes)
text_azim = ax.text2D(0.05, 0.90, f"Azim: {current_azimuth}", transform=ax.transAxes)
text_roll = ax.text2D(0.05, 0.85, f"Roll: {current_roll}", transform=ax.transAxes)

# List to store 3D landmarks and view angles for recording
recorded_data = []

# Function to update 3D view based on user input
def adjust_elevation(event):
    global current_elevation, text_elev
    current_elevation = (current_elevation + event.step) % 360
    ax.view_init(elev=current_elevation, azim=current_azimuth, roll=current_roll)
    text_elev.set_text(f"Elev: {current_elevation}")
    plt.draw()

def adjust_azimuth(event):
    global current_azimuth, text_azim
    current_azimuth = (current_azimuth + event.step) % 360
    ax.view_init(elev=current_elevation, azim=current_azimuth, roll=current_roll)
    text_azim.set_text(f"Azim: {current_azimuth}")
    plt.draw()

def adjust_roll(event):
    global current_roll, text_roll
    current_roll = (current_roll + event.step) % 360
    ax.view_init(elev=current_elevation, azim=current_azimuth, roll=current_roll)
    text_roll.set_text(f"Roll: {current_roll}")
    plt.draw()

# Flag to control the loop
running = True

# Function to exit the application and save recorded data
def exit_app(event):
    global running
    running = False
    plt.close(fig)
    # Save recorded landmarks and view angles to a CSV file
    with open('landmarks_and_view_angles.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # Write headers
        headers = ['Frame', 'Elevation', 'Azimuth', 'Roll']
        for i in range(33):  # 33 landmarks for MediaPipe Pose
            headers.extend([f'Landmark_{i}_x', f'Landmark_{i}_y', f'Landmark_{i}_z'])
        writer.writerow(headers)
        writer.writerows(recorded_data)

# Add buttons to adjust view angles
button_ax_elev_up = plt.axes([0.1, 0.02, 0.1, 0.075])
button_elev_up = Button(button_ax_elev_up, '+45 Elev')
button_elev_up.on_clicked(lambda event: adjust_elevation(type('Event', (object,), {'step': 45})))

button_ax_elev_down = plt.axes([0.1, 0.1, 0.1, 0.075])
button_elev_down = Button(button_ax_elev_down, '-45 Elev')
button_elev_down.on_clicked(lambda event: adjust_elevation(type('Event', (object,), {'step': -45})))

button_ax_azim_up = plt.axes([0.25, 0.02, 0.1, 0.075])
button_azim_up = Button(button_ax_azim_up, '+45 Azim')
button_azim_up.on_clicked(lambda event: adjust_azimuth(type('Event', (object,), {'step': 45})))

button_ax_azim_down = plt.axes([0.25, 0.1, 0.1, 0.075])
button_azim_down = Button(button_ax_azim_down, '-45 Azim')
button_azim_down.on_clicked(lambda event: adjust_azimuth(type('Event', (object,), {'step': -45})))

button_ax_roll_up = plt.axes([0.4, 0.02, 0.1, 0.075])
button_roll_up = Button(button_ax_roll_up, '+45 Roll')
button_roll_up.on_clicked(lambda event: adjust_roll(type('Event', (object,), {'step': 45})))

button_ax_roll_down = plt.axes([0.4, 0.1, 0.1, 0.075])
button_roll_down = Button(button_ax_roll_down, '-45 Roll')
button_roll_down.on_clicked(lambda event: adjust_roll(type('Event', (object,), {'step': -45})))

# Add an Exit button to terminate the application
button_ax_exit = plt.axes([0.8, 0.9, 0.1, 0.075])
exit_button = Button(button_ax_exit, 'Exit', color='red', hovercolor='darkred')
exit_button.on_clicked(exit_app)

# Main loop for streaming and visualization
try:
    frame_count = 0
    while running:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Process the RGB frame with MediaPipe Pose
        results = pose.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        # Draw the pose annotation on the color image
        annotated_image = color_image.copy()
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract landmark coordinates with depth
            x_vals = []
            y_vals = []
            z_vals = []
            landmarks_data = []

            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                # Get the 2D coordinates in the image
                x = int(landmark.x * color_image.shape[1])
                y = int(landmark.y * color_image.shape[0])

                # Ensure coordinates are within the frame
                if x >= 0 and x < depth_image.shape[1] and y >= 0 and y < depth_image.shape[0]:
                    depth_value = depth_image[y, x]

                    # Normalize coordinates for 3D visualization
                    x_vals.append(landmark.x - 0.5)  # Centered at 0
                    y_vals.append(0.5 - landmark.y)  # Inverted Y-axis
                    z_vals.append(depth_value / 1000.0)  # Convert mm to meters

                    # Record the landmark coordinates
                    landmarks_data.extend([landmark.x, landmark.y, landmark.z])

                    # Draw the landmark and depth info on the annotated image
                    # cv2.putText(annotated_image, f"ID {idx}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    # cv2.putText(annotated_image, f"{depth_value}mm", (x, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # Clear previous lines
            for line in lines:
                line.remove()
            lines.clear()

            # Plot lines connecting landmarks
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(x_vals) and end_idx < len(x_vals):
                    line = ax.plot(
                        [x_vals[start_idx], x_vals[end_idx]],
                        [y_vals[start_idx], y_vals[end_idx]],
                        [z_vals[start_idx], z_vals[end_idx]],
                        c='g'
                    )[0]
                    lines.append(line)

            # Update 3D scatter plot
            scatter._offsets3d = (x_vals, y_vals, z_vals)

            # Record the current frame's landmarks and view angles
            frame_data = [frame_count, current_elevation, current_azimuth, current_roll] + landmarks_data
            recorded_data.append(frame_data)
            frame_count += 1

            plt.draw()
            plt.pause(0.1)

        # Display the annotated image
        cv2.imshow('RealSense + MediaPipe Pose', annotated_image)

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
    plt.close(fig)
    # Save recorded landmarks and view angles to a CSV file if not saved already
    if running:
        with open('landmarks_and_view_angles.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            # Write headers
            headers = ['Frame', 'Elevation', 'Azimuth', 'Roll']
            for i in range(33):  # 33 landmarks for MediaPipe Pose
                headers.extend([f'Landmark_{i}_x', f'Landmark_{i}_y', f'Landmark_{i}_z'])
            writer.writerow(headers)
            writer.writerows(recorded_data)