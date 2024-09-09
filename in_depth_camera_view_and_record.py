import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os

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

# Directory and filenames for outputs
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
video_output_path = os.path.join(output_dir, "output_video.mp4")
json_output_path = os.path.join(output_dir, "landmarks_output.json")

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_output_path, fourcc, 30, (640, 480))

# List to store 3D landmarks for JSON output
landmark_data_json = []


# Function to record landmarks and save in JSON format
def record_landmarks(frame_count, results):
    landmarks_frame_data = []

    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmarks_frame_data.append({
                "id": idx,
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility
            })

    # Append the frame's landmarks to the JSON data
    landmark_data_json.append({"frame": frame_count, "landmarks": landmarks_frame_data})


# Main loop for streaming and visualization
try:
    frame_count = 0
    while True:
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

            # Record the landmarks for the current frame
            record_landmarks(frame_count, results)

            # Draw and update the plot
            plt.draw()
            plt.pause(0.001)

        # Write the frame to the video output
        out.write(annotated_image)

        # Display the annotated image in real-time
        cv2.imshow('RealSense + MediaPipe Pose', annotated_image)

        frame_count += 1

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming and release resources
    pipeline.stop()
    out.release()
    cv2.destroyAllWindows()
    plt.close(fig)

    # Save the landmark data to a JSON file
    with open(json_output_path, 'w') as json_file:
        json.dump(landmark_data_json, json_file, indent=4)

    print(f"Video saved to {video_output_path}")
    print(f"Landmarks saved to {json_output_path}")