import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define the MediaPipe pose connections (based on MediaPipe Pose)
pose_connections = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Head to right ear
    (0, 4), (4, 5), (5, 6), (6, 8),  # Head to left ear
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17),  # Right arm
    (12, 14), (14, 16), (16, 18),  # Left arm
    (11, 23), (12, 24), (23, 24),  # Torso
    (23, 25), (25, 27), (27, 29), (29, 31),  # Right leg
    (24, 26), (26, 28), (28, 30), (30, 32),  # Left leg
]

# Camera view settings for the plot
default_elevation = -90  # Elevation angle
default_azimuth = -90    # Azimuth angle
default_roll = 0     # Roll angle (not directly supported by matplotlib)

# Function to load landmark data from a JSON file
def load_landmark_data(json_file):
    with open(json_file, 'r') as f:
        landmark_data = json.load(f)
    return landmark_data

# Initialize the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = None
lines = []

# Set the initial camera view
ax.view_init(elev=default_elevation, azim=default_azimuth)

# Function to update the 3D plot for each frame
def update_graph(frame_index, landmark_data):
    global scatter, lines

    # Clear previous lines
    for line in lines:
        line.remove()
    lines.clear()

    # Get the current frame's landmarks
    landmarks = landmark_data[frame_index]['landmarks']

    # Extract x, y, z coordinates for each landmark
    x_vals = [lm['x'] for lm in landmarks]
    y_vals = [lm['y'] for lm in landmarks]
    z_vals = [lm['z'] for lm in landmarks]

    # Update scatter plot
    if scatter:
        scatter._offsets3d = (x_vals, y_vals, z_vals)
    else:
        scatter = ax.scatter(x_vals, y_vals, z_vals, c='b', marker='o')

    # Plot lines connecting the landmarks
    for start_idx, end_idx in pose_connections:
        line = ax.plot(
            [x_vals[start_idx], x_vals[end_idx]],
            [y_vals[start_idx], y_vals[end_idx]],
            [z_vals[start_idx], z_vals[end_idx]],
            c='g'
        )[0]
        lines.append(line)

    # Update axis limits and labels
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

# Animation function
def animate(frame_index):
    frame_index = frame_index % len(landmark_data)  # Loop when reaching the last frame
    update_graph(frame_index, landmark_data)

# Load landmark data from JSON file
# json_file_path = 'output/output_14-19-good-0/all_squats_landmarks.json'
# json_file_path = 'output/output_14-21-good-1-0/all_squats_landmarks.json'
# json_file_path = 'output/output_14-27-bad-2-0/all_squats_landmarks.json'
json_file_path = 'output/output_14-31-bad-3-0/all_squats_landmarks.json'
# json_file_path = 'output/output_14-37-bad-4-0/all_squats_landmarks.json'
# json_file_path = 'output/output_14-41-bad-5-0/all_squats_landmarks.json'
landmark_data = load_landmark_data(json_file_path)

# Create the animation
ani = FuncAnimation(fig, animate, frames=len(landmark_data), interval=10, repeat=True)

# Show the plot
plt.show()