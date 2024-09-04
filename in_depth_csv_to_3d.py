import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
import csv
import time

# Load recorded data from CSV
recorded_data = []
with open('landmarks_and_view_angles.csv', 'r') as file:
    reader = csv.reader(file)
    headers = next(reader)  # Skip header
    for row in reader:
        recorded_data.append([float(val) for val in row])

# Set up matplotlib for 3D visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Set axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Set axis limits
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(-0.5, 0.5)

# Initialize scatter plot for landmarks
scatter = ax.scatter([], [], [], c='b', marker='o')
lines = []  # To store line objects

# Initialize view angles
current_elevation = 90
current_azimuth = 90
current_roll = 180

# Display text for current elevation, azimuth, and roll
text_elev = ax.text2D(0.05, 0.95, f"Elev: {current_elevation}", transform=ax.transAxes)
text_azim = ax.text2D(0.05, 0.90, f"Azim: {current_azimuth}", transform=ax.transAxes)
text_roll = ax.text2D(0.05, 0.85, f"Roll: {current_roll}", transform=ax.transAxes)

# Flag to control the loop
running = True


# Function to update 3D view based on user input
def adjust_elevation(event):
    global current_elevation
    current_elevation = (current_elevation + event.step) % 360
    ax.view_init(elev=current_elevation, azim=current_azimuth)
    text_elev.set_text(f"Elev: {current_elevation}")
    plt.draw()


def adjust_azimuth(event):
    global current_azimuth
    current_azimuth = (current_azimuth + event.step) % 360
    ax.view_init(elev=current_elevation, azim=current_azimuth)
    text_azim.set_text(f"Azim: {current_azimuth}")
    plt.draw()


def adjust_roll(event):
    global current_roll
    current_roll = (current_roll + event.step) % 360
    text_roll.set_text(f"Roll: {current_roll}")  # Placeholder: roll isn't directly supported
    plt.draw()


# Function to exit the application
def exit_app(event):
    global running
    running = False


# Add buttons to adjust view angles
button_ax_elev_up = plt.axes([0.1, 0.02, 0.1, 0.035])
button_elev_up = Button(button_ax_elev_up, '+45 Elev')
button_elev_up.on_clicked(lambda event: adjust_elevation(type('Event', (object,), {'step': 45})))

button_ax_elev_down = plt.axes([0.1, 0.1, 0.1, 0.035])
button_elev_down = Button(button_ax_elev_down, '-45 Elev')
button_elev_down.on_clicked(lambda event: adjust_elevation(type('Event', (object,), {'step': -45})))

button_ax_azim_up = plt.axes([0.25, 0.02, 0.1, 0.035])
button_azim_up = Button(button_ax_azim_up, '+45 Azim')
button_azim_up.on_clicked(lambda event: adjust_azimuth(type('Event', (object,), {'step': 45})))

button_ax_azim_down = plt.axes([0.25, 0.1, 0.1, 0.035])
button_azim_down = Button(button_ax_azim_down, '-45 Azim')
button_azim_down.on_clicked(lambda event: adjust_azimuth(type('Event', (object,), {'step': -45})))

button_ax_roll_up = plt.axes([0.4, 0.02, 0.1, 0.035])
button_roll_up = Button(button_ax_roll_up, '+45 Roll')
button_roll_up.on_clicked(lambda event: adjust_roll(type('Event', (object,), {'step': 45})))

button_ax_roll_down = plt.axes([0.4, 0.1, 0.1, 0.035])
button_roll_down = Button(button_ax_roll_down, '-45 Roll')
button_roll_down.on_clicked(lambda event: adjust_roll(type('Event', (object,), {'step': -45})))

# Add an Exit button to terminate the application\
button_ax_exit = plt.axes([0.8, 0.9, 0.1, 0.035])
exit_button = Button(button_ax_exit, 'Exit', color='red', hovercolor='darkred')
exit_button.on_clicked(exit_app)

# Define connections that represent a human skeleton
pose_connections = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),  # Head and shoulders
    (9, 10),  # Hips
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 23), (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
    (12, 24), (24, 26), (26, 28), (28, 30), (30, 32)  # Right leg
]

# Main loop to continuously replay recorded data
while running:
    for frame_data in recorded_data:
        if not running:  # Check if exit button was pressed
            break

        # Ignore recorded view angles and only use landmark coordinates
        landmarks = frame_data[4:]  # Skip the first 4 entries (frame, elev, azim, roll)

        # Use the last adjusted view angles (current_elevation, current_azimuth)
        ax.view_init(elev=current_elevation, azim=current_azimuth)

        # Update scatter plot with landmark positions
        x_vals = landmarks[0::3]
        y_vals = landmarks[1::3]
        z_vals = landmarks[2::3]
        scatter._offsets3d = (x_vals, y_vals, z_vals)

        # Clear previous lines and draw new ones
        for line in lines:
            line.remove()
        lines.clear()
        for start, end in pose_connections:
            if start < len(x_vals) and end < len(x_vals):
                line = ax.plot(
                    [x_vals[start], x_vals[end]],
                    [y_vals[start], y_vals[end]],
                    [z_vals[start], z_vals[end]], c='g')[0]
                lines.append(line)

        # Update text
        text_elev.set_text(f"Elev: {current_elevation}")
        text_azim.set_text(f"Azim: {current_azimuth}")
        text_roll.set_text(f"Roll: {current_roll}")

        plt.draw()
        plt.pause(0.1)  # Adjust delay for playback speed

plt.show()