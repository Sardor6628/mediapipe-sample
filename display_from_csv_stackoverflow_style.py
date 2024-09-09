import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button

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
ax.set_xlim(0, 130)
ax.set_ylim(-100, 100)
ax.set_zlim(-100, 100)

# Default view settings for the 3D plot
default_elevation = 0  # Elevation angle
default_azimuth = 90  # Azimuth angle
default_roll = 270  # Roll angle
current_elevation = default_elevation
current_azimuth = default_azimuth
current_roll = default_roll
ax.view_init(elev=current_elevation, azim=current_azimuth)

# Flag to control the loop
running = True


# Function to update the 3D plot with new landmarks
def update_3d_plot(data_frame):
    global scatter, lines
    # Extract X, Y, Z coordinates of each point
    x_vals = []
    y_vals = []
    z_vals = []

    # Loop over each landmark point (assuming landmark columns are grouped by _x, _y, _z suffixes)
    for i in range(33):  # Assuming 33 landmarks
        x_vals.append(data_frame[f'Landmark_{i}_x'].values[0])
        y_vals.append(data_frame[f'Landmark_{i}_y'].values[0])
        z_vals.append(data_frame[f'Landmark_{i}_z'].values[0])

    # Clear old lines
    for line in lines:
        line.remove()
    lines.clear()

    # Plot lines connecting these points (if applicable)
    line = ax.plot(x_vals, y_vals, z_vals, c='g')[0]
    lines.append(line)

    # Update the scatter plot with new coordinates
    scatter._offsets3d = (x_vals, y_vals, z_vals)

    plt.draw()  # Update the plot with new data
    plt.pause(0.1)  # Pause briefly to allow for the plot to update


# Function to exit the application
def exit_app(event):
    global running
    running = False
    plt.close(fig)


# Functions to adjust the view
def adjust_elevation(event):
    global current_elevation
    current_elevation = (current_elevation + event.step) % 360
    ax.view_init(elev=current_elevation, azim=current_azimuth)
    plt.draw()


def adjust_azimuth(event):
    global current_azimuth
    current_azimuth = (current_azimuth + event.step) % 360
    ax.view_init(elev=current_elevation, azim=current_azimuth)
    plt.draw()


def adjust_roll(event):
    global current_roll
    current_roll = (current_roll + event.step) % 360
    ax.view_init(elev=current_elevation, azim=current_azimuth)
    plt.draw()


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

# Load the CSV data, dropping Elevation, Azimuth, and Roll columns
data_path = 'landmarks_and_view_angles.csv'  # Replace with your CSV file path
data = pd.read_csv(data_path).drop(columns=['Elevation', 'Azimuth', 'Roll'])

# Main loop to update the plot continuously until the exit button is pressed
while running:
    # Assuming the CSV file contains the necessary columns (Landmark_x, Landmark_y, Landmark_z for each frame)
    for i in range(len(data)):
        if not running:  # Check if the exit button was pressed
            break

        # Extract each row of the CSV as a frame
        frame_data = data.iloc[[i]]  # Access a single frame (row)

        # Update the 3D plot
        update_3d_plot(frame_data)

# Keep the plot open until explicitly closed
plt.show()

