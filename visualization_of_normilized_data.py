import read_from_csv as data_manipulation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button

NORMALIZED_DATA_PATH = "normalized_data/n_1_sih/N_SIH_dynamic_0_02.csv"
data = data_manipulation.get_final_list(NORMALIZED_DATA_PATH)
print(data)

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
default_azimuth = 90   # Azimuth angle
default_roll = 270     # Roll angle
current_elevation = default_elevation
current_azimuth = default_azimuth
current_roll = default_roll
ax.view_init(elev=current_elevation, azim=current_azimuth, roll=current_roll)

# Flag to control the loop
running = True

# Function to update the 3D plot with new landmarks
def update_3d_plot(data_frame):
    global scatter, lines
    # Extract X, Y, Z coordinates of each point
    x_vals = []
    y_vals = []
    z_vals = []

    for segment in data_frame:
        for coord in zip(segment['x'], segment['y'], segment['z']):
            x_vals.append(coord[0])
            y_vals.append(coord[1])
            z_vals.append(coord[2])

        # Plot lines connecting these points
        line = ax.plot(segment['x'], segment['y'], segment['z'], c='g')[0]
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
    ax.view_init(elev=current_elevation, azim=current_azimuth, roll=current_roll)
    plt.draw()

def adjust_azimuth(event):
    global current_azimuth
    current_azimuth = (current_azimuth + event.step) % 360
    ax.view_init(elev=current_elevation, azim=current_azimuth, roll=current_roll)
    plt.draw()

def adjust_roll(event):
    global current_roll
    current_roll = (current_roll + event.step) % 360
    ax.view_init(elev=current_elevation, azim=current_azimuth, roll=current_roll)
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

# Main loop to update the plot continuously until the exit button is pressed
while running:
    for frame_data in data:
        if not running:  # Check if the exit button was pressed
            break

        # Clear old lines
        for line in lines:
            line.remove()
        lines.clear()

        update_3d_plot(frame_data)

# Keep the plot open until explicitly closed
plt.show()