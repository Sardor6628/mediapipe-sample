import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd
import os


# Helper function to load MediaPipe data from a JSON file
def load_mediapipe_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data  # Return the full list of lists (each sublist contains one frame of data)


# Function to calculate vector between two landmarks
def calculate_vector(p1, p2):
    return np.array([p2['x'] - p1['x'], p2['y'] - p1['y'], p2['z'] - p1['z']])


# Function to create an orthogonal basis from the segment vector
def create_rotation_matrix(vector):
    z_axis = vector / np.linalg.norm(vector)  # Normalize to create the main axis
    # Arbitrary vector to create a perpendicular axis
    x_axis = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    # Normalize all axes
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=-1)  # Rotation matrix (3x3)


# Function to compute relative rotation matrix between two segments
def compute_relative_rotation_matrix(segment1, segment2):
    return np.dot(np.linalg.inv(segment1), segment2)


# Function to convert relative rotation matrix to Euler angles in the desired order
def rotation_matrix_to_euler_angles(rotation_matrix, axes_order='xyz'):
    r = R.from_matrix(rotation_matrix)
    return r.as_euler(axes_order, degrees=True)  # Return Euler angles in degrees (X, Y, Z)


# Main function to process the joint angles and return the data for each frame
def process_joint_angles(mediapipe_data):
    # Define landmark indices for hips, knees, and ankles
    RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 24, 26, 28
    LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 23, 25, 27

    # Initialize list to store the results
    joint_angles_data = []

    # Loop through each frame in the data
    for frame_index, frame_data in enumerate(mediapipe_data):
        # Extract landmarks from MediaPipe data for the current frame
        right_hip = frame_data[RIGHT_HIP]
        right_knee = frame_data[RIGHT_KNEE]
        right_ankle = frame_data[RIGHT_ANKLE]

        left_hip = frame_data[LEFT_HIP]
        left_knee = frame_data[LEFT_KNEE]
        left_ankle = frame_data[LEFT_ANKLE]

        # Calculate vectors for right leg segments
        right_thigh_vector = calculate_vector(right_hip, right_knee)
        right_shank_vector = calculate_vector(right_knee, right_ankle)

        # Calculate vectors for left leg segments
        left_thigh_vector = calculate_vector(left_hip, left_knee)
        left_shank_vector = calculate_vector(left_knee, left_ankle)

        # Create rotation matrices for right leg segments
        right_thigh_rotation = create_rotation_matrix(right_thigh_vector)
        right_shank_rotation = create_rotation_matrix(right_shank_vector)

        # Create rotation matrices for left leg segments
        left_thigh_rotation = create_rotation_matrix(left_thigh_vector)
        left_shank_rotation = create_rotation_matrix(left_shank_vector)

        # Compute relative rotation matrices (e.g., knee joint is between thigh and shank)
        right_knee_rotation_matrix = compute_relative_rotation_matrix(right_thigh_rotation, right_shank_rotation)
        left_knee_rotation_matrix = compute_relative_rotation_matrix(left_thigh_rotation, left_shank_rotation)

        # Convert relative rotation matrices to Euler angles for knees (X = flexion/extension, Y = abduction/adduction, Z = internal/external rotation)
        right_knee_angles = rotation_matrix_to_euler_angles(right_knee_rotation_matrix, axes_order='xyz')
        left_knee_angles = rotation_matrix_to_euler_angles(left_knee_rotation_matrix, axes_order='xyz')

        # Compute the right hip joint
        pelvis_vector = calculate_vector(left_hip, right_hip)  # Pelvis segment vector
        pelvis_rotation = create_rotation_matrix(pelvis_vector)

        # Relative rotation matrix for the right hip joint (between pelvis and right thigh)
        right_hip_rotation_matrix = compute_relative_rotation_matrix(pelvis_rotation, right_thigh_rotation)
        right_hip_angles = rotation_matrix_to_euler_angles(right_hip_rotation_matrix, axes_order='xyz')

        # Relative rotation matrix for the left hip joint (between pelvis and left thigh)
        left_hip_rotation_matrix = compute_relative_rotation_matrix(pelvis_rotation, left_thigh_rotation)
        left_hip_angles = rotation_matrix_to_euler_angles(left_hip_rotation_matrix, axes_order='xyz')

        # Compute the ankle plantar flexion/dorsiflexion (only X-axis is relevant for ankle)
        right_ankle_angles = rotation_matrix_to_euler_angles(right_shank_rotation, axes_order='xyz')[
            0]  # Extract only the X-axis (plantar flexion/dorsiflexion)
        left_ankle_angles = rotation_matrix_to_euler_angles(left_shank_rotation, axes_order='xyz')[
            0]  # Extract only the X-axis (plantar flexion/dorsiflexion)

        # Store the data for the current frame
        joint_angles_data.append({
            'frame': frame_index + 1,
            'rt_hip_flexion': right_hip_angles[0],  # X-axis (flexion/extension)
            'rt_hip_abduction': right_hip_angles[1],  # Y-axis (abduction/adduction)
            'rt_hip_rotation': right_hip_angles[2],  # Z-axis (internal/external rotation)
            'rt_knee_flexion': right_knee_angles[0],  # X-axis (flexion/extension)
            'rt_knee_abduction': right_knee_angles[1],  # Y-axis (abduction/adduction)
            'rt_knee_rotation': right_knee_angles[2],  # Z-axis (internal/external rotation)
            'rt_ankle_flexion': right_ankle_angles,  # X-axis (plantar flexion/dorsi flexion)
            'lt_hip_flexion': left_hip_angles[0],  # X-axis (flexion/extension)
            'lt_hip_abduction': left_hip_angles[1],  # Y-axis (abduction/adduction)
            'lt_hip_rotation': left_hip_angles[2],  # Z-axis (internal/external rotation)
            'lt_knee_flexion': left_knee_angles[0],  # X-axis (flexion/extension)
            'lt_knee_abduction': left_knee_angles[1],  # Y-axis (abduction/adduction)
            'lt_knee_rotation': left_knee_angles[2],  # Z-axis (internal/external rotation)
            'lt_ankle_flexion': left_ankle_angles,  # X-axis (plantar flexion/dorsi flexion)
        })

    # Return the joint angles data as a DataFrame
    return pd.DataFrame(joint_angles_data)


# Function to process multiple JSON files, save the results as CSV files, and print min/max values
def process_and_save_to_csv(json_files, output_dir):
    for json_file in json_files:
        # Load the MediaPipe data from the JSON file
        mediapipe_data = load_mediapipe_data(json_file)

        # Process the joint angles
        joint_angles_df = process_joint_angles(mediapipe_data)

        # Create the output CSV filename based on the JSON filename
        csv_filename = os.path.join(output_dir, os.path.basename(json_file).replace('.json', '_joint_angles.csv'))

        # Save the DataFrame to a CSV file
        joint_angles_df.to_csv(csv_filename, index=False)
        print(f"Saved joint angles for {json_file} to {csv_filename}")

        # Print the minimum and maximum values for each column
        print(f"\nMin and Max values for {json_file}:\n")
        for column in joint_angles_df.columns:
            if column != 'frame':  # Exclude frame column from min/max computation
                min_val = joint_angles_df[column].min()
                max_val = joint_angles_df[column].max()
                print(f"{column} -> Min: {min_val:.2f}, Max: {max_val:.2f},  df: {abs(max_val - min_val):.2f}")
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    # List of JSON files to process
    json_files = [
        '.Good_output/SIH_dynamic_0_front1_output/SIH_dynamic_0_front1_landmarks.json',
    ]

    # Output directory to save the CSV files
    output_dir = './output_csv'

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each JSON file and save the results as CSV
    process_and_save_to_csv(json_files, output_dir)