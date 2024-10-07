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


# Function to create a global rotation matrix from three vectors (e.g., pelvis or other body segments)
def create_rotation_matrix_from_vectors(v1, v2):
    z_axis = v1 / np.linalg.norm(v1)  # Normalize the main axis (typically the primary direction of the segment)
    x_axis = np.cross(v2, z_axis)  # Create orthogonal axes
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)  # Ensure the y-axis is orthogonal too
    return np.stack([x_axis, y_axis, z_axis], axis=-1)  # Global rotation matrix (3x3)


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
    PELVIS = 0  # Assuming pelvis center or average between hips is landmark 0, or calculate as midpoint of hips

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

        pelvis = frame_data[PELVIS]  # Assuming pelvis center is one landmark (or calculated as midpoint of hips)

        # Calculate vectors for each leg segment in the global coordinate system
        right_thigh_vector_global = calculate_vector(right_hip, right_knee)
        right_shank_vector_global = calculate_vector(right_knee, right_ankle)

        left_thigh_vector_global = calculate_vector(left_hip, left_knee)
        left_shank_vector_global = calculate_vector(left_knee, left_ankle)

        # Calculate pelvis orientation in the global coordinate system (e.g., using left and right hips to form the axes)
        pelvis_vector_x = calculate_vector(left_hip, right_hip)  # Hip width as x-axis for pelvis orientation
        pelvis_vector_z = calculate_vector(pelvis, right_hip)  # Approximation of pelvis-to-hip as z-axis
        pelvis_rotation_global = create_rotation_matrix_from_vectors(pelvis_vector_z, pelvis_vector_x)

        # Create global rotation matrices for each leg segment
        right_thigh_rotation_global = create_rotation_matrix_from_vectors(right_thigh_vector_global, pelvis_vector_x)
        right_shank_rotation_global = create_rotation_matrix_from_vectors(right_shank_vector_global,
                                                                          right_thigh_vector_global)

        left_thigh_rotation_global = create_rotation_matrix_from_vectors(left_thigh_vector_global, pelvis_vector_x)
        left_shank_rotation_global = create_rotation_matrix_from_vectors(left_shank_vector_global,
                                                                         left_thigh_vector_global)

        # Compute relative rotation matrices for the knee joints in global reference frame
        right_knee_rotation_matrix_global = compute_relative_rotation_matrix(right_thigh_rotation_global,
                                                                             right_shank_rotation_global)
        left_knee_rotation_matrix_global = compute_relative_rotation_matrix(left_thigh_rotation_global,
                                                                            left_shank_rotation_global)

        # Compute relative rotation matrices for the hips (pelvis vs. thigh)
        right_hip_rotation_matrix_global = compute_relative_rotation_matrix(pelvis_rotation_global,
                                                                            right_thigh_rotation_global)
        left_hip_rotation_matrix_global = compute_relative_rotation_matrix(pelvis_rotation_global,
                                                                           left_thigh_rotation_global)

        # Convert relative rotation matrices to Euler angles for knees and hips
        right_knee_angles = rotation_matrix_to_euler_angles(right_knee_rotation_matrix_global, axes_order='xyz')
        left_knee_angles = rotation_matrix_to_euler_angles(left_knee_rotation_matrix_global, axes_order='xyz')
        right_hip_angles = rotation_matrix_to_euler_angles(right_hip_rotation_matrix_global, axes_order='xyz')
        left_hip_angles = rotation_matrix_to_euler_angles(left_hip_rotation_matrix_global, axes_order='xyz')

        # Compute ankle plantar flexion/dorsiflexion (only X-axis is relevant for ankle)
        right_ankle_flexion = rotation_matrix_to_euler_angles(right_shank_rotation_global, axes_order='xyz')[0]
        left_ankle_flexion = rotation_matrix_to_euler_angles(left_shank_rotation_global, axes_order='xyz')[0]

        # Store the data for the current frame
        joint_angles_data.append({
            'frame': frame_index + 1,
            'rt_hip_sagittal': right_hip_angles[0],  # X-axis (flexion/extension)
            'rt_hip_frontal': right_hip_angles[1],  # Y-axis (abduction/adduction)
            'rt_hip_transe': right_hip_angles[2],  # Z-axis (internal/external rotation)
            'rt_knee_sagittal': right_knee_angles[0],  # X-axis (flexion/extension)
            'rt_knee_frontal': right_knee_angles[1],  # Y-axis (abduction/adduction)
            'rt_knee_transe': right_knee_angles[2],  # Z-axis (internal/external rotation)
            'rt_ank_sagittal': right_ankle_flexion,  # X-axis (plantar flexion/dorsi flexion)
            'lt_hip_sagittal': left_hip_angles[0],  # X-axis (flexion/extension)
            'lt_hip_frontal': left_hip_angles[1],  # Y-axis (abduction/adduction)
            'lt_hip_transe': left_hip_angles[2],  # Z-axis (internal/external rotation)
            'lt_knee_sagittal': left_knee_angles[0],  # X-axis (flexion/extension)
            'lt_knee_frontal': left_knee_angles[1],  # Y-axis (abduction/adduction)
            'lt_knee_transe': left_knee_angles[2],  # Z-axis (internal/external rotation)
            'lt_ank_sagittal': left_ankle_flexion,  # X-axis (plantar flexion/dorsi flexion)
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