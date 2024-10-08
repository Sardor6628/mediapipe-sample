import json
import pandas as pd
import numpy as np

# Function to load landmark data from a JSON file
def load_landmark_data(json_file):
    with open(json_file, 'r') as f:
        landmark_data = json.load(f)
    return landmark_data

# Helper function to calculate angle between three points (for general joint angles)
def find_angle(a, b, c, min_visibility=0.6):
    try:
        if a['visibility'] > min_visibility and b['visibility'] > min_visibility and c['visibility'] > min_visibility:
            ba = np.array([a['x'] - b['x'], a['y'] - b['y'], a['z'] - b['z']])
            bc = np.array([c['x'] - b['x'], c['y'] - b['y'], c['z'] - b['z']])
            angle = np.arccos(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))) * (180.0 / np.pi)
            return angle if angle <= 180 else 360 - angle
        else:
            return -1
    except Exception as e:
        return -1

# Normalize landmarks relative to the pelvis (midpoint of left and right hips)
def normalize_landmarks(landmarks):
    left_hip = np.array([landmarks[23]['x'], landmarks[23]['y'], landmarks[23]['z']])
    right_hip = np.array([landmarks[24]['x'], landmarks[24]['y'], landmarks[24]['z']])
    pelvis = (left_hip + right_hip) / 2

    normalized_landmarks = []
    for lm in landmarks:
        normalized_lm = {
            'id': lm['id'],
            'x': lm['x'] - pelvis[0],
            'y': lm['y'] - pelvis[1],
            'z': lm['z'] - pelvis[2],
            'visibility': lm['visibility']
        }
        normalized_landmarks.append(normalized_lm)

    return normalized_landmarks

# Helper function to calculate angles relative to the x, y, z axes
def calculate_joint_angle_relative_to_axes(b, c, min_visibility=0.6):
    if b['visibility'] > min_visibility and c['visibility'] > min_visibility:
        vector_bc = np.array([c['x'] - b['x'], c['y'] - b['y'], c['z'] - b['z']])
        vector_bc_normalized = vector_bc / np.linalg.norm(vector_bc)

        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])

        angle_x = np.degrees(np.arccos(np.dot(vector_bc_normalized, x_axis)))
        angle_y = np.degrees(np.arccos(np.dot(vector_bc_normalized, y_axis)))
        angle_z = np.degrees(np.arccos(np.dot(vector_bc_normalized, z_axis)))

        return angle_x, angle_y, angle_z
    else:
        return -1, -1, -1

# Flatten landmarks into a format suitable for a CSV
def flatten_landmarks(landmarks):
    flattened = {}
    for lm in landmarks:
        lm_id = lm['id']
        flattened[f'{lm_id}_x'] = lm['x']
        flattened[f'{lm_id}_y'] = lm['y']
        flattened[f'{lm_id}_z'] = lm['z']
        flattened[f'{lm_id}_v'] = lm['visibility']
    return flattened

# Generate CSV data from the normalized and processed landmarks
def generate_csv(landmark_data):
    data = []

    for json_object in landmark_data:
        # Extract and normalize landmarks
        landmarks = normalize_landmarks(json_object['landmarks'])

        # Calculate joint angles
        r_knee_angle = find_angle(get_dict_by_id(landmarks, 24), get_dict_by_id(landmarks, 26),
                                  get_dict_by_id(landmarks, 28))
        l_knee_angle = find_angle(get_dict_by_id(landmarks, 23), get_dict_by_id(landmarks, 25),
                                  get_dict_by_id(landmarks, 27))

        r_knee_angles_xyz = calculate_joint_angle_relative_to_axes(get_dict_by_id(landmarks, 26),
                                                                   get_dict_by_id(landmarks, 28))
        l_knee_angles_xyz = calculate_joint_angle_relative_to_axes(get_dict_by_id(landmarks, 25),
                                                                   get_dict_by_id(landmarks, 27))

        # Flatten normalized landmarks
        flattened_landmarks = flatten_landmarks(landmarks)

        # Add calculated angles to the flattened landmarks
        flattened_landmarks['r_knee_angle'] = float(r_knee_angle)
        flattened_landmarks['l_knee_angle'] = float(l_knee_angle)

        flattened_landmarks['r_knee_angle_x'] = r_knee_angles_xyz[0]
        flattened_landmarks['r_knee_angle_y'] = r_knee_angles_xyz[1]
        flattened_landmarks['r_knee_angle_z'] = r_knee_angles_xyz[2]

        flattened_landmarks['l_knee_angle_x'] = l_knee_angles_xyz[0]
        flattened_landmarks['l_knee_angle_y'] = l_knee_angles_xyz[1]
        flattened_landmarks['l_knee_angle_z'] = l_knee_angles_xyz[2]

        flattened_landmarks['frame'] = json_object['frame']

        data.append(flattened_landmarks)
    return data

# Save processed data to a CSV file
def generate_and_save_report_into_csv(base_path, path, seq):
    try:
        landmark_data = load_landmark_data(path)
        csv_data = generate_csv(landmark_data)
        df = pd.DataFrame(csv_data)
        df.to_csv(base_path + '/squat_' + str(seq) + '_output_landmarks.csv', index=False)
        print('Saved file into:', base_path + '/squat_' + str(seq) + '_output_landmarks.csv')
    except Exception as e:
        print("Error occurred while processing:", path, "Error:", e)

base_path = r"mediapipe-sample\output\Squat_Data\Squat_Data\Valid\output"

# Process multiple files
for index in range(0, 200):
    generate_and_save_report_into_csv(base_path, base_path + f"{index}.json", index)