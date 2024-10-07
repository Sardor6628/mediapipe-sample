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


# Function to calculate angles relative to the x, y, z axes
def calculate_joint_angle_relative_to_axes(b, c, min_visibility=0.6):
    if b['visibility'] > min_visibility and c['visibility'] > min_visibility:
        # Vector from joint b to joint c
        vector_bc = np.array([c['x'] - b['x'], c['y'] - b['y'], c['z'] - b['z']])
        # Normalize the vector
        vector_bc_normalized = vector_bc / np.linalg.norm(vector_bc)

        # Unit vectors for x, y, z axes
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])
        z_axis = np.array([0, 0, 1])

        # Calculate angles relative to x, y, z axes
        angle_x = np.degrees(np.arccos(np.dot(vector_bc_normalized, x_axis)))
        angle_y = np.degrees(np.arccos(np.dot(vector_bc_normalized, y_axis)))
        angle_z = np.degrees(np.arccos(np.dot(vector_bc_normalized, z_axis)))

        return angle_x, angle_y, angle_z
    else:
        return -1, -1, -1


def get_dict_by_id(json_list, search_id):
    # Using a loop to find the dictionary with the matching id
    for item in json_list:
        if item.get("id") == search_id:  # Use get() to avoid KeyError if 'id' doesn't exist
            return item
    return None  # Return None if no match is found


# Convert landmarks to flattened structure like 0_x, 0_y, 0_z, etc.
def flatten_landmarks(landmarks):
    flattened = {}
    for lm in landmarks:
        lm_id = lm['id']
        flattened[f'{lm_id}_x'] = lm['x']
        flattened[f'{lm_id}_y'] = lm['y']
        flattened[f'{lm_id}_z'] = lm['z']
        flattened[f'{lm_id}_v'] = lm['visibility']
    return flattened


def generate_csv(landmark_data):
    data = []

    for json_object in landmark_data:
        # Extract landmarks
        landmarks = json_object['landmarks']

        # Calculate knee angles (general 3-point angles)
        r_knee_angle = find_angle(get_dict_by_id(landmarks, 24), get_dict_by_id(landmarks, 26),
                                  get_dict_by_id(landmarks, 28))
        l_knee_angle = find_angle(get_dict_by_id(landmarks, 23), get_dict_by_id(landmarks, 25),
                                  get_dict_by_id(landmarks, 27))

        # Calculate joint angles relative to x, y, z axes (for hip, knee, ankle)

        # Right knee
        r_knee_angles_xyz = calculate_joint_angle_relative_to_axes(get_dict_by_id(landmarks, 26),
                                                                   get_dict_by_id(landmarks, 28))
        # Left knee
        l_knee_angles_xyz = calculate_joint_angle_relative_to_axes(get_dict_by_id(landmarks, 25),
                                                                   get_dict_by_id(landmarks, 27))

        # Right hip (using landmarks 24 and 26)
        r_hip_angles_xyz = calculate_joint_angle_relative_to_axes(get_dict_by_id(landmarks, 24),
                                                                  get_dict_by_id(landmarks, 26))
        # Left hip (using landmarks 23 and 25)
        l_hip_angles_xyz = calculate_joint_angle_relative_to_axes(get_dict_by_id(landmarks, 23),
                                                                  get_dict_by_id(landmarks, 25))

        # Right ankle (using landmarks 28 and 32)
        r_ankle_angles_xyz = calculate_joint_angle_relative_to_axes(get_dict_by_id(landmarks, 28),
                                                                    get_dict_by_id(landmarks, 32))
        # Left ankle (using landmarks 27 and 31)
        l_ankle_angles_xyz = calculate_joint_angle_relative_to_axes(get_dict_by_id(landmarks, 27),
                                                                    get_dict_by_id(landmarks, 31))

        # Flatten landmarks into fields like 0_x, 0_y, 0_z, 0_visibility
        flattened_landmarks = flatten_landmarks(landmarks)

        # Add angles to the flattened landmarks
        flattened_landmarks['r_knee_angle'] = float(r_knee_angle)
        flattened_landmarks['l_knee_angle'] = float(l_knee_angle)

        # Add relative angles to the x, y, z axes for the right knee
        flattened_landmarks['r_knee_angle_x'] = r_knee_angles_xyz[0]
        flattened_landmarks['r_knee_angle_y'] = r_knee_angles_xyz[1]
        flattened_landmarks['r_knee_angle_z'] = r_knee_angles_xyz[2]

        # Add relative angles to the x, y, z axes for the left knee
        flattened_landmarks['l_knee_angle_x'] = l_knee_angles_xyz[0]
        flattened_landmarks['l_knee_angle_y'] = l_knee_angles_xyz[1]
        flattened_landmarks['l_knee_angle_z'] = l_knee_angles_xyz[2]

        # Add relative angles for right and left hips
        flattened_landmarks['r_hip_angle_x'] = r_hip_angles_xyz[0]
        flattened_landmarks['r_hip_angle_y'] = r_hip_angles_xyz[1]
        flattened_landmarks['r_hip_angle_z'] = r_hip_angles_xyz[2]

        flattened_landmarks['l_hip_angle_x'] = l_hip_angles_xyz[0]
        flattened_landmarks['l_hip_angle_y'] = l_hip_angles_xyz[1]
        flattened_landmarks['l_hip_angle_z'] = l_hip_angles_xyz[2]

        # Add relative angles for right and left ankles
        flattened_landmarks['r_ankle_angle_x'] = r_ankle_angles_xyz[0]
        flattened_landmarks['r_ankle_angle_y'] = r_ankle_angles_xyz[1]
        flattened_landmarks['r_ankle_angle_z'] = r_ankle_angles_xyz[2]

        flattened_landmarks['l_ankle_angle_x'] = l_ankle_angles_xyz[0]
        flattened_landmarks['l_ankle_angle_y'] = l_ankle_angles_xyz[1]
        flattened_landmarks['l_ankle_angle_z'] = l_ankle_angles_xyz[2]

        # Add frame number
        flattened_landmarks['frame'] = json_object['frame']

        # Append the processed data
        data.append(flattened_landmarks)
    return data


def generate_and_save_report_into_csv(base_path, path,seq):
    try:
        landmark_data = load_landmark_data(path)
        csv_data = generate_csv(landmark_data)
        df = pd.DataFrame(csv_data)
        df.to_csv(base_path +'/squat_'+str(seq)+'_output_landmarks.csv', index=False)
        print('saved file into: ' + base_path + 'squat_'+str(seq)+'_output_landmarks.csv')
    except Exception as e:
        print("error occurred while working on ", base_path, "Error=>", e)


base_path ="output/sam-2"

for index in range(1,20):
    generate_and_save_report_into_csv(base_path,base_path+"/squat_"+str(index)+"_landmarks.json",index)