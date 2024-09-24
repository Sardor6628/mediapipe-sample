import json
import pandas as pd
import numpy as np



# Function to load landmark data from a JSON file
def load_landmark_data(json_file):
    with open(json_file, 'r') as f:
        landmark_data = json.load(f)
    return landmark_data

# Helper function to calculate angle between three points
def find_angle(a, b, c, min_visibility=0.8):
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

        # Calculate knee angles
        r_knee_angle = find_angle(get_dict_by_id(landmarks, 24), get_dict_by_id(landmarks, 26),
                                  get_dict_by_id(landmarks, 28))
        l_knee_angle = find_angle(get_dict_by_id(landmarks, 23), get_dict_by_id(landmarks, 25),
                                  get_dict_by_id(landmarks, 27))

        # Flatten landmarks into fields like 0_x, 0_y, 0_z, 0_visibility
        flattened_landmarks = flatten_landmarks(landmarks)

        # Add angles to the flattened landmarks
        flattened_landmarks['r_knee_angle'] = float(r_knee_angle)
        flattened_landmarks['l_knee_angle'] = float(l_knee_angle)
        flattened_landmarks['frame'] = json_object['frame']
        # Append the processed data
        data.append(flattened_landmarks)
    return data

def generate_and_save_report_into_csv(save_csv_path):
    json_file_path = save_csv_path+'all_squats_landmarks.json'
    try:
        landmark_data = load_landmark_data(json_file_path)
        csv_data=generate_csv(landmark_data)
        df = pd.DataFrame(csv_data)
        df.to_csv(save_csv_path + 'output_landmarks.csv', index=False)
        print('saved file into: ' + save_csv_path + 'output_landmarks.csv')
    except Exception as e:
        print("error occurred while working on ", json_file_path, "Error=>",e)



list_of_path=["output/squat_luke/output_14-19-good-0/", "output/squat_luke/output_14-18-good-0/", "output/squat_luke/output_14-27-bad-2-0/","output/squat_luke/output_14-31-bad-3-0/", "output/squat_luke/output_14-37-bad-4-0/","output/squat_luke/output_14-41-bad-5-0/"]

for list_item in list_of_path:
    generate_and_save_report_into_csv(list_item)
