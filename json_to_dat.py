import json
import numpy as np

# Joint ID mapping based on the previous example
JOINT_ID_MAP = {
    'lefthip': 6, 'leftknee': 8, 'leftfoot': 10,
    'righthip': 7, 'rightknee': 9, 'rightfoot': 11,
    'leftshoulder': 0, 'leftelbow': 2, 'leftwrist': 4,
    'rightshoulder': 1, 'rightelbow': 3, 'rightwrist': 5
}


# Function to load keypoints from JSON file based on the specified index map
def load_keypoints_from_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    frames = []
    for frame in data:
        frame_keypoints = []
        for joint_name, joint_id in JOINT_ID_MAP.items():
            joint_data = next((item for item in frame if item['id'] == joint_id), None)
            if joint_data:
                frame_keypoints.append([joint_data['x'], joint_data['y'], joint_data['z']])
            else:
                # If for any reason the joint is not found, append a placeholder
                frame_keypoints.append([0.0, 0.0, 0.0])

        # Ensure that we have exactly 12 keypoints (36 values)
        if len(frame_keypoints) == 12:
            frames.append(frame_keypoints)
        else:
            print(f"Warning: Frame does not contain 12 keypoints, skipping this frame.")

    return np.array(frames)


# Function to write keypoints to a .dat file
def write_keypoints_to_dat(filename, keypoints):
    with open(filename, 'w') as f:
        for frame in keypoints:
            frame_str = ' '.join([f"{coord:.6f}" for coords in frame for coord in coords])
            f.write(f"{frame_str}\n")


if __name__ == '__main__':
    json_filename = '.Good_output/SIH_dynamic_0_front1_output/SIH_dynamic_0_front1_landmarks.json'  # Replace with your actual JSON file
    dat_filename = 'output.dat'  # The name of the .dat file to output

    keypoints = load_keypoints_from_json(json_filename)
    write_keypoints_to_dat(dat_filename, keypoints)

    print(f"Successfully converted {json_filename} to {dat_filename}")