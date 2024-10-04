import json
import numpy as np

# Load JSON data
with open('.Good_output/SIH_dynamic_0_front1_output/SIH_dynamic_0_front1_landmarks.json', 'r') as f:
    frames = json.load(f)

# Define landmark IDs
RIGHT_HIP = 24
RIGHT_KNEE = 26
RIGHT_ANKLE = 28
RIGHT_SHANK_REF = 32  # Reference point on the shank; replace as needed


def get_landmark(frame, landmark_id):
    """Extracts landmark position as a vector from frame data."""
    for landmark in frame:
        if landmark["id"] == landmark_id:
            return np.array([landmark["x"], landmark["y"], landmark["z"]])
    return None


print(get_landmark(0, RIGHT_HIP))