import numpy as np
import sys
from scipy.signal import medfilt
import utils  # Importing the utilities you provided


# Read keypoints from file
def read_keypoints(filename):
    num_keypoints = 12
    kpts = []
    with open(filename, 'r') as fin:
        for line in fin:
            line = line.split()
            line = [float(s) for s in line]
            line = np.reshape(line, (num_keypoints, -1))
            kpts.append(line)
    return np.array(kpts)


# Convert keypoints to a dictionary format for easier manipulation
def convert_to_dictionary(kpts):
    keypoints_to_index = {
        'lefthip': 6, 'leftknee': 8, 'leftfoot': 10,
        'righthip': 7, 'rightknee': 9, 'rightfoot': 11
    }
    kpts_dict = {key: kpts[:, k_index] for key, k_index in keypoints_to_index.items()}
    kpts_dict['joints'] = list(keypoints_to_index.keys())
    return kpts_dict


# Add the hips keypoint (midpoint between left and right hip)
def add_hips(kpts):
    hips = (kpts['lefthip'] + kpts['righthip']) / 2
    kpts['hips'] = hips
    kpts['joints'].append('hips')
    return kpts


# Median filter to reduce noise in keypoints
def median_filter(kpts, window_size=3):
    filtered = {}
    filtered['joints'] = kpts['joints']  # Preserve the 'joints' key

    for joint in kpts['joints']:
        joint_kpts = kpts[joint]
        xs, ys, zs = joint_kpts[:, 0], joint_kpts[:, 1], joint_kpts[:, 2]
        xs = medfilt(xs, window_size)
        ys = medfilt(ys, window_size)
        zs = medfilt(zs, window_size)
        filtered[joint] = np.stack([xs, ys, zs], axis=-1)

    return filtered


# Calculate angles for hip and knee joints (3D rotation angles: x, y, z)
def calculate_joint_3D_angles(kpts):
    for framenum in range(kpts['lefthip'].shape[0]):
        # Left Hip: angles relative to the pelvis
        left_hip_knee = kpts['lefthip'][framenum] - kpts['leftknee'][framenum]
        right_hip_knee = kpts['righthip'][framenum] - kpts['rightknee'][framenum]

        left_hip_flexion = utils.Decompose_R_ZXY(utils.Get_R(left_hip_knee, np.array([1, 0, 0])))
        left_hip_adduction = utils.Decompose_R_ZXY(utils.Get_R(left_hip_knee, np.array([0, 1, 0])))
        left_hip_rotation = utils.Decompose_R_ZXY(utils.Get_R(left_hip_knee, np.array([0, 0, 1])))

        right_hip_flexion = utils.Decompose_R_ZXY(utils.Get_R(right_hip_knee, np.array([1, 0, 0])))
        right_hip_adduction = utils.Decompose_R_ZXY(utils.Get_R(right_hip_knee, np.array([0, 1, 0])))
        right_hip_rotation = utils.Decompose_R_ZXY(utils.Get_R(right_hip_knee, np.array([0, 0, 1])))

        # Knee Joint Angles
        left_knee_ankle = kpts['leftknee'][framenum] - kpts['leftfoot'][framenum]
        right_knee_ankle = kpts['rightknee'][framenum] - kpts['rightfoot'][framenum]

        left_knee_flexion = utils.Decompose_R_ZXY(utils.Get_R(left_hip_knee, left_knee_ankle))
        right_knee_flexion = utils.Decompose_R_ZXY(utils.Get_R(right_hip_knee, right_knee_ankle))

        # Print results for each frame
        print(f"Frame {framenum + 1}:")
        print(f" Left Hip Flexion: {np.degrees(left_hip_flexion[1]):.2f}°")
        print(f" Left Hip Adduction: {np.degrees(left_hip_adduction[1]):.2f}°")
        print(f" Left Hip Rotation: {np.degrees(left_hip_rotation[1]):.2f}°")

        print(f" Right Hip Flexion: {np.degrees(right_hip_flexion[1]):.2f}°")
        print(f" Right Hip Adduction: {np.degrees(right_hip_adduction[1]):.2f}°")
        print(f" Right Hip Rotation: {np.degrees(right_hip_rotation[1]):.2f}°")

        print(f" Left Knee Flexion: {np.degrees(left_knee_flexion[1]):.2f}°")
        print(f" Right Knee Flexion: {np.degrees(right_knee_flexion[1]):.2f}°")
        print()


# Main execution
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Call program with input pose file')
        quit()

    filename = sys.argv[1]
    kpts = read_keypoints(filename)

    kpts = convert_to_dictionary(kpts)
    kpts = add_hips(kpts)  # Add hips keypoint to the dictionary
    filtered_kpts = median_filter(kpts)  # Filtered keypoints

    # Calculate and display joint angles (x, y, z) for hip and knee
    calculate_joint_3D_angles(filtered_kpts)