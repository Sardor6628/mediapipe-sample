#!/usr/bin/python

# Import libraries
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import math
from scipy.io import savemat

# MediaPipe modules
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# Function to calculate angles between joints in 3D
def calculate_3d_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point (joint of interest)
    c = np.array(c)  # Last point
    # Vectors in 3D space
    ab = a - b
    bc = c - b
    # Normalize vectors
    ab_norm = ab / np.linalg.norm(ab)
    bc_norm = bc / np.linalg.norm(bc)
    # Calculate the angle between the two vectors
    angle = np.arccos(np.clip(np.dot(ab_norm, bc_norm), -1.0, 1.0))
    return np.degrees(angle)  # Convert radians to degrees


# Real-time processing function
def run():
    # Define the webcam feed
    cap = cv2.VideoCapture(1)  # Use webcam as input (0 refers to default webcam)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    # Set default resolution (can be adjusted)
    width, height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Initialize a list to store the data
    data = []

    # MediaPipe Pose instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Start real-time loop
        while cap.isOpened():
            ret, frame = cap.read()  # Capture frame-by-frame from the webcam
            if not ret:
                print("Failed to capture frame. Exiting...")
                break

            # Resize frame to chosen resolution
            frame = cv2.resize(frame, (width, height))

            # Convert the image to RGB (required for MediaPipe)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False  # Optimize processing

            # Detect the pose
            results = pose.process(image_rgb)

            # Revert the image to BGR for OpenCV display
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # Extract pose landmarks if available
            try:
                landmarks = results.pose_landmarks.landmark

                # Extract 3D coordinates of joints for left and right sides
                joints = {
                    "lt_hip": [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z],
                    "lt_knee": [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z],
                    "lt_ankle": [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z],
                    "rt_hip": [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z],
                    "rt_knee": [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z],
                    "rt_ankle": [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
                }

                # For left and right hips, knees, and ankles, calculate the sagittal, frontal, and transverse angles
                # Sagittal plane (y-axis movements), Frontal plane (x-axis movements), Transverse plane (z-axis movements)

                # For hips (using the knee and ankle for reference)
                lt_hip_sagittal = calculate_3d_angle(joints["lt_hip"], joints["lt_knee"],
                                                     [joints["lt_hip"][0], joints["lt_knee"][1],
                                                      joints["lt_hip"][2]])  # y-axis
                lt_hip_frontal = calculate_3d_angle(joints["lt_hip"], joints["lt_knee"],
                                                    [joints["lt_knee"][0], joints["lt_hip"][1],
                                                     joints["lt_hip"][2]])  # x-axis
                lt_hip_transverse = calculate_3d_angle(joints["lt_hip"], joints["lt_knee"],
                                                       [joints["lt_knee"][0], joints["lt_knee"][1],
                                                        joints["lt_hip"][2]])  # z-axis

                rt_hip_sagittal = calculate_3d_angle(joints["rt_hip"], joints["rt_knee"],
                                                     [joints["rt_hip"][0], joints["rt_knee"][1],
                                                      joints["rt_hip"][2]])  # y-axis
                rt_hip_frontal = calculate_3d_angle(joints["rt_hip"], joints["rt_knee"],
                                                    [joints["rt_knee"][0], joints["rt_hip"][1],
                                                     joints["rt_hip"][2]])  # x-axis
                rt_hip_transverse = calculate_3d_angle(joints["rt_hip"], joints["rt_knee"],
                                                       [joints["rt_knee"][0], joints["rt_knee"][1],
                                                        joints["rt_hip"][2]])  # z-axis

                # For knees (using the ankle for reference)
                lt_knee_sagittal = calculate_3d_angle(joints["lt_knee"], joints["lt_ankle"],
                                                      [joints["lt_knee"][0], joints["lt_ankle"][1],
                                                       joints["lt_knee"][2]])  # y-axis
                lt_knee_frontal = calculate_3d_angle(joints["lt_knee"], joints["lt_ankle"],
                                                     [joints["lt_ankle"][0], joints["lt_knee"][1],
                                                      joints["lt_knee"][2]])  # x-axis
                lt_knee_transverse = calculate_3d_angle(joints["lt_knee"], joints["lt_ankle"],
                                                        [joints["lt_ankle"][0], joints["lt_ankle"][1],
                                                         joints["lt_knee"][2]])  # z-axis

                rt_knee_sagittal = calculate_3d_angle(joints["rt_knee"], joints["rt_ankle"],
                                                      [joints["rt_knee"][0], joints["rt_ankle"][1],
                                                       joints["rt_knee"][2]])  # y-axis
                rt_knee_frontal = calculate_3d_angle(joints["rt_knee"], joints["rt_ankle"],
                                                     [joints["rt_ankle"][0], joints["rt_knee"][1],
                                                      joints["rt_knee"][2]])  # x-axis
                rt_knee_transverse = calculate_3d_angle(joints["rt_knee"], joints["rt_ankle"],
                                                        [joints["rt_ankle"][0], joints["rt_ankle"][1],
                                                         joints["rt_knee"][2]])  # z-axis

                # For ankles (using the foot for reference)
                lt_ank_sagittal = calculate_3d_angle(joints["lt_ankle"], joints["lt_knee"],
                                                     [joints["lt_ankle"][0], joints["lt_knee"][1],
                                                      joints["lt_ankle"][2]])  # y-axis
                lt_ank_frontal = calculate_3d_angle(joints["lt_ankle"], joints["lt_knee"],
                                                    [joints["lt_knee"][0], joints["lt_ankle"][1],
                                                     joints["lt_ankle"][2]])  # x-axis
                lt_ank_transverse = calculate_3d_angle(joints["lt_ankle"], joints["lt_knee"],
                                                       [joints["lt_knee"][0], joints["lt_knee"][1],
                                                        joints["lt_ankle"][2]])  # z-axis

                rt_ank_sagittal = calculate_3d_angle(joints["rt_ankle"], joints["rt_knee"],
                                                     [joints["rt_ankle"][0], joints["rt_knee"][1],
                                                      joints["rt_ankle"][2]])  # y-axis
                rt_ank_frontal = calculate_3d_angle(joints["rt_ankle"], joints["rt_knee"],
                                                    [joints["rt_knee"][0], joints["rt_ankle"][1],
                                                     joints["rt_ankle"][2]])  # x-axis
                rt_ank_transverse = calculate_3d_angle(joints["rt_ankle"], joints["rt_knee"],
                                                       [joints["rt_knee"][0], joints["rt_knee"][1],
                                                        joints["rt_ankle"][2]])  # z-axis

                # Append the calculated angles to the data list
                data.append({
                    'lt_hip_sagittal': lt_hip_sagittal, 'lt_hip_frontal': lt_hip_frontal,
                    'lt_hip_transe': lt_hip_transverse,
                    'lt_knee_sagittal': lt_knee_sagittal, 'lt_knee_frontal': lt_knee_frontal,
                    'lt_knee_transe': lt_knee_transverse,
                    'lt_ank_sagittal': lt_ank_sagittal, 'lt_ank_frontal': lt_ank_frontal,
                    'lt_ank_transe': lt_ank_transverse,
                    'rt_hip_sagittal': rt_hip_sagittal, 'rt_hip_frontal': rt_hip_frontal,
                    'rt_hip_transe': rt_hip_transverse,
                    'rt_knee_sagittal': rt_knee_sagittal, 'rt_knee_frontal': rt_knee_frontal,
                    'rt_knee_transe': rt_knee_transverse,
                    'rt_ank_sagittal': rt_ank_sagittal, 'rt_ank_frontal': rt_ank_frontal,
                    'rt_ank_transe': rt_ank_transverse
                })

                # Visualize landmarks on the frame
                mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            except AttributeError:
                pass

            # Display the resulting frame
            cv2.imshow('Real-time Pose Detection', image_bgr)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and close windows
        cap.release()
        cv2.destroyAllWindows()

        # Convert the collected data to a DataFrame and save it in the required format
        df = pd.DataFrame(data)
        df.to_csv('pose_angles_data.csv', index=False)
        print("Data saved successfully to 'pose_angles_data.csv'.")


# Main function to run the real-time processing
if __name__ == '__main__':
    run()