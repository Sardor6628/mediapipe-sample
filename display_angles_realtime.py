import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize MediaPipe drawing utilities for visualizing landmarks
mp_drawing = mp.solutions.drawing_utils

# Helper function to calculate the general knee angle between three points
def find_angle(a, b, c, min_visibility=0.6):
    try:
        if a.visibility > min_visibility and b.visibility > min_visibility and c.visibility > min_visibility:
            ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
            bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
            angle = np.arccos(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))) * (180.0 / np.pi)
            return angle if angle <= 180 else 360 - angle
        else:
            return -1
    except Exception as e:
        return -1

# Helper function to calculate angle between two joints relative to the x, y, z axes
def calculate_joint_angle_relative_to_axes(b, c, min_visibility=0.6):
    if b.visibility > min_visibility and c.visibility > min_visibility:
        # Vector from joint b to joint c
        vector_bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
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

# Function to process the frame and calculate both the general knee angle and the relative angles
def process_pose(landmarks, frame):
    # Get the landmarks required for the right and left knees
    right_hip = landmarks[24]
    right_knee = landmarks[26]
    right_ankle = landmarks[28]
    left_hip = landmarks[23]
    left_knee = landmarks[25]
    left_ankle = landmarks[27]

    # Calculate the right knee general angle
    r_knee_angle = find_angle(right_hip, right_knee, right_ankle)

    # Calculate the left knee general angle
    l_knee_angle = find_angle(left_hip, left_knee, left_ankle)

    # Calculate the right knee angles relative to the x, y, z axes
    r_knee_angle_x, r_knee_angle_y, r_knee_angle_z = calculate_joint_angle_relative_to_axes(right_knee, right_ankle)

    # Calculate the left knee angles relative to the x, y, z axes
    l_knee_angle_x, l_knee_angle_y, l_knee_angle_z = calculate_joint_angle_relative_to_axes(left_knee, left_ankle)

    # Get the dimensions of the frame
    height, width, _ = frame.shape

    # Display right knee angles (general and x, y, z) on the bottom right side of the frame
    cv2.putText(frame, f"R Knee Angle: {r_knee_angle:.2f}", (width - 300, height - 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"R Knee X: {r_knee_angle_x:.2f}", (width - 300, height - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"R Knee Y: {r_knee_angle_y:.2f}", (width - 300, height - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"R Knee Z: {r_knee_angle_z:.2f}", (width - 300, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display left knee angles (general and x, y, z) on the bottom left side of the frame
    cv2.putText(frame, f"L Knee Angle: {l_knee_angle:.2f}", (30, height - 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"L Knee X: {l_knee_angle_x:.2f}", (30, height - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"L Knee Y: {l_knee_angle_y:.2f}", (30, height - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"L Knee Z: {l_knee_angle_z:.2f}", (30, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Start capturing video from the webcam
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame from BGR to RGB as MediaPipe requires RGB input
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect pose landmarks
    results = pose.process(image)

    # Convert the image back to BGR for OpenCV display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # If landmarks are detected, process and display them
    if results.pose_landmarks:
        # Draw pose landmarks on the frame
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
        )

        # Calculate and display the knee angles (general and x, y, z)
        process_pose(results.pose_landmarks.landmark, image)

    # Display the frame with landmarks and knee angles
    cv2.imshow('MediaPipe Pose - Knee Angles', image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()