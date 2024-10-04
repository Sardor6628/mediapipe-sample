import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to define the LCS for a segment with consistent y-axis and z-axis directions
def create_lcs(origin, proximal, ref_point, enforce_y_direction=None, enforce_z_direction=None):
    y_axis = (proximal - origin) / np.linalg.norm(proximal - origin)
    temp_vector = ref_point - origin
    z_axis = np.cross(y_axis, temp_vector)
    z_axis /= np.linalg.norm(z_axis)

    # Enforce consistent y-axis direction if specified
    if enforce_y_direction is not None:
        if np.dot(y_axis, enforce_y_direction) < 0:
            y_axis = -y_axis
            z_axis = -z_axis  # Flip z-axis to maintain a right-handed coordinate system

    # Enforce consistent z-axis direction if specified
    if enforce_z_direction is not None:
        if np.dot(z_axis, enforce_z_direction) < 0:
            z_axis = -z_axis

    # Recalculate x-axis to ensure orthogonality
    x_axis = np.cross(y_axis, z_axis)
    return x_axis, y_axis, z_axis

# Convert normalized landmark coordinates to pixel coordinates
def to_pixel_coords(coord, image_shape):
    h, w, _ = image_shape
    return int(coord[0] * w), int(coord[1] * h)

# Display XYZ coordinates on the frame
def display_coordinates(frame, coord, label, position):
    text = f"{label}: ({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})"
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# Calculate knee angles on sagittal, frontal, and transverse planes
def calculate_knee_angles(y_knee, x_knee, z_knee, enforced_y_direction, enforced_z_direction):
    sagittal_angle = np.degrees(np.arccos(np.clip(np.dot(y_knee, enforced_y_direction), -1.0, 1.0)))
    # Sagittal plane (flexion/extension): angle between y_knee and enforced_y_direction

    # Frontal plane (adduction/abduction): angle between x_knee and enforced_y_direction

    # Transverse plane (internal/external rotation): angle between z_knee and enforced_z_direction

    frontal_angle = np.degrees(np.arccos(np.clip(np.dot(x_knee, enforced_z_direction), -1.0, 1.0)))

    transverse_angle = np.degrees(np.arccos(np.clip(np.dot(z_knee, enforced_y_direction), -1.0, 1.0)))

    return sagittal_angle, frontal_angle, transverse_angle

# Capture video
cap = cv2.VideoCapture(1)

# Define a global direction for y-axis and z-axis
enforced_y_direction = np.array([0, 1, 0])  # Upward direction
enforced_z_direction = np.array([1, 0, 0])  # Rightward direction

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert the BGR image to RGB and process with MediaPipe Pose
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extract joint landmarks for right hip, knee, ankle, and reference points
        r_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z])
        r_knee = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y,
                           landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].z])
        r_ankle = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].z])

        # Use RIGHT_SHOULDER as reference for hip to improve consistency
        r_shoulder_ref = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z])
        # Use RIGHT_FOOT_INDEX as the reference point for the knee and ankle
        r_shank_ref = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y,
                                landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].z])

        # Define LCS for hip, knee, and ankle with enforced y-axis and z-axis directions
        x_hip, y_hip, z_hip = create_lcs(r_knee, r_hip, r_shoulder_ref, enforce_y_direction=enforced_y_direction, enforce_z_direction=enforced_z_direction)
        x_knee, y_knee, z_knee = create_lcs(r_knee, r_ankle, r_shank_ref, enforce_y_direction=enforced_y_direction, enforce_z_direction=enforced_z_direction)
        x_ankle, y_ankle, z_ankle = create_lcs(r_ankle, r_knee, r_shank_ref, enforce_y_direction=enforced_y_direction, enforce_z_direction=enforced_z_direction)

        # Calculate knee angles
        sagittal_angle, frontal_angle, transverse_angle = calculate_knee_angles(y_knee, x_knee, z_knee, enforced_y_direction, enforced_z_direction)

        # Draw LCS lines for each joint (Hip, Knee, Ankle)
        lcs_length = 0.1  # Scale factor for LCS lines

        # Draw hip LCS
        hip_origin_pixel = to_pixel_coords(r_hip, frame.shape)
        cv2.line(frame, hip_origin_pixel, to_pixel_coords(r_hip + lcs_length * x_hip, frame.shape), (0, 0, 255), 2)  # X-axis in red
        cv2.line(frame, hip_origin_pixel, to_pixel_coords(r_hip + lcs_length * y_hip, frame.shape), (0, 255, 0), 2)  # Y-axis in green
        cv2.line(frame, hip_origin_pixel, to_pixel_coords(r_hip + lcs_length * z_hip, frame.shape), (255, 0, 0), 2)  # Z-axis in blue

        # Draw knee LCS
        knee_origin_pixel = to_pixel_coords(r_knee, frame.shape)
        cv2.line(frame, knee_origin_pixel, to_pixel_coords(r_knee + lcs_length * x_knee, frame.shape), (0, 0, 255), 2)  # X-axis in red
        cv2.line(frame, knee_origin_pixel, to_pixel_coords(r_knee + lcs_length * y_knee, frame.shape), (0, 255, 0), 2)  # Y-axis in green
        cv2.line(frame, knee_origin_pixel, to_pixel_coords(r_knee + lcs_length * z_knee, frame.shape), (255, 0, 0), 2)  # Z-axis in blue

        # Draw ankle LCS
        ankle_origin_pixel = to_pixel_coords(r_ankle, frame.shape)
        cv2.line(frame, ankle_origin_pixel, to_pixel_coords(r_ankle + lcs_length * x_ankle, frame.shape), (0, 0, 255), 2)  # X-axis in red
        cv2.line(frame, ankle_origin_pixel, to_pixel_coords(r_ankle + lcs_length * y_ankle, frame.shape), (0, 255, 0), 2)  # Y-axis in green
        cv2.line(frame, ankle_origin_pixel, to_pixel_coords(r_ankle + lcs_length * z_ankle, frame.shape), (255, 0, 0), 2)  # Z-axis in blue

        # Display XYZ coordinates for hip, knee, and ankle
        display_coordinates(frame, r_hip, "Hip", (hip_origin_pixel[0], hip_origin_pixel[1] - 10))
        display_coordinates(frame, r_knee, "Knee", (knee_origin_pixel[0], knee_origin_pixel[1] - 10))
        display_coordinates(frame, r_ankle, "Ankle", (ankle_origin_pixel[0], ankle_origin_pixel[1] - 10))

        # Display calculated knee angles near the knee
        angle_text = f"S: {sagittal_angle:.2f}, F: {frontal_angle:.2f}, T: {transverse_angle:.2f}"
        cv2.putText(frame, angle_text, (knee_origin_pixel[0], knee_origin_pixel[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow("Pose with LCS, Angles, and Coordinates", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()