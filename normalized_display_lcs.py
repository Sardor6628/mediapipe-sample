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
    if np.dot(z_axis, enforce_z_direction) < 0:
        z_axis = -z_axis

    # Recalculate x-axis to ensure orthogonality
    x_axis = np.cross(y_axis, z_axis)
    return x_axis, y_axis, z_axis

# Calculate knee angles on sagittal, frontal, and transverse planes
def calculate_knee_angles(y_knee, x_knee, z_knee, enforced_y_direction, enforced_z_direction):
    sagittal_angle = np.degrees(np.arccos(np.clip(np.dot(y_knee, enforced_y_direction), -1.0, 1.0)))
    frontal_angle = np.degrees(np.arccos(np.clip(np.dot(x_knee, enforced_z_direction), -1.0, 1.0)))
    transverse_angle = np.degrees(np.arccos(np.clip(np.dot(z_knee, enforced_y_direction), -1.0, 1.0)))
    return sagittal_angle, frontal_angle, transverse_angle

# Normalize landmarks relative to the pelvis
def normalize_landmarks(landmarks):
    pelvis_x = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2
    pelvis_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
    pelvis_z = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z) / 2

    normalized_landmarks = []
    for lm in landmarks:
        normalized_landmarks.append({
            'x': lm.x - pelvis_x,
            'y': lm.y - pelvis_y,
            'z': lm.z - pelvis_z
        })
    return normalized_landmarks

# Capture video
cap = cv2.VideoCapture(1)

# Global enforcement of y-axis and z-axis direction
enforced_y_direction = np.array([0, 1, 0])
enforced_z_direction = np.array([1, 0, 0])

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert the BGR image to RGB and process with MediaPipe Pose
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Normalize the landmarks relative to the pelvis
        normalized_landmarks = normalize_landmarks(landmarks)

        # Extract joint landmarks for right hip, knee, and ankle from normalized data
        r_hip = np.array([normalized_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]['x'],
                          normalized_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]['y'],
                          normalized_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]['z']])
        r_knee = np.array([normalized_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]['x'],
                           normalized_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]['y'],
                           normalized_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]['z']])
        r_ankle = np.array([normalized_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]['x'],
                            normalized_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]['y'],
                            normalized_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]['z']])
        r_shoulder_ref = np.array([normalized_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]['x'],
                                   normalized_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]['y'],
                                   normalized_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]['z']])
        r_shank_ref = np.array([normalized_landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]['x'],
                                normalized_landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]['y'],
                                normalized_landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]['z']])

        # Calculate LCS for hip, knee, and ankle from normalized data
        x_knee, y_knee, z_knee = create_lcs(r_knee, r_ankle, r_shank_ref, enforced_y_direction, enforced_z_direction)

        # Calculate knee angles using normalized data
        sagittal_angle, frontal_angle, transverse_angle = calculate_knee_angles(y_knee, x_knee, z_knee, enforced_y_direction, enforced_z_direction)

        # Display knee angles near the knee
        angle_text = f"S: {sagittal_angle:.2f}, F: {frontal_angle:.2f}, T: {transverse_angle:.2f}"
        knee_origin_pixel = (50, 50)  # Position to display the angles
        cv2.putText(frame, angle_text, knee_origin_pixel, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow("Knee Angles", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()