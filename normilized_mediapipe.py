import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize drawing utils
mp_drawing = mp.solutions.drawing_utils

# Start capturing video input (0 is the default webcam)
cap = cv2.VideoCapture(1)

def normalize_landmarks(landmarks):
    """
    Normalizes the landmarks relative to the pelvis (hips).
    :param landmarks: List of landmarks (33 landmarks with x, y, z values)
    :return: Normalized landmarks
    """
    # Get the left and right hip landmarks (landmark 24 and 23)
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # Calculate pelvis as the midpoint of the two hips
    pelvis_x = (left_hip.x + right_hip.x) / 2
    pelvis_y = (left_hip.y + right_hip.y) / 2
    pelvis_z = (left_hip.z + right_hip.z) / 2

    # Normalize all landmarks relative to the pelvis
    normalized_landmarks = []
    for lm in landmarks:
        normalized_landmarks.append({
            'x': lm.x - pelvis_x,
            'y': lm.y - pelvis_y,
            'z': lm.z - pelvis_z
        })

    return normalized_landmarks

def draw_landmarks(image, landmarks, normalized_landmarks):
    """
    Draw both original and normalized landmarks on the image.
    :param image: The image to draw on
    :param landmarks: Original landmarks
    :param normalized_landmarks: Normalized landmarks
    """
    h, w, _ = image.shape

    for idx, lm in enumerate(normalized_landmarks):
        x = int(lm['x'] * w + w / 2)
        y = int(lm['y'] * h + h / 2)

        # Draw normalized landmarks as blue circles
        cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

        # Display the coordinates on the screen
        if idx > 23:
            cv2.putText(image, f"x={lm['x']:.2f}, y={lm['y']:.2f}, z={lm['z']:.2f}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Draw original landmarks (for comparison) in green
    mp_drawing.draw_landmarks(image, landmarks, mp_pose.POSE_CONNECTIONS)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Ignoring empty camera frame.")
        continue

    # Convert the image to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect the pose
    result = pose.process(image_rgb)

    # If landmarks are detected
    if result.pose_landmarks:
        # Get the original landmarks
        landmarks = result.pose_landmarks.landmark

        # Normalize the landmarks relative to the pelvis
        normalized_landmarks = normalize_landmarks(landmarks)

        # Draw both original and normalized landmarks on the image
        draw_landmarks(frame, result.pose_landmarks, normalized_landmarks)

    # Display the result
    cv2.imshow('Normalized Landmarks in Real-time', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pose.close()