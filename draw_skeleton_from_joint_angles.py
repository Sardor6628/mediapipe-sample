import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)


# Capture video from the webcam
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Check if landmarks are detected
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Right shoulder flexion/extension angle (neck, right shoulder, right elbow)
        neck = [landmarks[mp_pose.PoseLandmark.NOSE.value].x * frame.shape[1],
                landmarks[mp_pose.PoseLandmark.NOSE.value].y * frame.shape[0]]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * frame.shape[1],
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * frame.shape[0]]

        # Calculate shoulder flexion/extension angle
        shoulder_angle = calculate_angle(neck, right_shoulder, right_elbow)

        # Right elbow flexion/extension angle (right shoulder, right elbow, right wrist)
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * frame.shape[1],
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * frame.shape[0]]

        # Calculate elbow flexion/extension angle
        elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Display the angles on the screen
        cv2.putText(frame, f"Shoulder Angle: {int(shoulder_angle)} degrees",
                    (int(right_shoulder[0]), int(right_shoulder[1]) - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Elbow Angle: {int(elbow_angle)} degrees",
                    (int(right_elbow[0]), int(right_elbow[1]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw the landmarks and connections
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Draw circles on key points
        cv2.circle(frame, (int(neck[0]), int(neck[1])), 5, (0, 255, 0), -1)
        cv2.circle(frame, (int(right_shoulder[0]), int(right_shoulder[1])), 5, (0, 255, 0), -1)
        cv2.circle(frame, (int(right_elbow[0]), int(right_elbow[1])), 5, (0, 255, 0), -1)
        cv2.circle(frame, (int(right_wrist[0]), int(right_wrist[1])), 5, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow('Bench Press Joint Angle Estimation', frame)

    # Break the loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()