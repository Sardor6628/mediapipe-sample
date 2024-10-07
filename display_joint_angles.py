import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set up pose detection
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize the camera
cap = cv2.VideoCapture(1)

# List to store joint angle data
joint_data = []

# Helper function to calculate angles in different planes
def calculate_joint_angles(p1, p2, p3):
    """Returns joint angles for sagittal, frontal, and transverse planes."""
    # Vectors from p1 to p2 (proximal to distal) and p2 to p3
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p2)

    # Normalize vectors to avoid scaling effects
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    # Sagittal plane (Y-Z): flexion/extension
    sagittal_angle = np.arctan2(v1[1], v1[2])

    # Frontal plane (X-Y): abduction/adduction
    frontal_angle = np.arctan2(v1[0], v1[1])

    # Transverse plane (X-Z): internal/external rotation
    transverse_angle = np.arctan2(v1[0], v1[2])

    return np.degrees(sagittal_angle), np.degrees(frontal_angle), np.degrees(transverse_angle)

# Pelvis angle calculation: reference to global coordinate system
def calculate_pelvis_angles(hip, opposite_hip, shoulder):
    """Calculates pelvis angles relative to the global coordinate system."""
    # Pelvis vector as the vector between the two hips
    pelvis_vector = np.array(hip) - np.array(opposite_hip)
    shoulder_vector = np.array(shoulder) - np.array(hip)

    # Sagittal plane (Y-Z): Flexion/Extension
    sagittal_angle = np.arctan2(shoulder_vector[1], shoulder_vector[2])

    # Frontal plane (X-Y): Lateral tilt
    frontal_angle = np.arctan2(pelvis_vector[0], pelvis_vector[1])

    # Transverse plane (X-Z): Rotation (twisting of pelvis)
    transverse_angle = np.arctan2(pelvis_vector[0], pelvis_vector[2])

    return np.degrees(sagittal_angle), np.degrees(frontal_angle), np.degrees(transverse_angle)

def display_text_on_screen(frame, text, position):
    """Helper function to display text on the screen at a given position."""
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def main():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Draw MediaPipe landmarks and connections on the frame
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()  # Removed connection drawing spec
            )

            # Extract landmarks
            landmarks = results.pose_landmarks.landmark

            # Extract key points for left and right sides
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP].z]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].z]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].z]

            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].z]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].z]

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z]

            # Pelvis angle calculation
            lt_plv_sagittal, lt_plv_frontal, lt_plv_transe = calculate_pelvis_angles(left_hip, right_hip, left_shoulder)
            rt_plv_sagittal, rt_plv_frontal, rt_plv_transe = calculate_pelvis_angles(right_hip, left_hip, right_shoulder)

            # Calculate joint angles for both sides (using hip-knee-ankle vectors)
            lt_hip_sagittal, lt_hip_frontal, lt_hip_transe = calculate_joint_angles(left_hip, left_knee, left_ankle)
            lt_knee_sagittal, lt_knee_frontal, lt_knee_transe = calculate_joint_angles(left_knee, left_ankle, [0,0,0])
            lt_ank_sagittal, lt_ank_frontal, lt_ank_transe = calculate_joint_angles(left_knee, left_ankle, [0,0,0])

            rt_hip_sagittal, rt_hip_frontal, rt_hip_transe = calculate_joint_angles(right_hip, right_knee, right_ankle)
            rt_knee_sagittal, rt_knee_frontal, rt_knee_transe = calculate_joint_angles(right_knee, right_ankle, [0,0,0])
            rt_ank_sagittal, rt_ank_frontal, rt_ank_transe = calculate_joint_angles(right_knee, right_ankle, [0,0,0])

            # Save the joint data in a dictionary
            joint_data.append({
                'lt_hip_sagittal': lt_hip_sagittal, 'lt_hip_frontal': lt_hip_frontal, 'lt_hip_transe': lt_hip_transe,
                'lt_knee_sagittal': lt_knee_sagittal, 'lt_knee_frontal': lt_knee_frontal, 'lt_knee_transe': lt_knee_transe,
                'lt_ank_sagittal': lt_ank_sagittal, 'lt_ank_frontal': lt_ank_frontal, 'lt_ank_transe': lt_ank_transe,
                'rt_hip_sagittal': rt_hip_sagittal, 'rt_hip_frontal': rt_hip_frontal, 'rt_hip_transe': rt_hip_transe,
                'rt_knee_sagittal': rt_knee_sagittal, 'rt_knee_frontal': rt_knee_frontal, 'rt_knee_transe': rt_knee_transe,
                'rt_ank_sagittal': rt_ank_sagittal, 'rt_ank_frontal': rt_ank_frontal, 'rt_ank_transe': rt_ank_transe,
                'lt_plv_sagittal': lt_plv_sagittal, 'lt_plv_frontal': lt_plv_frontal, 'lt_plv_transe': lt_plv_transe,
                'rt_plv_sagittal': rt_plv_sagittal, 'rt_plv_frontal': rt_plv_frontal, 'rt_plv_transe': rt_plv_transe
            })

            # Display left side values (on the left side of the screen)
            display_text_on_screen(frame, f'Lt Hip Sagittal: {lt_hip_sagittal:.2f}', (50, 50))
            display_text_on_screen(frame, f'Lt Hip Frontal: {lt_hip_frontal:.2f}', (50, 80))
            display_text_on_screen(frame, f'Lt Hip Transverse: {lt_hip_transe:.2f}', (50, 110))

            display_text_on_screen(frame, f'Lt Knee Sagittal: {lt_knee_sagittal:.2f}', (50, 140))
            display_text_on_screen(frame, f'Lt Knee Frontal: {lt_knee_frontal:.2f}', (50, 170))
            display_text_on_screen(frame, f'Lt Knee Transverse: {lt_knee_transe:.2f}', (50, 200))

            display_text_on_screen(frame, f'Lt Ankle Sagittal: {lt_ank_sagittal:.2f}', (50, 230))
            display_text_on_screen(frame, f'Lt Ankle Frontal: {lt_ank_frontal:.2f}', (50, 260))
            display_text_on_screen(frame, f'Lt Ankle Transverse: {lt_ank_transe:.2f}', (50, 290))

            display_text_on_screen(frame, f'Lt Pelvis Sagittal: {lt_plv_sagittal:.2f}', (50, 320))
            display_text_on_screen(frame, f'Lt Pelvis Frontal: {lt_plv_frontal:.2f}', (50, 350))
            display_text_on_screen(frame, f'Lt Pelvis Transverse: {lt_plv_transe:.2f}', (50, 380))

            # Display right side values (on the right side of the screen)
            display_text_on_screen(frame, f'Rt Hip Sagittal: {rt_hip_sagittal:.2f}', (400, 50))
            display_text_on_screen(frame, f'Rt Hip Frontal: {rt_hip_frontal:.2f}', (400, 80))
            display_text_on_screen(frame, f'Rt Hip Transverse: {rt_hip_transe:.2f}', (400, 110))

            display_text_on_screen(frame, f'Rt Knee Sagittal: {rt_knee_sagittal:.2f}', (400, 140))
            display_text_on_screen(frame, f'Rt Knee Frontal: {rt_knee_frontal:.2f}', (400, 170))
            display_text_on_screen(frame, f'Rt Knee Transverse: {rt_knee_transe:.2f}', (400, 200))

            display_text_on_screen(frame, f'Rt Ankle Sagittal: {rt_ank_sagittal:.2f}', (400, 230))
            display_text_on_screen(frame, f'Rt Ankle Frontal: {rt_ank_frontal:.2f}', (400, 260))
            display_text_on_screen(frame, f'Rt Ankle Transverse: {rt_ank_transe:.2f}', (400, 290))

            display_text_on_screen(frame, f'Rt Pelvis Sagittal: {rt_plv_sagittal:.2f}', (400, 320))
            display_text_on_screen(frame, f'Rt Pelvis Frontal: {rt_plv_frontal:.2f}', (400, 350))
            display_text_on_screen(frame, f'Rt Pelvis Transverse: {rt_plv_transe:.2f}', (400, 380))

        # Display the frame
        cv2.imshow('Pose Landmarks with Joint Angles', frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save joint data to CSV
    df = pd.DataFrame(joint_data)
    df.to_csv('joint_angles_vicon_like_with_pelvis.csv', index=False)
    print("Joint angle data saved to 'joint_angles_vicon_like_with_pelvis.csv'.")

if __name__ == "__main__":
    main()