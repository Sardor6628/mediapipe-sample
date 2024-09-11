import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp
import json
import os

# Helper function to calculate angle between three points
def find_angle(a, b, c, min_visibility=0.8):
    if a.visibility > min_visibility and b.visibility > min_visibility and c.visibility > min_visibility:
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        angle = np.arccos(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))) * (180.0 / np.pi)
        return angle if angle <= 180 else 360 - angle
    else:
        return -1

def leg_state(angle):
    if angle < 0:
        return 0  # Not detected
    elif angle < 105:
        return 1  # Squatting
    elif angle < 130:
        return 2  # Transitioning
    else:
        return 3  # Standing

def initialize_pose():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    return pose

def initialize_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    return pipeline, config

def start_streaming(pipeline, config):
    pipeline.start(config)

def initialize_output(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    video_output_path = os.path.join(output_dir, "output_video.mp4")
    json_output_path = os.path.join(output_dir, "landmarks_output.json")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, 30, (640, 480))
    return out, video_output_path, json_output_path

def record_landmarks(frame_count, results, landmark_data_json):
    landmarks_frame_data = []
    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmarks_frame_data.append({
                "id": idx,
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility
            })
    landmark_data_json.append({"frame": frame_count, "landmarks": landmarks_frame_data})

def process_frame(pipeline, pose, max_retries=5):
    retries = 0
    while retries < max_retries:
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            retries += 1
            continue
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        results = pose.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        return depth_image, color_image, results
    return None, None, None

def display_status(annotated_image, status_text, color):
    cv2.putText(annotated_image, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

def draw_landmark_connections(annotated_image, results):
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

def save_output(video_writer, frame, annotated_image):
    video_writer.write(annotated_image)
    cv2.imshow('RealSense + MediaPipe Pose', annotated_image)

def draw_squat_count(annotated_image, squat_count):
    cv2.putText(annotated_image, f"Squat: {squat_count}", (annotated_image.shape[1] - 200, annotated_image.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

def check_body_visibility(results, min_visible_landmarks=30, visibility_threshold=0.3):
    if results.pose_landmarks:
        visible_landmarks = [l for l in results.pose_landmarks.landmark if l.visibility >= visibility_threshold]
        if len(visible_landmarks) >= min_visible_landmarks:
            return True  # Full body is tracked
        return False  # Partial tracking
    return None  # No body tracked

def main():
    # Initialize components
    pose = initialize_pose()
    pipeline, config = initialize_realsense()
    start_streaming(pipeline, config)
    output_dir = "output"
    out, video_output_path, json_output_path = initialize_output(output_dir)
    landmark_data_json = []
    squat_count = 0
    last_state = 3  # Assume standing initially

    try:
        frame_count = 0
        while True:
            depth_image, color_image, results = process_frame(pipeline, pose)
            if depth_image is None or color_image is None:
                continue
            annotated_image = color_image.copy()
            draw_landmark_connections(annotated_image, results)
            body_visibility = check_body_visibility(results)

            if body_visibility is None:
                display_status(annotated_image, "No body tracked", (0, 0, 255))
            elif body_visibility:
                display_status(annotated_image, "Full body tracked", (0, 255, 0))
            else:
                display_status(annotated_image, "Full body couldn't be tracked", (0, 0, 255))

            if results.pose_landmarks:
                r_knee_angle = find_angle(results.pose_landmarks.landmark[24], results.pose_landmarks.landmark[26], results.pose_landmarks.landmark[28])
                l_knee_angle = find_angle(results.pose_landmarks.landmark[23], results.pose_landmarks.landmark[25], results.pose_landmarks.landmark[27])
                print(f"Right Knee Angle: {r_knee_angle:.2f}, Left Knee Angle: {l_knee_angle:.2f}")
                r_state = leg_state(r_knee_angle)
                l_state = leg_state(l_knee_angle)
                print(f"Right Leg State: {r_state}, Left Leg State: {l_state}")

                if r_state == 1 and l_state == 1 and last_state == 3:
                    print("Transitioning to Squat")
                    last_state = 1
                elif r_state == 3 and l_state == 3 and last_state == 1:
                    squat_count += 1
                    print(f"Squat Count: {squat_count}")
                    last_state = 3

            draw_squat_count(annotated_image, squat_count)
            save_output(out, frame_count, annotated_image)
            record_landmarks(frame_count, results, landmark_data_json)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    finally:
        pipeline.stop()
        out.release()
        cv2.destroyAllWindows()
        with open(json_output_path, 'w') as json_file:
            json.dump(landmark_data_json, json_file, indent=4)

        print(f"Video saved to {video_output_path}")
        print(f"Landmarks saved to {json_output_path}")

if __name__ == "__main__":
    main()