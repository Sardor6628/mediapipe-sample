import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Process the RGB frame with MediaPipe Pose
        results = pose.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        # Draw the pose annotation on the color image
        annotated_image = color_image.copy()
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extracting landmark coordinates with depth
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                # Get the 2D coordinates in the image
                x = int(landmark.x * color_image.shape[1])
                y = int(landmark.y * color_image.shape[0])

                # Ensure coordinates are within the frame
                if x >= 0 and x < depth_image.shape[1] and y >= 0 and y < depth_image.shape[0]:
                    depth_value = depth_image[y, x]

                    # Print landmark index, pixel coordinates, and depth value
                    # print(f"Landmark {idx}: (x={x}, y={y}, depth={depth_value} mm)")

                    # Draw the landmark and depth info on the annotated image
                    # cv2.putText(annotated_image, f"ID {idx}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    # cv2.putText(annotated_image, f"{depth_value}mm", (x, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Display the annotated image
        cv2.imshow('RealSense + MediaPipe Pose', annotated_image)

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()