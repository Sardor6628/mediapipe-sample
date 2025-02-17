import cv2
import numpy as np
import time
import pyrealsense2 as rs
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Smallest model for real-time inference

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB stream
pipeline.start(config)

# Depth measurement thresholds (meters)
START_DISTANCE_THRESHOLD = 0.5  # When the person starts moving
MARKER_DISTANCE_THRESHOLD = 3.0  # When they reach 3 meters
RETURN_DISTANCE_THRESHOLD = 0.5  # When they return to the chair

# Tracking variables
start_time = None
end_time = None
reached_marker = False
person_detected = False

# OpenCV Full Screen Setup
cv2.namedWindow("3M TUG Test Tracker", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("3M TUG Test Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("System Ready. Waiting for person to start...")

while True:
    # Get frames from RealSense camera
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    # Convert frames to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Run YOLO object detection on color image
    results = model(color_image)

    for result in results:
        for obj in result.boxes.data:
            x1, y1, x2, y2, conf, cls = obj.tolist()

            if int(cls) == 0:  # Class 0 is 'person' in the COCO dataset
                person_detected = True
                person_x_center = int((x1 + x2) / 2)
                person_y_center = int((y1 + y2) / 2)

                # Get depth measurement at detected person's center
                depth = depth_frame.get_distance(person_x_center, person_y_center)

                # Draw bounding box and display distance
                cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(color_image, f"Depth: {depth:.2f}m", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Start test when the person **starts moving forward**
                if start_time is None and depth > START_DISTANCE_THRESHOLD:
                    start_time = time.time()
                    print("Test started!")

                # Detect when the person reaches the **3-meter marker**
                if not reached_marker and depth > MARKER_DISTANCE_THRESHOLD:
                    reached_marker = True
                    print("Reached 3M marker!")

                # Stop test when person **returns and sits down**
                if reached_marker and depth < RETURN_DISTANCE_THRESHOLD:
                    end_time = time.time()
                    total_time = end_time - start_time
                    print(f"Test completed! Time: {total_time:.2f} seconds")

                    # Reset for next test
                    reached_marker = False
                    start_time = None
                    end_time = None
                    time.sleep(2)  # Small delay before allowing new test

    # Display the camera feed in full screen
    cv2.imshow("3M TUG Test Tracker", color_image)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop RealSense pipeline and close OpenCV window
pipeline.stop()
cv2.destroyAllWindows()