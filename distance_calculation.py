import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("best.pt")

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
# Create a named window before the loop
cv2.namedWindow("YOLOv8 + RealSense Depth", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("YOLOv8 + RealSense Depth", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Start streaming
profile = pipeline.start(config)

# Get depth sensor scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_option(rs.option.depth_units)  # Convert depth to meters

try:
    while True:
        # Get frames from the camera
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert frames to NumPy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Run YOLOv8 object detection
        results = model(color_image, conf=0.1)

        # Draw detections and measure distances
        for result in results:
            for box in result.boxes.xyxy:  # Bounding boxes
                x1, y1, x2, y2 = map(int, box[:4])  # Convert to integers

                # Get the center of the bounding box
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                # Retrieve depth value at center of object
                distance = depth_image[center_y, center_x] * depth_scale

                # Draw bounding box and label
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{result.names[int(result.boxes.cls[0])]}: {distance:.2f}m"
                cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show the frame with detection and distance
        cv2.imshow("YOLOv8 + RealSense Depth", color_image)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()