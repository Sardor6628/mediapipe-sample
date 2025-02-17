from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model
model = YOLO("detect_wrist/train/weights/best.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 on the frame
    results = model(frame, conf=0.1)

    # Show results
    for result in results:
        annotated_frame = result.plot()  # Draw detections

    cv2.imshow("YOLOv8 Real-time Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()