import cv2
import time
import mediapipe as mp

# Choose camera index (0 for default camera, 1 for the next camera, etc.)
camera_index = 0

mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils

# Change the camera index here to switch cameras
capture = cv2.VideoCapture(camera_index)

# Initializing current time and previous time for calculating the FPS
previousTime = 0
currentTime = 0

def draw_landmarks(image, landmarks, connections):
    if landmarks:
        for idx, landmark in enumerate(landmarks.landmark):
            if idx >10:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.putText(image, f'{idx}: ({landmark.x:.2f}, {landmark.y:.2f})', (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        if connections:
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                if start_idx >= 10 and end_idx >= 10:
                    start = landmarks.landmark[start_idx]
                    end = landmarks.landmark[end_idx]
                    start_point = (int(start.x * image.shape[1]), int(start.y * image.shape[0]))
                    end_point = (int(end.x * image.shape[1]), int(end.y * image.shape[0]))
                    cv2.line(image, start_point, end_point, (0, 255, 0), 2)

while capture.isOpened():
    # capture frame by frame
    ret, frame = capture.read()

    if not ret:
        print("Failed to grab frame")
        break

    # resizing the frame for better view
    frame = cv2.resize(frame, (800, 600))

    # Converting the from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Making predictions using holistic model
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True

    # Converting back the RGB image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Drawing the landmarks and displaying values
    draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Calculating the FPS
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # Displaying FPS on the image
    cv2.putText(image, str(int(fps)) + " FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting image
    cv2.imshow("Body and Hand Landmarks", image)

    # Enter key 'q' to break the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# When all the process is done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()