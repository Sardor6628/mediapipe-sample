import cv2
import time
import mediapipe as mp
import json
import os

def process_video(video_path):
    # Initialize Mediapipe holistic model
    mp_holistic = mp.solutions.holistic
    holistic_model = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # Capture video from the given path
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print("Error opening video file")
        return

    # Get video details
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = f'./{video_name}_output'
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, f'{video_name}_processed.mp4')

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Initialize time variables for calculating FPS
    previousTime = 0

    # Initialize JSON data structure
    landmark_data = []

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        # Convert the frame from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Make predictions using holistic model
        image.flags.writeable = False
        results = holistic_model.process(image)
        image.flags.writeable = True

        # Convert back the RGB image to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks and display values
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            # Record landmark data
            landmarks = results.pose_landmarks.landmark
            frame_landmarks = [{'id': idx, 'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility} for idx, lm in enumerate(landmarks)]
            landmark_data.append(frame_landmarks)

        # Calculate FPS
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        # Display FPS on the image
        cv2.putText(image, str(int(fps)) + " FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        # Write the frame to the output video
        out.write(image)

        # Display the resulting image
        cv2.imshow("Body and Hand Landmarks", image)

        # Press 'q' to exit
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Release resources
    capture.release()
    out.release()
    cv2.destroyAllWindows()

    # Write landmark data to JSON file
    json_output_path = os.path.join(output_dir, f'{video_name}_landmarks.json')
    with open(json_output_path, 'w') as f:
        json.dump(landmark_data, f, indent=4)

    print(f'Processed video saved at {output_video_path}')
    print(f'Landmark data saved at {json_output_path}')

# Example usage
video_path = 'SIH_dynamic_0_right2.MOV'
process_video(video_path)