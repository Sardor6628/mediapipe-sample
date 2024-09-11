import tkinter as tk
import cv2
import mediapipe as mp
from PIL import Image, ImageTk
import os

# Suppress TensorFlow-related warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Mediapipe setup for pose detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Tkinter setup
root = tk.Tk()
root.title("Real-time Video & Mediapipe Pose Detection")
root.geometry('800x600')

# Function to start Mediapipe real-time pose detection in the new window
def process_video():
    # Create a new window for video feed
    new_window = tk.Toplevel(root)
    new_window.title("Pose Detection")
    new_window.geometry('800x600')

    # Add a label for the video feed in the new window
    new_video_label = tk.Label(new_window)
    new_video_label.pack()

    # Initialize OpenCV for video capture and Mediapipe pose detection
    cap = cv2.VideoCapture(0)  # Access webcam
    pose = mp_pose.Pose()

    def update_frame():
        success, frame = cap.read()
        if success:
            # Convert the frame to RGB (Tkinter-compatible)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform pose detection
            results = pose.process(rgb_frame)

            # Draw pose landmarks on the frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(rgb_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Convert the OpenCV image (numpy array) to PIL image for Tkinter
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the new_video_label with the new frame
            new_video_label.imgtk = imgtk  # Keep a reference to avoid garbage collection
            new_video_label.configure(image=imgtk)

        # Continue updating the video frame every 10 ms
        new_video_label.after(10, update_frame)

    # Start updating frames in the new window
    update_frame()

    # Close the video capture when the window is closed
    def close_video():
        cap.release()
        new_window.destroy()

    # Add a close button to close the window and stop the video
    close_button = tk.Button(new_window, text="Close Window", command=close_video)
    close_button.pack(pady=20)

# Button to start video processing
try:
    start_video_button = tk.Button(root, text="Start Pose Detection", command=process_video)
    start_video_button.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    root.mainloop()
except Exception as e:
    print('Error:', e)
finally:
    root.destroy()
