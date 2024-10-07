import numpy as np
import json
import os


def convert_npy_folder_to_single_json(base_folder_path, output_json_path):
    # List to hold all frames data
    all_frames_data = []

    # Loop over all files in the folder
    for file_name in sorted(os.listdir(base_folder_path)):
        if file_name.endswith('.npy'):
            # Get full file path
            npy_file_path = os.path.join(base_folder_path, file_name)

            # Extract frame number from the file name (remove the extension and convert to int)
            frame_number = int(os.path.splitext(file_name)[0])
            if frame_number==0:
                continue

            # Load the npy file
            data = np.load(npy_file_path)

            # Number of landmarks (each landmark has 4 values: x, y, z, visibility)
            num_landmarks = len(data) // 4

            # List to hold this frame's landmarks data
            frame_data = {
                "frame": frame_number,
                "landmarks": []
            }

            # Process the data in chunks of 4 values (x, y, z, visibility) for each landmark
            for i in range(num_landmarks):
                landmark = {
                    "id": i,
                    "x": data[i * 4],
                    "y": data[i * 4 + 1],
                    "z": data[i * 4 + 2],
                    "visibility": data[i * 4 + 3]
                }
                frame_data["landmarks"].append(landmark)

            # Append this frame's data to the all_frames_data list
            all_frames_data.append(frame_data)

    # Write the full dataset to a single JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(all_frames_data, json_file, indent=4)

    print(f"All frames data saved to {output_json_path}.")


# Example usage
base_folder_path = 'output/Squat_Data/Squat_Data/Valid/'  # Replace with your folder's base URL or path

for index in range(120):
    output_json_path = os.path.join(base_folder_path, f'output/{index}.json')  # JSON file to save
    convert_npy_folder_to_single_json(base_folder_path+f"{index}", output_json_path)
    print("Conversion of all npy files to a single JSON is complete!")
print("Conversion of all npy files to a single JSON is complete!")
