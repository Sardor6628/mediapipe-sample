import os
import pandas as pd
import numpy as np


# Function to aggregate squat data from a single CSV file
def aggregate_squat_data(file_path):
    df = pd.read_csv(file_path)

    # Calculate statistical aggregates for key angles and their components
    squat_aggregate = {
        'mean_r_knee_angle': np.mean(df['r_knee_angle']),
        'min_r_knee_angle': np.min(df['r_knee_angle']),
        'max_r_knee_angle': np.max(df['r_knee_angle']),
        'std_r_knee_angle': np.std(df['r_knee_angle']),

        'mean_l_knee_angle': np.mean(df['l_knee_angle']),
        'min_l_knee_angle': np.min(df['l_knee_angle']),
        'max_l_knee_angle': np.max(df['l_knee_angle']),
        'std_l_knee_angle': np.std(df['l_knee_angle']),

        'mean_r_knee_angle_x': np.mean(df['r_knee_angle_x']),
        'min_r_knee_angle_x': np.min(df['r_knee_angle_x']),
        'max_r_knee_angle_x': np.max(df['r_knee_angle_x']),
        'std_r_knee_angle_x': np.std(df['r_knee_angle_x']),

        'mean_r_knee_angle_y': np.mean(df['r_knee_angle_y']),
        'min_r_knee_angle_y': np.min(df['r_knee_angle_y']),
        'max_r_knee_angle_y': np.max(df['r_knee_angle_y']),
        'std_r_knee_angle_y': np.std(df['r_knee_angle_y']),

        'mean_r_knee_angle_z': np.mean(df['r_knee_angle_z']),
        'min_r_knee_angle_z': np.min(df['r_knee_angle_z']),
        'max_r_knee_angle_z': np.max(df['r_knee_angle_z']),
        'std_r_knee_angle_z': np.std(df['r_knee_angle_z']),

        'mean_l_knee_angle_x': np.mean(df['l_knee_angle_x']),
        'min_l_knee_angle_x': np.min(df['l_knee_angle_x']),
        'max_l_knee_angle_x': np.max(df['l_knee_angle_x']),
        'std_l_knee_angle_x': np.std(df['l_knee_angle_x']),

        'mean_l_knee_angle_y': np.mean(df['l_knee_angle_y']),
        'min_l_knee_angle_y': np.min(df['l_knee_angle_y']),
        'max_l_knee_angle_y': np.max(df['l_knee_angle_y']),
        'std_l_knee_angle_y': np.std(df['l_knee_angle_y']),

        'mean_l_knee_angle_z': np.mean(df['l_knee_angle_z']),
        'min_l_knee_angle_z': np.min(df['l_knee_angle_z']),
        'max_l_knee_angle_z': np.max(df['l_knee_angle_z']),
        'std_l_knee_angle_z': np.std(df['l_knee_angle_z']),

        'mean_r_hip_angle_x': np.mean(df['r_hip_angle_x']),
        'min_r_hip_angle_x': np.min(df['r_hip_angle_x']),
        'max_r_hip_angle_x': np.max(df['r_hip_angle_x']),
        'std_r_hip_angle_x': np.std(df['r_hip_angle_x']),

        'mean_r_hip_angle_y': np.mean(df['r_hip_angle_y']),
        'min_r_hip_angle_y': np.min(df['r_hip_angle_y']),
        'max_r_hip_angle_y': np.max(df['r_hip_angle_y']),
        'std_r_hip_angle_y': np.std(df['r_hip_angle_y']),

        'mean_r_hip_angle_z': np.mean(df['r_hip_angle_z']),
        'min_r_hip_angle_z': np.min(df['r_hip_angle_z']),
        'max_r_hip_angle_z': np.max(df['r_hip_angle_z']),
        'std_r_hip_angle_z': np.std(df['r_hip_angle_z']),

        'mean_l_hip_angle_x': np.mean(df['l_hip_angle_x']),
        'min_l_hip_angle_x': np.min(df['l_hip_angle_x']),
        'max_l_hip_angle_x': np.max(df['l_hip_angle_x']),
        'std_l_hip_angle_x': np.std(df['l_hip_angle_x']),

        'mean_l_hip_angle_y': np.mean(df['l_hip_angle_y']),
        'min_l_hip_angle_y': np.min(df['l_hip_angle_y']),
        'max_l_hip_angle_y': np.max(df['l_hip_angle_y']),
        'std_l_hip_angle_y': np.std(df['l_hip_angle_y']),

        'mean_l_hip_angle_z': np.mean(df['l_hip_angle_z']),
        'min_l_hip_angle_z': np.min(df['l_hip_angle_z']),
        'max_l_hip_angle_z': np.max(df['l_hip_angle_z']),
        'std_l_hip_angle_z': np.std(df['l_hip_angle_z']),

        'mean_r_ankle_angle_x': np.mean(df['r_ankle_angle_x']),
        'min_r_ankle_angle_x': np.min(df['r_ankle_angle_x']),
        'max_r_ankle_angle_x': np.max(df['r_ankle_angle_x']),
        'std_r_ankle_angle_x': np.std(df['r_ankle_angle_x']),

        'mean_r_ankle_angle_y': np.mean(df['r_ankle_angle_y']),
        'min_r_ankle_angle_y': np.min(df['r_ankle_angle_y']),
        'max_r_ankle_angle_y': np.max(df['r_ankle_angle_y']),
        'std_r_ankle_angle_y': np.std(df['r_ankle_angle_y']),

        'mean_r_ankle_angle_z': np.mean(df['r_ankle_angle_z']),
        'min_r_ankle_angle_z': np.min(df['r_ankle_angle_z']),
        'max_r_ankle_angle_z': np.max(df['r_ankle_angle_z']),
        'std_r_ankle_angle_z': np.std(df['r_ankle_angle_z']),

        'mean_l_ankle_angle_x': np.mean(df['l_ankle_angle_x']),
        'min_l_ankle_angle_x': np.min(df['l_ankle_angle_x']),
        'max_l_ankle_angle_x': np.max(df['l_ankle_angle_x']),
        'std_l_ankle_angle_x': np.std(df['l_ankle_angle_x']),

        'mean_l_ankle_angle_y': np.mean(df['l_ankle_angle_y']),
        'min_l_ankle_angle_y': np.min(df['l_ankle_angle_y']),
        'max_l_ankle_angle_y': np.max(df['l_ankle_angle_y']),
        'std_l_ankle_angle_y': np.std(df['l_ankle_angle_y']),

        'mean_l_ankle_angle_z': np.mean(df['l_ankle_angle_z']),
        'min_l_ankle_angle_z': np.min(df['l_ankle_angle_z']),
        'max_l_ankle_angle_z': np.max(df['l_ankle_angle_z']),
        'std_l_ankle_angle_z': np.std(df['l_ankle_angle_z']),
    }

    # You can add more features if needed
    return squat_aggregate


# Function to process a list of CSV files and prepare data for AI training
def process_squat_files(good_squat_files, bad_squat_files):
    all_squat_data = []

    # Process good squats
    for file_path in good_squat_files:
        squat_data = aggregate_squat_data(file_path)
        squat_data['label'] = 1  # Label for good squats
        all_squat_data.append(squat_data)

    # Process bad squats
    for file_path in bad_squat_files:
        squat_data = aggregate_squat_data(file_path)
        squat_data['label'] = 0  # Label for bad squats
        all_squat_data.append(squat_data)

    # Convert to DataFrame for easy manipulation and saving
    squat_dataset = pd.DataFrame(all_squat_data)
    return squat_dataset


good_squat_files = [
    'output/squat_luke/separated_by_squat/output_14-20-good-0-1/squat_1_output_landmarks.csv',
    "output/squat_luke/separated_by_squat/output_14-20-good-0-1/squat_2_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-20-good-0-1/squat_3_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-20-good-0-1/squat_4_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-20-good-0-1/squat_5_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-20-good-0-1/squat_6_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-20-good-0-1/squat_7_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-20-good-0-1/squat_8_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-20-good-0-1/squat_9_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-23-good-1-1/squat_2_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-23-good-1-1/squat_3_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-23-good-1-1/squat_4_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-23-good-1-1/squat_5_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-23-good-1-1/squat_6_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-23-good-1-1/squat_7_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-23-good-1-1/squat_8_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-23-good-1-1/squat_9_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-23-good-1-1/squat_10_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-23-good-1-1/squat_11_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-23-good-1-1/squat_12_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-23-good-1-1/squat_13_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-23-good-1-1/squat_14_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-23-good-1-1/squat_15_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-23-good-1-1/squat_16_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-23-good-1-1/squat_17_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-23-good-1-1/squat_18_output_landmarks.csv",


]

bad_squat_files = [

    "output/squat_luke/separated_by_squat/output_14-23-good-1-1/squat_19_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-29-bad-2-1/squat_1_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-29-bad-2-1/squat_2_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-29-bad-2-1/squat_3_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-29-bad-2-1/squat_4_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-29-bad-2-1/squat_5_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-32-bad-3-1/squat_1_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-32-bad-3-1/squat_2_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-32-bad-3-1/squat_3_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-32-bad-3-1/squat_4_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-32-bad-3-1/squat_5_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-32-bad-3-1/squat_6_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-32-bad-3-1/squat_7_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-32-bad-3-1/squat_8_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-32-bad-3-1/squat_9_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-38-bad-4-1/squat_1_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-38-bad-4-1/squat_2_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-38-bad-4-1/squat_3_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-38-bad-4-1/squat_4_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-38-bad-4-1/squat_5_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-38-bad-4-1/squat_6_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-38-bad-4-1/squat_7_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-38-bad-4-1/squat_8_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-38-bad-4-1/squat_9_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-38-bad-4-1/squat_1_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-38-bad-4-1/squat_2_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-38-bad-4-1/squat_3_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-38-bad-4-1/squat_4_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-38-bad-4-1/squat_5_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-38-bad-4-1/squat_6_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-38-bad-4-1/squat_7_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-38-bad-4-1/squat_8_output_landmarks.csv",
    "output/squat_luke/separated_by_squat/output_14-38-bad-4-1/squat_9_output_landmarks.csv",

]  # Replace with your file paths

good_base_path="output/Squat_Data/Squat_Data/Valid/output"
bad_base_path="output/Squat_Data/Squat_Data/Invalid/output"
for i in range(1,120):
    good_squat_files.append(f"{good_base_path}/squat_{i}_output_landmarks.csv")
    bad_squat_files.append(f"{bad_base_path}/squat_{i}_output_landmarks.csv")


squat_dataset = process_squat_files(good_squat_files, bad_squat_files)

# Save the prepared dataset to CSV for later use
squat_dataset.to_csv('aggregated_squat_data.csv', index=False)

print(squat_dataset.head())
