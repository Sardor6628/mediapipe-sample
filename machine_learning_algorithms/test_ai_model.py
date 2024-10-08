import pandas as pd
import joblib
import numpy as np

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



# Load the saved model
for index in range(1,20):
    model = joblib.load("squat_classifier_model.pkl")

    # Provide the correct file path
    file_path = f'output_10-21squat_{index}_output_landmarks.csv'
    # Aggregate the ne/w squat data
    aggregated_data = aggregate_squat_data(file_path)

    # Convert the aggregated data into a DataFrame for prediction
    aggregated_df = pd.DataFrame([aggregated_data])

    # Make the prediction
    prediction = model.predict(aggregated_df)

    # Output the prediction result
    result = "Good Squat" if prediction[0] == 1 else "Bad Squat"
    print(f"Prediction result for {index}: {result}")