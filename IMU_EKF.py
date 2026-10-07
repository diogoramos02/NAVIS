"""
===============================================================================
File: IMU_EKF.py
Author: Diogo Ramos
Institution: Instituto Superior Técnico (ECE - AI & Robotics)
Date: September 2026

Project: Master's Thesis - 3D UAV Trajectory Tracking and Pose Estimation

Description:
This script implements an Extended Kalman Filter (EKF) using an IMU mounted on
the drone to estimate the 3D trajectory of a UAV relative to a moving ship 
deck fused with stereo vision measurements (YOLO bounding box detections).
===============================================================================
"""


# Import libraries
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Import helper scripts
from utils import *
from CV_helpers import *
from extended_EKF_helpers import *


# Path to the directory containing the data acquired from the simulation
#
# Directory should have the following structure:
# DIRECTORY_PATH/
# ├── drone_imu.csv
# ├── capture_log.csv
# ├── trajectory_settings.json
# └── images/
#     ├── left_0001.bmp
#     ├── left_0002.bmp
#     ├── ... 
#     ├── right_0001.bmp
#     ├── right_0002.bmp        
#     └── ... 
DIRECTORY_PATH = r'C:\Captures'

# Path to the .json file with the camera parameters
CAM_SETTINGS_PATH = r'C:\Users\Diogo Ramos\Desktop\NAVIS_DiogoRamos\NAVIS\EKF\config\camera_settings.json'

# Load camera settings
try:
    with open(CAM_SETTINGS_PATH, 'r') as file:
        cam_settings = json.load(file)
except FileNotFoundError:
        raise FileNotFoundError(f"Camera settings file not found at: {CAM_SETTINGS_PATH}")
except json.JSONDecodeError:
        raise ValueError(f"Camera settings file contains invalid JSON: {CAM_SETTINGS_PATH}")

EKF_SETTINGS_PATH = r'C:\Users\Diogo Ramos\Desktop\NAVIS_DiogoRamos\NAVIS\EKF\final_code\ekf_settings.json'

# Load ekf settings
try:
    with open(EKF_SETTINGS_PATH, 'r') as file:
        ekf_settings = json.load(file)
except FileNotFoundError:
        raise FileNotFoundError(f"Camera settings file not found at: {EKF_SETTINGS_PATH}")
except json.JSONDecodeError:
        raise ValueError(f"Camera settings file contains invalid JSON: {EKF_SETTINGS_PATH}")


# Retrieve the necessary paths to the data files and image folder
DRONE_IMU_PATH,SHIP_IMU_PATH,IMAGES_PATH, TRAJECTORY_SETTINGS_PATH = get_paths(DIRECTORY_PATH)

# Load the trajectory settings
try:
    with open(TRAJECTORY_SETTINGS_PATH, 'r') as file:
        trajectory_settings = json.load(file)
except FileNotFoundError:
    raise FileNotFoundError(f"Trajectory settings file not found at: {TRAJECTORY_SETTINGS_PATH}")
except json.JSONDecodeError:
    raise ValueError(f"Trajectory settings file contains invalid JSON: {TRAJECTORY_SETTINGS_PATH}")

# Path to the trained YOLO model
MODEL_PATH = r'C:\Users\Diogo Ramos\Desktop\NAVIS_DiogoRamos\NAVIS\EKF\best.pt'
model = YOLO(MODEL_PATH)
detector_conf = ekf_settings["detector"]["confidence"]

# Load and format data
df_ship, df_drone = format_data(SHIP_IMU_PATH, DRONE_IMU_PATH, trajectory_settings)

# Align timestamps
df_drone, filtered_df_ship = timestamp_alignment(df_drone, df_ship)

# Get camera equations
intrinsics = get_intrinsics(cam_settings)
left_extrinsics = get_extrinsics(cam_settings, "left", 0, 0, 0)
right_extrinsics = get_extrinsics(cam_settings, "right", 0, 0, 0)
left_projection = get_projection_matrix(intrinsics, left_extrinsics)
right_projection = get_projection_matrix(intrinsics, right_extrinsics)

# Get intial position estimation and trajectory starting timestamp
x_0, starting_timestamp = get_initial_position_estimate(cam_settings, filtered_df_ship, left_projection, right_projection, intrinsics, IMAGES_PATH, model, detector_conf, False)
previous_timestamp = starting_timestamp

# Combine the dataframes for the main loop and eliminate the rows before the first detection
combined_dataframe = combine_dataframes(filtered_df_ship, df_drone)
df_drone, combined_dataframe = crop_timestamps(combined_dataframe, df_drone, starting_timestamp)
previous_timestamp = 0.0

# Extract variance values from config
position_variance = ekf_settings["initial_covariance"]["position"]
velocity_variance = ekf_settings["initial_covariance"]["velocity"]
orientation_variance = ekf_settings["initial_covariance"]["IMU_orientation"]
bias_variance = ekf_settings["initial_covariance"]["bias"]
pixel_variance = ekf_settings["measurement_noise"]["pixel"]

accel_variance = ekf_settings["process_noise"]["accelerometer"]
gyro_variance = ekf_settings["process_noise"]["gyroscope"]
imu_accel = np.zeros(3)
imu_gyro  = np.zeros(3)

# Initialize the EKF state and state covariance
x = initialize_state(trajectory_settings, x_0)
P = intialize_state_covariance(position_variance, velocity_variance, orientation_variance, bias_variance, 2)

# Initialize measurement noise covariance matrix
R = intialize_R_matrix(pixel_variance)

# Initialize list to log the results
results = []

print(f"Begining loop. The initial state is{x}")

for idx, row  in combined_dataframe.iterrows():

    # Predict the state update using IMU output values
    if(row['entity'] == 'drone'):

        # Calculate how much time has passed since the last iteration 
        new_timestamp = row['timestamp']
        dt = new_timestamp - previous_timestamp
        Q = get_extended_Q_matrix(dt, accel_variance, gyro_variance)
        imu_accel = np.array([row['acc_x'], row['acc_y'], row['acc_z']])
        imu_gyro = np.array([row['gyro_x'], row['gyro_y'], row['gyro_z']])
        x ,P = predict_imu(x, P, imu_accel, imu_gyro, dt, Q)
        previous_timestamp = new_timestamp

    # Correct the state using the images
    if(row['entity'] == 'ship'):

        # Open both images for the current frame number and check their integrity
        frame_num = int(float(row['frame']))
        frame_str = f"{frame_num:04d}"
        left_img_path = os.path.join(IMAGES_PATH, f"left_{frame_str}.bmp")
        right_img_path = os.path.join(IMAGES_PATH, f"right_{frame_str}.bmp")

        if not os.path.exists(left_img_path) or not os.path.exists(right_img_path):
            warnings.warn(
                f"Image missing for frame {frame_str}. Skipping iteration. "
                f"Checked paths: \n  - {left_img_path}\n  - {right_img_path}", 
                UserWarning
            )
            continue

        if os.path.getsize(left_img_path) == 0 or os.path.getsize(right_img_path) == 0:
            warnings.warn(
                f"Image empty for frame {frame_str}. Skipping iteration. "
                f"Checked images: \n  - {left_img_path}\n  - {right_img_path}", 
                UserWarning
            )
            continue

        try:
            res_left = model(left_img_path, conf=detector_conf, verbose=False)[0] 
            res_right = model(right_img_path, conf=detector_conf, verbose=False)[0]
        except Exception as e:
            warnings.warn(
                f"Warning: Corrupted image at frame {frame_str}. Model error: {e}. Skipping.", 
                UserWarning
            )
            results = log_results(results, x, P, new_timestamp)
            previous_timestamp = new_timestamp
            continue

        # Check if the drone is detected in the images
        left_detected = len(res_left.boxes) > 0
        right_detected = len(res_right.boxes) > 0

        if left_detected:
            # Calculate how much time has passed since the last iteration 
            new_timestamp = row['timestamp']
            dt = new_timestamp - previous_timestamp
            Q = get_extended_Q_matrix(dt, accel_variance, gyro_variance)
            if(dt>0):
                # To aplly the update, the state must be predicted at that exact timestamp
                #
                # Since the images don't have associated IMU readings, the last values are reused
                x ,P = predict_imu(x, P, imu_accel, imu_gyro, dt, Q)

            # Apply the correction to the state and state covariance
            x, P = correction_step(x, P, intrinsics, left_extrinsics, res_left, R)
            previous_timestamp = new_timestamp


        if right_detected:
            # Calculate how much time has passed since the last iteration. If the drone is detected in both images, dt = 0.
            new_timestamp = row['timestamp']
            dt = new_timestamp - previous_timestamp
            Q = get_extended_Q_matrix(dt, accel_variance, gyro_variance)
            if(dt>0):
                # To aplly the update, the state must be predicted at that exact timestamp
                #
                # Since the images don't have associated IMU readings, the last values are reused
                x ,P = predict_imu(x, P, imu_accel, imu_gyro, dt, Q)

            # Apply the correction to the state and state covariance
            x, P = correction_step(x, P, intrinsics, right_extrinsics,res_right, R)
            previous_timestamp = new_timestamp

    results = log_results(results, x, P, new_timestamp)

print("Loop completed, saving results and plotting graphs\n\n")

# Convert drone position to the Worl frame which is the ship frame
df_drone['pos_x'] = df_drone['pos_x'] - df_drone['ship_pos_x']
df_drone['pos_y'] = df_drone['pos_y'] - df_drone['ship_pos_y']
#df_drone['pos_z'] = df_drone['pos_z'] - df_drone['ship_pos_z']

# Convert the results to a dataframe
df_ekf = pd.DataFrame(results)

# The rotations plots use Euler angles, so we convert the quaternions to euler angles in both dataframes
ekf_quats = df_ekf[['ekf_qx', 'ekf_qy', 'ekf_qz', 'ekf_qw']].values
ekf_euler = Rotation.from_quat(ekf_quats).as_euler('xyz', degrees=True)
df_ekf['ekf_roll']  = ekf_euler[:, 0]
df_ekf['ekf_pitch'] = ekf_euler[:, 1]
df_ekf['ekf_yaw']   = ekf_euler[:, 2]

gt_quats = df_drone[['q_x', 'q_y', 'q_z', 'q_w']].values
gt_euler = Rotation.from_quat(gt_quats).as_euler('xyz', degrees=True)
df_drone['gt_roll']  = gt_euler[:, 0]
df_drone['gt_pitch'] = gt_euler[:, 1]
df_drone['gt_yaw']   = gt_euler[:, 2]

df_ekf = df_ekf.sort_values(by='timestamp').reset_index(drop=True)

# For each row of the results dataframe get the row of the drone dataframe with the nearest timestamp
df_gt_matched = match_nearest(df_drone, df_ekf, source_col='timestamp', target_col='timestamp')

#Save the dataframes to csv files
df_ekf.to_csv(r"results\imu_ekf.csv", index=False)
df_gt_matched.to_csv(r"results\imu_gt.csv", index=False)

# Plot position
plot_ekf_comparison(
    df_ekf=df_ekf,
    df_gt_matched=df_gt_matched,
    title="EKF Position Estimate vs Ground Truth",
    labels=["X - North (m)", "Y - East (m)", "Z - Down (m)"],
    ekf_cols=["ekf_px", "ekf_py", "ekf_pz"],
    gt_cols=["pos_x", "pos_y", "pos_z"],
    var_cols=["var_px", "var_py", "var_pz"],
    filename=r"results\IMU_ekf_position.png",
    gt_color="green",
    ekf_color="red"
)

# Plot Velocity
plot_ekf_comparison(
    df_ekf=df_ekf,
    df_gt_matched=df_gt_matched,
    title="EKF Velocity Estimate vs Ground Truth",
    labels=["Vx - North (m/s)", "Vy - East (m/s)", "Vz - Down (m/s)"],
    ekf_cols=["ekf_vx", "ekf_vy", "ekf_vz"],
    gt_cols=["vel_x", "vel_y", "vel_z"],
    var_cols=["var_vx", "var_vy", "var_vz"],
    filename=r"results\IMU_ekf_velocity.pdf",
    gt_color="blue",
    ekf_color="orange" 
)

# Plot Orientation
plot_ekf_comparison(
    df_ekf=df_ekf,
    df_gt_matched=df_gt_matched,
    title="EKF Orientation Estimate vs Ground Truth",
    labels=["Roll (deg)", "Pitch (deg)", "Yaw (deg)"],
    ekf_cols=["ekf_roll", "ekf_pitch", "ekf_yaw"],
    gt_cols=["gt_roll", "gt_pitch", "gt_yaw"],
    var_cols=["var_roll", "var_pitch", "var_yaw"],
    filename=r"results\IMU_ekf_orientation.pdf",
    gt_color="purple",
    ekf_color="cyan"
)