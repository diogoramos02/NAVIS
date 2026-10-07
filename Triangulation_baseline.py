"""
===============================================================================
File: Cameras_only.py
Author: Diogo Ramos
Institution: Instituto Superior Técnico (ECE - AI & Robotics)
Date: September 2026

Project: Master's Thesis - 3D UAV Trajectory Tracking and Pose Estimation

Description:
This script implements an Extended Kalman Filter (EKF) using a Constant 
Velocity model to estimate the 3D trajectory of a UAV relative to a moving ship 
deck. It uses stereo vision measurements (YOLO bounding box detections).
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
from simple_EKF_helpers import *


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
x_0, starting_timestamp = get_initial_position_estimate(cam_settings, filtered_df_ship, left_projection, right_projection,intrinsics, IMAGES_PATH, model, detector_conf, use_MRU=False)
previous_timestamp = starting_timestamp

# Combine the dataframes for the main loop and eliminate the rows before the first detection
combined_dataframe = combine_dataframes(filtered_df_ship, df_drone)
df_drone, combined_dataframe = crop_timestamps(combined_dataframe, df_drone, starting_timestamp)
previous_timestamp = 0.0

print(f"Begining loop. The initial position is{x_0}")

results= []

# Main loop
for idx, row in combined_dataframe.iterrows():

    # This system does not use the IMU measuremnts so the lines with IMU data are ignored
    if(row['entity'] == 'drone'):
        continue

    new_timestamp = row['timestamp']

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

    # Apply the detection model and check the correctness of the detection.
    # If there is an error with the model we still log the results after the prediction step but without any correction
    try:
        res_left = model(left_img_path, conf=detector_conf, verbose=False)[0] 
        res_right = model(right_img_path, conf=detector_conf, verbose=False)[0]
    except Exception as e:
        warnings.warn(
            f"Warning: Corrupted image at frame {frame_str}. Model error: {e}. Skipping.", 
            UserWarning
        )
        continue

    # Check if the drone is detected in the images
    left_detected = len(res_left.boxes) > 0
    right_detected = len(res_right.boxes) > 0

    if left_detected and right_detected:
        best_left_idx = res_left.boxes.conf.argmax()
        best_right_idx = res_right.boxes.conf.argmax()

        u_l = float(res_left.boxes.xywh[best_left_idx][0])
        v_l = float(res_left.boxes.xywh[best_left_idx][1])

        u_r = float(res_right.boxes.xywh[best_right_idx][0])
        v_r = float(res_right.boxes.xywh[best_right_idx][1])

        x = triangulate_position(u_l, v_l, left_projection, u_r, v_r, right_projection)

        results.append({
            'timestamp' : new_timestamp,
            'ekf_px': x[0],
            'ekf_py': x[1],
            'ekf_pz': x[2]
        })

print("Loop completed, saving results and plotting graphs\n\n")

# Convert drone position to the Worl frame which is the ship frame
df_drone['pos_x'] = df_drone['pos_x'] - df_drone['ship_pos_x']
df_drone['pos_y'] = df_drone['pos_y'] - df_drone['ship_pos_y']

# Convert the results to a dataframe
df_ekf = pd.DataFrame(results)
df_ekf = df_ekf.sort_values(by='timestamp').reset_index(drop=True)

# For each row of the results dataframe get the row of the drone dataframe with the nearest timestamp
df_gt_matched = match_nearest(df_drone, df_ekf, source_col='timestamp', target_col='timestamp')

# Save the dataframes to csv files
df_ekf.to_csv(r"results\baseline_ekf.csv", index=False)
df_gt_matched.to_csv(r"results\baseline_gt.csv", index=False)


# Plot position
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle("EKF Position Estimate vs Ground Truth", fontsize=14)

labels    = ["X - North (m)", "Y - East (m)", "Z - Down (m)"]
ekf_cols  = ["ekf_px", "ekf_py", "ekf_pz"]
gt_cols   = ["pos_x",  "pos_y",  "pos_z"]

for ax, label, ekf_col, gt_col in zip(
        axes, labels, ekf_cols, gt_cols):

    t = df_ekf["timestamp"] - df_ekf["timestamp"].iloc[0]  # relative time from 0

    # Ground truth
    ax.plot(t, df_gt_matched[gt_col], 
            color="green", linewidth=2, label="Ground Truth")

    # EKF estimate
    ax.plot(t, df_ekf[ekf_col], 
            color="red", linestyle="--", linewidth=1.5, label="EKF Estimate")

    ax.set_ylabel(label)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig(r"results\baseline_position.pdf", dpi=150)
