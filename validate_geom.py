import os
import cv2
import numpy as np
import pandas as pd
from helpers_3 import *

DIRECTORY_PATH = r'C:\Captures'
DRONE_IMU_PATH = os.path.join(DIRECTORY_PATH, 'drone_imu.csv')
SHIP_IMU_PATH = os.path.join(DIRECTORY_PATH, 'capture_log.csv')
IMAGES_PATH = os.path.join(DIRECTORY_PATH, 'images')


print("Loading and formatting data...")
# Pulls formatted data using your existing helper functions
df_ship, df_drone = format_data(SHIP_IMU_PATH, DRONE_IMU_PATH, DIRECTORY_PATH)

# Normalize timestamps to start at 0
trajectory_start_time = df_drone['timestamp'].min()
df_ship['timestamp'] = df_ship['timestamp'] - trajectory_start_time
df_drone['timestamp'] = df_drone['timestamp'] - trajectory_start_time

# Filter ship data to match drone trajectory bounds
df_ship = df_ship[(df_ship['timestamp'] >= 0) & 
                  (df_ship['timestamp'] <= df_drone['timestamp'].max())].copy()

intrinsics = get_intrinsics()

print("Starting Ground-Truth Reprojection Viewer...")


for index, ship_row in df_ship.iterrows():

    
    
    frame_num = int(float(ship_row['frame']))

    if frame_num < 500:
        continue

    frame_str = f"{frame_num:04d}"
    
    left_img_path = os.path.join(IMAGES_PATH, f"left_{frame_str}.bmp")
    right_img_path = os.path.join(IMAGES_PATH, f"right_{frame_str}.bmp")
    
    if not os.path.exists(left_img_path) or not os.path.exists(right_img_path):
        continue
        
    img_left = cv2.imread(left_img_path)
    img_right = cv2.imread(right_img_path)
    
    if img_left is None or img_right is None:
        continue

    # 1. Sync timestamps: Find the exact drone position for this camera frame
    drone_idx = (df_drone['timestamp'] - ship_row['timestamp']).abs().idxmin()
    drone_row = df_drone.loc[drone_idx]

    # 2. Build the 3D Ground Truth Point in the "World Frame"
    # According to helpers_3.py, the World Origin is the Ship's X/Y position, 
    # but retains the absolute Z coordinate.
    rel_x = drone_row['pos_x'] - ship_row['x']
    rel_y = drone_row['pos_y'] - ship_row['y']
    abs_z = drone_row['pos_z'] - ship_row['z']  # Keep absolute Z for reprojection  
    
    pt_world_homog = np.array([rel_x, rel_y, abs_z, 1.0])

    # 3. Get projection matrices using the ship's current attitude
    left_ext = get_extrinsics('left', ship_row['pitch'], ship_row['roll'])
    right_ext = get_extrinsics('right', ship_row['pitch'], ship_row['roll'])
    
    P_left = get_projection_matrix(intrinsics, left_ext)
    P_right = get_projection_matrix(intrinsics, right_ext)

    # 4. Project 3D point onto the 2D Left Image
    uv_left = P_left @ pt_world_homog
    if uv_left[2] > 0: # Ensure point is in front of the camera
        u_l = int(uv_left[0] / uv_left[2])
        v_l = int(uv_left[1] / uv_left[2])
        cv2.circle(img_left, (u_l, v_l), radius=8, color=(0, 0, 255), thickness=-1)
        # Draw a crosshair for visibility
        cv2.drawMarker(img_left, (u_l, v_l), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    # 5. Project 3D point onto the 2D Right Image
    uv_right = P_right @ pt_world_homog
    if uv_right[2] > 0:
        u_r = int(uv_right[0] / uv_right[2])
        v_r = int(uv_right[1] / uv_right[2])
        cv2.circle(img_right, (u_r, v_r), radius=8, color=(0, 0, 255), thickness=-1)
        cv2.drawMarker(img_right, (u_r, v_r), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    # 6. Display the results
    stereo_view = np.hstack((img_left, img_right))
    stereo_view = cv2.resize(stereo_view, (0, 0), fx=0.5, fy=0.5)
    
    cv2.putText(stereo_view, f"Frame: {frame_str} | Ship Pitch: {np.degrees(ship_row['pitch']):.2f} deg | Ship Roll: {np.degrees(ship_row['roll']):.2f} deg", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    
    cv2.imshow("Ground Truth Reprojection Geometry Test", stereo_view)
    
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()