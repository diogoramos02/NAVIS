import numpy as np
import os
import json
import cv2
from scipy.spatial.transform import Rotation as Rotation
import pandas as pd
import warnings


def get_intrinsics(cam_settings):
    """Retrieves the camera settings parameters and calculates intrinsic matrix

    Args:
        cam_settings : Loaded json file where camera settings are stored.

    Returns:
        intrinsics (numpy.array): 3x3 intrinsic parameter 

    Raises:
        FileNotFoundError: If the file is not found.
        ValueError: If the file is empty or corrupted.
    """

    fx = cam_settings['intrinsics']['fx']
    fy = cam_settings['intrinsics']['fy']
    cx = cam_settings['intrinsics']['cx']
    cy = cam_settings['intrinsics']['cy']

    intrinsics = np.array([[fx, 0, cx],[0, fy, cy],[0,0,1]])

    return intrinsics


def get_extrinsics(cam_settings, cam, p, r, heave):
    """
    Computes the world-to-camera extrinsic matrix in OpenCV convention.
    x is right, y is down, z is forward from the center of the camera

    Three frames:
        World Frame  {W}: Origin at ship origin. No rotation (identity).
        Ship Frame   {S}: Origin at ship origin. Rotated by ship pitch and roll only (yaw=0).
        Camera Frame {C}: Offset from ship origin by camera mount position and orientation.

    Args:
        cam_settings : Loaded json file where camera settings are stored
        cam (str) : camera name key in camera_settings.json ('left' or 'right')
        p : ship pitch in NED radians
        r : ship roll  in NED radians
        heave : ship heave in meters

    Returns:
        E_cv (numpy.array): (3x4) world-to-camera matrix in OpenCV axis convention

    Raises:
        FileNotFoundError: If the file is not found.
        ValueError: If the file is empty or corrupted.
    """

    key = f'extrinsics_{cam}'

    # Camera mount parameters (in ship body frame)
    tx = cam_settings[key]['tx']
    ty = cam_settings[key]['ty']
    tz = cam_settings[key]['tz']

    pitch_off = cam_settings[key]['pitch_offset_deg']
    roll_off  = cam_settings[key]['roll_offset_deg']
    yaw_off   = cam_settings[key]['yaw_offset_deg']

    # T_WS: World → Ship
    #
    # Ship origin coincides with world origin (translation = 0)
    # Only pitch and roll applied, yaw = 0
    R_WS = Rotation.from_euler('ZYX', [0.0, p, r], degrees=False).as_matrix()

    T_WS = np.eye(4)
    T_WS[:3, :3] = R_WS
    T_WS[:3,  3] = np.array([0.0, 0.0, heave])   # ship origin = world origin

    # T_SC: Ship → Camera
    #
    # Camera is mounted at (tx, ty, tz) in ship body frame
    # with a fixed orientation offset (roll_off, pitch_off, yaw_off)
    R_SC = Rotation.from_euler('ZYX', [yaw_off, pitch_off, roll_off], degrees=True).as_matrix()

    T_SC = np.eye(4)
    T_SC[:3, :3] = R_SC
    T_SC[:3,  3] = np.array([tx, ty, tz])

    # T_WC: World → Camera (computed by composition)
    T_WC = T_WS @ T_SC

    # T_CW: Invert to get world-to-camera transform
    T_CW = np.linalg.inv(T_WC)
    T = T_CW[:3, :]          

    # NED → OpenCV axis permutation so the extrinsics and intrinsic matrix can be directly multiplied
    # NED:    X=North(forward)  Y=East(right)  Z=Down
    # OpenCV: X=Right           Y=Down         Z=Forward(optical axis)
    E_cv = np.zeros((3, 4))
    E_cv[0, :] = T[1, :]
    E_cv[1, :] = T[2, :]
    E_cv[2, :] = T[0, :]

    return E_cv


def get_projection_matrix(K, T):
    """Computes the projection matrix

    Args:
        K (numpy.array) : (3x3) Intrinsic parameter matrix
        T (numpy.array) : (3x4) Extrinsic parameter matrix

    Returns:
        P (numpy.array) : (3x4) Projection matrix
    """
    P = K @ T
    
    return P


def triangulate_position(u_l, v_l , P_l, u_r, v_r, P_r):
    """ From a pair of detections, return the triangulated 3D point
    
    """

    pt_l = np.array([[u_l], [v_l]], dtype=np.float32)
    pt_r = np.array([[u_r], [v_r]], dtype=np.float32)

    pt_4D = cv2.triangulatePoints(P_l, P_r, pt_l, pt_r)

    pt_3D = pt_4D[:3, :] / pt_4D[3, :]

    return pt_3D.flatten()


def get_initial_position_estimate(cam_settings, filtered_df_ship, P_l, P_r, intrinsics, IMAGES_PATH, model, confidence, use_MRU: bool = True):
    """Computes an inital estimation of the drone's position via triangulation
       when the drone is first detected in both frames 

    Args:
        cam_settings : Loaded json file where camera settings are stored
        filtered_df_ship (pandas dataframe) : Filtered dataframe containing the ship pose data and the image frame numbers timestamps
        P_l (numpy.array) : (3x4) Left camera projection matrix
        P_r (numpy.array) : (3x4) Rigth camera projection matrix
        intrinsics (numpy.array) : (3x3) Contains extrinsic parameters
        IMAGES_PATH (str) : Path to the folder containing the images
        model : Loaded YOLO model 
        use_MRU (bool) : If True, use the MRU data to recalculate new extrincs and projections for each frame  

    Returns:
        pt_3D (numpy.array) : Point coordinates in world frame
        starting_timestamp : timestamp which markes the start of the trajectory calculations
    """
    u_l, v_l, u_r, v_r, starting_timestamp, left_extrinsics, right_extrinsics = find_inital_frame(cam_settings, filtered_df_ship, IMAGES_PATH, model, confidence)

    pt_l = np.array([[u_l], [v_l]], dtype=np.float32)
    pt_r = np.array([[u_r], [v_r]], dtype=np.float32)

    # If the MRU is in use, calculate new projection matrix
    if(use_MRU):
        P_l = get_projection_matrix(intrinsics, left_extrinsics)
        P_r = get_projection_matrix(intrinsics, right_extrinsics)

    # Returns homogenous coordinates point in 4D (X,Y,Z,W)
    pt_4D = cv2.triangulatePoints(P_l, P_r, pt_l, pt_r)

    # Convert to 3D cartesian coordinates
    pt_3D = pt_4D[:3, :] / pt_4D[3, :]

    return pt_3D.flatten(), starting_timestamp


def find_inital_frame(cam_settings, filtered_df_ship, IMAGES_PATH, model, confidence):
    """Find the frame where the drone is first detected in both images.
       Return the pixel coordinates od the drone in both image and the image timestamp

    Args:
        cam_settings : Loaded json file where camera settings are stored
        filtered_df_ship (pandas dataframe) : Filtered dataframe containing the ship pose data and the image frame numbers timestamps
        IMAGES_PATH (str) : Path to the folder containing the images
        model : Loaded YOLO model

    Returns:
        u_l, v_l : Drone position in the left image in pixel coordinates
        u_r, v_r : Drone position in the right image in pixel coordinates
        starting_timestamp : timestamp which markes the start of the trajectory calculations
    """

    for _, row in  filtered_df_ship.iterrows():

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

        res_left = model(left_img_path,conf=confidence, verbose=False)[0]
        res_right = model(right_img_path,conf=confidence, verbose=False)[0]

        left_detected = len(res_left.boxes) > 0
        right_detected = len(res_right.boxes) > 0

        if left_detected and right_detected:
            print(row['frame'])
            starting_timestamp = row['timestamp']
            best_res_left = res_left
            best_res_right = res_right
            left_extrinsics = get_extrinsics(cam_settings,'left',row['pitch'], row['roll'], row['z'])
            right_extrinsics = get_extrinsics(cam_settings,'right', row['pitch'], row['roll'], row['z'])   
            break

    
    if starting_timestamp is None:
        raise ValueError("EKF Initialization Failed: Could not find any frame with valid stereo detections.")

    best_left_idx = best_res_left.boxes.conf.argmax()
    best_right_idx = best_res_right.boxes.conf.argmax()

    u_l = float(best_res_left.boxes.xywh[best_left_idx][0])
    v_l = float(best_res_left.boxes.xywh[best_left_idx][1])
    u_r = float(best_res_right.boxes.xywh[best_right_idx][0])
    v_r = float(best_res_right.boxes.xywh[best_right_idx][1])

    return u_l, v_l, u_r, v_r, starting_timestamp, left_extrinsics, right_extrinsics