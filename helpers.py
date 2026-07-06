import numpy as np
import pandas as pd
import json
from scipy.spatial.transform import Rotation as R
import cv2
import io
import os

K = np.array([[],[],[]])
T = np.array([[],[],[]])


def get_starting_position(directory_path):

    trajectory_settings_path = os.path.join(directory_path, 'trajectory_settings.json')
    with open(trajectory_settings_path, 'r') as file:
        trajectory_settings = json.load(file)

    px = trajectory_settings['starting_position']['x']
    py = trajectory_settings['starting_position']['y']
    pz = trajectory_settings['starting_position']['z']

    return np.array([px, py, pz])


def get_intrinsics():

    with open(r'C:\Thesis\NAVIS\camera_settings.json', 'r') as file:
        cam_settings = json.load(file)

    fx = cam_settings['intrinsics']['fx']
    fy = cam_settings['intrinsics']['fy']
    cx = cam_settings['intrinsics']['cx']
    cy = cam_settings['intrinsics']['cy']

    intrinsics = np.array([[fx, 0, cx],[0, fy, cy],[0,0,1]])

    return intrinsics


### Derive intrinsic matrix for a camera ###
### Takes in variables in conventional ###
def get_extrinsics(cam, px, py, pz, p, y, r):
    """
    px, py, pz : ship position in NED meters (already converted)
    p, y, r    : ship pitch, yaw, roll in NED, IN DEGREES (not pre-converted to radians)
    """

    with open(r'C:\Thesis\NAVIS\camera_settings.json', 'r') as file:
        cam_settings = json.load(file)

    key = f'extrinsics_{cam}'

    tx = cam_settings[key]['tx']
    ty = cam_settings[key]['ty']
    tz = cam_settings[key]['tz']
    pitch_off = cam_settings[key]['pitch_offset_deg']
    roll_off  = cam_settings[key]['roll_offset_deg']
    yaw_off   = cam_settings[key]['yaw_offset_deg']

    center_pos = np.array([px, py, pz])

    # Ship body → world rotation, standard aerospace ZYX order
    R_WC = R.from_euler('ZYX', [y, p, r], degrees=True).as_matrix()

    T_WC = np.eye(4)
    T_WC[:3, :3] = R_WC
    T_WC[:3, 3]  = center_pos

    # Camera mount offset relative to ship body, same ZYX convention
    cam_local_pos = np.array([tx, ty, tz])
    R_CK = R.from_euler('ZYX', [yaw_off, pitch_off, roll_off], degrees=True).as_matrix()

    T_CK = np.eye(4)
    T_CK[:3, :3] = R_CK
    T_CK[:3, 3]  = cam_local_pos

    # Camera pose in world frame
    T_WK = T_WC @ T_CK

    # World-to-camera transform (what you need for projection)
    T_KW = np.linalg.inv(T_WK)

    extrinsic_matrix = T_KW[:3, :]

    return extrinsic_matrix


# Takes pixel coordinates and projection matrixes for each cameras detection ###

def get_inital_position_estimate(u_l, v_l , P_l, u_r, v_r, P_r):

    pt_l = np.array([[u_l], [v_l]], dtype=np.float32)
    pt_r = np.array([[u_r], [v_r]], dtype=np.float32)

    #Returns homogenous point in 4D (X,Y,Z,W)
    pt_4D = cv2.triangulatePoints(P_l, P_r, pt_l, pt_r)

    pt_3D = pt_4D[:3, :] / pt_4D[3, :]

    return pt_3D.flatten()


### --- Take intrinsics and extrinsics matrix and return OpenCV projection matrix --- ###
def get_projection_matrix(K, T):

    E_cv = np.zeros((3, 4))
    
    # OpenCV X (Right)   <- NED Y
    E_cv[0, :] = T[1, :] 
    
    # OpenCV Y (Down)    <- NED Z
    E_cv[1, :] = T[2, :] 
    
    # OpenCV Z (Forward) <- NED X
    E_cv[2, :] = T[0, :] 
    
    # 3. Calculate Projection Matrix: P = K * E
    P = K @ E_cv
    
    return P




### --- Pre-processing steps --- ###

def format_data(capture_log_path, drone_imu_path, directory_path):

    df_ship = pd.read_csv(capture_log_path, skiprows = 1, names=['frame', 'timestamp', 'xyz_string', 'pyr_string'])
    df_drone = pd.read_csv(drone_imu_path)

    ## Format Drone df ##

    p = get_starting_position(directory_path)

    df_drone['pos_x'] += p[0]
    df_drone['pos_y'] += p[1]
    df_drone['pos_z'] += p[2]

    ## Format ship df ##

    df_ship[['x', 'y', 'z']] = df_ship['xyz_string'].str.extract(r'X=([-\d.]+)\s+Y=([-\d.]+)\s+Z=([-\d.]+)')
    df_ship[['pitch', 'yaw', 'roll']] = df_ship['pyr_string'].str.extract(r'P=([-\d.]+)\s+Y=([-\d.]+)\s+R=([-\d.]+)')

    df_ship = df_ship.drop(columns=['xyz_string', 'pyr_string'])

    cols_to_convert = ['timestamp','x', 'y', 'z', 'pitch', 'yaw', 'roll']
    df_ship[cols_to_convert] = df_ship[cols_to_convert].apply(pd.to_numeric)


    df_ship['x'] = df_ship['x'] / 100
    df_ship['y'] = df_ship['y'] / 100
    df_ship['z'] = -df_ship['z'] / 100

    df_ship['roll'] = np.radians(-df_ship['roll'])
    df_ship['yaw'] = np.radians(df_ship['yaw'])
    df_ship['pitch'] = np.radians(-df_ship['pitch'])

    return df_ship, df_drone
    

### ---  EKF HELPER FUNCTIONS --- ###

def get_F_matrix(dt):

    F = np.eye(6)
    F[0,3] = dt
    F[1,4] = dt
    F[2,5] = dt

    return F


def get_B_matrix(dt):

    B = np.zeros((6,3))

    dt2 = (dt ** 2)/2

    B[0,0] = dt2
    B[1,1] = dt2
    B[2,2] = dt2
    B[3,0] = dt
    B[4,0] = dt
    B[5,0] = dt

    return B
                 

def get_Q_matrix( B, sigma_accel):


    Q = B @ sigma_accel @ B.T

    return Q

def initialize_state_covariance(position_variance, velocity_variance):

    P = np.eye(6)

    P[0,0] = position_variance
    P[1,1] = position_variance * 5
    P[2,2] = position_variance

    P[3,3] = velocity_variance
    P[4,4] = velocity_variance * 5
    P[5,5]= velocity_variance

    return P



def get_jacobian(x, T, K):
    
    fx = K[0, 0]
    fy = K[1, 1]

    E_cv = np.zeros((3, 4))
    # OpenCV X (Right)   <- NED Y
    E_cv[0, :] = T[1, :] 
    # OpenCV Y (Down)    <- NED Z
    E_cv[1, :] = T[2, :] 
    # OpenCV Z (Forward) <- NED X
    E_cv[2, :] = T[0, :] 

    p_cam = E_cv @ np.array([x[0], x[1], x[2], 1.0])

    x_cam = p_cam[0]
    y_cam = p_cam[1]
    z_cam = p_cam[2]

    if z_cam < 0: 
        return None
    
    J_proj = np.array([
        [fx/z_cam, 0, -fx*x_cam/ z_cam ** 2],
        [0, fy/z_cam, -fy * y_cam/ z_cam**2]
    ])

    J_pos = J_proj @ E_cv[:, :3]

    H = np.zeros((2,6))

    H[:, :3] = J_pos

    return H

def get_Kalman_gain(H, R, state_cov):

    K = state_cov @ H.T @ np.linalg.inv(H @ state_cov @ H.T + R)

    return K



def match_nearest(df_source, df_target, source_col='timestamp', target_col='timestamp'):
    """For each row in df_target, find the nearest timestamp in df_source."""
    matched_rows = []
    for ts in df_target[target_col]:
        idx = (df_source[source_col] - ts).abs().idxmin()
        matched_rows.append(df_source.loc[idx])
    return pd.DataFrame(matched_rows).reset_index(drop=True)