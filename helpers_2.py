import numpy as np
import pandas as pd
import json
from scipy.spatial.transform import Rotation as Rotation
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

    with open(r'C:\Users\Diogo Ramos\Desktop\NAVIS_DiogoRamos\NAVIS\EKF\config\camera_settings.json', 'r') as file:
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

    with open(r'C:\Users\Diogo Ramos\Desktop\NAVIS_DiogoRamos\NAVIS\EKF\config\camera_settings.json', 'r') as file:
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
    R_WC = Rotation.from_euler('ZYX', [y, p, r], degrees=True).as_matrix()

    T_WC = np.eye(4)
    T_WC[:3, :3] = R_WC
    T_WC[:3, 3]  = center_pos

    # Camera mount offset relative to ship body, same ZYX convention
    cam_local_pos = np.array([tx, ty, tz])
    R_CK = Rotation.from_euler('ZYX', [yaw_off, pitch_off, roll_off], degrees=True).as_matrix()

    T_CK = np.eye(4)
    T_CK[:3, :3] = R_CK
    T_CK[:3, 3]  = cam_local_pos

    # Camera pose in world frame
    T_WK = T_WC @ T_CK

    # World-to-camera transform (what you need for projection)
    T_KW = np.linalg.inv(T_WK)

    T = T_KW[:3, :]

    E_cv = np.zeros((3, 4))
    
    # OpenCV X (Right)   <- NED Y
    E_cv[0, :] = T[1, :] 
    
    # OpenCV Y (Down)    <- NED Z
    E_cv[1, :] = T[2, :] 
    
    # OpenCV Z (Forward) <- NED X
    E_cv[2, :] = T[0, :]


    return E_cv




def get_inital_position_estimate(u_l, v_l , P_l, u_r, v_r, P_r):

    pt_l = np.array([[u_l], [v_l]], dtype=np.float32)
    pt_r = np.array([[u_r], [v_r]], dtype=np.float32)

    #Returns homogenous point in 4D (X,Y,Z,W)
    pt_4D = cv2.triangulatePoints(P_l, P_r, pt_l, pt_r)

    pt_3D = pt_4D[:3, :] / pt_4D[3, :]

    proj_l_homogeneous = P_l @ pt_4D
    proj_l = proj_l_homogeneous[:2, :] / proj_l_homogeneous[2, :]
    
    proj_r_homogeneous = P_r @ pt_4D
    proj_r = proj_r_homogeneous[:2, :] / proj_r_homogeneous[2, :]
    
    error_l = np.linalg.norm(pt_l - proj_l)
    error_r = np.linalg.norm(pt_r - proj_r)
    
    # 4. Average the error across both cameras
    mean_reproj_error = (error_l + error_r) / 2.0

    return pt_3D.flatten(), mean_reproj_error



def get_projection_matrix(K, T):

    P = K @ T
    
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


def combines_dataframes(df_ship, df_drone):
    
    df_ship['entity'] = 'ship'
    df_drone['entity'] = 'drone'

    df_combined = pd.concat([df_ship, df_drone], ignore_index=True)
    df_combined = df_combined.sort_values(by='airsim_timestamp').reset_index(drop=True)

    return df_combined




def get_Kalman_gain(H, R, state_cov):

    K = state_cov @ H.T @ np.linalg.inv(H @ state_cov @ H.T + R)

    return K



def match_nearest(df_source, df_target, source_col='airsim_timestamp', target_col='timestamp'):
    """For each row in df_target, find the nearest timestamp in df_source."""
    matched_rows = []
    for ts in df_target[target_col]:
        idx = (df_source[source_col] - ts).abs().idxmin()
        matched_rows.append(df_source.loc[idx])
    return pd.DataFrame(matched_rows).reset_index(drop=True)


def skew_symmetric(v):

    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def predict_imu(x, P, imu_accel, imu_gyro, dt, Q):

    p = x[:3]
    v = x[3:6]
    q = x[6:10]
    ba = x[10:13] # Accel bias
    bg = x[13:16] # Gyro bias

    # 2. Correct the IMU readings by removing the bias BEFORE integrating
    accel_true = imu_accel - ba
    gyro_true = imu_gyro - bg

    rot = Rotation.from_quat(q)
    R_mat = rot.as_matrix()

    g = np.array([0.0, 0.0, 9.81])

    accel_world = R_mat @ accel_true + g

    p_new = p + v * dt + 0.5 * accel_world * dt**2
    v_new = v + accel_world * dt
    delta_rot = Rotation.from_rotvec(gyro_true * dt)
    rot_new = rot * delta_rot
    q_new = rot_new.as_quat()


    state_new = np.concatenate((p_new, v_new, q_new, ba, bg))

    # 5. 15x15 Error-State Jacobian (F)
    F = np.eye(15)
    
    # Kinematics
    F[0:3, 3:6] = np.eye(3) * dt
    F[0:3, 6:9] = -0.5 * R_mat @ skew_symmetric(accel_true) * dt**2
    F[3:6, 6:9] = -R_mat @ skew_symmetric(accel_true) * dt
    F[6:9, 6:9] = np.eye(3) - skew_symmetric(gyro_true) * dt
    
    # Bias Jacobians (How bias drift affects position/velocity/orientation)
    F[0:3, 9:12] = -0.5 * R_mat * dt**2
    F[3:6, 9:12] = -R_mat * dt
    F[6:9, 12:15] = -np.eye(3) * dt

    P_new = F @ P @ F.T + Q

    return state_new, P_new


def get_jacobian_extended_state(x, T, K):
    
    fx = K[0, 0]
    fy = K[1, 1]
 
    p_cam = T @ np.array([x[0], x[1], x[2], 1.0])

    x_cam = p_cam[0]
    y_cam = p_cam[1]
    z_cam = p_cam[2]

    if z_cam < 0: 
        return None
    
    J_proj = np.array([
        [fx/z_cam, 0, -fx*x_cam/ z_cam ** 2],
        [0, fy/z_cam, -fy * y_cam/ z_cam**2]
    ])

    J_pos = J_proj @ T[:, :3]

    H = np.zeros((2,15))

    H[:, :3] = J_pos

    return H


def initialize__extended_state_covariance(pos_var, vel_var, ori_var, bias_var):
    P = np.zeros((15, 15))
    np.fill_diagonal(P[0:3, 0:3], pos_var)
    P[1,1] = P[1,1] * 3 
    np.fill_diagonal(P[3:6, 3:6], vel_var)
    np.fill_diagonal(P[6:9, 6:9], ori_var)
    np.fill_diagonal(P[9:15, 9:15], bias_var)
    return P



def get_extended_Q_matrix(dt, sigma_accel, sigma_gyro):
    Q = np.zeros((15, 15))
    
    dt2 = dt**2
    dt3 = dt**3
    dt4 = dt**4

    baseline_pos_noise = 1e-6

    # 1. Accelerometer Noise (Affects Position, Velocity, and their correlation)
    # Variance of position error = (1/4) * sigma_accel * dt^4
    np.fill_diagonal(Q[0:3, 0:3], (sigma_accel * dt4) / 4.0 + baseline_pos_noise)
    
    # Variance of velocity error = sigma_accel * dt^2
    np.fill_diagonal(Q[3:6, 3:6], sigma_accel * dt2)
    
    # Cross-correlation between Position and Velocity = (1/2) * sigma_accel * dt^3
    # We must fill both upper and lower symmetric blocks
    pos_vel_cov = (sigma_accel * dt3) / 2.0
    np.fill_diagonal(Q[0:3, 3:6], pos_vel_cov)  
    np.fill_diagonal(Q[3:6, 0:3], pos_vel_cov)

    # 2. Gyroscope Noise (Affects Orientation)
    np.fill_diagonal(Q[6:9, 6:9], sigma_gyro * dt2)

    # 3. Bias Random Walk (Allows biases to drift slowly)
    np.fill_diagonal(Q[9:12, 9:12], 1e-4 * dt) # Accel Bias
    np.fill_diagonal(Q[12:15, 12:15], 1e-4 * dt) # Gyro Bias

    return Q





