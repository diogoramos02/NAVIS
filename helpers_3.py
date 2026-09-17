import numpy as np
import pandas as pd
import json
from scipy.spatial.transform import Rotation as Rotation
import cv2

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

def get_extrinsics(cam, p, r):
    """
    Computes the world-to-camera extrinsic matrix in OpenCV convention.

    Three frames:
        World Frame  {W}: Origin at ship origin. No rotation (identity).
        Ship Frame   {S}: Origin at ship origin. Rotated by ship pitch and roll only (yaw=0).
        Camera Frame {C}: Offset from ship origin by camera mount position and orientation.

    Args:
        cam : camera name key in camera_settings.json ('left' or 'right')
        p   : ship pitch in NED radians
        r   : ship roll  in NED radians
        (ship position is always world origin, yaw is always zero)

    Returns:
        E_cv : (3x4) world-to-camera matrix in OpenCV axis convention
    """

    with open(r'C:\Users\Diogo Ramos\Desktop\NAVIS_DiogoRamos\NAVIS\EKF\config\camera_settings.json', 'r') as file:
        cam_settings = json.load(file)

    key = f'extrinsics_{cam}'

    # ── Camera mount parameters (in ship body frame) ──────────────────────────
    # Position offset is relative to (0, 12, 3.5) in NED ship body frame
    # Note: (0, 12, -3.5) in Unreal → (0, 12, +3.5) in NED (Z flips)
    tx = cam_settings[key]['tx'] #+ 0.0
    ty = cam_settings[key]['ty'] #+ 12.0
    tz = cam_settings[key]['tz'] #- 3.5

    pitch_off = cam_settings[key]['pitch_offset_deg']
    roll_off  = cam_settings[key]['roll_offset_deg']
    yaw_off   = cam_settings[key]['yaw_offset_deg']

    # ── T_WS: World → Ship ────────────────────────────────────────────────────
    # Ship origin coincides with world origin (translation = 0)
    # Only pitch and roll applied, yaw = 0
    R_WS = Rotation.from_euler('ZYX', [0.0, p, r], degrees=False).as_matrix()

    T_WS = np.eye(4)
    T_WS[:3, :3] = R_WS
    T_WS[:3,  3] = np.array([0.0, 0.0, 0.0])   # ship origin = world origin

    # ── T_SC: Ship → Camera ───────────────────────────────────────────────────
    # Camera is mounted at (tx, ty, tz) in ship body frame
    # with a fixed orientation offset (roll_off, pitch_off, yaw_off)
    R_SC = Rotation.from_euler('ZYX', [yaw_off, pitch_off, roll_off], degrees=True).as_matrix()

    T_SC = np.eye(4)
    T_SC[:3, :3] = R_SC
    T_SC[:3,  3] = np.array([tx, ty, tz])

    # ── T_WC: World → Camera (full chain) ────────────────────────────────────
    T_WC = T_WS @ T_SC

    # ── T_CW: Invert to get world-to-camera transform ────────────────────────
    T_CW = np.linalg.inv(T_WC)
    T    = T_CW[:3, :]                          # keep only the 3x4 block

    # ── NED → OpenCV axis permutation ────────────────────────────────────────
    # NED:    X=North(forward)  Y=East(right)  Z=Down
    # OpenCV: X=Right           Y=Down         Z=Forward(optical axis)
    E_cv = np.zeros((3, 4))
    E_cv[0, :] = T[1, :]   # OpenCV X (right)   ← NED Y (east)
    E_cv[1, :] = T[2, :]   # OpenCV Y (down)    ← NED Z (down)
    E_cv[2, :] = T[0, :]   # OpenCV Z (forward) ← NED X (north)

    return E_cv


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

    df_ship['roll'] = np.radians(df_ship['roll'])
    df_ship['yaw'] = np.radians(df_ship['yaw'])
    df_ship['pitch'] = np.radians(df_ship['pitch'])

    return df_ship, df_drone



def combines_dataframes(df_ship, df_drone):
    
    df_ship['entity'] = 'ship'
    df_drone['entity'] = 'drone'

    df_combined = pd.concat([df_ship, df_drone], ignore_index=True)
    df_combined = df_combined.sort_values(by='timestamp').reset_index(drop=True)

    return df_combined
    

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
    B[4,1] = dt
    B[5,2] = dt

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
    np.fill_diagonal(P[3:6, 3:6], vel_var)
    np.fill_diagonal(P[6:9, 6:9], ori_var)
    np.fill_diagonal(P[9:15, 9:15], bias_var)
    return P


def correct_tilt(x, P, imu_accel, R_accel_noise):
    """ Corrects Roll and Pitch using gravity observation """
    q = x[6:10]
    rot = Rotation.from_quat(q)
    R_mat = rot.as_matrix()

    # Normalize the accelerometer reading to get the "down" direction
    accel_norm = np.linalg.norm(imu_accel)
    if accel_norm == 0: return x, P
    
    # Expected gravity vector in world frame (NED)
    g_world = np.array([0.0, 0.0, 1.0])
    
    # Actual gravity vector observed in world frame based on current orientation
    g_obs = R_mat @ (imu_accel / accel_norm)

    # Cross product gives the angular error between expected and observed
    # This is a neat trick to directly calculate the error state for theta
    error_theta = np.cross(g_obs, g_world)

    # We only want to correct Roll (x) and Pitch (y). Yaw (z) is unobservable by gravity.
    error_theta[2] = 0.0 

    # H matrix for this specific update maps directly to orientation
    H = np.zeros((3, 15))
    H[:, 6:9] = np.eye(3) 
    
    # Kalman Gain
    S = H @ P @ H.T + R_accel_noise
    K = P @ H.T @ np.linalg.inv(S)

    # Error state
    error_state = K @ error_theta

    # Apply corrections
    delta_theta = error_state[6:9]
    rot_error = Rotation.from_rotvec(delta_theta)
    rot_new = rot * rot_error
    q_new = rot_new.as_quat()

    # We also allow it to update the gyro biases slightly
    bg_new = x[13:16] + error_state[12:15]

    # Re-pack state (Position, Velocity, Accel Bias remain untouched)
    x_new = np.concatenate((x[0:6], q_new, x[10:13], bg_new))

    # Update Covariance
    P_new = (np.eye(15) - K @ H) @ P
    P_new = (P_new + P_new.T) / 2.0

    return x_new, P_new


def get_extended_Q_matrix(dt, sigma_accel, sigma_gyro):
    Q = np.zeros((15, 15))
    
    dt2 = dt**2
    dt3 = dt**3
    dt4 = dt**4

    # 1. Accelerometer Noise (Affects Position, Velocity, and their correlation)
    # Variance of position error = (1/4) * sigma_accel * dt^4
    np.fill_diagonal(Q[0:3, 0:3], (sigma_accel * dt4) / 4.0)
    
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