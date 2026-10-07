import numpy as np
import json
from CV_helpers import *

def initialize_state(trajectory_settings, inital_position_estimate):
    """Intializes the EKF state with the initial position estimate and the velocity in the direction of the world frame origin

    Args:
        trajeotry_settings (str) : Loaded json file where trajectory settings are stored.
        intial_position_estimate (np.array) : (3x1) Position (in {W}) triangulated from the first frame where the drone is detected in both images 

    Returns:
        x (np.array): (16x1) Initial State of the Extended Kalman Filter with position, velocity, orientation and biases

    Raises:
        FileNotFoundError: If the required file is not found in the specified directory.
        ValueError: If the file is empty or corrupted.
    """

    distance = np.linalg.norm(inital_position_estimate)

    # Project the velocity from the starting position estimate towards the origin.
    # The velocity module is assumed to be known.
    velocity = trajectory_settings["velocity"]
    vx = -velocity * inital_position_estimate[0]/distance
    vy = -velocity * inital_position_estimate[1]/distance
    vz = -velocity * inital_position_estimate[2]/distance

    # Initial estimation for the yaw assumes the drone in facing direectl towards the ship
    # Pitch and roll are initialized as zero
    yaw = np.arctan2(-inital_position_estimate[1], -inital_position_estimate[0])
    r = Rotation.from_euler('ZYX', [yaw, 0.0, 0.0], degrees=False)
    q = r.as_quat()

    # The bias are initialized as zero
    x = np.array([inital_position_estimate[0], inital_position_estimate[1], inital_position_estimate[2], vx, vy, vz,
                    q[0], q[1], q[2], q[3], 0.0, 0.0, 0.0, 0.0, 0.0 , 0.0])

    return x


def intialize_state_covariance(position_variance, velocity_variance, ori_variance, bias_variance, depth_scale_factor = 1):
    """Intializes the EKF state covariance matrix.
       The depth scale factor is used tu increase the variance of the depth axis position and velocity.
       A binocular camera system has less precision in the depth direction so it might be good to increase the variance.

    Args:
        position_variance : Initial value of position variance 
        velocity_variance : Initial value of velocity variance
        ori_variance : Initial value of orientation variance in degrees
        bias_variance : Initial value of bias variance 
        depth_scale_factor : Factor by which the depth axis variables' variances are to be scaled 

    Returns:
        P (np.array): (15x15) Initial state covariance matrix
    """

    P = np.eye(15)

    P[0,0] = position_variance
    P[1,1] = position_variance * depth_scale_factor
    P[2,2] = position_variance

    P[3,3] = velocity_variance
    P[4,4] = velocity_variance * depth_scale_factor
    P[5,5] = velocity_variance

    P[6,6] = ori_variance
    P[7,7] = ori_variance
    P[8,8] = ori_variance * 3

    np.fill_diagonal(P[9:15, 9:15], bias_variance)

    return P


def get_extended_Q_matrix(dt, sigma_accel, sigma_gyro):
    """Intializes the process noise covariance matrix.

    Args:
        sigma_accel : Accelerometer covariance matrix 
        sigma_gyro : Gryoscope covariance matrix

    Returns:
        Q (np.array): (15x15) Proces noise covariance
    """

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
    np.fill_diagonal(Q[9:12, 9:12], 1e-6 * dt) # Accel Bias
    np.fill_diagonal(Q[12:15, 12:15], 1e-6 * dt) # Gyro Bias

    return Q


def intialize_R_matrix(pixel_variance):
    """Intializes the EKF measurement noise covariance matrix.

    Args:
        pixel_variance : Initial value of pixel variance

    Returns:
        R (np.array): (2x2) Initial measurement covariance matrix
    """

    R = np.eye(2) * pixel_variance

    return R


def skew_symmetric(v):
    """Calculates and returns the skew-symmetric matrix of a vector

    Args:
        v : 3D vector

    Returns:
        (np.array): (3x3) Skew-symmetric matrix of vector v
    """
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def predict_imu(x, P, imu_accel, imu_gyro, dt, Q):
    """Performs the prediciton step of the EKF by integrating the IMU measurements

    Args:
        x (np.array) : (16x1) EKF state
        P (np.array) : (15x15) EKF state covariance matrix
        imu_accel : IMU accelerometer measurements
        imu_gyro : IMU gyroscope measurements
        dt : time step between iterations
        Q : Process noise covariance matrix 

    Returns:
        x_new (np.array) : (16x1) Updated EKF state
        P_new (np.array) : (15x15) Updated state covariance matrix
    """

    # 1. Extract the current state variables
    p = x[:3]
    v = x[3:6]
    q = x[6:10]
    ba = x[10:13]
    bg = x[13:16]

    # 2. Correct the IMU readings by removing the bias BEFORE integrating
    accel_true = imu_accel - ba
    gyro_true = imu_gyro - bg

    # 3. Calculate rotation matrix, extract the world frame acceleration by removing the gravity
    rot = Rotation.from_quat(q)
    R_mat = rot.as_matrix()
    g = np.array([0.0, 0.0, 9.81])
    accel_world = R_mat @ accel_true + g

    # 4. Integrate IMU meaasurements and calculate state updates
    p_new = p + v * dt + 0.5 * accel_world * dt**2
    v_new = v + accel_world * dt
    delta_rot = Rotation.from_rotvec(gyro_true * dt)
    rot_new = rot * delta_rot
    q_new = rot_new.as_quat()
    x_new = np.concatenate((p_new, v_new, q_new, ba, bg))

    # 5. Calculate the 15x15 Error-State Jacobian (F)
    F = np.eye(15)

    F[0:3, 3:6] = np.eye(3) * dt
    F[0:3, 6:9] = -0.5 * R_mat @ skew_symmetric(accel_true) * dt**2
    F[3:6, 6:9] = -R_mat @ skew_symmetric(accel_true) * dt
    F[6:9, 6:9] = np.eye(3) - skew_symmetric(gyro_true) * dt
    
    F[0:3, 9:12] = -0.5 * R_mat * dt**2
    F[3:6, 9:12] = -R_mat * dt
    F[6:9, 12:15] = -np.eye(3) * dt

    P_new = F @ P @ F.T + Q

    return x_new, P_new

def log_results(results, x, P, timestamp):
    """Logs the current state and state covariance values

    Args:
        results : List with the logged results
        x : current state
        P : current state covariance matrix
        timestamp : current iteration timestamp
         
    Returns:
        results : Updated list with the logged results
    """    
    results.append({
        'timestamp' : timestamp,
        # EKF state
        'ekf_px'    : x[0],
        'ekf_py'    : x[1],
        'ekf_pz'    : x[2],
        'ekf_vx'    : x[3],
        'ekf_vy'    : x[4],
        'ekf_vz'    : x[5],
        'ekf_qx'    : x[6],
        'ekf_qy'    : x[7],
        'ekf_qz'    : x[8],
        'ekf_qw'    : x[9],
        'ekf_bax'   : x[10],
        'ekf_bay'   : x[11],
        'ekf_baz'   : x[12],
        'ekf_bgx'   : x[13],
        'ekf_bgy'   : x[14],
        'ekf_bgz'   : x[15],
        # Covariance diagonal (uncertainty)
        'var_px'    : P[0, 0],
        'var_py'    : P[1, 1],
        'var_pz'    : P[2, 2],
        'var_vx'    : P[3, 3],
        'var_vy'    : P[4, 4],
        'var_vz'    : P[5, 5],
        'var_roll'    : P[6, 6],
        'var_pitch'    : P[7, 7],
        'var_yaw'    : P[8, 8],
        'var_bax'   :P[9,9],
        'var_bay'   :P[10,10],
        'var_baz'   :P[11,11],
        'var_bgx'   :P[12,12],
        'var_bgy'   :P[13,13],
        'var_bgz'   :P[14,14],

    })

    return results



def get_jacobian(x, T, K):
    """
    Computes the measurement Jacobian (2x15) for an Extended Kalman Filter,
    mapping the 6D state to 2D image pixel coordinates.

    Args:
        x : Current EKF state
        T : Extrinsic parameter matrix
        K : Intrinsic parameter matrix
         
    Returns:
        H : Measurement Jacobian
    """ 

    # Extract focal lengths from the Intrinsic Matrix
    fx = K[0, 0]
    fy = K[1, 1]

    # Transform the 3D world position estimate into the camera's local coordinate frame
    p_cam = T @ np.array([x[0], x[1], x[2], 1.0])

    x_cam = p_cam[0]
    y_cam = p_cam[1]
    z_cam = p_cam[2]

    if z_cam < 0: 
        return None

    # Compute the projection Jacobian
    # Evaluates how pixel coordinates (u, v) change with respect to camera frame coordinates (X_c, Y_c, Z_c)
    J_proj = np.array([
        [fx/z_cam, 0, -fx*x_cam/ z_cam ** 2],
        [0, fy/z_cam, -fy * y_cam/ z_cam**2]
    ])

    # Apply the chain rule
    # Multiply by the rotation matrix (T[:, :3]) to map the camera frame gradients back to the World frame
    J_pos = J_proj @ T[:, :3]

    # Velocity has no impact on pixel coordinates
    H = np.zeros((2,15))
    H[:, :3] = J_pos

    return H


def get_Kalman_gain(H, R, P):
    """
    Computes the Kalman gain matrix

    Args:
        H : Measurement Jacobian
        R : Measurement noise covariance matrix
        P : State covariance matrix
         
    Returns:
        K : Kalman gain
    """ 
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)

    return K


def correction_step(x, P, intrinsics, extrinsics, detection_results, R):
    """
    Computes the update step of the EKF based on the drone detection in one of the images

    Args:
        x : Current EKF state
        P : State covariance matrix
        intrinsics : Intrinsic parameter matrix
        extrinsics : Extrinsic parameter matrix
        detection_results : output of the detector
        R : Measurement noise covariance matrix
         
    Returns:
        x : Updated EKF state
        P : Updated state covariance matrix
    """ 

    #Project the drone postion to get the expected pixel coordinates in the image
    projection = get_projection_matrix(intrinsics, extrinsics)
    p_homogenous = np.array([x[0], x[1], x[2], 1.0])
    pixel_coords = projection @ p_homogenous

    if(pixel_coords[2]>0):
        u = pixel_coords[0] / pixel_coords[2]
        v = pixel_coords[1] / pixel_coords[2]

        best_idx = detection_results.boxes.conf.argmax()
        u_l = float(detection_results.boxes.xywh[best_idx][0])
        v_l = float(detection_results.boxes.xywh[best_idx][1])

        H = get_jacobian(x, extrinsics, intrinsics)

        #Update the state based on the innovation
        if H is not None:

            K = get_Kalman_gain(H, R, P)

            # Calculate the error state and the corrections to be applied
            error_state =  K @ (np.array([u_l, v_l]) - np.array([u, v]))
            delta_p = error_state[0:3]
            delta_v = error_state[3:6]
            delta_theta = error_state[6:9]
            delta_ba = error_state[9:12]
            delta_bg = error_state[12:15]

            # Apply the corrections to the state variables
            p_new = x[0:3] + delta_p
            v_new = x[3:6] + delta_v
            ba_new = x[10:13] + delta_ba
            bg_new = x[13:16] + delta_bg

            # Orientation corrections are calculated differently
            rot_current = Rotation.from_quat(x[6:10])
            rot_error = Rotation.from_rotvec(delta_theta)
            rot_new = rot_current * rot_error
            q_new = rot_new.as_quat()

            # Pack the new state
            x = np.concatenate((p_new, v_new, q_new, ba_new, bg_new))

            # Calculate the new state covariance
            I_KH = np.eye(15) - K @ H
            P = I_KH @ P @ I_KH.T + K @ R @ K.T


    return x, P


def correct_ahrs(x, P, quat_measured, R_ahrs):
    """
    Corrects the EKF state using AHRS quaternion measurements.

    Args:
        x (np.array): (16x1) EKF nominal state
        P (np.array): (15x15) EKF state covariance matrix
        quat_measured (np.array): (4x1) Measured quaternion [x, y, z, w]
        R_ahrs (np.array): (3x3) AHRS measurement noise covariance matrix
         
    Returns:
        x : Updated 16D EKF state
        P : Updated 15x15 state covariance matrix
    """ 
    # Extract current estimated orientation
    rot_est = Rotation.from_quat(x[6:10])
    rot_meas = Rotation.from_quat(quat_measured)

    # Calculate innovation: the angular error between estimate and measurement in the body frame
    rot_error_meas = rot_est.inv() * rot_meas
    innovation = rot_error_meas.as_rotvec()

    # The AHRS directly observes the 3D orientation error state
    H = np.zeros((3, 15))
    H[:, 6:9] = np.eye(3)

    # Calculate Kalman Gain
    S = H @ P @ H.T + R_ahrs
    K = P @ H.T @ np.linalg.inv(S)

    # Calculate error state
    error_state = K @ innovation

    # Extract corrections
    delta_p = error_state[0:3]
    delta_v = error_state[3:6]
    delta_theta = error_state[6:9]
    delta_ba = error_state[9:12]
    delta_bg = error_state[12:15]

    # Apply translation and bias corrections
    p_new = x[0:3] + delta_p
    v_new = x[3:6] + delta_v
    ba_new = x[10:13] + delta_ba
    bg_new = x[13:16] + delta_bg

    # Apply orientation correction
    rot_current = Rotation.from_quat(x[6:10])
    rot_error = Rotation.from_rotvec(delta_theta)
    rot_new = rot_current * rot_error
    q_new = rot_new.as_quat()

    # Pack the updated state
    x = np.concatenate((p_new, v_new, q_new, ba_new, bg_new))

    # Update state covariance (Joseph form)
    I_KH = np.eye(15) - K @ H
    P = I_KH @ P @ I_KH.T + K @ R_ahrs @ K.T

    return x, P
