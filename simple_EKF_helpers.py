import numpy as np
import json
from CV_helpers import *

def initialize_state(trajectory_settings, inital_position_estimate):
    """Intializes the EKF state with the initial position estimate and the velocity in the direction of the world frame origin

    Args:
        trajeotry_settings (str) : Loaded json file where trajectory settings are stored.
        intial_position_estimate (np.array) : (3x1) Position (in {W}) triangulated from the first frame where the drone is detected in both images 

    Returns:
        x (np.array): (6x1) Initial State of the Extended Kalman Filter

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

    x = np.array([inital_position_estimate[0], inital_position_estimate[1], inital_position_estimate[2], vx, vy, vz])

    return x


def intialize_state_covariance(position_variance, velocity_variance, depth_scale_factor = 1):
    """Intializes the EKF state covariance matrix.
       The depth scale factor is used tu increase the variance of the depth axis position and velocity.
       A binocular camera system has less precision in the depth direction so it might be good to increase the variance.

    Args:
        position_variance : Initial value of position variance 
        velocity_variance : Initial value of velocity variance
        depth_scale_factor : Factor by which the depth axis variables' variances are to be scaled 

    Returns:
        P (np.array): (6x6) Initial state covariance matrix
    """

    P = np.eye(6)

    P[0,0] = position_variance
    P[1,1] = position_variance * depth_scale_factor
    P[2,2] = position_variance

    P[3,3] = velocity_variance
    P[4,4] = velocity_variance * depth_scale_factor
    P[5,5] = velocity_variance

    return P

def intialize_R_matrix(pixel_variance):
    """Intializes the EKF measurement noise covariance matrix.

    Args:
        pixel_variance : Initial value of pixel variance

    Returns:
        R (np.array): (2x2) Initial measurement covariance matrix
    """

    R = np.eye(2) * pixel_variance

    return R


def get_F_matrix(dt):
    """Retrieves the state transition matrix

    Args:
        dt : timestamp between iterations

    Returns:
        F (np.array): (6x6) State transition matrix
    """

    F = np.eye(6)
    F[0,3] = dt
    F[1,4] = dt
    F[2,5] = dt

    return F


def get_Q_matrix(dt, acceleration_variance):
    """Retrieves the process noise covariance matrix

    Args:
        dt : timestamp between iterations
        sigma_accel : acceleration covariance matrix

    Returns:
        Q (np.array): (6x6) Process noise covariance
    """
    acceleration_covariance = np.eye(3) * acceleration_variance

    B = get_B_matrix(dt)
    Q = B @ acceleration_covariance @ B.T

    return Q


def get_B_matrix(dt):
    """Retrieves the B matrix, which maps the acceleration into the state space

    Args:
        dt : timestamp between iterations

    Returns:
        B (np.array): (6x3) Maps acceleration to state space
    """    

    B = np.zeros((6,3))

    dt2 = (dt ** 2)/2

    B[0,0] = dt2
    B[1,1] = dt2
    B[2,2] = dt2
    B[3,0] = dt
    B[4,1] = dt
    B[5,2] = dt

    return B


def log_results(results, x, P, frame_str, timestamp):
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
        'frame'     : frame_str,
        'timestamp' : timestamp,
        # EKF state
        'ekf_px'    : x[0],
        'ekf_py'    : x[1],
        'ekf_pz'    : x[2],
        'ekf_vx'    : x[3],
        'ekf_vy'    : x[4],
        'ekf_vz'    : x[5],
        # Covariance diagonal (uncertainty)
        'var_px'    : P[0, 0],
        'var_py'    : P[1, 1],
        'var_pz'    : P[2, 2],
        'var_vx'    : P[3, 3],
        'var_vy'    : P[4, 4],
        'var_vz'    : P[5, 5],
    })

    return results


def get_jacobian(x, T, K):
    """
    Computes the measurement Jacobian (2x6) for an Extended Kalman Filter,
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
    H = np.zeros((2,6))
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

        #Calculate the Kalman Gain to perform the update
        H = get_jacobian(x, extrinsics, intrinsics)
        K = get_Kalman_gain(H, R, P)

        #Update the state based on the innovation
        x = x + K @ (np.array([u_l, v_l]) - np.array([u, v]))
        I = np.eye(6)
        P = (I - K @ H) @ P @ (I - K @ H).T + (K @ R @ K.T)

    return x, P

