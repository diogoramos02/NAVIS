import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from helpers_2 import *
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

########################################################
#                                                      #
#                   FILE PROCESSING                    #
#                                                      #
########################################################


# PATH TO TRAJECTORY DIRECTORY #
DIRECTORY_PATH = r'C:\Trajectories\trajectory3'

# PATH TO DRONE IMU DATA #
DRONE_IMU_PATH = os.path.join(DIRECTORY_PATH, 'drone_imu.csv')

# PATH TO SHIP IMU DATA #
SHIP_IMU_PATH = os.path.join(DIRECTORY_PATH, 'capture_log.csv')

# PATH TO IMAGES #
IMAGES_PATH = os.path.join(DIRECTORY_PATH, 'images')

# PATH TO YOLO MODEL #
MODEL_PATH = r'C:\Users\Diogo Ramos\Desktop\NAVIS_DiogoRamos\NAVIS\EKF\best.pt'



########################################################
#                                                      #
#                   DATA EXTRACTION                    #
#                                                      #
########################################################

results = []

df_ship, df_drone = format_data(SHIP_IMU_PATH, DRONE_IMU_PATH, DIRECTORY_PATH)

model = YOLO(MODEL_PATH)

intrinsics = get_intrinsics()
left_extrinsics = get_extrinsics('left',0,12,-3.5,0,0,0) 
right_extrinsics = get_extrinsics('right',0,12, -3.5,0,0,0) 


#Get the projection matrixes for triangulation in OpenCV format
left_projection = get_projection_matrix(intrinsics, left_extrinsics)
right_projection = get_projection_matrix(intrinsics, right_extrinsics)


########################################################
#                                                      #
#          FORMATING THE TIMESTAMPS                    #
#                                                      #
########################################################

starting_position = get_starting_position(DIRECTORY_PATH)

trajectory_start_time = df_drone['airsim_timestamp'].min()
trajectory_end_time = df_drone['airsim_timestamp'].max()


filtered_df_ship = df_ship[df_ship['timestamp'] >= trajectory_start_time].copy()
filtered_df_ship = filtered_df_ship[filtered_df_ship['timestamp'] <= trajectory_end_time].copy()
filtered_df_ship['timestamp'] = filtered_df_ship['timestamp'] - trajectory_start_time
filtered_df_ship = filtered_df_ship.rename(
    columns={"timestamp": "airsim_timestamp"}
)

df_drone['airsim_timestamp'] = df_drone['airsim_timestamp'] - trajectory_start_time

previous_timestamp = None

########################################################
#                                                      #
#          CONCATENATING DATA FRAMES                   #
#                                                      #
########################################################


df_combined = combines_dataframes(filtered_df_ship, df_drone)


########################################################
#                                                      #
#           FINDING INITIAL EKF FRAME                  #
#                                                      #
########################################################

successful_initial_frame = None
left_results = None
right_results = None

for index, row in filtered_df_ship.iterrows():

    frame_num = int(float(row['frame']))
    frame_str = f"{frame_num:04d}"
    print(f"Analyzing frame: {frame_num}")


    left_img_path = os.path.join(IMAGES_PATH, f"left_{frame_str}.bmp")
    right_img_path = os.path.join(IMAGES_PATH, f"right_{frame_str}.bmp")

    if not os.path.exists(left_img_path) or not os.path.exists(right_img_path):
        print(f"Warning: Images for frame {frame_str} not found. Skipping.")
        continue    
    
    res_left = model(left_img_path, verbose=False)[0] 
    res_right = model(right_img_path, verbose=False)[0]

    left_detected = len(res_left.boxes) > 0
    right_detected = len(res_right.boxes) > 0

    if left_detected and right_detected:
        successful_initial_frame = frame_str
        left_results = res_left
        right_results = res_right
        previous_timestamp = row['airsim_timestamp']

        break
    

print(f"Successful initial frame is: {successful_initial_frame}")
    
df_combined = df_combined[df_combined['airsim_timestamp'] > previous_timestamp].reset_index(drop=True)    
df_drone = df_drone[df_drone['airsim_timestamp'] > previous_timestamp].reset_index(drop=True)    


#tsart = df_drone['airsim_timestamp'].min()
df_drone['airsim_timestamp'] = df_drone['airsim_timestamp'] - previous_timestamp
df_combined['airsim_timestamp'] = df_combined['airsim_timestamp'] - previous_timestamp

previous_timestamp = 0.0
########################################################
#                                                      #
#        Implement handling of no detection            #
#                                                      #
########################################################


########################################################
#                                                      #
# EXTRACT THE CENTER OF THE BOX WITH BEST SCORE        #
#                                                      #
########################################################

best_left_idx = res_left.boxes.conf.argmax()
best_right_idx = res_right.boxes.conf.argmax()

u_l = float(res_left.boxes.xywh[best_left_idx][0])
v_l = float(res_left.boxes.xywh[best_left_idx][1])

u_r = float(res_right.boxes.xywh[best_right_idx][0])
v_r = float(res_right.boxes.xywh[best_right_idx][1])

intital_position_estimate, reproj_err = get_inital_position_estimate(u_l, v_l, left_projection, u_r, v_r, right_projection)

print(f"Initial position estimate is: {intital_position_estimate}")
print(f"Reproj error is: {reproj_err}")


########################################################
#                                                      #
#          SET THE INITIAL STATE OF THE EKF            #
#                                                      #
########################################################

#For the velocity, how to set initial estimate?
x = np.array([intital_position_estimate[0], intital_position_estimate[1], intital_position_estimate[2], -1.0, -2.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

########################################################
#                                                      #
#           SET THE PARAMETERS OF THE EKF              #
#                                                      #
########################################################

P = initialize__extended_state_covariance(20.0, 1.0, 0.01, 0.001)
R = np.eye(2) * 9.0  # Pixel variance


sigma_accel = 1.4
sigma_gyro = 0.1

imu_accel = np.zeros(3)
imu_gyro  = np.zeros(3)

for idx, row in df_combined.iterrows():
    
    
    new_timestamp = row['airsim_timestamp']
    dt = new_timestamp - previous_timestamp

    #PREDICT#
    


    if(row['entity'] == 'drone'):
        new_timestamp = row['airsim_timestamp']
        dt = new_timestamp - previous_timestamp
        Q = get_extended_Q_matrix(dt, sigma_accel, sigma_gyro)
        imu_accel = np.array([row['acc_x'], row['acc_y'], row['acc_z']])
        imu_gyro = np.array([row['gyro_x'], row['gyro_y'], row['gyro_z']])
        x ,P = predict_imu(x, P, imu_accel, imu_gyro, dt, Q)
        previous_timestamp = new_timestamp


    if(row['entity'] == 'ship'):

        print(f"Time: {row['airsim_timestamp']}... x: {x[0]:.2f}, y: {x[1]:.2f}, z: {x[2]:.2f}")

        left_projection = get_projection_matrix(intrinsics, left_extrinsics)
        right_projection = get_projection_matrix(intrinsics, right_extrinsics)

        # CORRECT #
        frame_num = int(float(row['frame']))
        frame_str = f"{frame_num:04d}"

        left_img_path = os.path.join(IMAGES_PATH, f"left_{frame_str}.bmp")
        right_img_path = os.path.join(IMAGES_PATH, f"right_{frame_str}.bmp")

        if not os.path.exists(left_img_path) or not os.path.exists(right_img_path):
            print(f"Warning: Images for frame {frame_str} not found. Skipping.")
            continue
        
        # 2. Check if the files are entirely empty (0 bytes)
        if os.path.getsize(left_img_path) == 0 or os.path.getsize(right_img_path) == 0:
            print(f"Warning: Images for frame {frame_str} are empty (0 bytes). Skipping.")
            continue
        
        # 3. Try to process the images, catch OpenCV decoding errors if they are corrupted
        try:
            res_left = model(left_img_path, verbose=False)[0] 
            res_right = model(right_img_path, verbose=False)[0]
        except Exception as e:
            print(f"Warning: Corrupted image at frame {frame_str}. Model error: {e}. Skipping.")
            continue
    

        left_detected = len(res_left.boxes) > 0
        right_detected = len(res_right.boxes) > 0


        if left_detected:
            new_timestamp = row['airsim_timestamp']
            dt = new_timestamp - previous_timestamp
            Q = get_extended_Q_matrix(dt, sigma_accel, sigma_gyro)
            x ,P = predict_imu(x, P, imu_accel, imu_gyro, dt, Q)

            p_homogenous = np.array([x[0], x[1], x[2], 1.0])
            pixel_coords_left = left_projection @ p_homogenous

            if(pixel_coords_left[2]>0):
                u = pixel_coords_left[0] / pixel_coords_left[2]
                v = pixel_coords_left[1] / pixel_coords_left[2]

            best_left_idx = res_left.boxes.conf.argmax()
            u_l = float(res_left.boxes.xywh[best_left_idx][0])
            v_l = float(res_left.boxes.xywh[best_left_idx][1])

            H = get_jacobian_extended_state(x, left_extrinsics, intrinsics)

            innovation = np.array([u_l, v_l]) - np.array([u, v])
            NIS = innovation.T @ np.linalg.inv(H @ P @ H.T + R) @ innovation
            print(f"NIS: {NIS}") 
            if H is not None:
                K = get_Kalman_gain(H, R, P)

                error_state =  K @ (np.array([u_l, v_l]) - np.array([u, v]))

                delta_p = error_state[0:3]
                print(f"Delta p left: {delta_p}")
                delta_v = error_state[3:6]
                delta_theta = error_state[6:9]
                delta_ba = error_state[9:12]
                delta_bg = error_state[12:15]

                p_new = x[0:3] + delta_p
                v_new = x[3:6] + delta_v
                ba_new = x[10:13] + delta_ba
                bg_new = x[13:16] + delta_bg

                rot_current = Rotation.from_quat(x[6:10])
                rot_error = Rotation.from_rotvec(delta_theta)

                rot_new = rot_current * rot_error
                q_new = rot_new.as_quat()

                x = np.concatenate((p_new, v_new, q_new, ba_new, bg_new))

                I_KH = np.eye(15) - K @ H
                P = I_KH @ P @ I_KH.T + K @ R @ K.T

            previous_timestamp = new_timestamp



        if right_detected:
            new_timestamp = row['airsim_timestamp']
            dt = new_timestamp - previous_timestamp
            Q = get_extended_Q_matrix(dt, sigma_accel, sigma_gyro)
            if(dt>0):
                x ,P = predict_imu(x, P, imu_accel, imu_gyro, dt, Q)

            p_homogenous = np.array([x[0], x[1], x[2], 1.0])
            pixel_coords_right = right_projection @ p_homogenous

            if(pixel_coords_right[2]>0):
                u = pixel_coords_right[0] / pixel_coords_right[2]
                v = pixel_coords_right[1] / pixel_coords_right[2]

            best_right_idx = res_right.boxes.conf.argmax()
            u_r = float(res_right.boxes.xywh[best_right_idx][0])
            v_r = float(res_right.boxes.xywh[best_right_idx][1])

            H = get_jacobian_extended_state(x, right_extrinsics, intrinsics)

            innovation = np.array([u_r, v_r]) - np.array([u, v])
            NIS = innovation.T @ np.linalg.inv(H @ P @ H.T + R) @ innovation
            print(f"NIS: {NIS}") 

            if H is not None:
                K = get_Kalman_gain(H, R, P)

                error_state =  K @ (np.array([u_r, v_r]) - np.array([u, v]))

                delta_p = error_state[0:3]
                print(f"Delta p right: {delta_p}")
                delta_v = error_state[3:6]
                delta_theta = error_state[6:9]
                delta_ba = error_state[9:12]
                delta_bg = error_state[12:15]

                p_new = x[0:3] + delta_p
                v_new = x[3:6] + delta_v
                ba_new = x[10:13] + delta_ba
                bg_new = x[13:16] + delta_bg

                rot_current = Rotation.from_quat(x[6:10])
                rot_error = Rotation.from_rotvec(delta_theta)

                rot_new = rot_current * rot_error
                q_new = rot_new.as_quat()

                x = np.concatenate((p_new, v_new, q_new, ba_new, bg_new))

                I_KH = np.eye(15) - K @ H
                P = I_KH @ P @ I_KH.T + K @ R @ K.T

            previous_timestamp = new_timestamp

    results.append({
        'timestamp' : new_timestamp,
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
 


df_ekf = pd.DataFrame(results)
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
df_ekf.to_csv("ekf_output.csv", index=False)
print(df_ekf.head())


df_drone['pos_x'] = df_drone['pos_x'] - df_drone['ship_pos_x']
df_drone['pos_y'] = df_drone['pos_y'] - df_drone['ship_pos_y']
df_drone['pos_z'] = df_drone['pos_z'] - df_drone['ship_pos_z']


########################################################
#                                                      #
#                 Forat ground truth                   #
#                                                      #
########################################################
df_gt_matched = match_nearest(df_drone, df_ekf, source_col='airsim_timestamp', target_col='timestamp')


########################################################
#                                                      #
#                 Plot Results                         #
#                                                      #
########################################################


fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle("EKF Position Estimate vs Ground Truth", fontsize=14)

labels    = ["X - North (m)", "Y - East (m)", "Z - Down (m)"]
ekf_cols  = ["ekf_px", "ekf_py", "ekf_pz"]
gt_cols   = ["pos_x",  "pos_y",  "pos_z"]
var_cols  = ["var_px", "var_py", "var_pz"]

for ax, label, ekf_col, gt_col, var_col in zip(
        axes, labels, ekf_cols, gt_cols, var_cols):

    t = df_ekf["timestamp"] - df_ekf["timestamp"].iloc[0]  # relative time from 0

    # Ground truth
    ax.plot(t, df_gt_matched[gt_col], 
            color="green", linewidth=2, label="Ground Truth")

    # EKF estimate
    ax.plot(t, df_ekf[ekf_col], 
            color="red", linestyle="--", linewidth=1.5, label="EKF Estimate")


    # Uncertainty band (±2 standard deviations = 95% confidence)
    std = np.sqrt(df_ekf[var_col])
    ax.fill_between(t,
                    df_ekf[ekf_col] - 2 * std,
                    df_ekf[ekf_col] + 2 * std,
                    color="red", alpha=0.15, label="±2σ uncertainty")

    ax.set_ylabel(label)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("ekf_position.png", dpi=150)

fig_vel, axes_vel = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig_vel.suptitle("EKF Velocity Estimate vs Ground Truth", fontsize=14)

vel_labels   = ["Vx - North (m/s)", "Vy - East (m/s)", "Vz - Down (m/s)"]
vel_ekf_cols = ["ekf_vx", "ekf_vy", "ekf_vz"]  # Update these if your columns differ
vel_gt_cols  = ["vel_x",  "vel_y",  "vel_z"]   # Update these if your columns differ
vel_var_cols = ["var_vx", "var_vy", "var_vz"]  # Update these if your columns differ

for ax, label, ekf_col, gt_col, var_col in zip(
        axes_vel, vel_labels, vel_ekf_cols, vel_gt_cols, vel_var_cols):

    # Ground truth
    ax.plot(t, df_gt_matched[gt_col], 
            color="blue", linewidth=2, label="Ground Truth")

    # EKF estimate
    ax.plot(t, df_ekf[ekf_col], 
            color="orange", linestyle="--", linewidth=1.5, label="EKF Estimate")

    # Uncertainty band
    std = np.sqrt(df_ekf[var_col])
    ax.fill_between(t,
                    df_ekf[ekf_col] - 2 * std,
                    df_ekf[ekf_col] + 2 * std,
                    color="orange", alpha=0.15, label="±2σ uncertainty")

    ax.set_ylabel(label)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("ekf_velocity.png", dpi=150)

fig_ori, axes_ori = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig_ori.suptitle("EKF Orientation Estimate vs Ground Truth", fontsize=14)

ori_labels   = ["Roll (deg)", "Pitch (deg)", "Yaw (deg)"]
ekf_ori_cols = ["ekf_roll", "ekf_pitch", "ekf_yaw"]
gt_ori_cols  = ["gt_roll", "gt_pitch", "gt_yaw"]
ori_var_cols = ["var_roll", "var_pitch", "var_yaw"]

for ax, label, ekf_col, gt_col, var_col in zip(axes_ori, ori_labels, ekf_ori_cols, gt_ori_cols, ori_var_cols):

    # Ground truth
    ax.plot(t, df_gt_matched[gt_col], 
            color="purple", linewidth=2, label="Ground Truth")
        
    # EKF estimate
    ax.plot(t, df_ekf[ekf_col], 
            color="cyan", linestyle="--", linewidth=1.5, label="EKF Estimate")

    # Uncertainty band (±2 standard deviations)
    # Variance is in rad^2. Convert to std in degrees.
    std_deg = np.sqrt(df_ekf[var_col]) * (180.0 / np.pi)
    ax.fill_between(t,
                    df_ekf[ekf_col] - 2 * std_deg,
                    df_ekf[ekf_col] + 2 * std_deg,
                    color="cyan", alpha=0.15, label="±2σ uncertainty")

    ax.set_ylabel(label)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("ekf_orientation.png", dpi=150)

fig_bi, axes_bi = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig_bi.suptitle("EKF Bias Estimate", fontsize=14)

bi_labels   = ["bx", "by", "bz"]
ba_ekf_cols = ["ekf_bax", "ekf_bay", "ekf_baz"]  # Update these if your columns differ
bg_ekf_cols  = ["ekf_bgx",  "ekf_bgy",  "ekf_bgz"]   # Update these if your columns differ
bi_var_cols = ["var_bax", "var_bay", "var_baz"]  # Update these if your columns differ

for ax, label, ekf_col, gt_col, var_col in zip(
        axes_bi, bi_labels, ba_ekf_cols, bg_ekf_cols, bi_var_cols):

    # Ground truth
    ax.plot(t, df_ekf[gt_col], 
            color="blue", linewidth=2, label="Ground Truth")

    # EKF estimate
    ax.plot(t, df_ekf[ekf_col], 
            color="orange", linestyle="--", linewidth=1.5, label="EKF Estimate")

    # Uncertainty band
    std = np.sqrt(df_ekf[var_col])
    ax.fill_between(t,
                    df_ekf[ekf_col] - 2 * std,
                    df_ekf[ekf_col] + 2 * std,
                    color="orange", alpha=0.15, label="±2σ uncertainty")

    ax.set_ylabel(label)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("ekf_bias.png", dpi=150)



# Show both plots
plt.show()

# Position error over time
pos_err = np.sqrt(
    (df_ekf["ekf_px"] - df_gt_matched["pos_x"])**2 +
    (df_ekf["ekf_py"] - df_gt_matched["pos_y"])**2 +
    (df_ekf["ekf_pz"] - df_gt_matched["pos_z"])**2
)

print(f"Mean position error : {pos_err.mean():.3f} m")
print(f"Max position error  : {pos_err.max():.3f} m")
print(f"Std position error  : {pos_err.std():.3f} m")