import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from helpers_3 import *
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
left_extrinsics = get_extrinsics('left',0,0) 
right_extrinsics = get_extrinsics('right',0,0) 


#Get the projection matrixes for triangulation in OpenCV format
left_projection = get_projection_matrix(intrinsics, left_extrinsics)
right_projection = get_projection_matrix(intrinsics, right_extrinsics)


########################################################
#                                                      #
#          FORMATING THE TIMESTAMPS                    #
#                                                      #
########################################################

starting_position = get_starting_position(DIRECTORY_PATH)

trajectory_start_time = df_drone['timestamp'].min()
trajectory_end_time = df_drone['timestamp'].max()


filtered_df_ship = df_ship[df_ship['timestamp'] >= trajectory_start_time].copy()
filtered_df_ship = filtered_df_ship[filtered_df_ship['timestamp'] <= trajectory_end_time].copy()
filtered_df_ship['timestamp'] = filtered_df_ship['timestamp'] - trajectory_start_time

df_drone['timestamp'] = df_drone['timestamp'] - trajectory_start_time

previous_timestamp = None


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
        previous_timestamp = row['timestamp']

        break
    

print(f"Successful initial frame is: {successful_initial_frame}")
    
    
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

intital_position_estimate = get_inital_position_estimate(u_l, v_l, left_projection, u_r, v_r, right_projection)

print(f"Initial position estimate is: {intital_position_estimate}")


########################################################
#                                                      #
#          SET THE INITIAL STATE OF THE EKF            #
#                                                      #
########################################################

#For the velocity, how to set initial estimate?
x = np.array([intital_position_estimate[0], intital_position_estimate[1], intital_position_estimate[2], -2, -2, 0])


########################################################
#                                                      #
#           SET THE PARAMETERS OF THE EKF              #
#                                                      #
########################################################
acceleration_variance = 1.0
acceleration_covariance = acceleration_variance * np.eye(3)

P = initialize_state_covariance(position_variance=9.0, velocity_variance=0.2)
pixel_variance = 9.0
R = np.eye(2) * pixel_variance


for idx, row in filtered_df_ship.iterrows():

    if int(float(row['frame'])) <= int(successful_initial_frame):
        continue


    new_timestamp = row['timestamp']
    dt = new_timestamp - previous_timestamp

    #PREDICT#
    F = get_F_matrix(dt)
    B = get_B_matrix(dt)
    Q = get_Q_matrix(B, acceleration_covariance)

    x = F @ x
    P = F @ P @ F.T + Q

    # PROJECT #

    ship_pitch = row["pitch"]
    ship_roll = row["roll"]


    # Calculate new extrinsics if we consider the moiving ship #

    left_extrinsics = get_extrinsics('left',ship_pitch,ship_roll) 
    right_extrinsics = get_extrinsics('right',ship_pitch,ship_roll) 

    # Get the new projection matrices #

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
        results.append({
        'frame'     : frame_str,
        'timestamp' : new_timestamp,
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
        # Detection flags
        'left_detected'  : len(res_left.boxes)  > 0,
        'right_detected' : len(res_right.boxes) > 0,
        })

        previous_timestamp = new_timestamp
        continue
    

    left_detected = len(res_left.boxes) > 0
    right_detected = len(res_right.boxes) > 0


    if left_detected:

        p_homogenous = np.array([x[0], x[1], x[2], 1.0])
        pixel_coords_left = left_projection @ p_homogenous

        if(pixel_coords_left[2]>0):
            u = pixel_coords_left[0] / pixel_coords_left[2]
            v = pixel_coords_left[1] / pixel_coords_left[2]

            best_left_idx = res_left.boxes.conf.argmax()
            u_l = float(res_left.boxes.xywh[best_left_idx][0])
            v_l = float(res_left.boxes.xywh[best_left_idx][1])

            H = get_jacobian(x, left_extrinsics, intrinsics)
            K = get_Kalman_gain(H, R, P)

            x = x + K @ (np.array([u_l, v_l]) - np.array([u, v]))
            I = np.eye(6)
            P = (I - K @ H) @ P @ (I - K @ H).T + (K @ R @ K.T)



    if right_detected:

        p_homogenous = np.array([x[0], x[1], x[2], 1.0])
        pixel_coords_right = right_projection @ p_homogenous

        if(pixel_coords_right[2]>0):
            u = pixel_coords_right[0] / pixel_coords_right[2]
            v = pixel_coords_right[1] / pixel_coords_right[2]

            best_right_idx = res_right.boxes.conf.argmax()
            u_r = float(res_right.boxes.xywh[best_right_idx][0])
            v_r = float(res_right.boxes.xywh[best_right_idx][1])

            H = get_jacobian(x, right_extrinsics, intrinsics)
            K = get_Kalman_gain(H, R, P)

            x = x + K @ (np.array([u_r, v_r]) - np.array([u, v]))
            I = np.eye(6)
            P = (I - K @ H) @ P @ (I - K @ H).T + (K @ R @ K.T)

    results.append({
        'frame'     : frame_str,
        'timestamp' : new_timestamp,
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
        # Detection flags
        'left_detected'  : len(res_left.boxes)  > 0,
        'right_detected' : len(res_right.boxes) > 0,
    })

    previous_timestamp = new_timestamp


df_ekf = pd.DataFrame(results)
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
df_gt_matched = match_nearest(df_drone, df_ekf, source_col='timestamp', target_col='timestamp')


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
plt.savefig("simple_ekf_position.png", dpi=150)

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
plt.savefig("simple_ekf_velocity.png", dpi=150)

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