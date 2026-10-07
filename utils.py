import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#from helpers import *
import numpy as np
import pandas as pd
import json
from ultralytics import YOLO
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rotation



def get_paths(DIRECTORY_PATH):
    """Fetches the paths for the simulation data and images

    Args:
        DIRECTORY_PATH (str): Path to the folder where data is stored.

    Returns:
        DRONE_IMU_PATH (str): Path to the csv file containing drone and ship inertial data acquired in airsim.
        SHIP_IMU_PATH (str): Path to the csv file containing ship position data and image timestamps acquired from Unreal Engine blueprint.
        IMAGES_PATH (str): Path to the subdirectory containing the images acquired from the Unreal Engine blueprint.
        TRAJECTORY_SETTINGS_PATH (str): Path to the json file containing trajectory settings.

    Raises:
        FileNotFoundError: If any of the required files or directories are not found in the specified directory.
    """

    DRONE_IMU_PATH = os.path.join(DIRECTORY_PATH, 'drone_imu.csv')
    SHIP_IMU_PATH = os.path.join(DIRECTORY_PATH, 'capture_log.csv')
    IMAGES_PATH = os.path.join(DIRECTORY_PATH, 'images')
    TRAJECTORY_SETTINGS_PATH = os.path.join(DIRECTORY_PATH, 'trajectory_settings.json')

    if not os.path.isfile(DRONE_IMU_PATH):
        raise FileNotFoundError(f"Drone IMU file not found at: {DRONE_IMU_PATH}")
        
    if not os.path.isfile(SHIP_IMU_PATH):
        raise FileNotFoundError(f"Ship IMU capture log not found at: {SHIP_IMU_PATH}")
        
    if not os.path.isdir(IMAGES_PATH):
        raise FileNotFoundError(f"Images directory not found at: {IMAGES_PATH}")
    
    if not os.path.isfile(TRAJECTORY_SETTINGS_PATH):
        raise FileNotFoundError(f"Trajectory settings file not found at: {TRAJECTORY_SETTINGS_PATH}")

    return DRONE_IMU_PATH,SHIP_IMU_PATH,IMAGES_PATH, TRAJECTORY_SETTINGS_PATH


def format_data(capture_log_path, drone_imu_path, trajectory_settings):
    """Process the ship and drone data to ensure units and frames of reference are consistent
    
        Args:
            capture_log_path (str): Path to the csv file where ship position data and image timestamps are stored.
            drone_imu_path (str): Path to the csv file where drone and ship intertial data is stored.
            trajeotry_settings : Loaded json file where trajectory settings are stored.
    
        Returns:
            df_ship (pandas.DataFrame): Dataframe containing the ship pose data and the image frame numbers timestamps
            df_drone (pandas.DataFrame): Dataframe containing the drone and ship's pose and inertial data. 

        Raises:
            FileNotFoundError: If any of the required files are not found in the specified directory.
            ValueError: IF any of the files are empty or corrupted.
        """

    # Load ship data
    try:
        df_ship = pd.read_csv(capture_log_path, skiprows=1, names=['frame', 'timestamp', 'xyz_string', 'pyr_string'])
    except FileNotFoundError:
        raise FileNotFoundError(f"Ship capture log not found at: {capture_log_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Ship capture log is empty: {capture_log_path}")
    except pd.errors.ParserError:
        raise ValueError(f"Ship capture log is corrupted or not a valid CSV: {capture_log_path}")    

    # Load drone data
    try:
        df_drone = pd.read_csv(drone_imu_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Drone IMU data not found at: {drone_imu_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Drone IMU data is empty: {drone_imu_path}")

    # Apply the starting position offset to the logged drone position
    # 
    # Airsim takes the "settings.json" multirotor starting position as the frame origin for the logged positions.
    px = trajectory_settings['starting_position']['x']
    py = trajectory_settings['starting_position']['y']
    pz = trajectory_settings['starting_position']['z']
    
    df_drone['pos_x'] += px
    df_drone['pos_y'] += py
    df_drone['pos_z'] += pz

    # Extract the ship's position and orientation from the strings logged via the Unreal Engine blueprint.
    #
    # Convert the data logged as strings to numerical values.
    df_ship[['x', 'y', 'z']] = df_ship['xyz_string'].str.extract(r'X=([-\d.]+)\s+Y=([-\d.]+)\s+Z=([-\d.]+)')
    df_ship[['pitch', 'yaw', 'roll']] = df_ship['pyr_string'].str.extract(r'P=([-\d.]+)\s+Y=([-\d.]+)\s+R=([-\d.]+)')
    df_ship = df_ship.drop(columns=['xyz_string', 'pyr_string'])

    cols_to_convert = ['timestamp','x', 'y', 'z', 'pitch', 'yaw', 'roll']
    df_ship[cols_to_convert] = df_ship[cols_to_convert].apply(pd.to_numeric)

    # Convert the position data from the Unreal Engine (NEU, left-handed) frame to the standard (NED,right-handed)
    # by inverting the z-axis and switching to meters (Unreal Engine units are centimeters).
    df_ship['x'] = df_ship['x'] / 100
    df_ship['y'] = df_ship['y'] / 100
    df_ship['z'] = -df_ship['z'] / 100

    # The "Rotator" object underlying the "Get World Rotation" function match NED convention if the z-axis is inverted
    # It uses left-handed convetntion for rotations around the z-axis and right-handed convention for rotations around the x-axis and y-axis
    #
    #  The only necessary transformation is from degrees to radians.
    df_ship['roll'] = np.radians(df_ship['roll'])
    df_ship['yaw'] = np.radians(df_ship['yaw'])
    df_ship['pitch'] = np.radians(df_ship['pitch'])

    return df_ship, df_drone


def timestamp_alignment(df_drone, df_ship):
    """Align the timestamp from the dataframes
       The airsim python script and the unreal engine blueprint use the same physics clock, but start counting from different moments.
       The timestamps are aligned by matching the point in the trajectory where the ships pitch is at its max.
       The timestamps are normalized to the start of the trajectory.
       The ship dataframe is filtered to discard images and data from before the trajectory starts and after the trajectory ends.
        
        Args:
            df_ship (pandas.DataFrame): Dataframe containing the ship pose data and the image frame numbers timestamps
            df_drone (pandas.DataFrame): Dataframe containing the drone and ship's pose and inertial data.
    
        Returns:
            filtered_df_ship (pandas.DataFrame): Filtered dataframe containing the ship pose data and the image frame numbers timestamps
            df_drone (pandas.DataFrame): Dataframe containing the drone and ship's pose and inertial data. 
        """

    # Extract the pitch data from the ship blueprint    
    bp_pitch = df_ship['pitch']

    # The ship orientation is stored as quaternions in the drone dataframe
    # so the pitch has to be calculated
    quats = np.vstack([df_drone['ship_q_x'], df_drone['ship_q_y'], df_drone['ship_q_z'], df_drone['ship_q_w']]).T
    eulers = Rotation.from_quat(quats).as_euler('ZYX', degrees=False)
    py_pitch = pd.Series(eulers[:, 1])

    # Find the index where the pitch is at its max in each dataframe
    peak_idx_blueprint = bp_pitch.idxmax()
    peak_idx_python = py_pitch.idxmax()

    # Calculate the offset between the timestamps
    peak_time_blueprint = df_ship['timestamp'].iloc[peak_idx_blueprint] 
    peak_time_python = df_drone['airsim_timestamp'].iloc[peak_idx_python] 
    clock_offset = peak_time_python - peak_time_blueprint

    # Add the offset so the timestamps align
    df_ship['timestamp'] = df_ship['timestamp'] + clock_offset

    # Extract trajectory bounds using the highly accurate physics clock
    trajectory_start_time = df_drone['airsim_timestamp'].min()
    trajectory_end_time = df_drone['airsim_timestamp'].max()

    # Filter the ship data to bounds
    filtered_df_ship = df_ship[(df_ship['timestamp'] >= trajectory_start_time) & (df_ship['timestamp'] <= trajectory_end_time)].copy()

    # Normalize both dataframes to start at t=0 so the rest of simple_EKF works unchanged.
    # We overwrite df_drone['timestamp'] with the synced airsim_timestamp to guarantee precision.
    filtered_df_ship['timestamp'] = filtered_df_ship['timestamp'] - trajectory_start_time
    df_drone['timestamp'] = df_drone['airsim_timestamp'] - trajectory_start_time

    return df_drone, filtered_df_ship


def combine_dataframes(df_ship, df_drone):
    """Combine the dataframes so the EKF loop can iterate over a single dataframe. 
       The rows are diffentiated using the 'entity' column to distinguish if they come from the UE blueprint log or from the airsim script log
    
        Args:
            df_ship (pandas.DataFrame): Dataframe containing the ship pose data and the image frame numbers timestamps
            df_drone (pandas.DataFrame): Dataframe containing the drone and ship's pose and inertial data.
    
        Returns:
            df_combined (pandas.DataFrame): Combined dataframe containing all the data 
        """
    
    df_ship['entity'] = 'ship'
    df_drone['entity'] = 'drone'

    # Combine the dataframes and sort by timestamps
    df_combined = pd.concat([df_ship, df_drone], ignore_index=True)
    df_combined = df_combined.sort_values(by='timestamp').reset_index(drop=True)

    return df_combined


def crop_timestamps(df_combined, df_drone, starting_timestamp):
    """Combine the dataframes so the EKF loop can iterate over a single dataframe. 
    
        Args:
            df_combined (pandas.DataFrame): Combined dataframe containing all the data 
            df_drone (pandas.DataFrame): Dataframe containing the drone and ship's pose and inertial data.
            df_combined (pandas.DataFrame): Combined dataframe containing all the data 
    
        Returns:
            df_combined (pandas.DataFrame): Combined dataframe containing all the data
            df_drone (pandas.DataFrame): Dataframe containing the drone and ship's pose and inertial data.
        """
        
    df_combined = df_combined[df_combined['timestamp'] > starting_timestamp].reset_index(drop=True)    
    df_drone = df_drone[df_drone['timestamp'] >starting_timestamp].reset_index(drop=True)    

    df_drone['timestamp'] = df_drone['timestamp'] - starting_timestamp
    df_combined['timestamp'] = df_combined['timestamp'] - starting_timestamp

    return df_drone, df_combined


def match_nearest(df_source, df_target, source_col='timestamp', target_col='timestamp'):
    """
    Matches two dataframes' rows by comparing two columns. 
    For each row in in df_target, which has an element target_col,
    choses the row in df_source, where the element source_col is nearest to target_col.

    Args:
        df_source (pandas.DataFrame): dataframe where to look for the nearest element
        df_target (pandas.DataFrame): dataframe with the rows to be matched
        source_col : column of df_source to be compared
        target_col : column of df_target to be compared

    Returns:
        (pandas.DataFrame): with the selected rows from df_source already in order    

    """
    matched_rows = []
    for ts in df_target[target_col]:
        idx = (df_source[source_col] - ts).abs().idxmin()
        matched_rows.append(df_source.loc[idx])
    return pd.DataFrame(matched_rows).reset_index(drop=True)


def plot_ekf_comparison(df_ekf, df_gt_matched, title, labels, ekf_cols, gt_cols, var_cols, filename, gt_color="green", ekf_color="red", show_plot=False):
    """
    Generates and saves a 3-subplot figure comparing EKF estimates to Ground Truth as a PDF vector graphic.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title, fontsize=14)

    # Relative time from 0
    t = df_ekf["timestamp"] - df_ekf["timestamp"].iloc[0]  

    for ax, label, ekf_col, gt_col, var_col in zip(axes, labels, ekf_cols, gt_cols, var_cols):
        # Ground truth
        ax.plot(t, df_gt_matched[gt_col], 
                color=gt_color, linewidth=2, label="Ground Truth")

        # EKF estimate
        ax.plot(t, df_ekf[ekf_col], 
                color=ekf_color, linestyle="--", linewidth=1.5, label="EKF Estimate")

        # Uncertainty band (±2 standard deviations = 95% confidence)
        std = np.sqrt(df_ekf[var_col])
        ax.fill_between(t,
                        df_ekf[ekf_col] - 2 * std,
                        df_ekf[ekf_col] + 2 * std,
                        color=ekf_color, alpha=0.15, label="±2σ uncertainty")

        ax.set_ylabel(label)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    
    # Enforce .pdf extension for vector graphics
    if not filename.lower().endswith('.pdf'):
        filename = filename.rsplit('.', 1)[0] + '.pdf'
        
    # Save as PDF vector format
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def get_position_error(df_ekf, df_gt):
    """
    Calculates the position error during the trajectory 

    Args:
        df_ekf (pandas.DataFrame) : dataframe with the data estimated by the EKF during the trajectory
        df_gt (pandas.DataFrame) : dataframe with the ground truth data

    Returns:
        p_error (np.array) : array with the position error during the trajectory
    """

    p_error = np.sqrt(
    (df_ekf["ekf_px"] - df_gt["pos_x"])**2 +
    (df_ekf["ekf_py"] - df_gt["pos_y"])**2 +
    (df_ekf["ekf_pz"] - df_gt["pos_z"])**2)

    return p_error


def plot_error_vs_distance(df_ekf, df_gt_matched, filename="error_vs_distance.pdf"):
    """
    Calculates the distance from the drone to the camera center 
    and plots the EKF position error with respect to that distance.
    """
    # 1. Calculate Position Error
    p_error = get_position_error(df_ekf, df_gt_matched)
    
    # 2. Define the Camera's Local Ship Coordinates
    # Local position of the camera center in the ship frame {S}
    p_cam_ship = np.array([0.0, 12.0, -4.0]) 

    # 3. Transform the Camera to the EKF World Frame {W}
    # Extract ship quaternions for the entire trajectory
    quats = df_gt_matched[['ship_q_x', 'ship_q_y', 'ship_q_z', 'ship_q_w']].values
    rot_ship = Rotation.from_quat(quats)
    
    # Apply rotation: R_WS * p_cam_ship
    p_cam_rotated = rot_ship.apply(p_cam_ship)
    
    # Apply translation: 
    # The EKF World Frame is ship-relative horizontally (X=0, Y=0), 
    # but uses absolute Heave (Z = ship_pos_z).
    ship_translations = np.column_stack((
        np.zeros(len(df_gt_matched)), 
        np.zeros(len(df_gt_matched)), 
        df_gt_matched['ship_pos_z'].values
    ))
    
    # Final camera position in {W} for every row
    p_cam_world = p_cam_rotated + ship_translations

    # 4. Extract EKF Drone Position
    p_drone_ekf = df_ekf[['ekf_px', 'ekf_py', 'ekf_pz']].values

    # 5. Calculate Euclidean Distance between Drone and Camera
    distances = np.linalg.norm(p_drone_ekf - p_cam_world, axis=1)

    # 6. Plotting
    plt.figure(figsize=(10, 6))
    
    # Using a scatter plot because distance doesn't decrease perfectly linearly over time
    plt.scatter(distances, p_error, c='blue', alpha=1.0, edgecolors='k', s=2)
    
    plt.title("EKF Position Error vs. Distance to Camera")
    plt.xlabel("Distance from Camera Center (m)")
    plt.ylabel("Position Error (m)")
    plt.grid(True, alpha=0.3)
    
    # Invert X-axis if you want it to visually represent the drone approaching the ship 
    # (moving from right to left on the graph)
    plt.gca().invert_xaxis()
    
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.show()

    return p_error, distances


def plot_multiple_errors_vs_distance(ekf_dict, filename=r"results\compare_errors.pdf"):
    """
    Plots position error vs distance for multiple EKF approaches using a connected line plot.
    
    Args:
        ekf_dict: Dictionary formatted as {"Approach Name": (df_ekf, df_gt)}
        filename: Path to save the output plot
    """
    plt.figure(figsize=(10, 6))
    
    # Define distinct colors for up to 5 approaches
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (label_name, (df_ekf, df_gt)) in enumerate(ekf_dict.items()):
        # 1. Position Error
        p_error = np.sqrt(
            (df_ekf["ekf_px"] - df_gt["pos_x"])**2 +
            (df_ekf["ekf_py"] - df_gt["pos_y"])**2 +
            (df_ekf["ekf_pz"] - df_gt["pos_z"])**2
        )
        
        # 2. Camera Position in World Frame
        p_cam_ship = np.array([0.0, 12.0, -4.0]) 
        quats = df_gt[['ship_q_x', 'ship_q_y', 'ship_q_z', 'ship_q_w']].values
        rot_ship = Rotation.from_quat(quats)
        p_cam_rotated = rot_ship.apply(p_cam_ship)
        
        ship_translations = np.column_stack((
            np.zeros(len(df_gt)), 
            np.zeros(len(df_gt)), 
            df_gt['ship_pos_z'].values
        ))
        p_cam_world = p_cam_rotated + ship_translations
        
        # 3. Euclidean Distance
        p_drone_gt = df_gt[['pos_x', 'pos_y', 'pos_z']].values
        distances = np.linalg.norm(p_drone_gt - p_cam_world, axis=1)
        
        # 4. Plot onto the shared axis
        # Using plt.plot to connect points chronologically
        plt.plot(distances, p_error, color=colors[i % len(colors)], 
                 marker='o', markersize=2, linestyle='-', linewidth=1.0, 
                 alpha=0.7, label=label_name)

    # 5. Styling
    plt.title("EKF Position Error vs. Distance to Camera")
    plt.xlabel("Distance from Camera Center (m)")
    plt.ylabel("Position Error (m)")
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis() # Invert X-axis so distance decreases left-to-right
    
    # Customize the legend for Line2D objects to make them readable
    leg = plt.legend(loc="upper left")
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)
        handle.set_linewidth(2.0)    # Make the line thicker in the legend
        handle.set_markersize(6.0)   # Make the dot larger in the legend

    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.show()