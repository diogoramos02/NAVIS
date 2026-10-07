import subprocess
import sys
import os
import pandas as pd
from utils import *


# List the scripts in the order you want them to run
scripts = [
    "Triangulation_MRU.py",
    "Cameras_with_MRU.py",
    "IMU_MRU.py",
    "AHRS_EKF.py"
]

# Create the results directory if it does not already exist
os.makedirs("results", exist_ok=True)

for script in scripts:
    print(f"--- Starting {script} ---")
    try:
        # sys.executable ensures the scripts run in the exact same Python environment
        # check=True ensures that if a script crashes, the main script stops and raises an error
        subprocess.run([sys.executable, script], check=True)
        print(f"--- Finished {script} ---\n")
        
    except subprocess.CalledProcessError as e:
        print(f"Execution stopped. {script} failed with exit code {e.returncode}.")
        break

ekf_data = {
        "Baseline with MRU": (pd.read_csv(r"results\baseline_mru_ekf.csv"), pd.read_csv(r"results\baseline_mru_gt.csv")),
        "Cameras with MRU":    (pd.read_csv(r"results\cameras_mru_ekf.csv"), pd.read_csv(r"results\cameras_mru_gt.csv")),
        "Drone IMU with MRU":   (pd.read_csv(r"results\imu_mru_ekf.csv"), pd.read_csv(r"results\imu_mru_gt.csv")),
        "AHRS":   (pd.read_csv(r"results\ahrs_ekf.csv"), pd.read_csv(r"results\ahrs_gt.csv"))
    }

plot_multiple_errors_vs_distance(ekf_data, filename=r"results\error_comparison.pdf")