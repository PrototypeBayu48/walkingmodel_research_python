import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt

leadfoot = 1
forceactive = 1

# ==========================================================
# 1. Configuration & Smart Loading
# ==========================================================
marker_file = os.getcwd() + "/experimental_data/fukuchi_2018/WBDSascii/WBDS01walkT01mkr.txt"
force_file = os.getcwd() + "/experimental_data/fukuchi_2018/WBDSascii/WBDS01walkT01grf.txt" 

mass = 74.3
g = 9.81
treadmill_speed = 0.49

print("Loading continuous 150Hz Kinematics and 300Hz Kinetics...")
df_mkr = pd.read_csv(marker_file, sep='\t')
df_grf = pd.read_csv(force_file, sep='\t')

t_mkr = df_mkr.iloc[:, 0].values  
if t_mkr[1] - t_mkr[0] >= 1.0: t_mkr = (t_mkr - t_mkr[0]) / 150.0

t_grf = df_grf.iloc[:, 0].values
if t_grf[1] - t_grf[0] >= 1.0: t_grf = (t_grf - t_grf[0]) / 300.0

l_cop_prog_raw = df_grf['COPx1'].values / 1000.0
r_cop_prog_raw = df_grf['COPx2'].values / 1000.0

# ==========================================================
# 2. Construct Center of Mass (150 Hz)
# ==========================================================
prog_cols = ['R.ASISX', 'L.ASISX', 'R.PSISX', 'L.PSISX']
lat_cols  = ['R.ASISZ', 'L.ASISZ', 'R.PSISZ', 'L.PSISZ']

com_progression = df_mkr[prog_cols].mean(axis=1).values / 1000.0  
com_lateral = df_mkr[lat_cols].mean(axis=1).values / 1000.0      

com_progression_unrolled = com_progression + (treadmill_speed * t_mkr)

# ==========================================================
# 3. Ground Truth Stance Detection (Using Vertical Force)
# ==========================================================
# In this specific dataset, Y is the vertical axis for the force plates
l_fy_raw = df_grf['Fy1'].values
r_fy_raw = df_grf['Fy2'].values

# ==========================================================
# 4. Filter the Sensor Noise (Biomechanical Butterworth)
# ==========================================================
l_cop_lat_raw = df_grf['COPz1'].values / 1000.0
r_cop_lat_raw = df_grf['COPz2'].values / 1000.0

def filter_continuous(signal, cutoff=10.0, fs=300.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, signal)

l_fy = filter_continuous(l_fy_raw)
r_fy = filter_continuous(r_fy_raw)

if forceactive == 1:
    force_threshold = 0.07*mass*g  
    l_active = l_fy > force_threshold
    r_active = r_fy > force_threshold
else:
    l_active = (np.abs(l_cop_lat_raw) > 0.0001) | (np.abs(l_cop_lat_raw) > 0.0001)
    r_active = (np.abs(r_cop_lat_raw) > 0.0001) | (np.abs(r_cop_lat_raw) > 0.0001)

def smooth_stance_phases(cop_array, active_mask, cutoff=10.0, fs=300.0, order=4):
    smoothed = np.copy(cop_array)
    edges = np.diff(active_mask.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    
    if active_mask[0]: starts = np.insert(starts, 0, 0)
    if active_mask[-1]: ends = np.append(ends, len(cop_array))
        
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    
    for s, e in zip(starts, ends):
        if e - s > 30:  
            smoothed[s:e] = filtfilt(b, a, cop_array[s:e])
    return smoothed

print("Applying 10Hz Butterworth Filter to CoP data...")
# Overwrite the raw noisy data with the smoothed data
if forceactive == 1:
    l_cop_prog_raw = smooth_stance_phases(l_cop_prog_raw, l_active)
    l_cop_lat_raw  = smooth_stance_phases(l_cop_lat_raw, l_active)
    r_cop_prog_raw = smooth_stance_phases(r_cop_prog_raw, r_active)
    r_cop_lat_raw  = smooth_stance_phases(r_cop_lat_raw, r_active)
else:
    l_cop_prog_raw = smooth_stance_phases(l_cop_prog_raw, np.abs(l_cop_prog_raw) > 0.0001)
    l_cop_lat_raw  = smooth_stance_phases(l_cop_lat_raw, np.abs(l_cop_lat_raw) > 0.0001)
    r_cop_prog_raw = smooth_stance_phases(r_cop_prog_raw, np.abs(r_cop_prog_raw) > 0.0001)
    r_cop_lat_raw  = smooth_stance_phases(r_cop_lat_raw, np.abs(r_cop_lat_raw) > 0.0001)

# ==========================================================
# 5. Extract and Unroll Smoothed CoP
# ==========================================================
num_force_frames = len(t_grf)
cop_lateral_global = np.full(num_force_frames, np.nan)
cop_progression_global = np.full(num_force_frames, np.nan)

cop_l_lateral_global = np.full(num_force_frames, np.nan)
cop_l_progression_global = np.full(num_force_frames, np.nan)
cop_r_lateral_global = np.full(num_force_frames, np.nan)
cop_r_progression_global = np.full(num_force_frames, np.nan)

if forceactive != 1:
    l_active = (np.abs(l_cop_lat_raw) > 0.0001) | (np.abs(l_cop_lat_raw) > 0.0001)
    r_active = (np.abs(r_cop_lat_raw) > 0.0001) | (np.abs(r_cop_lat_raw) > 0.0001)

def process_text_plate(cop_prog, cop_lat, t_array, active_mask):
    cop_prog_unrolled = np.copy(cop_prog)
    cop_prog_unrolled[active_mask] += (treadmill_speed * t_array[active_mask])
    return cop_prog_unrolled, cop_lat

l_p, l_l = process_text_plate(l_cop_prog_raw, l_cop_lat_raw, t_grf, l_active)
r_p, r_l = process_text_plate(r_cop_prog_raw, r_cop_lat_raw, t_grf, r_active)

# Single Support
l_only = l_active & ~r_active
r_only = r_active & ~l_active

cop_lateral_global[l_only] = l_l[l_only]
cop_progression_global[l_only] = l_p[l_only]
cop_lateral_global[r_only] = r_l[r_only]
cop_progression_global[r_only] = r_p[r_only]

cop_l_lateral_global[l_only] = l_l[l_only]
cop_l_progression_global[l_only] = l_p[l_only]
#cop_l_lateral_global[l_active] = l_l[l_active]
#cop_l_progression_global[l_active] = l_p[l_active]

cop_r_lateral_global[r_only] = r_l[r_only]
cop_r_progression_global[r_only] = r_p[r_only]
#cop_r_lateral_global[r_active] = r_l[r_active]
#cop_r_progression_global[r_active] = r_p[r_active]

# Double Support
double_supp = l_active & r_active
if leadfoot == 1:
    left_is_leading = l_p[double_supp] > r_p[double_supp]
    right_is_leading = ~left_is_leading

    ds_indices = np.where(double_supp)[0]
    left_lead_idx = ds_indices[left_is_leading]
    right_lead_idx = ds_indices[right_is_leading]

    cop_progression_global[left_lead_idx] = l_p[left_lead_idx]
    cop_lateral_global[left_lead_idx] = l_l[left_lead_idx]

    cop_progression_global[right_lead_idx] = r_p[right_lead_idx]
    cop_lateral_global[right_lead_idx] = r_l[right_lead_idx]
else:
    #cop_lateral_global[double_supp] = (l_l[double_supp] + r_l[double_supp]) / 2.0
    #cop_progression_global[double_supp] = (l_p[double_supp] + r_p[double_supp]) / 2.0

    # Calculate total vertical force during the overlap
    total_force = l_fy[double_supp] + r_fy[double_supp]

    # Dynamically weight the X and Z coordinates based on weight transfer
    cop_progression_global[double_supp] = (
        (l_p[double_supp] * l_fy[double_supp]) + 
        (r_p[double_supp] * r_fy[double_supp])
    ) / total_force

    cop_lateral_global[double_supp] = (
        (l_l[double_supp] * l_fy[double_supp]) + 
        (r_l[double_supp] * r_fy[double_supp])
    ) / total_force

valid_mask = ~np.isnan(cop_progression_global)
cop_lateral_clean = cop_lateral_global[valid_mask]
cop_progression_clean = cop_progression_global[valid_mask]

cop_l_lateral_clean = cop_l_lateral_global[~np.isnan(cop_l_lateral_global)]
cop_l_progression_clean = cop_l_progression_global[~np.isnan(cop_l_progression_global)]
cop_r_lateral_clean = cop_r_lateral_global[~np.isnan(cop_r_lateral_global)]
cop_r_progression_clean = cop_r_progression_global[~np.isnan(cop_r_progression_global)]

# ==========================================================
# 6. Generate the Figure
# ==========================================================
plt.figure()
plt.plot(com_lateral, com_progression_unrolled, color='red', linewidth=3, zorder=5, label='CoM Trajectory')
plt.plot(cop_lateral_clean, cop_progression_clean, 'k--', linewidth=1.5, label='CoP Trajectory')
plt.plot(cop_lateral_clean, cop_progression_clean, 'bo', markersize=2)
plt.plot(cop_l_lateral_clean, cop_l_progression_clean, 'C0o', markersize=2, label='Left Foot')
plt.plot(cop_r_lateral_clean, cop_r_progression_clean, 'C1o', markersize=2, label='Right Foot')
plt.xlabel('Z (m)', fontsize=12)
plt.ylabel('X (m)', fontsize=12)
plt.legend(loc='upper right')
plt.show()



