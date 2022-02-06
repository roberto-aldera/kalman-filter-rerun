import pdb
import numpy as np
import sys
import math
import pandas as pd
from pathlib import Path
import settings
import matplotlib.pyplot as plt


def get_interpolated_ins(ins_df, ro_df):
    ins_se3s = get_poses_from_df(ins_df)
    ins_pose = np.identity(4)
    ro_idx = 0

    accumulated_ins_se3s = []
    ins_se3s_timestamps = []

    for i in range(len(ins_df)):
        # Check input is within suitable range
        if ins_df.timestamp.iloc[i] < ro_df.timestamp.min():
            print(
                "Input query time must be greater than earliest RO timestamp, skipping frame:", i)
        elif ins_df.timestamp.iloc[i] > ro_df.timestamp.max():
            print(
                "Input query time must be less than latest RO timestamp, skipping frame:", i)
        else:
            # Check if INS timestamp is less than the next RO timestamp
            if ins_df.timestamp.iloc[i] < ro_df.timestamp.iloc[ro_idx]:
                # If it's less, add it to the accumulating pose we're building
                ins_pose = ins_pose @ ins_se3s[i]
            else:
                # If it's more, interpolate it, and add that partial pose - then write out that pose
                time_delta = ro_df.timestamp.iloc[ro_idx] - \
                    ins_df.timestamp.iloc[i-1]
                assert(time_delta > 0)

                # Get value of INS signal interpolated between this lower bound and the next index (upper bound)
                interpolation_val = 1 - time_delta / \
                    (ins_df.timestamp.iloc[i] - ins_df.timestamp.iloc[i-1])

                ins_x = get_interpolated_instance(
                    ins_df.x, i-1, interpolation_val)
                ins_y = get_interpolated_instance(
                    ins_df.y, i-1, interpolation_val)
                ins_yaw = get_interpolated_instance(
                    ins_df.yaw, i-1, interpolation_val)

                partial_ins_pose = np.identity(4)
                partial_ins_pose[0, 0] = np.cos(ins_yaw)
                partial_ins_pose[0, 1] = -np.sin(ins_yaw)
                partial_ins_pose[1, 0] = np.sin(ins_yaw)
                partial_ins_pose[1, 1] = np.cos(ins_yaw)
                partial_ins_pose[0, 3] = ins_x
                partial_ins_pose[1, 3] = ins_y

                ins_pose = ins_pose @ partial_ins_pose

                accumulated_ins_se3s.append(ins_pose)
                ins_se3s_timestamps.append(ro_df.timestamp.iloc[ro_idx])
                ins_pose = np.identity(4)  # reset pose for next accumulation
                ro_idx += 1

    return pd.DataFrame({"timestamp": [timestamp for timestamp in ins_se3s_timestamps],
                         "x": [se3[0, 3] for se3 in accumulated_ins_se3s],
                         "y": [se3[1, 3] for se3 in accumulated_ins_se3s],
                         "yaw": [math.atan2(se3[1, 0], se3[0, 0]) for se3 in accumulated_ins_se3s]})


def get_poses_from_df(df):
    x_vals = df.x
    y_vals = df.y
    th_vals = df.yaw

    se3s = []
    for i in range(len(df.index)):
        th = th_vals[i]
        pose = np.identity(4)
        pose[0, 0] = np.cos(th)
        pose[0, 1] = -np.sin(th)
        pose[1, 0] = np.sin(th)
        pose[1, 1] = np.cos(th)
        pose[0, 3] = x_vals[i]
        pose[1, 3] = y_vals[i]
        se3s.append(pose)
    return se3s


def get_interpolated_instance(quantity, lower_bound_idx, interpolation_val):
    return quantity[lower_bound_idx] + interpolation_val*(
        quantity[lower_bound_idx+1] - quantity[lower_bound_idx])


def apply_smoothing_filter(df):
    import scipy.signal as signal
    N = 4    # Filter order
    Wn = 0.15  # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    df.x = signal.filtfilt(B, A, df.x)

    # Y requires higher frequency cut-off to preserve signal
    N = 4    # Filter order
    Wn = 0.2  # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    df.y = signal.filtfilt(B, A, df.y)

    # Yaw requires higher frequency cut-off to preserve signal
    N = 4    # Filter order
    Wn = 0.4  # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    df.yaw = signal.filtfilt(B, A, df.yaw)
    return df


def plot_show_smoothing_effect(output_file, df_ro, df_ins):
    # Using this to experiment with smoothing (Butterworth) filter parameters
    import scipy.signal as signal

    # First, design the Buterworth filter
    N = 4    # Filter order
    Wn = 0.2  # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    smooth_data = signal.filtfilt(B, A, df_ins.y)

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))

    ax.plot(df_ro.y, '-', color=settings.colours.ro, label="RO")
    ax.plot(df_ins.y, '-', color=settings.colours.ins, label="INS")
    ax.plot(smooth_data, color="tab:blue", label="Smoothed INS")
    ax.set_xlabel("xtitle")
    ax.set_ylabel("ytitle")
    ax.legend()
    ax.grid()

    plt.savefig(output_file)
    plt.close()

    print("Plots generated and written to:", output_file)


def main():
    print("Running script...")
    output_dir = Path("/Users/roberto/data/kalman-filter-rerun")

    ro_csv = f"{output_dir}/ro_poses.csv"
    ins_csv = f"{output_dir}/ins_poses.csv"
    df_ro = pd.read_csv(ro_csv)
    df_ins = pd.read_csv(ins_csv)

    df_ro = df_ro[:3000]
    df_ins = df_ins[:40000]

    df_interpolated_ins = get_interpolated_ins(df_ins, df_ro)

    plot_show_smoothing_effect(
        f"{output_dir}/with_smoothing.pdf", df_ro[:600], df_interpolated_ins[:600])

    df_smoothed_ins = apply_smoothing_filter(df_interpolated_ins)
    df_smoothed_ins.to_csv(
        f"/Users/roberto/data/kalman-filter-rerun/processed_ins.csv", index=False)


if __name__ == "__main__":
    main()
