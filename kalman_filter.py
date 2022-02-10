import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import settings
from pathlib import Path

LINE_WIDTH = 0.3


def get_kalman_filter_output(df_ro, df_labels):
    def A(dt): return np.array([[0, 0, 0, dt, 0, 0],
                                [0, 0, 0, 0, dt, 0],
                                [0, 0, 0, 0, 0, dt],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 1]])

    def C(x): return np.array([[np.cos(x[2]), -np.sin(x[2]), 0, 0, 0, 0],
                               [np.sin(x[2]), np.cos(x[2]), 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0]])

    R = np.eye(6, 6) * 1  # motion model uncertainty
    Q = np.eye(3, 3) * 1  # sensor (RO) uncertainty
    E = np.zeros([6, 6])

    def Y(i): return np.array(
        [df_ro.x.iloc[i], df_ro.y.iloc[i], df_ro.yaw.iloc[i]])

    t = len(df_ro.timestamp)
    X = np.array([0, 0, 0, 0, 0, 0]).T
    X_outputs = []
    for i in range(1, t):
        oldX = np.array(X)
        y = Y(i)
        dt = (df_ro.timestamp.iloc[i] - df_ro.timestamp.iloc[i-1])*1e-6

        X = np.matmul(A(dt), X)
        E = np.matmul(np.matmul(A(dt), E), A(dt).T) + R

        if(df_labels.bad_ro_state.iloc[i] == -1.0):
            k_1 = np.matmul(E, C(oldX).T)
            k_2 = np.matmul(np.matmul(C(oldX), E), C(oldX).T) + Q
            K = np.matmul(k_1, np.linalg.pinv(k_2))
            X2 = X + np.matmul(K, y - np.matmul(C(oldX), X))
            E2 = np.matmul(np.eye(6, 6) - np.matmul(K, C(oldX)), E)

            m = X - X2
            v = np.matmul(m.T, np.matmul(np.linalg.pinv(E2 - E), m))
            if(abs(v) < 3.8):  # chi-squared check - perform update if this passes
                X = X2
                E = E2
        X_outputs.append(X)

    return pd.DataFrame({"timestamp": df_ro.timestamp[1:],
                         "x": [state[0] for state in X_outputs],
                         "y": [state[1] for state in X_outputs],
                         "yaw": [state[2] for state in X_outputs]})


def quick_plot(output_file, df_ro, df_kf):
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))

    ax.plot(df_ro.x, '-', color=settings.colours.ro,
            lw=LINE_WIDTH, label="RO_x")
    ax.plot(df_kf.x, '-', color="tab:red", lw=LINE_WIDTH, label="KF")
    ax.set_xlabel("xtitle")
    ax.set_ylabel("ytitle")
    ax.legend()
    ax.grid()

    plt.savefig(output_file)
    plt.close()

    print("Plots generated and written to:", output_file)


def check_labels_alignment(output_file, df_labels, df_ro):
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
    start_val = 950  # 2800
    num_instances = 200
    offset_guess = 0
    ax.plot(df_ro.x[start_val:start_val + num_instances], '-', color=settings.colours.ro,
            lw=LINE_WIDTH, label="RO_x")
    ax.plot(np.arange(start_val+offset_guess, start_val+num_instances, 1), df_labels.bad_ro_state[start_val:start_val + num_instances - offset_guess], 'x', color="tab:red",
            lw=LINE_WIDTH, label="label")
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
    df_ro = pd.read_csv(ro_csv)
    df_labels = pd.read_csv(f"{output_dir}/svm-labels.csv")

    # Labels from 2019 data require 2 additional elements (alignment checked in plotting function)
    buffer_labels = pd.DataFrame({"bad_ro_state": np.array([-1, -1])})
    df_labels = df_labels.append(buffer_labels, ignore_index=True)
    # check_labels_alignment(
    #     f"{output_dir}/label_alignment.pdf", df_labels, df_ro)

    df_kf = get_kalman_filter_output(df_ro, df_labels)
    df_kf.to_csv(
        f"/Users/roberto/data/kalman-filter-rerun/kalman_filter_poses.csv", index=False)
    quick_plot(f"{output_dir}/kf.pdf", df_ro, df_kf)


if __name__ == "__main__":
    main()
