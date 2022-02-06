import pdb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import settings
from pathlib import Path


def plot_single_parameter(output_file, df_ro, df_ins):
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))

    ax.plot(df_ro.x, '-', color=settings.colours.ro, label="RO")
    ax.plot(df_ins.x, '-', color=settings.colours.ins, label="INS")
    ax.set_xlabel("xtitle")
    ax.set_ylabel("ytitle")
    ax.legend()
    ax.grid()

    plt.savefig(output_file)
    plt.close()

    print("Plots generated and written to:", output_file)


def plot_x_y_yaw(output_file, df_ro, df_ins):

    ro_time_diffs = np.diff(df_ro.timestamp) * 1e-6
    ins_time_diffs = np.diff(df_ins.timestamp) * 1e-6

    _, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 10))

    ax = axes[0]
    # ax.plot(df_ro.x[1:]/ro_time_diffs, ',', color=colours.ro, label="RO_x")
    # ax.plot(df_ins.x[1:]/ins_time_diffs, ',', color=colours.ins, label="INS_x")
    ax.plot(df_ro.x, ',', color=settings.colours.ro, label="RO_x")
    ax.plot(df_ins.x, ',', color=settings.colours.ins, label="INS_x")
    ax.set_xlabel("xtitle")
    ax.set_ylabel("ytitle")
    ax.legend()
    ax.grid()

    ax = axes[1]
    ax.plot(df_ro.y, ',', color=settings.colours.ro, label="RO_y")
    ax.plot(df_ins.y, ',', color=settings.colours.ins, label="INS_y")
    ax.set_xlabel("xtitle")
    ax.set_ylabel("ytitle")
    ax.legend()
    ax.grid()

    ax = axes[2]
    ax.plot(df_ro.yaw, ',', color=settings.colours.ro, label="RO_yaw")
    ax.plot(df_ins.yaw, ',', color=settings.colours.ins, label="INS_yaw")
    ax.set_xlabel("xtitle")
    ax.set_ylabel("ytitle")
    ax.legend()
    ax.grid()

    plt.savefig(output_file)
    plt.close()

    print("Plots generated and written to:", output_file)


def plot_labels(df_labels, output_file):
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))

    ax.plot(df_labels, '.-', color="tab:red", label="labels")
    ax.set_xlabel("xtitle")
    ax.set_ylabel("ytitle")
    ax.legend()
    ax.grid()

    plt.savefig(output_file)
    plt.close()

    print("Plots generated and written to:", output_file)


def plot_ro_with_labels(df_ro, df_labels, output_file):
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))

    for i in range(len(df_labels)):
        if df_labels.bad_ro_state.iloc[i] == 1.0:
            ax.plot(i, df_ro.x.iloc[i], ',', color="tab:red")
        else:
            ax.plot(i, df_ro.x.iloc[i], ',', color="tab:green")
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
    ins_csv = f"{output_dir}/processed_ins.csv"
    df_ins = pd.read_csv(ins_csv)

    # df_ro = df_ro[:1000]
    # df_ins = df_ins[:1000]

    plot_single_parameter(f"{output_dir}/single_parameter.pdf", df_ro, df_ins)
    plot_x_y_yaw(f"{output_dir}/fig.pdf", df_ro, df_ins)

    df_labels = pd.read_csv(f"{output_dir}/svm-labels.csv")
    plot_labels(df_labels, f"{output_dir}/labels.pdf")
    plot_ro_with_labels(df_ro, df_labels, f"{output_dir}/ro-with-labels.pdf")


if __name__ == "__main__":
    main()
