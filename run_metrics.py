import numpy as np
from pathlib import Path
import shutil
from argparse import ArgumentParser
from pyslam.metrics import TrajectoryMetrics
from pyslam.visualizers import TrajectoryVisualizer
import pandas as pd
from liegroups import SE3
import settings
import matplotlib.pyplot as plt


def get_metrics(ro_df, ins_df, output_path):
    gt_se3s, _ = get_poses_and_timestamps_from_df(ins_df)
    aux0_se3s, _ = get_poses_and_timestamps_from_df(ro_df)

    # making global poses from the relative poses
    gt_global_se3s = [np.identity(4)]
    for i in range(1, len(gt_se3s)):
        gt_global_se3s.append(gt_global_se3s[i - 1] @ gt_se3s[i])
    gt_global_SE3s = get_se3s_from_raw_se3s(gt_global_se3s)

    aux0_global_se3s = [np.identity(4)]
    for i in range(1, len(aux0_se3s)):
        aux0_global_se3s.append(aux0_global_se3s[i - 1] @ aux0_se3s[i])
    aux0_global_SE3s = get_se3s_from_raw_se3s(aux0_global_se3s)

    segment_lengths = [100, 200, 300, 400, 500, 600, 700, 800]

    tm_gt_aux0 = TrajectoryMetrics(gt_global_SE3s, aux0_global_SE3s)
    # print_trajectory_metrics(tm_gt_aux0, segment_lengths, data_name="RO")

    # Save metrics to text file and make plots
    output_path_for_metrics = output_path / "trajectory_metrics"
    if output_path_for_metrics.exists() and output_path_for_metrics.is_dir():
        shutil.rmtree(output_path_for_metrics)
    output_path_for_metrics.mkdir(parents=True)

    save_trajectory_metrics_to_file(
        output_path_for_metrics, {"RO": tm_gt_aux0}, segment_lengths)

    visualiser = TrajectoryVisualizer(
        {"RO": tm_gt_aux0})
    visualiser.plot_segment_errors(figsize=(10, 4), segs=segment_lengths, legend_fontsize=8,
                                   outfile="%s%s" % (output_path_for_metrics, "/segment_errors.pdf"))
    visualiser.plot_topdown(which_plane='yx',  # this was yx, a custom flip to conform to MRG convention, instead of xy
                            outfile="%s%s" % (output_path_for_metrics, "/topdown.pdf"), figsize=(6, 6))


def save_trajectory_metrics_to_file(output_path, tm_gt_est_dict, segment_lengths):
    print("Calculating trajectory metrics to save to file...")
    results_file = output_path / "trajectory_metrics.txt"
    with open(results_file, "w") as text_file:
        for data_name, tm_gt_est in tm_gt_est_dict.items():
            print(f"{data_name} metrics:", file=text_file)
            print(
                f"Segment error - lengths (m), translation (m), rotation (deg) \n {tm_gt_est.segment_errors(segment_lengths, rot_unit='deg')[1]}",
                file=text_file)
            print(
                f"Mean segment error: translation (m), rotation (deg) \n {np.mean(tm_gt_est.segment_errors(segment_lengths, rot_unit='deg')[1], axis=0)[1:]} \n",
                file=text_file)


def print_trajectory_metrics(tm_gt_est, segment_lengths, data_name="this"):
    print("\nTrajectory Metrics for", data_name, "set:")
    print("average segment_error:", np.mean(tm_gt_est.segment_errors(
        segment_lengths, rot_unit='deg')[1], axis=0)[1:])
    print("mean_err:", tm_gt_est.mean_err(rot_unit='deg'))
    print("rms_err:", tm_gt_est.rms_err(rot_unit='deg'))


def get_poses_and_timestamps_from_df(df):
    x_vals = df.x
    y_vals = df.y
    th_vals = df.yaw
    timestamps = df.timestamp

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
    return se3s, timestamps


def get_se3s_from_raw_se3s(raw_se3s):
    """
    Transform from raw se3 matrices into fancier SE3 type
    """
    se3s = []
    for pose in raw_se3s:
        se3s.append(SE3.from_matrix(np.asarray(pose)))
    return se3s


def plot_x_y_yaw(output_file, ro_df, ins_df):
    _, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 10))

    ax = axes[0]
    ax.plot(ro_df.x, ',',
            color=settings.colours.ro, label="RO")
    ax.plot(ins_df.x, ',',
            color=settings.colours.ins, label="INS")
    ax.set_title("Title")
    ax.set_xlabel("x label")
    ax.set_ylabel("y label")
    ax.legend()
    ax.grid()

    ax = axes[1]
    ax.plot(ro_df.y, ',',
            color=settings.colours.ro, label="RO")
    ax.plot(ins_df.y, ',',
            color=settings.colours.ins, label="INS")
    ax.set_xlabel("x label")
    ax.set_ylabel("y label")
    ax.legend()
    ax.grid()

    ax = axes[2]
    ax.plot(ro_df.yaw, ',',
            color=settings.colours.ro, label="RO")
    ax.plot(ins_df.yaw, ',',
            color=settings.colours.ins, label="INS")
    ax.set_xlabel("x label")
    ax.set_ylabel("y label")
    ax.legend()
    ax.grid()

    plt.savefig(output_file)
    plt.close()

    print("Plots generated and written to:", output_file)


def parse_arguments(args):
    parser = ArgumentParser(description="Plot INS vs RO outputs.")
    parser.add_argument('--ro_csv', default="", type=Path,
                        help="RO relative pose CSV")
    parser.add_argument('--ins_csv', default="", type=Path,
                        help="INS relative pose CSV")
    parser.add_argument('--output_dir', default="", type=Path,
                        help="Output directory to store figures")
    parser.add_argument('--overwrite', default=False, action="store_true",
                        help="Overwrite exported plots if this flag is set")

    return parser.parse_args(args)


def main(arg_list=None):
    args = parse_arguments(arg_list)

    output_path = args.output_dir / "tmp-plots"
    output_path.mkdir(parents=True, exist_ok=args.overwrite)

    ro_df = pd.read_csv(args.ro_csv)
    ins_df = pd.read_csv(args.ins_csv)

    plot_x_y_yaw(f"{output_path}/x_y_yaw.pdf", ro_df, ins_df)
    get_metrics(ro_df, ins_df, output_path)


if __name__ == '__main__':
    main()  # pragma: no cover