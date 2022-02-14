import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
import settings
import matplotlib.pyplot as plt
import pdb

LINE_WIDTH = 1.5
MARKER_SIZE = 7

plt.rc('text', usetex=False)
plt.rc('font', family='serif')


def plot_speeds(ins_df, ro_df, kf_aux1_df, kf_aux2_df, kf_aux3_df,  output_path):
    # Kalman filtered data only starts from second frame, so need to crop the others
    ins_df = ins_df.iloc[1:].reset_index(drop=True)
    ro_df = ro_df.iloc[1:].reset_index(drop=True)

    gt_speeds, gt_timestamps = get_speeds_and_elapsed_timestamps_from_df(
        ins_df)
    aux0_speeds, aux0_timestamps = get_speeds_and_elapsed_timestamps_from_df(
        ro_df)
    aux1_speeds, aux1_timestamps = get_speeds_and_elapsed_timestamps_from_df(
        kf_aux1_df)
    aux2_speeds, aux2_timestamps = get_speeds_and_elapsed_timestamps_from_df(
        kf_aux2_df)
    aux3_speeds, aux3_timestamps = get_speeds_and_elapsed_timestamps_from_df(
        kf_aux3_df)

    # check timestamps all line up before proceeding
    assert(gt_timestamps[0] == aux0_timestamps[0])
    assert(gt_timestamps[0] == aux1_timestamps[0])
    assert(gt_timestamps[0] == aux2_timestamps[0])
    assert(gt_timestamps[0] == aux3_timestamps[0])

    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    ax.plot(gt_timestamps, gt_speeds, '+-', lw=LINE_WIDTH, ms=MARKER_SIZE+3,
            color=settings.colours.ins, label="Ground truth")
    ax.plot(aux0_timestamps, aux0_speeds, '.-', lw=LINE_WIDTH, ms=MARKER_SIZE,
            color=settings.colours.ro, label="RO")
    ax.plot(aux1_timestamps, aux1_speeds, '.-', lw=LINE_WIDTH, ms=MARKER_SIZE,
            color=settings.colours.kf_aux1, label=settings.AUX1_NAME)
    ax.plot(aux2_timestamps, aux2_speeds, '.-', lw=LINE_WIDTH, ms=MARKER_SIZE,
            color=settings.colours.kf_aux2, label=settings.AUX2_NAME)
    ax.plot(aux3_timestamps, aux3_speeds, '.-', lw=LINE_WIDTH, ms=MARKER_SIZE,
            color=settings.colours.kf_aux3, label=settings.AUX3_NAME)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    window_size = 30
    start_index = 705
    ax.set_xlim(start_index, start_index+window_size)
    ax.set_ylim(0, 25)
    ax.legend()
    ax.grid()
    plt.title("Comparison of speed estimates for configurations 2, 3 and 5")

    plt.savefig(f"{output_path}/tmp.pdf", bbox_inches='tight')
    plt.close()


def get_speeds_and_elapsed_timestamps_from_df(df):
    translations = np.linalg.norm([df.x.to_numpy(), df.y.to_numpy()], axis=0)
    time_diff = np.diff(df.timestamp)/1e6
    # time_diff = np.insert(time_diff, 0, 0.25)
    speeds = translations[1:]/time_diff
    elapsed_timestamps = np.cumsum(time_diff)

    return speeds, elapsed_timestamps


def parse_arguments(args):
    parser = ArgumentParser(description="Plot speeds.")
    parser.add_argument('--ins_csv', default=settings.INS_CSV, type=Path,
                        help="INS relative pose CSV")
    parser.add_argument('--ro_csv', default=settings.RO_CSV, type=Path,
                        help="RO relative pose CSV")
    parser.add_argument('--output_dir', default="", type=Path,
                        help="Output directory to store figures")
    parser.add_argument('--overwrite', default=False, action="store_true",
                        help="Overwrite exported plots if this flag is set")

    return parser.parse_args(args)


def main(arg_list=None):
    args = parse_arguments(arg_list)

    output_path = args.output_dir / "speed-plots"
    output_path.mkdir(parents=True, exist_ok=args.overwrite)

    ins_df = pd.read_csv(args.ins_csv)
    ro_df = pd.read_csv(args.ro_csv)
    kf_aux1_df = pd.read_csv(settings.AUX1_CSV)
    kf_aux2_df = pd.read_csv(settings.AUX2_CSV)
    kf_aux3_df = pd.read_csv(settings.AUX3_CSV)

    plot_speeds(ins_df, ro_df, kf_aux1_df, kf_aux2_df, kf_aux3_df, output_path)


if __name__ == '__main__':
    main()  # pragma: no cover
