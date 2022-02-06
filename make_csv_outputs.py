import pdb
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import pandas as pd

# Include paths - need these for interfacing with custom protobufs
sys.path.insert(-1, "/Users/roberto/code/corelibs/src/tools-python")
sys.path.insert(-1, "/Users/roberto/code/corelibs/build/datatypes")
from mrg.logging import MonolithicDecoder                                       # nopep8
from mrg.transform.conversions import se3_to_components, build_se3_transform    # nopep8
from mrg.adaptors.transform import PbSerialisedTransformToPython                # nopep8


def process_poses(relative_pose_path, pose_type):

    # open monolithic and iterate frames
    print("reading relative_poses_path: " + relative_pose_path)
    monolithic_decoder = MonolithicDecoder(
        relative_pose_path)

    # iterate mono
    se3s = []
    timestamps = []
    for pb_serialised_transform, _, _ in monolithic_decoder:
        serialised_transform = PbSerialisedTransformToPython(
            pb_serialised_transform)
        se3s.append(serialised_transform[0])
        timestamps.append(serialised_transform[1])
    df = pd.DataFrame({"timestamp": timestamps,
                       "x": [se3[0, 3] for se3 in se3s],
                       "y": [se3[1, 3] for se3 in se3s],
                       "yaw": [math.atan2(se3[1, 0], se3[0, 0]) for se3 in se3s]})
    df.to_csv(
        f"/Users/roberto/data/kalman-filter-rerun/{pose_type}.csv", index=False)


def main():
    print("Running script...")

    ro_relative_poses_path = "/Users/roberto/data/odometry-comparisons/rugged_ro/" \
        "2018-06-21-16-24-39-long-hanborough-to-ori-V4-radar-leopon-trial-sunny-long-range/" \
        "motion_estimation/standard-ro/radar_motion_estimation.monolithic"

    ins_relative_poses_path = "/Users/roberto/data/odometry-comparisons/rugged_ro/" \
        "2018-06-21-16-24-39-long-hanborough-to-ori-V4-radar-leopon-trial-sunny-long-range/" \
        "motion_estimation/ground-truth/flattened_novatel_generated_poses.monolithic"

    process_poses(ro_relative_poses_path, "ro_poses")
    process_poses(ins_relative_poses_path, "ins_poses")


if __name__ == "__main__":
    main()
