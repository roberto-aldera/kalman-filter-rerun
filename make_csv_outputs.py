# Comparing RO to INS data
import pdb
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Include paths - need these for interfacing with custom protobufs
sys.path.insert(-1, "/Users/roberto/code/corelibs/src/tools-python")
sys.path.insert(-1, "/Users/roberto/code/corelibs/build/datatypes")
from mrg.logging import MonolithicDecoder                                       # nopep8
from mrg.transform.conversions import se3_to_components, build_se3_transform    # nopep8
from mrg.adaptors.transform import PbSerialisedTransformToPython                # nopep8


RO_relative_poses_path = "/Users/roberto/data/odometry-comparisons/rugged_ro/" \
    "2018-06-21-16-24-39-long-hanborough-to-ori-V4-radar-leopon-trial-sunny-long-range/" \
    "motion_estimation/standard-ro/radar_motion_estimation.monolithic"

# RO - open monolithic and iterate frames
print("reading RO_relative_poses_path: " + RO_relative_poses_path)
monolithic_decoder = MonolithicDecoder(
    RO_relative_poses_path)

# iterate mono
RO_se3s = []
RO_timestamps = []
for pb_serialised_transform, _, _ in monolithic_decoder:
    serialised_transform = PbSerialisedTransformToPython(
        pb_serialised_transform)
    RO_se3s.append(serialised_transform[0])
    RO_timestamps.append(serialised_transform[1])

pdb.set_trace()


def main():
    print("Running script...")


if __name__ == "__main__":
    main()
