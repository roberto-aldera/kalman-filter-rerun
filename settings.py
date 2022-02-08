from collections import namedtuple

Colours = namedtuple("colour", "ins, ro, kf")
colours = Colours(ins="black", ro="tab:blue", kf="tab:orange")

RO_CSV = "/Users/roberto/data/kalman-filter-rerun/ro_poses.csv"
INS_CSV = "/Users/roberto/data/kalman-filter-rerun/processed_ins.csv"
KF_CSV = "/Users/roberto/data/kalman-filter-rerun/kalman_filter_poses.csv"
