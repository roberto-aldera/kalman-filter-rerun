from collections import namedtuple

Colours = namedtuple("colour", "ins, ro, kf_aux1, kf_aux2, kf_aux3")
colours = Colours(ins="black", ro="tab:blue",
                  kf_aux1="tab:green", kf_aux2="tab:red", kf_aux3="tab:brown")

INS_CSV = "/Users/roberto/data/kalman-filter-rerun/relative-poses/processed_ins.csv"
RO_CSV = "/Users/roberto/data/kalman-filter-rerun/relative-poses/ro_poses.csv"

# Experiment 1: config 1, 2, 3
# AUX1_CSV = "/Users/roberto/data/kalman-filter-rerun/relative-poses/kf_R1_Q1_poses.csv"
# AUX2_CSV = "/Users/roberto/data/kalman-filter-rerun/relative-poses/kf_R0p5_Q1_poses.csv"
# AUX3_CSV = "/Users/roberto/data/kalman-filter-rerun/relative-poses/kf_R1_Q0p5_poses.csv"

# Experiment 2: config 3, 4, 5
# AUX1_CSV = "/Users/roberto/data/kalman-filter-rerun/relative-poses/kf_R1_Q0p5_poses.csv"
# AUX2_CSV = "/Users/roberto/data/kalman-filter-rerun/relative-poses/kf_R1_Q0p3_poses.csv"
# AUX3_CSV = "/Users/roberto/data/kalman-filter-rerun/relative-poses/kf_R1_Q0p1_poses.csv"

# Speed plot for config 2, 3, 5
AUX1_CSV = "/Users/roberto/data/kalman-filter-rerun/relative-poses/kf_R0p5_Q1_poses.csv"
AUX2_CSV = "/Users/roberto/data/kalman-filter-rerun/relative-poses/kf_R1_Q0p5_poses.csv"
AUX3_CSV = "/Users/roberto/data/kalman-filter-rerun/relative-poses/kf_R1_Q0p1_poses.csv"

AUX1_NAME = "Configuration 2"
AUX2_NAME = "Configuration 3"
AUX3_NAME = "Configuration 5"
