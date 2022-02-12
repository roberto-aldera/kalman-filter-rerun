from collections import namedtuple

Colours = namedtuple("colour", "ins, ro, kf_aux1, kf_aux2, kf_aux3")
colours = Colours(ins="black", ro="tab:blue",
                  kf_aux1="tab:orange", kf_aux2="tab:green", kf_aux3="tab:red")

INS_CSV = "/Users/roberto/data/kalman-filter-rerun/relative-poses/processed_ins.csv"
RO_CSV = "/Users/roberto/data/kalman-filter-rerun/relative-poses/ro_poses.csv"
# AUX1_CSV = "/Users/roberto/data/kalman-filter-rerun/relative-poses/kf_R1_Q1_poses.csv"
# AUX2_CSV = "/Users/roberto/data/kalman-filter-rerun/relative-poses/kf_R0p5_Q1_poses.csv"
# AUX3_CSV = "/Users/roberto/data/kalman-filter-rerun/relative-poses/kf_R1_Q0p5_poses.csv"
AUX1_CSV = "/Users/roberto/data/kalman-filter-rerun/relative-poses/kf_R1_Q0p5_poses.csv"
AUX2_CSV = "/Users/roberto/data/kalman-filter-rerun/relative-poses/kf_R1_Q0p3_poses.csv"
AUX3_CSV = "/Users/roberto/data/kalman-filter-rerun/relative-poses/kf_R1_Q0p1_poses.csv"
