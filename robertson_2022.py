from main import *

data = read_data("SCPTu-08.xlsx")
data = process_dissipation_test(data)
data = process_sbt(data)

# pre_graph_Ic(data, 80)
# graph_basic_plots(data, 80)
data = combine_cpt_vel(data)
graph_vs_plots(data, 80)