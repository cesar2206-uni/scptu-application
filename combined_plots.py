from main import *

process = [
    "21CP01.xlsx",
    "21CP02.xlsx",
    "21CP03.xlsx",
    "21CP04.xlsx",
    "21CP05.xlsx"
    ]

processed_data = []
for row in process:
    data = read_data(row)
    data = process_dissipation_test(data)
    data = process_sbt(data)
    processed_data.append(data)

# multiply_graph_qt(processed_data, 30)

# multiply_graph_Qtn(processed_data, 30)
# multiply_graph_Ic(processed_data, 30)
# multiply_graph_IB(processed_data, 30)

multiply_graph_COV_Qtn(processed_data, "Sand", 30)
multiply_graph_COV_qt(processed_data, "Sand", 30)