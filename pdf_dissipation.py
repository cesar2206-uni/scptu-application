from main import *

process = [
    ["SCPTu-03.xlsx", 70],
]

for row in process:
    data = read_data(row[0])
    data_1 = process_dissipation_test(data)
    graph_dissipation_test(data_1, row[1], option = 1)
    graph_dissipation_test(data_1, row[1], option = 2)