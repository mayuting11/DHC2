import xlrd

init_env_settings = (1, 135, 19)
max_map_lenght = 135
max_map_width = 19
obstacle_list = []
file_name = 'Floor2'
file_path = 'D:\pythonProject\DHC\D3QTP' + '//' + file_name + '.xls'
# file_path='.\Floor2.xls'
workbook = xlrd.open_workbook(file_path)
sheets = workbook.sheet_names()
worksheet = workbook.sheet_by_name(sheets[0])
row_obs = 14
for i in range(1, worksheet.nrows):
    if worksheet.cell_value(i, row_obs) != '':
        obstacle_list.append([int(worksheet.cell_value(i, row_obs)), int(worksheet.cell_value(i, row_obs + 1))])
# print(obstacle_list)

goal_num = 2
goal_list = []
row_goal = 16
goal_people_quantity = []
for i in range(1, worksheet.nrows):
    if worksheet.cell_value(i, row_goal) != '':
        goal_list.append([int(worksheet.cell_value(i, row_goal)), int(worksheet.cell_value(i, row_goal + 1)),
                          int(worksheet.cell_value(i, row_goal + 2))])
for i in range(goal_num):
    goal_people_quantity.append(0)
# print(goal_list)

supply_list = []
row_supply = 19
for i in range(1, worksheet.nrows):
    if worksheet.cell_value(i, row_supply) != '':
        supply_list.append([int(worksheet.cell_value(i, row_supply)), int(worksheet.cell_value(i, row_supply + 1)),
                            int(worksheet.cell_value(i, row_supply + 2)), int(worksheet.cell_value(i, row_supply + 3))])
supply_num = len(supply_list)
# print(supply_num)

dic_node = {}
row_node = 0
for i in range(1, worksheet.nrows):
    if worksheet.cell_value(i, row_node) != '':
        dic_node[int(worksheet.cell_value(i, row_node))] = (
            int(worksheet.cell_value(i, row_node + 2)), int(worksheet.cell_value(i, row_node + 1)))

delay_dic = {}
row_delay = 5
for i in range(1, worksheet.nrows):
    if worksheet.cell_value(i, row_delay) != '':
        delay_dic[dic_node[int(worksheet.cell_value(i, row_delay))], dic_node[
            int(worksheet.cell_value(i, row_delay + 1))]] = [
            [int(worksheet.cell_value(i, row_delay + 2)), int(worksheet.cell_value(i, row_delay + 4)),
             int(worksheet.cell_value(i, row_delay + 6))],
            [worksheet.cell_value(i, row_delay + 3), worksheet.cell_value(i, row_delay + 5),
             worksheet.cell_value(i, row_delay + 7)]]

save_path = './models_pre'
