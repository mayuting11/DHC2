import copy
import xlwt
import xlrd
import sort_function
import initilization
import configs_pretrain

file_name_0 = 'tables_0'
file_path_0 = 'D:\pythonProject\DHC\D3QTP' + '//' + file_name_0 + '.xls'

workbook_0 = xlrd.open_workbook(file_path_0)
sheets_0 = workbook_0.sheet_names()
worksheet_0 = workbook_0.sheet_by_name(sheets_0[0])
column = 0
num_col = worksheet_0.ncols
dir_dic_0 = {}
for i in range(0, worksheet_0.nrows):
    if worksheet_0.cell_value(i, column) != '':
        dir_dic_0[worksheet_0.cell_value(i, column)] = []
        for j in range(1, num_col, 3):
            if worksheet_0.cell_value(i, j) != '':
                dir_dic_0[worksheet_0.cell_value(i, column)].append(
                    [worksheet_0.cell_value(i, j), worksheet_0.cell_value(i, j + 1),
                     worksheet_0.cell_value(i, j + 2)])

file_name_1 = 'tables_1'
file_path_1 = 'D:\pythonProject\DHC\D3QTP' + '//' + file_name_1 + '.xls'

workbook_1 = xlrd.open_workbook(file_path_1)
sheets_1 = workbook_1.sheet_names()
worksheet_1 = workbook_1.sheet_by_name(sheets_1[0])
column = 0
num_col = worksheet_1.ncols
dir_dic_1 = {}
for i in range(0, worksheet_1.nrows):
    if worksheet_1.cell_value(i, column) != '':
        dir_dic_1[worksheet_1.cell_value(i, column)] = []
        for j in range(1, num_col, 3):
            if worksheet_1.cell_value(i, j) != '':
                dir_dic_1[worksheet_1.cell_value(i, column)].append(
                    [worksheet_1.cell_value(i, j), worksheet_1.cell_value(i, j + 1),
                     worksheet_1.cell_value(i, j + 2)])

goals_list = configs_pretrain.goal_list
goals_indice = list()
for i in goals_list:
    goals_indice.append((i[1], i[2]))
dir_dic = initilization.initialization(goals_indice)

for i in dir_dic_0:
    a = i.lstrip('(')
    a = a.rstrip(')')
    a = a.split(",")
    for j in dir_dic_0[i]:
        flag1 = False
        list_d_0 = []
        # print(i[4])
        for entry in copy.deepcopy(dir_dic[(int(a[0]), int(a[1]))]):
            if j[0] == entry[0]:
                flag1 = True
                list_d_0.append(entry)
        if not flag1:
            dir_dic[(int(a[0]), int(a[1]))].append(j)
        else:
            for p in list_d_0:
                if p[1] is not None:
                    c = j[1].lstrip('(')
                    c = c.rstrip(')')
                    c = c.split(",")
                    if ((int(c[0]), int(c[1])) != p[1]) and (j not in dir_dic[(int(a[0]), int(a[1]))]):
                        dir_dic[(int(a[0]), int(a[1]))].append(j)
        for q in copy.deepcopy(dir_dic[(int(a[0]), int(a[1]))]):
            if q[1] is not None:
                e = j[1].lstrip('(')
                e = e.rstrip(')')
                e = e.split(",")
                if (q[0] == j[0]) and (q[2] != j[2]) and (q[1][0] == int(e[0])) and (q[1][1] == int(e[1])):
                    dir_dic[(int(a[0]), int(a[1]))].remove(q)
            else:
                dir_dic[(int(a[0]), int(a[1]))].remove(q)
    for k in dir_dic_1[i]:
        flag2 = False
        list_d = []
        for m in copy.deepcopy(dir_dic[(int(a[0]), int(a[1]))]):
            if m[0] == k[0]:
                flag2 = True
                list_d.append(m)
        if not flag2:
            dir_dic[(int(a[0]), int(a[1]))].append(k)
        else:
            for n in list_d:
                if n[1] is not None:
                    g = k[1].lstrip('(')
                    g = g.rstrip(')')
                    g = g.split(",")
                    if n[1] != (int(g[0]), int(g[1])):
                        dir_dic[(int(a[0]), int(a[1]))].append(k)
        for entry in copy.deepcopy(dir_dic[(int(a[0]), int(a[1]))]):
            if entry[1] is not None:
                l = k[1].lstrip('(')
                l = l.rstrip(')')
                l = l.split(",")
                if (entry[0] == k[0]) and (entry[2] != k[2]) and (entry[1][0] == int(l[0])) and (entry[1][1] == int(l[1])):
                    dir_dic[(int(a[0]), int(a[1]))].remove(entry)
            else:
                dir_dic[(int(a[0]), int(a[1]))].remove(entry)
    dir_dic[(int(a[0]), int(a[1]))].sort(key=sort_function.take_first, reverse=True)

file_name = 'tables_fea'
file_path = 'D:\pythonProject\DHC\D3QTP' + '//' + file_name + '.xls'

sheet_name = 'tables_fea'
workbook = xlwt.Workbook(encoding='utf-8', style_compression=0)  # 新建一个工作簿
sheet = workbook.add_sheet(sheet_name, cell_overwrite_ok=True)  # 在工作簿中新建一个表格
row = 0
for i in dir_dic:
    sheet.write(row, 0, str(i))
    for j in range(1, len(dir_dic[i]), 3):
        sheet.write(row, int(j), dir_dic[i][j][0])
        sheet.write(row, int(j + 1), str(dir_dic[i][j][1]))
        sheet.write(row, int(j + 2), dir_dic[i][j][2])
    row += 1
workbook.save(file_path)
print("successful write")
