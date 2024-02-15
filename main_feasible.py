import get_feasible_map
import xlwt

file_name = 'tables_feasible_0'
file_path = 'D:\pythonProject\DHC\D3QTP' + '//' + file_name + '.xls'
dir_dic = get_feasible_map.generate_tables()

sheet_name = 'TABLES_feasible_0'
workbook = xlwt.Workbook(encoding='utf-8', style_compression=0)  # 新建一个工作簿
sheet = workbook.add_sheet(sheet_name, cell_overwrite_ok=True)  # 在工作簿中新建一个表格
row = 0
for i in dir_dic:
    sheet.write(row, 0, str(i))
    for j in range(len(dir_dic[i])):
        sheet.write(row, int(3 * j + 1), dir_dic[i][j][0])
        sheet.write(row, int(3 * j + 2), str(dir_dic[i][j][1]))
        sheet.write(row, int(3 * j + 3), dir_dic[i][j][2])
    row += 1
workbook.save(file_path)
print("successful write")
