import sort_function
import copy


# def merge_fea(d, e, edge, dir_dict):
#     flag1 = False
#     list_d = []
#     for entry in copy.deepcopy(dir_dict[edge[0]]):
#         if entry[0] == d:
#             flag1 = True
#             list_d.append(entry)
#     if not flag1:
#         dir_dict[edge[0]].append([d, edge[1], e])
#     else:
#         for entry in list_d:
#             v_prime = entry[1]
#             if edge[1] != v_prime:
#                 dir_dict[edge[0]].append([d, edge[1], e])
#     unique = []
#     for entry in copy.deepcopy(dir_dict[edge[0]]):
#         if (entry[0] == d) and (entry[1] == edge[1]) and (entry[2] != e):
#             dir_dict[edge[0]].remove(entry)
#     for entry in copy.deepcopy(dir_dict[edge[0]]):
#         if entry not in unique:
#             unique.append(entry)
#     dir_dict[edge[0]] = unique
#     dir_dict[edge[0]].sort(key=sort_function.take_first, reverse=True)
#     return dir_dict

def merge_fea(d, e, edge, dir_dict):
    flag = False
    list_v = []
    for entry in copy.deepcopy(dir_dict[edge[0]]):
        if entry[1] == edge[1]:
            flag = True
            list_v.append(entry)
    if not flag:
        dir_dict[edge[0]].append([d, edge[1], e])
    else:
        flag1 = False
        list_d = []
        for entry in list_v:
            if entry[0] == d:
                flag1 = True
                list_d.append(entry)
        if not flag1:
            dir_dict[edge[0]].append([d, edge[1], e])
    dir_dict[edge[0]].sort(key=sort_function.take_first, reverse=True)
    return dir_dict

