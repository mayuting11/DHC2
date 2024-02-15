import copy
import merge_fea

def relax(edge, dir_dict, inf, delay_dic):
    for i in copy.deepcopy(dir_dict[edge[1]]):
        # print(dir_dict[edge[0]])
        for j in range(3):
            flag_safe_fea = False
            if delay_dic[tuple(edge)][1][j] > 0:
                d_u = i[0] + delay_dic[tuple(edge)][0][j]
                e_u = 0
                for k in range(len(delay_dic[tuple(edge)][1])):
                    if delay_dic[tuple(edge)][1][k] > 0:
                        for l in dir_dict[edge[1]]:
                            if (d_u - delay_dic[tuple(edge)][0][k] >= l[0]) and (l[0] != inf):
                                e_u += delay_dic[tuple(edge)][1][k] * (delay_dic[tuple(edge)][0][k] + l[2])
                                flag_safe_fea = True
                                break
                        if not flag_safe_fea:
                            e_u = inf
                if e_u < inf:
                    # print('starting merge')
                    merge_fea.merge_fea(d_u, e_u, edge, dir_dict)
    # print('finish relaxing')
    return dir_dict
