import configs_pretrain


def initialization(goals_indice, map_length=configs_pretrain.init_env_settings[1],
                   map_width=configs_pretrain.init_env_settings[2]):
    dir_dic = {}
    inf = 2 ** 31 - 1
    for i in range(map_width):
        for j in range(map_length):
            if (i, j) not in goals_indice:
                dir_dic[(i, j)] = [[inf, None, inf]]
            else:
                dir_dic[(i, j)] = [[0, (i, j), 0]]
    return dir_dic
