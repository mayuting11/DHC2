import xlrd

############################################################
####################    environment     ####################
############################################################
num_agents = 38
obs_radius = 4
list_fac_off_route = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
fac_off_route = list_fac_off_route[0]
reward_fn = dict(move=-0.2,
                 stay_off_goal=-2.75,
                 oscillate=-2.5,
                 off_route=-2.25 * fac_off_route,
                 stay_on_goal_a=0,
                 stay_on_goal_b=-0.6,
                 stay_on_goal_c=-0.6,
                 stay_on_goal_d=-2.1,
                 pick_up_a=-4.5,
                 pick_up_b=0,
                 pick_up_c=-4.5,
                 pick_up_d=-4.5,
                 collision=-5,
                 infeasible_move=-4.5,
                 finish=+60)

init_env_settings = (38, 135, 19)
max_map_lenght = 135
max_map_width = 19
max_num_agents = 38

obstacle_list = []
file_name = 'Floor2'
file_path = '/home/lzimny/DHC/D3QTP' + '//' + file_name + '.xls'
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

file_name = 'tables_feasible'
file_path = '/home/lzimny/DHC/D3QTP' + '//' + file_name + '.xls'
workbook = xlrd.open_workbook(file_path)
sheets = workbook.sheet_names()
worksheet = workbook.sheet_by_name(sheets[0])
dir_dic = {}
column_dir = 0
for i in range(worksheet.nrows):
    if worksheet.cell_value(i, column_dir) != '':
        a = worksheet.cell_value(i, column_dir).lstrip('(')
        a = a.rstrip(')')
        a = a.split(",")
        dir_dic[(int(a[0]), int(a[1]))] = []
        for j in range(1, worksheet.ncols, 3):
            if worksheet.cell_value(i, j) != '':
                if worksheet.cell_value(i, j+1) != 'None':
                    b = worksheet.cell_value(i, j + 1).lstrip('(')
                    b = b.rstrip(')')
                    b = b.split(",")
                    dir_dic[(int(a[0]), int(a[1]))].append(
                        [worksheet.cell_value(i, j), (int(b[0]), int(b[1])),
                         worksheet.cell_value(i, j + 2)])

obs_shape = (8, 2 * obs_radius + 1, 2 * obs_radius + 1)
agent_state_shape = (4, 1)
action_dim = 6

############################################################
####################         DQN        ####################
############################################################
# FC layer parameter
Agent_REPR_SIZE = 12

# basic training setting
num_actors = 1
log_interval = 10
training_times = 600000
save_interval = 20000
gamma = 0.95
batch_size = 192
learning_starts = 50000
target_network_update_freq = 2000
save_path = './models'
max_episode_length = 512
seq_len = 8
load_model = None
max_episode_length = max_episode_length
actor_update_steps = 400


# gradient norm clipping
grad_norm_dqn = 40

# n-step forward
forward_steps = 2

# global buffer
episode_capacity = 1024

# prioritized replay
prioritized_replay_alpha = 0.6
prioritized_replay_beta = 0.4

# dqn network setting
cnn_channel = 128
hidden_dim = 256
latent_dim = seq_len*(2*obs_radius-1)*(2*obs_radius-1)
obs_all_size = hidden_dim+Agent_REPR_SIZE

test_seed = 0
num_test_cases = 200
# map length, map_width, number of agents
test_env_settings = (
    (135, 19, 2), (135, 19, 4), (135, 19, 6), (135, 19, 8), (135, 19, 10), (135, 19, 12), (135, 19, 14),
    (135, 19, 16), (135, 19, 18), (135, 19, 20), (135, 19, 22), (135, 19, 24), (135, 19, 26),
    (135, 19, 28), (135, 19, 30), (135, 19, 32), (135, 19, 34), (135, 19, 36), (135, 19, 38),
    (135, 19, 40), (135, 19, 42), (135, 19, 44), (135, 19, 46), (135, 19, 48), (135, 19, 50)
)
