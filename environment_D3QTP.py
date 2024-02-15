import random
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import configs_D3QTP

# import initilization
# import relax
# import relax_fea

plt.ion()  # Transform the display mode of matplotlib into interactive mode.

# define the five actions: wait, pick up, up, down, left, right
action_list = np.array([[0, 0, 0], [0, 0, 1], [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=int)

color_map = np.array([[255, 255, 255],  # white: available space
                      [190, 190, 190],  # gray: obstacle space
                      [0, 191, 255],  # blue: agent starting positions
                      [255, 165, 0],  # orange: goal waypoints
                      [0, 250, 154],  # green: agents after arriving at goal waypoints
                      [255, 48, 48],  # red: hazard waypoints
                      [255, 215, 0]])  # supply waypoints


def map_partition(map):
    '''
    partitioning map into independent partitions
    '''
    # Return the indices of the empty grid;
    empty_pos = np.argwhere(map == 0).astype(int).tolist()
    empty_pos = [tuple(pos) for pos in empty_pos]
    if not empty_pos:
        raise RuntimeError('no empty position')

    partition_list = list()
    while empty_pos:
        # select start positions of agents
        start_pos = empty_pos.pop()  # remove the last element from the empty grid list and treat it as start_pos
        open_list = list()
        open_list.append(start_pos)  # append the start_pos to the list open_list
        close_list = list()
        while open_list:
            x, y = open_list.pop(0)  # remove the first element of the list and return it to x and y
            up = x - 1, y  # determine the relative position of the agent's up
            if up[0] >= 0 and map[up] == 0 and up in empty_pos:
                empty_pos.remove(up)
                open_list.append(up)
            down = x + 1, y  # determine the relative position of the agent's down
            if (down[0] < (map.shape[0] - 1)) and (map[down] == 0) and (down in empty_pos):
                empty_pos.remove(down)
                open_list.append(down)
            left = x, y - 1  # determine the relative position of the agent's left
            if left[1] >= 0 and map[left] == 0 and left in empty_pos:
                empty_pos.remove(left)
                open_list.append(left)
            right = x, y + 1  # determine the relative position of the agent's right
            if (right[1] < map.shape[1] - 1) and (map[right] == 0) and (right in empty_pos):
                empty_pos.remove(right)
                open_list.append(right)
            close_list.append((x, y))
        partition_list.append(close_list)  # partition_list is a list [[()],[()],...,[()]]
    return partition_list


# set hazard degree
def get_hazard_degree(map_width, map_length):
    hazard_dic = {}
    for i in range(map_width):
        for j in range(map_length):
            random.seed(str(i) + str(j))
            hazard_dic[(i, j)] = random.uniform(0.5, 2.5)
    return hazard_dic


class Environment:
    def __init__(self, num_agents: int = configs_D3QTP.init_env_settings[0],
                 map_length: int = configs_D3QTP.init_env_settings[1],
                 map_width: int = configs_D3QTP.init_env_settings[2], obs_radius: int = configs_D3QTP.obs_radius,
                 reward_fn: dict = configs_D3QTP.reward_fn,
                 obstacle_list: list = configs_D3QTP.obstacle_list, num_goals: int = configs_D3QTP.goal_num,
                 goals_list: list = configs_D3QTP.goal_list, supply_list: list = configs_D3QTP.supply_list,
                 num_supply: int = configs_D3QTP.supply_num,
                 goal_people_quantity: list = configs_D3QTP.goal_people_quantity,
                 delay_distribution: dict = configs_D3QTP.delay_dic, curriculum=False,
                 init_env_settings_set=configs_D3QTP.init_env_settings):

        # set the number of agents and the size of the map
        self.curriculum = curriculum
        self.num_goals = num_goals
        self.obstacle_list = obstacle_list
        self.goals_list = goals_list
        self.supply_list = supply_list
        self.num_supply = num_supply
        self.goal_people_quantity = goal_people_quantity
        self.map_length = map_length
        self.map_width = map_width
        self.hazard_dic = get_hazard_degree(self.map_width, self.map_length)
        self.delay_distribution = delay_distribution

        if curriculum:
            self.env_set = [init_env_settings_set]
            self.num_agents = init_env_settings_set[0]
            self.map_size = (init_env_settings_set[2], init_env_settings_set[1])
        else:
            self.num_agents = num_agents
            self.map_size = (map_width, map_length)

        # determine obstacle positions
        self.map = np.random.choice(2, self.map_size, p=[1, 0]).astype(int)
        for i in self.obstacle_list:
            obstacle_pos = i[0], i[1]
            self.map[obstacle_pos] = int(1)

        # Return a list that stores the empty positions for each possible agent position (i.e., empty position) [[(),(),...,()],[(),(),...,()],...,[(),(),...,()]]
        partition_list = map_partition(self.map)
        # Delete the list denoting that an agent has no feasible move at current position
        partition_list = [partition for partition in partition_list if len(partition) >= 2]

        self.agents_pos = np.empty((self.num_agents, 2), dtype=int)  # set the positions of agents
        self.goals_pos = np.empty((self.num_goals, 2), dtype=int)  # set the goal positions
        self.supply_pos = np.empty((self.num_supply, 2), dtype=int)  # set the supply positions
        self.agents_state = np.empty(self.num_agents, dtype=bool)  # set the states of agents
        self.agents_deadline = np.empty(self.num_agents, dtype=int)  # set the states of agents
        self.goals_state = np.empty(self.num_goals, dtype=bool)  # set the states of goals
        self.supply_quantity = np.empty(self.num_supply, dtype=int)  # set the quantity of PLSE at each supply waypoint

        pos_num = sum([len(partition) for partition in
                       partition_list])  # calculate the sum of the number of all feasible move of all agents

        # Assign original position for each agent
        for i in range(self.num_agents):
            pos_idx = random.randint(0, pos_num - 1)  # return an integer in the range [0, pos_num-1]
            partition_idx = 0
            for partition in partition_list:
                if pos_idx >= len(partition):
                    pos_idx -= len(partition)  # equal to pos_idx=pos_idx-len(partition)
                    partition_idx += 1  # equal to  partition_idx=partition_idx+1
                else:
                    break  # break will stop the deepest loop and start to execute the code in the next line

            pos = random.choice(
                partition_list[partition_idx])  # randomly select a tuple from partition_list[partition_idx]
            partition_list[partition_idx].remove(pos)  # remove the first pos from partition_list[partition_idx]
            self.agents_pos[i] = np.asarray(pos, dtype=int)  # convert a tuple into an array
            partition_list = [partition for partition in partition_list if len(partition) - 1 >= 2]
            pos_num = sum([len(partition) for partition in partition_list])

        # Determine goal positions
        for i in range(self.num_goals):
            pos = (self.goals_list[i][1], self.goals_list[i][2])
            self.goals_pos[i] = np.asarray(pos, dtype=int)

        # Determine supply positions
        for i in range(self.num_supply):
            pos = (self.supply_list[i][1], self.supply_list[i][2])
            self.supply_pos[i] = np.asarray(pos, dtype=int)

        # Determine the state for each agent
        for i in range(self.num_agents):
            self.agents_state[i] = 0

        # Determine the deadline for each agent
        for i in range(self.num_agents):
            self.agents_deadline[i] = int(1800)

        # Determine the state for each goal
        for i in range(self.num_goals):
            if self.goal_people_quantity[i] <= 223:
                self.goals_state[i] = True
            else:
                self.goals_state[i] = False

        # Determine the quantity for each supply waypoint
        for i in range(self.num_supply):
            self.supply_quantity[i] = self.supply_list[i][3]

        self.obs_radius = obs_radius
        self.reward_fn = reward_fn
        # print('obtain reward_fn')
        # self.get_fastest_waypoint()
        # print('obtain the fastest waypoints')
        self.get_fea_waypoint()
        # print('obtain the feasible waypoints')
        self.steps = 0

        # Save the observation for each potential action for the agent
        self.last_actions = np.zeros((self.num_agents, 6, 2 * obs_radius + 1, 2 * obs_radius + 1), dtype=bool)

    # update the environment settings
    def update_env_settings_set(self, new_env_settings_set):
        self.env_set = new_env_settings_set

    # reset the environment (similar to the def __init__())
    def reset(self, num_agents=configs_D3QTP.num_agents, map_length=configs_D3QTP.max_map_lenght, map_width=configs_D3QTP.max_map_width):

        if self.curriculum:
            rand = random.choice(self.env_set)  # randomly choose an environment set
            self.num_agents = rand[0]
            self.map_size = (rand[2], rand[1])

        elif num_agents is not None and map_length is not None and map_width is not None:
            self.num_agents = num_agents
            self.map_size = (map_width, map_length)

        self.map = np.random.choice(2, self.map_size, p=[1, 0]).astype(np.float32)
        for i in self.obstacle_list:
            obstacle_pos = i[0], i[1]
            self.map[obstacle_pos] = np.float32(1)

        partition_list = map_partition(self.map)
        partition_list = [partition for partition in partition_list if len(partition) >= 2]

        self.agents_pos = np.empty((self.num_agents, 2), dtype=int)
        self.goals_pos = np.empty((self.num_goals, 2), dtype=int)
        pos_num = sum([len(partition) for partition in partition_list])
        for i in range(self.num_agents):
            pos_idx = random.randint(0, pos_num - 1)
            partition_idx = 0
            for partition in partition_list:
                if pos_idx >= len(partition):
                    pos_idx -= len(partition)
                    partition_idx += 1
                else:
                    break
            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.agents_pos[i] = np.asarray(pos, dtype=int)
            partition_list = [partition for partition in partition_list if len(partition) - 1 >= 2]
            pos_num = sum([len(partition) for partition in partition_list])

        for i in range(self.num_goals):
            pos = (self.goals_list[i][1], self.goals_list[i][2])
            self.goals_pos[i] = np.asarray(pos, dtype=int)

        self.steps = 0
        # self.get_fastest_waypoint()
        self.get_fea_waypoint()

        # Save the observation for each potential action for the agent
        self.last_actions = np.zeros((self.num_agents, 6, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1),
                                     dtype=bool)
        return self.observe()

    # load the map, the agent positions, and the goal positions
    def load(self, map: np.ndarray, agents_pos: np.ndarray, goals_pos: np.ndarray, agents_state: np.ndarray):
        self.map = np.copy(map)  # return a deep copy of map
        self.agents_pos = np.copy(agents_pos)
        self.goals_pos = np.copy(goals_pos)
        self.num_agents = agents_pos.shape[0]
        self.agents_state = np.copy(agents_state)
        self.map_size = (self.map.shape[0], self.map.shape[1])
        self.steps = 0
        self.imgs = []
        # self.get_fastest_waypoint()
        self.get_fea_waypoint()
        self.last_actions = np.zeros((self.num_agents, 6, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1),
                                     dtype=bool)

    def get_fea_waypoint(self):
        self.feas_map = np.zeros((self.num_agents, self.map_width, self.map_length), dtype=bool)
        fea_map = [[] for i in range(self.num_agents)]
        tables_fea = configs_D3QTP.dir_dic
        for j in range(self.num_agents):
            if tables_fea[tuple(self.agents_pos[j])]:
                for i in tables_fea[tuple(self.agents_pos[j])]:
                    if (self.agents_deadline[j] >= i[0]) and (i[1] not in fea_map[j]):
                        fea_map[j].append(i[1])

        for i in range(self.num_agents):
            if fea_map[i]:
                for j in fea_map[i]:
                    self.feas_map[i][j[0]][j[1]] = 1
        self.feas_map = np.pad(self.feas_map,
                               ((0, 0), (self.obs_radius, self.obs_radius), (self.obs_radius, self.obs_radius)),
                               'constant', constant_values=0)

    # the action set for the agent is a list consisting of 0,1,2,3,4,5indicating stay,pick-up,up,down,left,right.
    def step(self, actions: List[int]):
        assert len(actions) == self.num_agents, 'only {} actions as input while {} agents in environment'.format(
            len(actions), self.num_agents)
        assert all([5 >= action_idx >= 0 for action_idx in actions]), 'action index out of range'
        checking_list = [i for i in range(self.num_agents)]
        rewards = []
        next_pos = np.copy(self.agents_pos)
        next_agent_state = np.copy(self.agents_state)
        next_agent_deadline = np.copy(self.agents_deadline)

        for agent_id in checking_list.copy():
            if actions[agent_id] == 0:
                # print('the action for {} is {}'.format(agent_id, actions[agent_id]))
                # print('the state for {} is {}'.format(agent_id, self.agents_state[agent_id]))
                if np.array_equal(self.agents_pos[agent_id], self.goals_pos[0]):
                    if self.goals_state[0] == True and self.agents_state[agent_id] == 1:
                        rewards.append(self.reward_fn['stay_on_goal_a'])
                        next_agent_deadline[agent_id] -= 1
                    if self.goals_state[0] == True and self.agents_state[agent_id] == 0:
                        rewards.append(self.reward_fn['stay_on_goal_b'])
                        next_agent_deadline[agent_id] -= 1
                    if self.goals_state[0] == False and self.agents_state[agent_id] == 1:
                        rewards.append(self.reward_fn['stay_on_goal_c'])
                        next_agent_deadline[agent_id] -= 1
                    if self.goals_state[0] == False and self.agents_state[agent_id] == 0:
                        rewards.append(self.reward_fn['stay_on_goal_d'])
                        next_agent_deadline[agent_id] -= 1
                if np.array_equal(self.agents_pos[agent_id], self.goals_pos[1]):
                    if self.goals_state[1] == True and self.agents_state[agent_id] == 1:
                        rewards.append(self.reward_fn['stay_on_goal_a'])
                        next_agent_deadline[agent_id] -= 1
                    if self.goals_state[1] == True and self.agents_state[agent_id] == 0:
                        rewards.append(self.reward_fn['stay_on_goal_b'])
                        next_agent_deadline[agent_id] -= 1
                    if self.goals_state[1] == False and self.agents_state[agent_id] == 1:
                        rewards.append(self.reward_fn['stay_on_goal_c'])
                        next_agent_deadline[agent_id] -= 1
                    if self.goals_state[1] == False and self.agents_state[agent_id] == 0:
                        rewards.append(self.reward_fn['stay_on_goal_d'])
                        next_agent_deadline[agent_id] -= 1
                else:
                    rewards.append(self.reward_fn['stay_off_goal'])
                    next_agent_deadline[agent_id] -= 1
                checking_list.remove(agent_id)

            elif actions[agent_id] == 1:
                if self.agents_state[agent_id]:
                    rewards.append(self.reward_fn['pick_up_a'])
                    self.supply_quantity[np.where(self.supply_pos == self.agents_pos[agent_id])[0][0]] -= 1
                    next_agent_deadline[agent_id] -= 1
                    if self.agents_pos[agent_id] in self.supply_pos:
                        self.supply_quantity[np.where(self.supply_pos == self.agents_pos[agent_id])[0][0]] -= 1

                else:
                    if self.agents_pos[agent_id] in self.supply_pos:
                        if self.supply_quantity[np.where(self.supply_pos == self.agents_pos[agent_id])[0][0]] > 0:
                            rewards.append(self.reward_fn['pick_up_b'])
                            self.supply_quantity[np.where(self.supply_pos == self.agents_pos[agent_id])[0][0]] -= 1
                            next_agent_state[agent_id] = 1
                            next_agent_deadline[agent_id] -= 1
                        else:
                            rewards.append(self.reward_fn['pick_up_c'])
                            next_agent_deadline[agent_id] -= 1
                    else:
                        rewards.append(self.reward_fn['pick_up_c'])
                        next_agent_deadline[agent_id] -= 1
                checking_list.remove(agent_id)

            else:
                fea_waypoint = []
                tab_fea = configs_D3QTP.dir_dic
                for l in tab_fea[tuple(next_pos[agent_id])]:
                    if (next_agent_deadline[agent_id] >= l[0]) and (l[1] not in fea_waypoint):
                        fea_waypoint.append(l[1])

                next_pos[agent_id][0] += action_list[actions[agent_id]][0]
                next_pos[agent_id][1] += action_list[actions[agent_id]][1]
                if (tuple(self.agents_pos[agent_id]), tuple(next_pos[agent_id])) in self.delay_distribution.keys():
                    if fea_waypoint:
                        if (next_pos[agent_id][0], next_pos[agent_id][1]) in fea_waypoint:
                            delay_distr = self.delay_distribution[tuple(self.agents_pos[agent_id]), tuple(next_pos[agent_id])]
                            delay_actual = random.choice(delay_distr[0])
                            next_agent_deadline[agent_id] -= delay_actual
                            if tuple(next_pos[agent_id]) in self.hazard_dic.keys():
                                rewards.append(self.reward_fn['move'] * delay_actual * self.hazard_dic[
                                    tuple(next_pos[agent_id])])
                            else:
                                rewards.append(self.reward_fn['collision'])
                        else:
                            delay_distr = self.delay_distribution[
                                tuple(self.agents_pos[agent_id]), tuple(next_pos[agent_id])]
                            delay_actual = random.choice(delay_distr[0])
                            next_agent_deadline[agent_id] -= delay_actual
                            if tuple(next_pos[agent_id]) in self.hazard_dic.keys():
                                rewards.append(self.reward_fn['move'] * delay_actual * self.hazard_dic[
                                    tuple(next_pos[agent_id])] + self.reward_fn['infeasible_move'])
                            else:
                                rewards.append(self.reward_fn['collision'])
                    else:
                        delay_distr = self.delay_distribution[
                            tuple(self.agents_pos[agent_id]), tuple(next_pos[agent_id])]
                        delay_actual = random.choice(delay_distr[0])
                        next_agent_deadline[agent_id] -= delay_actual
                        if tuple(next_pos[agent_id]) in self.hazard_dic.keys():
                            rewards.append(self.reward_fn['move'] * delay_actual * self.hazard_dic[
                                tuple(next_pos[agent_id])] + self.reward_fn['infeasible_move'])
                        else:
                            rewards.append(self.reward_fn['collision'])

                else:
                    rewards.append(self.reward_fn['collision'])

        # first round check whether the agent is out of range or collides with obstacles
        for agent_id in checking_list.copy():
            if (np.any(next_pos[agent_id] <= 0)) or (next_pos[agent_id][0] >= self.map_size[0]) or (next_pos[agent_id][
                                                                                                      1] >= self.map_size[1]):
                rewards[agent_id] = self.reward_fn['collision']
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)

            elif self.map[tuple(next_pos[agent_id])] == 1:
                rewards[agent_id] = self.reward_fn['collision']
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)

        # second round check, agent swapping (交换) conflict
        no_conflict = False
        while not no_conflict:
            no_conflict = True
            for agent_id in checking_list:
                target_agent_id = np.where(np.all(next_pos[agent_id] == self.agents_pos, axis=1))[
                    0]  # stores agents' ordinal number with which the current agent collides

                if target_agent_id:
                    target_agent_id = target_agent_id.item()  # return the element value of 1-tensor
                    assert target_agent_id != agent_id, 'logic bug'  # if the expression is false, raise an assertion error

                    if np.array_equal(next_pos[target_agent_id],
                                      self.agents_pos[agent_id]):  # collision is as follows: agent⇄agent
                        assert target_agent_id in checking_list, 'target_agent_id should be in checking list'

                        next_pos[agent_id] = self.agents_pos[agent_id]
                        rewards[agent_id] = self.reward_fn['collision']

                        next_pos[target_agent_id] = self.agents_pos[target_agent_id]
                        rewards[target_agent_id] = self.reward_fn['collision']

                        checking_list.remove(agent_id)
                        checking_list.remove(target_agent_id)

                        no_conflict = False
                        break

        # third round check, agent collision conflict
        no_conflict = False
        while not no_conflict:
            no_conflict = True
            for agent_id in checking_list:
                collide_agent_id = np.where(np.all(next_pos == next_pos[agent_id], axis=1))[0].tolist()

                if len(collide_agent_id) > 1:
                    all_in_checking = True  # flag indicating that all agents who collide with the current agent are in checking_list
                    for id in collide_agent_id.copy():
                        if id not in checking_list:
                            all_in_checking = False
                            collide_agent_id.remove(id)

                    if all_in_checking:
                        collide_agent_pos = next_pos[collide_agent_id].tolist()
                        for pos, id in zip(collide_agent_pos,
                                           collide_agent_id):  # pack the elements in the iterable objects as a tuple and return an object
                            pos.append(id)
                        collide_agent_pos.sort(key=lambda x: x[0] + x[1] * self.map_size[
                            1])  # sort based on x[0] * self.map_size[0] + x[1] from small to big
                        collide_agent_id.remove(collide_agent_pos[0][
                                                    2])  # only one agent can move as planned and other agents return to the last state

                    next_pos[collide_agent_id] = self.agents_pos[collide_agent_id]
                    for id in collide_agent_id:
                        rewards[id] = self.reward_fn['collision']

                    for id in collide_agent_id:
                        checking_list.remove(id)

                    no_conflict = False
                    break

        self.agents_pos = np.copy(next_pos)
        self.steps += 1

        # check done
        flag_done = True
        for i in range(self.num_agents):
            if self.agents_pos[i] not in self.goals_pos:
                flag_done = False
        if flag_done:
            done = True
            rewards = [self.reward_fn['finish'] for _ in range(self.num_agents)]
        else:
            done = False

        info = {'step': self.steps - 1}

        # make sure no overlapping agents
        if any(agent_pos not in self.goals_pos and np.all(agent_pos == other_agent_pos)
               for i, agent_pos in enumerate(self.agents_pos)
               for j, other_agent_pos in enumerate(self.agents_pos) if i != j):
            print(self.steps)
            print(self.map)
            print(self.agents_pos)
            raise RuntimeError('unique')

        # update last actions
        self.last_actions = np.zeros((self.num_agents, 6, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1),
                                     dtype=bool)
        self.last_actions[np.arange(self.num_agents), np.array(actions)] = 1

        return self.observe(), rewards, done, info

    def observe(self):
        '''
        return observation and position for each agent
        obs: shape (num_agents, 11, 2*obs_radius+1, 2*obs_radius+1)
            layer 1: agent map
            layer 2: obstacle map
            layer 3: goal position map
            layer 4: goal state map
            layer 5: supply position map
            layer 6: supply quantity map
            # layer 7: fastest map
            layer 7: feasible map
            layer 8: hazard degree map
            # layer 9-14: one-hot representation of agent's last action
            layer 9: agent's remaining deadline, position, state

        pos: used for calculating communication task
        '''
        obs = np.zeros((self.num_agents, 8, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1))
        obs_sec = np.zeros((self.num_agents, 4, 1))

        # 0 represents obstacle to match 0 padding in CNN
        obstacle_map = np.pad(self.map, self.obs_radius, 'constant', constant_values=0)  # pad the self.map with 0

        agent_map = np.zeros((self.map_size), dtype=bool)
        agent_map[self.agents_pos[:, 0], self.agents_pos[:, 1]] = 1
        agent_map = np.pad(agent_map, self.obs_radius, 'constant', constant_values=0)

        goal_pos_map = np.zeros((self.map_size), dtype=bool)
        goal_pos_map[self.goals_pos[:, 0], self.goals_pos[:, 1]] = 1
        goal_pos_map = np.pad(goal_pos_map, self.obs_radius, 'constant', constant_values=0)
        goal_state_map = np.zeros((self.map_size), dtype=bool)
        for i in range(self.num_goals):
            if self.goal_people_quantity[i] <= 223:
                goal_state_map[self.goals_pos[i][0], self.goals_pos[i][1]] = 1
        goal_state_map = np.pad(goal_state_map, self.obs_radius, 'constant', constant_values=0)

        supply_pos_map = np.zeros((self.map_size), dtype=bool)
        supply_pos_map[self.supply_pos[:, 0], self.supply_pos[:, 1]] = 1
        supply_pos_map = np.pad(supply_pos_map, self.obs_radius, 'constant', constant_values=0)
        supply_quan_map = np.zeros((self.map_size), dtype=int)
        for i in range(self.num_supply):
            supply_quan_map[self.supply_pos[i][0], self.supply_pos[i][1]] = self.supply_quantity[i]
        supply_quan_map = np.pad(supply_quan_map, self.obs_radius, 'constant', constant_values=0)

        hazard_map = np.zeros((self.map_size), dtype=int)
        for i in self.hazard_dic:
            hazard_map[i[0], i[1]] = self.hazard_dic[i]
        hazard_map = np.pad(hazard_map, self.obs_radius, 'constant', constant_values=0)

        for i, agent_pos in enumerate(
                self.agents_pos):  # combine an iterable object as a sequence containing the element and index
            x, y = agent_pos
            obs[i, 0] = agent_map[x:x + 2 * self.obs_radius + 1, y:y + 2 * self.obs_radius + 1]
            obs[i, 0, self.obs_radius, self.obs_radius] = 0
            obs[i, 1] = obstacle_map[x:x + 2 * self.obs_radius + 1, y:y + 2 * self.obs_radius + 1]
            obs[i, 2] = goal_pos_map[x:x + 2 * self.obs_radius + 1, y:y + 2 * self.obs_radius + 1]
            obs[i, 3] = goal_state_map[x:x + 2 * self.obs_radius + 1, y:y + 2 * self.obs_radius + 1]
            obs[i, 4] = supply_pos_map[x:x + 2 * self.obs_radius + 1, y:y + 2 * self.obs_radius + 1]
            obs[i, 5] = supply_quan_map[x:x + 2 * self.obs_radius + 1, y:y + 2 * self.obs_radius + 1]
            # obs[i, 6] = self.fas_map[i][x:x + 2 * self.obs_radius + 1, y:y + 2 * self.obs_radius + 1]
            obs[i, 6] = self.feas_map[i][x:x + 2 * self.obs_radius + 1, y:y + 2 * self.obs_radius + 1]
            obs[i, 7] = hazard_map[x:x + 2 * self.obs_radius + 1, y:y + 2 * self.obs_radius + 1]

            # the input about the agent state
            obs_sec[i, 0] = x
            obs_sec[i, 1] = y
            obs_sec[i, 2] = self.agents_deadline[i]
            obs_sec[i, 3] = self.agents_state[i]

        return obs, np.copy(self.agents_pos), obs_sec

    def render(self):
        if not hasattr(self, 'fig'):  # check whether an object contains a specified attribute 'fig'
            self.fig = plt.figure()

        mapp = np.copy(self.map)
        for agent_id in range(self.num_agents):
            if self.agents_pos[agent_id] in self.goals_pos:
                mapp[tuple(self.agents_pos[agent_id])] = 4
            else:
                mapp[tuple(self.agents_pos[agent_id])] = 2
        for goal_id in range(self.num_goals):
            mapp[tuple(self.goals_pos[goal_id])] = 3
        for supply_id in range(self.num_supply):
            mapp[tuple(self.supply_pos[supply_id])] = 6
        for i in self.hazard_dic:
            if self.hazard_dic[i] == 2.5:
                mapp[i] = 5

        mapp = mapp.astype(np.uint8)  # modify the datatype as np.unit8

        # add text in plot
        self.imgs.append([])
        if hasattr(self, 'texts'):
            for i, (agent_x, agent_y) in enumerate(self.agents_pos):
                self.texts[i].set_position((agent_x, agent_y))
                self.texts[i].set_text(i)
        else:
            self.texts = []
            for i, (agent_x, agent_y) in enumerate(self.agents_pos):
                text = plt.text(agent_x, agent_y, i, color='black', ha='center', va='center')
                self.texts.append(text)
            for i, (goal_x, goal_y) in enumerate(self.goals_pos):
                plt.text(goal_x, goal_y, i, color='black', ha='center', va='center')

        plt.imshow(color_map[mapp], animated=True)
        plt.show()
        plt.pause(0.5)

    def close(self, save=False):
        plt.close()
        del self.fig
