import numpy as np
import configs_D3QTP


# used for prioritized experience replay
class SumTree:
    def __init__(self, capacity: int):
        layer = 1
        while 2 ** (layer - 1) < capacity:
            layer += 1
        assert 2 ** (layer - 1) == capacity, 'capacity only allow n**2 size'
        self.layer = layer
        self.tree = np.zeros(2 ** layer - 1, dtype=np.float64)
        self.capacity = capacity
        self.size = 0

    def sum(self):
        assert np.sum(self.tree[-self.capacity:]) - self.tree[0] < 0.1, 'sum is {} but root is {}'.format(
            np.sum(self.tree[-self.capacity:]), self.tree[0])
        return self.tree[0]

    def __getitem__(self, idx: int):
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity - 1 + idx]

    def batch_sample(self, batch_size: int):
        p_sum = self.tree[0]
        interval = p_sum / batch_size

        prefixsums = np.arange(0, p_sum, interval, dtype=np.float64) + np.random.uniform(0, interval, batch_size)

        idxes = np.zeros(batch_size, dtype=int)
        for _ in range(self.layer - 1):
            nodes = self.tree[idxes * 2 + 1]
            idxes = np.where(prefixsums < nodes, idxes * 2 + 1, idxes * 2 + 2)
            prefixsums = np.where(idxes % 2 == 0, prefixsums - self.tree[idxes - 1], prefixsums)

        priorities = self.tree[idxes]
        idxes -= self.capacity - 1

        assert np.all(priorities > 0), 'idx: {}, priority: {}'.format(idxes, priorities)
        assert np.all(idxes >= 0) and np.all(idxes < self.capacity)

        return idxes, priorities

    def batch_update(self, idxes: np.ndarray, priorities: np.ndarray):
        idxes += self.capacity - 1
        self.tree[idxes] = priorities

        for _ in range(self.layer - 1):
            idxes = (idxes - 1) // 2
            idxes = np.unique(idxes)
            self.tree[idxes] = self.tree[2 * idxes + 1] + self.tree[2 * idxes + 2]

        # check
        assert np.sum(self.tree[-self.capacity:]) - self.tree[0] < 0.1, 'sum is {} but root is {}'.format(
            np.sum(self.tree[-self.capacity:]), self.tree[0])


class LocalBuffer:
    # limit the attributes of the object
    __slots__ = (
        'actor_id', 'map_len', 'map_wid', 'num_agents', 'obs_buf', 'obs_agent_buf', 'act_buf', 'rew_buf', 'hid_buf',
        'q_buf', 'capacity',
        'size', 'done')

    def __init__(self, actor_id: int, num_agents: int, map_len: int, map_wid: int, init_obs: np.ndarray,
                 init_obs_agent: np.ndarray,
                 capacity: int = configs_D3QTP.max_episode_length,
                 obs_shape=configs_D3QTP.obs_shape, hidden_dim=configs_D3QTP.hidden_dim,
                 action_dim=configs_D3QTP.action_dim, obs_agent_shape=configs_D3QTP.agent_state_shape):
        """
        buffer for each episode
        """
        self.actor_id = actor_id
        self.num_agents = num_agents
        self.map_len = map_len
        self.map_wid = map_wid
        self.obs_buf = np.zeros((capacity + 1, num_agents, *obs_shape), dtype=bool)
        self.act_buf = np.zeros((capacity), dtype=np.uint8)
        self.rew_buf = np.zeros((capacity), dtype=np.float16)
        self.hid_buf = np.zeros((capacity, num_agents, hidden_dim), dtype=np.float32)
        self.q_buf = np.zeros((capacity + 1, action_dim), dtype=np.float32)
        self.capacity = capacity
        self.size = 0
        self.obs_agent_buf = np.zeros((capacity + 1, num_agents, *obs_agent_shape), dtype=bool)
        self.obs_buf[0] = init_obs
        self.obs_agent_buf[0] = init_obs_agent

    def __len__(self):
        return self.size

    def add(self, q_val: np.ndarray, action: int, reward: float, next_obs: np.ndarray, next_obs_agent: np.ndarray,
            hidden: np.ndarray):
        assert self.size < self.capacity

        self.act_buf[self.size] = action
        self.rew_buf[self.size] = reward
        self.obs_buf[self.size + 1] = next_obs
        self.obs_agent_buf[self.size + 1] = next_obs_agent
        self.q_buf[self.size] = q_val
        self.hid_buf[self.size] = hidden
        self.size += 1

    def finish(self, last_q_val=None):
        # last q value is None if done
        if last_q_val is None:
            done = True
        else:
            done = False
            self.q_buf[self.size] = last_q_val

        self.obs_buf = self.obs_buf[:self.size + 1]
        self.obs_agent_buf = self.obs_agent_buf[:self.size + 1]
        self.act_buf = self.act_buf[:self.size]
        self.rew_buf = self.rew_buf[:self.size]
        self.hid_buf = self.hid_buf[:self.size]
        self.q_buf = self.q_buf[:self.size + 1]

        # calculate td errors for prioritized experience replay
        td_errors = np.zeros(self.capacity, dtype=np.float32)
        q_max = np.max(self.q_buf[:self.size], axis=1)
        ret = self.rew_buf.tolist() + [0 for _ in range(configs_D3QTP.forward_steps - 1)]
        reward = np.convolve(ret, [0.95 ** (configs_D3QTP.forward_steps - 1 - i) for i in
                                   range(configs_D3QTP.forward_steps)],
                             'valid') + q_max
        q_val = self.q_buf[np.arange(self.size), self.act_buf]
        td_errors[:self.size] = np.abs(reward - q_val).clip(1e-4)

        return self.actor_id, self.num_agents, self.map_len, self.map_wid, self.obs_buf, self.act_buf, self.rew_buf, self.hid_buf, td_errors, done, self.size, self.obs_agent_buf
