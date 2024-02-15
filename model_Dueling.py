import numpy as np
import torch
import torch.nn as nn  # neural network which is a module that can construct the neural network
import torch.nn.functional as F  # most layers in the neural network have a corresponding function in "functional"
from torch.cuda.amp import autocast  # automatically mix the accuracy
import configs_D3QTP


# create a block with two convolutional layers, which is used three times in our model
class ResBlock(nn.Module):  # inherit from nn.Module
    def __init__(self, channel):
        super().__init__()  # inherit __init__ attribute from class nn.Module
        # the input channels, the out channels, the kernel size(3*3), stride(1), padding(1)
        self.block1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.block2 = nn.Conv2d(channel, channel, 3, 1, 1)

    def forward(self, x):  # forward propagation function
        identity = x
        x = self.block1(x)
        x = F.relu(x)
        x = self.block2(x)
        x += identity  # x=x+identity
        x = F.relu(x)

        return x


class FCBlock(nn.Module):
    def __init__(self, agent_shape1, agent_repr_shape1):
        super().__init__()
        self.fc1 = nn.Linear(agent_shape1, agent_repr_shape1)  # the input shape, the output shape

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        return x


class ConcatBlock(nn.Module):
    def __init__(self, obs_all_channel):
        super().__init__()
        self.fc2 = nn.Linear(obs_all_channel, obs_all_channel)
        self.fc3 = nn.Linear(obs_all_channel, obs_all_channel)

    def forward(self, x):
        x = torch.cat(x, 1)
        identity = x
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x += identity
        x = F.relu(x)

        return x


class Network(nn.Module):
    def __init__(self, input_shape=configs_D3QTP.obs_shape, cnn_channel=configs_D3QTP.cnn_channel,
                 hidden_dim=configs_D3QTP.hidden_dim, agent_state_shape=configs_D3QTP.agent_state_shape[0],
                 agent_repre_shape=configs_D3QTP.Agent_REPR_SIZE, agent_obs_all_dim=configs_D3QTP.obs_all_size,
                 latent_dim=configs_D3QTP.latent_dim):

        super().__init__()

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.agent_state_shape = agent_state_shape
        self.agent_representation_shape = agent_repre_shape
        self.agent_obs_all_dim = agent_obs_all_dim

        # A sequential container.
        # Modules will be added to it in the order they are passed in the constructor.
        # Alternatively, an ordered dict of modules can be passed in the constructor.
        # the number of input channels: 8, the number of output channels: 128, kernel_size: 3, stride: 1.
        # equal to nn.ReLU (inplace=True) change the original input.
        # nn.Flatten(start_dim=1, end_dim=-1) conduct flattening from the second dimension.
        self.obs_encoder = nn.Sequential(nn.Conv2d(self.input_shape[0], cnn_channel, 3, 1),
                                         nn.ReLU(True),
                                         ResBlock(cnn_channel),
                                         ResBlock(cnn_channel),
                                         nn.Conv2d(cnn_channel, configs_D3QTP.seq_len, 1, 1),
                                         nn.ReLU(True),
                                         nn.Flatten())

        # latent_dim= 16 * 5 * 5=400 (input_dim), hidden_dim = 256 (hidden_dim)
        self.recurrent = nn.GRUCell(self.latent_dim, self.hidden_dim)
        self.agent_state_encoder = FCBlock(self.agent_state_shape, self.agent_representation_shape)
        self.concatenation = ConcatBlock(self.agent_obs_all_dim)
        # dueling q structure
        # construct the fully connected layer，the parameters are respectively in_features and out_features
        self.adv = nn.Linear(self.agent_obs_all_dim, 6)  # advantages
        self.state = nn.Linear(self.agent_obs_all_dim, 1)  # state-value
        self.hidden = None

        for _, m in self.named_modules():  # return the iterators of all modules: _module name and m module itself
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):  # determine whether m is a known module, yes-True
                nn.init.xavier_uniform_(m.weight)  # xavier initialization makes var of input and output equal
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # pad the tensor m.bias with 0

    @torch.no_grad()  # don't compute gradient and don't conduct back propagation
    def step(self, obs, obs_agent):
        # print(obs.shape)
        # print(obs_agent.shape)
        latent = self.obs_encoder(obs)  # pass obs into observation encoder module and latent is the encoded observation
        obs_agent = np.squeeze(obs_agent, 2)
        agent_state = self.agent_state_encoder(obs_agent)

        if self.hidden is None:
            self.hidden = self.recurrent(latent)
        else:
            self.hidden = self.recurrent(latent, self.hidden)

        # print('The shape of step_agent_state{}'.format(agent_state.shape))
        # print('The shape of step_hidden{}'.format(self.hidden.shape))
        # concatenation
        obs_final = self.concatenation([agent_state, self.hidden])

        adv_val = self.adv(obs_final)
        state_val = self.state(obs_final)
        q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)
        actions = torch.argmax(q_val, 1).tolist()  # return the index of the maximum in the second dimension

        return actions, q_val.numpy(), self.hidden.numpy()

    def reset(self):
        self.hidden = None

    # implement half-precise to save memory and accelerate training
    @autocast()
    def forward(self, obs, obs_agent, steps, hidden):
        max_steps = obs.size(1)
        num_agents = configs_D3QTP.num_agents
        # print('The shape of agent_state{}'.format(obs_agent.shape))
        obs_agent = obs_agent.squeeze(3)
        obs_agent = obs_agent.contiguous().view(-1, self.agent_state_shape)
        agent_state = self.agent_state_encoder(obs_agent)
        agent_state = agent_state.view(configs_D3QTP.batch_size, num_agents, -1)
        # transpose the second and third dimension of obs
        # print('The shape of observation{}'.format(obs.shape))
        # print('The shape of agent_state{}'.format(agent_state.shape))
        obs = obs.transpose(1, 2)
        obs = obs.contiguous().view(-1, *self.input_shape)  # view equals to reshape, -1 denotes an uncertain row number
        latent = self.obs_encoder(obs)
        latent = latent.view(configs_D3QTP.batch_size * num_agents, max_steps, self.latent_dim).transpose(0, 1)

        agent_state_buffer = []
        agent_state_buffer.append(agent_state[:, 0])  # x[:,n] take the nth data of all sets
        agent_state_buffer = torch.stack(agent_state_buffer).transpose(0, 1)
        agent_state = agent_state_buffer.squeeze(1)
        # print('The shape of agent_state_buffer{}'.format(agent_state_buffer.shape))

        hidden_buffer = []
        # print("Hidden shape before loop:", hidden.shape)
        # print("Latent shape:", latent.shape)
        for i in range(max_steps):
            # print("Latent[i] shape:", latent[i].shape)
            # print("Hidden shape after loop:", hidden.shape)
            hidden = self.recurrent(latent[i], hidden)
            hidden = hidden.view(configs_D3QTP.batch_size, num_agents, self.hidden_dim)
            hidden_buffer.append(hidden[:, 0])  # x[:,n] take the nth data of all sets
            hidden = hidden.view(configs_D3QTP.batch_size * num_agents, self.hidden_dim)

        # hidden_buffer shape: max_steps * batch_size * self.hidden_dim
        # stack a sequence of matrices into a matrix according to time series
        # after transpose, the hidden_buffer shape becomes batch_size * max_steps * self.hidden_dim
        hidden_buffer = torch.stack(hidden_buffer).transpose(0, 1)
        # print('The shape of hidden_buffer{}'.format(hidden_buffer.shape))

        # hidden shape: batch_size (192) x self.hidden_dim (256)
        # torch.arange() returns a one dimension tensor whose elements are taken every step intervals from [start,end)
        hidden = hidden_buffer[torch.arange(configs_D3QTP.batch_size), steps - 1]
        # print('The shape of hidden{}'.format(hidden.shape))

        # print('The shape of agent_state{}'.format(agent_state.shape))
        # print('The shape of hidden{}'.format(hidden.shape))
        obs_final = self.concatenation([agent_state, hidden])

        adv_val = self.adv(obs_final)
        state_val = self.state(obs_final)
        q_val = state_val + adv_val - adv_val.mean(1, keepdim=True)

        return q_val
