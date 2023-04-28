import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

from reinforce import REINFORCE
from data import load_data, process_data

state_dim = 64
hidden_dim = 128
action_dim = 5
learning_rate = 0.001
gamma = 0.98
device = 'cpu'

agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma, device)

datalist1 = load_data("datav1\\01.json")
datalist2 = load_data("datav1\\02.json")
datalist = [datalist1, datalist2]

epoch_num = 1

for i in range(epoch_num):
    for record in datalist:
        transition_dict = {
                    'states': [],
                    'actions': [],
                    'rewards': [],
                }
        for data in tqdm(record):
            data = process_data(data)
            state = data['state']
            action = data['action']
            reward = data['reward']
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['rewards'].append(reward+5)
            # print(transition_dict['actions'])
            agent.warmup(transition_dict)

agentpath = "agentv1.pth"
torch.save(agent.policy_net.state_dict(), agentpath)
