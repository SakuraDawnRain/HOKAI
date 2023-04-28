import scrcpy
import cv2
from adbutils import adb
import os
import sys
import torch
import time
import win32api, win32con, win32gui
from pynput import keyboard
import numpy as np

from find import find_oppo, find_self
from detect import Detector, check_opponent_state, check_opponent_alive, check_self_alive
from actions import act
from reinforce import PolicyNet, REINFORCE
from data import get_oppo_data, get_self_data
from pyminitouch import MNTDevice

map_size = 285

# If you already know the device serial
client = scrcpy.Client(device="DEVICE SERIAL")
# You can also pass an ADBClient instance to it
adb.connect("127.0.0.1:7555")
client = scrcpy.Client(device=adb.device_list()[0])

state_dim = 64
hidden_dim = 128
action_dim = 5
learning_rate = 0.001
gamma = 0.98
device = 'cpu'

net = PolicyNet(state_dim, hidden_dim, action_dim)
net.load_state_dict(torch.load("PolicyNetv1.1"))
agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma, device)
agent.policy_net = net

# mask = cv2.imread("mask.png")
# mask = cv2.resize(mask, (map_size, map_size))
global detect_counter
detect_counter = 0

global move
move = {
    "w" : False,
    "a" : False,
    "s" : False,
    "d" : False
}

global last_move
last_move = {
    "w" : False,
    "a" : False,
    "s" : False,
    "d" : False
}

# state list
global self_alive
global oppo_alive

self_alive = True
oppo_alive = True

global score
score = 0

global transition_dict
transition_dict = {
                    'states': [],
                    'actions': [],
                    'rewards': [],
                }

def cal_score(self_state, oppo_state):
    global self_alive
    global oppo_alive
    global score
    if self_alive and not self_state:
        print("you are dead")
        score = score - 500
        print("current score: ", score)
        self_alive = self_state
        return -500

    if oppo_alive and not oppo_state:
        print("kill enemy")
        score = score + 100
        print("current score: ", score)
        oppo_alive = oppo_state
        return 100

    if not self_alive and self_state:
        self_alive = self_state

    if not oppo_alive and oppo_state:
        oppo_alive = oppo_state

    return 0



def on_frame(frame):
    # If you set non-blocking (default) in constructor, the frame event receiver 
    # may receive None to avoid blocking event.
    if frame is not None:
        # frame is an bgr numpy ndarray (cv2' default format)
        map = frame[0:map_size, 0:map_size]
        
        cv2.imshow("map", map)
        global self_pos 
        global enemy_pos 
        global both_pos
        global detect_counter
        detect_counter += 1

        global transition_dict
        
        if detect_counter % 10 == 0:
            self_data = get_self_data(map).tolist()
            oppo_data = get_oppo_data(map).tolist()
            model_input = torch.FloatTensor([self_data+oppo_data])
            model_output = agent.take_action(model_input)
            
            action = [False, False, False, False, False]
            action[model_output] = True
            # print(action)
            act(action)

            self_state = check_self_alive(frame)
            oppo_state = check_opponent_alive(frame)
            reward = cal_score(self_state, oppo_state)
            transition_dict["actions"].append(model_output)
            transition_dict["states"].append(self_data+oppo_data)
            transition_dict["rewards"].append(reward)
            agent.update(transition_dict)
        
        if detect_counter % 2000 == 0:
            print("model_saved")
            agentpath = "agentv0_"+str(detect_counter)+".pth"
            torch.save(agent.policy_net.state_dict(), agentpath)

        
    cv2.waitKey(1)

client.add_listener(scrcpy.EVENT_FRAME, on_frame)
client.start()


