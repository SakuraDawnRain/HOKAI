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
from torchvision import transforms
import random

from find import find_oppo, find_self
from detect import Detector, check_opponent_state, check_opponent_alive, check_self_alive
from actions import act
from reinforce import PolicyNet, PolicyNetPlus, PolicCNN
from data import get_oppo_data, get_self_data, get_processed_map, get_pos, get_processed_centermap, compression, center_oppo_count
from reinforce import REINFORCE

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

Net = PolicCNN(in_channels=1, action_dim=5)
Net.load_state_dict(torch.load("policycnns//PolicCNN2.0.pth"))
transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((28, 28))])
agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma, device)
agent.policy_net = Net

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

global user_def_reward
user_def_reward = 0

global steps
steps = 0

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
        global detect_counter
        detect_counter += 1

        global transition_dict
        global user_def_reward

        centermap = map[map_size//4:map_size-(map_size//4), map_size//4:map_size-(map_size//4)]
        cv2.imshow("center", centermap)
        oppo_count = center_oppo_count(centermap)
        centermap = get_processed_centermap(centermap)
        cv2.imshow("processed", centermap)
        
        if detect_counter % 30 == 0:
            self_data = get_self_data(map).tolist()
            oppo_data = get_oppo_data(map).tolist()
            

            self_data = compression(get_self_data(map).tolist())
            oppo_data = compression(get_oppo_data(map).tolist())
            model_input = torch.FloatTensor([self_data+oppo_data])
            # state = transforms(map).unsqueeze(0)
            self_pos = get_pos(self_data)
            action = [False, False, False, False, False]
            # print(self_pos)
            if self_pos>4:
                choice = 0 if random.random()<0.5 else 3
                action[choice] = True
                act(action)
            elif self_pos<4:
                choice = 1 if random.random()<0.5 else 2
                action[choice] = True
                act(action)
            else:
                attack_reward = 0
                model_input = transforms(centermap)
                choice = agent.take_action(model_input)
                if oppo_count>300:
                    if choice==4:
                        attack_reward += 100
                        print("give attack reward")
                action[choice] = True
                act(action)
                self_state = check_self_alive(frame)
                oppo_state = check_opponent_alive(frame)
                reward = cal_score(self_state, oppo_state)

                transition_dict["actions"].append(choice)
                transition_dict["states"].append(centermap)
                transition_dict["rewards"].append(reward+user_def_reward+attack_reward)
                agent.update(transition_dict)
                # print("user def reward:", user_def_reward)
                user_def_reward = 0

                global steps
                steps += 1
                print("step", steps)

            

        
    cv2.waitKey(1)

client.add_listener(scrcpy.EVENT_FRAME, on_frame)
client.start(threaded=True)

def on_press(key):
    global user_def_reward
    try:
        # print('alphanumeric key {0} pressed'.format(key.char))
        if isinstance(key, keyboard.KeyCode):
            if key.char=='c':
                user_def_reward += 50
                print("correct")
            if key.char=='n':
                user_def_reward += -50
                print("not correct")
    except AttributeError:
        print('special key {0} pressed'.format(key))

def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        
        torch.save(Net.state_dict(), "PolicCNN.pth")
        print("model saved")
        return False

# Collect events until released
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
