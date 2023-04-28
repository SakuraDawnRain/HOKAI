import scrcpy
import cv2
from adbutils import adb
import os
import sys
import torch
import time
import win32api, win32con, win32gui
from pynput import keyboard
from torchvision.models import resnet18
from torchvision import transforms

from find import find_oppo, find_self
from detect import Detector, check_opponent_state, check_opponent_alive, check_self_alive
from actions import act
from reinforce import PolicyNet, PolicyNetPlus
from data import get_oppo_data, get_self_data, get_processed_map
from pyminitouch import MNTDevice

map_size = 285

# If you already know the device serial
client = scrcpy.Client(device="DEVICE SERIAL")
# You can also pass an ADBClient instance to it
adb.connect("127.0.0.1:7555")
client = scrcpy.Client(device=adb.device_list()[0])

state_dim = 16
# hidden_dim = 256
hidden_dim = 16 # Mini Net
action_dim = 5
learning_rate = 0.01
gamma = 0.98

Net = PolicyNet(state_dim, hidden_dim, action_dim)
# Net.load_state_dict(torch.load("PolicyNetMini"))

# PolicyResNet = resnet18()
# PolicyResNet.fc = torch.nn.Sequential(torch.nn.Linear(in_features=512, out_features=action_dim), torch.nn.Softmax(dim=1))
# PolicyResNet.load_state_dict(torch.load("PolicyResNet"))
# transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])

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

def compression(data):
    result = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(8):
        result[i] = sum(data[i*4:i*4+4])
    return result

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

        
        
        if detect_counter % 10 == 0:
            self_data = compression(get_self_data(map).tolist())
            oppo_data = compression(get_oppo_data(map).tolist())
            model_input = torch.FloatTensor([self_data+oppo_data])
            # state = transforms(map).unsqueeze(0)
            pred = Net(model_input)
            action_dist = torch.distributions.Categorical(pred)
            choice = action_dist.sample()
            
            action = [False, False, False, False, False]
            action[choice] = True
            print(action)
            act(action)

            

            # moving(client, move)

            # print(move)

            # # if move['w'] and not last_move['w']:
            # if move['w']:
            #     move_w(True)
            # elif not move['w'] and last_move['w']:
            #     move_w(False)
            
            # # if move['a'] and not last_move['a']:
            # if move['a']:
            #     move_a(True)
            # elif not move['a'] and last_move['a']:
            #     move_a(False)

            # # if move['s'] and not last_move['s']:
            # if move['s']:
            #     move_s(True)
            # elif not move['s'] and last_move['s']:
            #     move_s(False)

            # # if move['d'] and not last_move['d']:
            # if move['d']:
            #     move_d(True)
            # elif not move['d'] and last_move['d']:
            #     move_d(False)

            # client.control.scroll(275, 705, 100, 0)

            

            
            last_move = move
        
    cv2.waitKey(1)

client.add_listener(scrcpy.EVENT_FRAME, on_frame)
client.start()


