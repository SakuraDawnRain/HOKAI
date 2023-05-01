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
import random

from find import find_oppo, find_self
from detect import Detector, check_opponent_state, check_opponent_alive, check_self_alive
from actions import act
from reinforce import PolicyNet, PolicyNetPlus, PolicCNN
from data import get_oppo_data, get_self_data, get_processed_map, get_pos, get_processed_centermap

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

Net = PolicCNN(in_channels=1, action_dim=5)
Net.load_state_dict(torch.load("PolicCNN.pth"))
transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((28, 28))])

# Net = PolicyNet(state_dim, hidden_dim, action_dim)
# Net.load_state_dict(torch.load("PolicyNetMini"))

# PolicyResNet = resnet18()
# PolicyResNet.fc = torch.nn.Sequential(torch.nn.Linear(in_features=512, out_features=action_dim), torch.nn.Softmax(dim=1))
# PolicyResNet.load_state_dict(torch.load("PolicyResNet"))

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
        
        global self_pos 
        global detect_counter
        detect_counter += 1
        centermap = map[map_size//4:map_size-(map_size//4), map_size//4:map_size-(map_size//4)]
        cv2.imshow("center", centermap)
        centermap = get_processed_centermap(centermap)
        cv2.imshow("processed", centermap)

        # input_map = cv2.resize(centermap, (8, 8))
        
        if detect_counter % 4 == 0:
            self_data = compression(get_self_data(map).tolist())
            oppo_data = compression(get_oppo_data(map).tolist())
            model_input = torch.FloatTensor([self_data+oppo_data])
            # state = transforms(map).unsqueeze(0)
            self_pos = get_pos(self_data)
            action = [False, False, False, False, False]
            print(self_pos)
            if self_pos>4:
                choice = 0 if random.random()<0.5 else 3
            else:
                pred = Net(transforms(centermap))
                print(pred)
                action_dist = torch.distributions.Categorical(pred)
                choice = action_dist.sample()
            action[choice] = True
            
            act(action)
        
    cv2.waitKey(1)

client.add_listener(scrcpy.EVENT_FRAME, on_frame)
client.start()


