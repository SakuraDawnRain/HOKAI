import scrcpy
import cv2
from adbutils import adb
from pynput import keyboard
import numpy as np
import torch
import random

from find import find_oppo, find_self
from detect import Detector, check_opponent_state, check_opponent_alive, check_self_alive
from actions import act
from data import get_oppo_data, get_self_data, make_data, save_data, get_processed_centermap, get_pos
from reinforce import PolicyNet, PolicyNetPlus, PolicCNN
# from torchvision.models import resnet18
from torchvision import transforms

map_size = 285

global detect_counter
detect_counter = 0

# state list
global self_alive
global oppo_alive

self_alive = True
oppo_alive = True

# action list
global w
global a
global s
global d
global att

w = False
a = False
s = False
d = False
att = False

# score
global score
score = 0

global datalist
datalist = []

global filename
filename = "01.json"

global steps
steps = 0

# If you already know the device serial
client = scrcpy.Client(device="DEVICE SERIAL")
# You can also pass an ADBClient instance to it

adb.connect("127.0.0.1:7555")
client = scrcpy.Client(device=adb.device_list()[0])

state_dim = 16
hidden_dim = 16 # Mini Net
action_dim = 5
learning_rate = 0.1
gamma = 0.98
device = 'cpu'

Net = PolicCNN(in_channels=1, action_dim=5)
transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((28, 28))])
# Net = PolicyNet(state_dim, hidden_dim, action_dim)
# Net.load_state_dict(torch.load("PolicyNetMini"))

# Net = PolicyNetPlus(state_dim, hidden_dim, action_dim)
# Net.load_state_dict(torch.load("PolicyNetPlus"))

# PolicyResNet = resnet18()
# PolicyResNet.fc = torch.nn.Sequential(torch.nn.Linear(in_features=512, out_features=action_dim), torch.nn.Softmax(dim=1))

# PolicyResNet.load_state_dict(torch.load("PolicyResNet"))

Loss = torch.nn.BCELoss()
optimizer = torch.optim.Adam(Net.parameters(),lr=learning_rate)

def compression(data):
    result = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(8):
        result[i] = sum(data[i*4:i*4+4])
    return result

def on_frame(frame):
    if frame is not None:
        # frame is an bgr numpy ndarray (cv2' default format)
        global self_pos 
        global detect_counter
        global steps
        detect_counter += 1

        map = frame[0:map_size, 0:map_size]
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
                optimizer.zero_grad()
                pred = Net(transforms(centermap))
        
                action = torch.FloatTensor([[w, a, s, d, att]])
                loss = Loss(pred, action)
                loss.backward()
                optimizer.step()
                steps += 1
                print("step:", steps)
                print("loss:", loss.item())

    cv2.waitKey(1)


client.add_listener(scrcpy.EVENT_FRAME, on_frame)
client.start(threaded=True)

sleep_time = 0.2

def on_press(key):
    global att
    global w
    global a
    global s
    global d
    try:
        # print('alphanumeric key {0} pressed'.format(key.char))
        if isinstance(key, keyboard.KeyCode):
            if key.char=='j':
                att = True
            elif key.char=='w':
                w = True
            elif key.char=='a':
                a = True
            elif key.char=='s':
                s = True
            elif key.char=='d':
                d = True

    except AttributeError:
        print('special key {0} pressed'.format(key))

def on_release(key):
    # print('{0} released'.format(key))
    global att
    global w
    global a
    global s
    global d
    if isinstance(key, keyboard.KeyCode):
        if key.char=='j':
            att = False
        elif key.char=='w':
            w = False
        elif key.char=='a':
            a = False
        elif key.char=='s':
            s = False
        elif key.char=='d':
            d = False
        
    if key == keyboard.Key.esc:
        # Stop listener
        
        torch.save(Net.state_dict(), "PolicCNN.pth")
        print("model saved")
        return False

# Collect events until released
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()


