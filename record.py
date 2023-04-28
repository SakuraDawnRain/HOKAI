import scrcpy
import cv2
from adbutils import adb
from pynput import keyboard
import numpy as np

from find import find_oppo, find_self
from detect import Detector, check_opponent_state, check_opponent_alive, check_self_alive
from actions import act
from data import get_oppo_data, get_self_data, make_data, save_data

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

# If you already know the device serial
client = scrcpy.Client(device="DEVICE SERIAL")
# You can also pass an ADBClient instance to it

adb.connect("127.0.0.1:7555")
client = scrcpy.Client(device=adb.device_list()[0])

def cal_score(self_state, oppo_state):
    global self_alive
    global oppo_alive
    global score
    if self_alive and not self_state:
        print("you are dead")
        score = score - 50
        print("current score: ", score)
        self_alive = self_state
        return -50

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
    if frame is not None:
        # frame is an bgr numpy ndarray (cv2' default format)
        global detect_counter

        map = frame[0:map_size, 0:map_size]
        self_data = get_self_data(map).tolist()
        oppo_data = get_oppo_data(map).tolist()
        
        self_state = check_self_alive(frame)
        oppo_state = check_opponent_alive(frame)
            

        detect_counter += 1
        state = self_data + oppo_data
        action = [w, a, s, d, att]
        # print(action)
        
        reward = cal_score(self_state, oppo_state)
        data = make_data(state, action, reward)
        global datalist
        datalist.append(data)

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
        global datalist
        global filename
        save_data(datalist, filename)
        return False

# Collect events until released
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()


