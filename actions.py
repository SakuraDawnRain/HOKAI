import scrcpy
import time
import win32api, win32con, win32gui

# Manual Move
FrameClass = "Qt5152QWindowIcon"
FrameTitle = "王者荣耀 - MuMu模拟器"
hwnd = win32gui.FindWindow(FrameClass, FrameTitle)

def RightClick(pos):
    cx, cy = pos
    long_position = win32api.MAKELONG(cx, cy)
    win32api.SendMessage(hwnd, win32con.WM_RBUTTONDOWN, win32con.MK_RBUTTON, long_position)
    win32api.SendMessage(hwnd, win32con.WM_RBUTTONUP, win32con.MK_RBUTTON, long_position)

def LeftClick(pos):
    cx, cy = pos
    long_position = win32api.MAKELONG(cx, cy)
    win32api.SendMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, long_position)
    win32api.SendMessage(hwnd, win32con.WM_LBUTTONUP, win32con.MK_LBUTTON, long_position)

# UP
w_pos = (640, 320)

# LEFT 
a_pos = (540, 420)

# DOWN
s_pos = (640, 520)

# RIGHT
d_pos = (740, 420)

# Attack
att_pos = (1100, 650)

# Up - W
def move_w(on):
    if on:
        win32api.keybd_event(87,0,0,0)
    else:
        win32api.keybd_event(87,0,win32con.KEYEVENTF_KEYUP,0)

# Left - A
def move_a(on):
    if on:
        win32api.keybd_event(65,0,0,0)
    else:
        win32api.keybd_event(65,0,win32con.KEYEVENTF_KEYUP,0)

# Down - S
def move_s(on):
    if on:
        win32api.keybd_event(83,0,0,0)
    else:
        win32api.keybd_event(83,0,win32con.KEYEVENTF_KEYUP,0)
    
# Right - D
def move_d(on):
    if on:
        win32api.keybd_event(68,0,0,0)
    else:
        win32api.keybd_event(68,0,win32con.KEYEVENTF_KEYUP,0)  

# Act
def act(action):
    if action[4]:
        LeftClick(att_pos)
    elif action[0]:
        RightClick(w_pos)
    elif action[1]:
        RightClick(a_pos)
    elif action[2]:
        RightClick(s_pos)
    elif action[3]:
        RightClick(d_pos)

# no use
def moving(move):
    if len(move.keys()) != 4:
        pass
    elif move['w']:
        RightClick(w_pos)
    elif move['a']:
        RightClick(a_pos)
    elif move['s']:
        RightClick(s_pos)
    elif move['d']:
        RightClick(d_pos)

def attack(client):
    # Mousedown
    client.control.touch(1390, 765, scrcpy.ACTION_DOWN)
    # Mouseup
    client.control.touch(1390, 765, scrcpy.ACTION_UP)
    # print("attack")

def skill1(client):
    # Mousedown
    client.control.touch(1123, 791, scrcpy.ACTION_DOWN)
    # Mouseup
    client.control.touch(1123, 791, scrcpy.ACTION_UP)
    # print("attack")


# Auto Actions

def auto_buy(client):
    # Mousedown
    client.control.touch(162, 361, scrcpy.ACTION_DOWN)
    # Mouseup
    client.control.touch(162, 361, scrcpy.ACTION_UP)

def upgrade_skill1(client):
    # Mousedown
    client.control.touch(1007, 708, scrcpy.ACTION_DOWN)
    # Mouseup
    client.control.touch(1007, 708, scrcpy.ACTION_UP)

def upgrade_skill2(client):
    # Mousedown
    client.control.touch(1106, 526, scrcpy.ACTION_DOWN)
    # Mouseup
    client.control.touch(1106, 526, scrcpy.ACTION_UP)

def upgrade_skill3(client):
    # Mousedown
    client.control.touch(1272, 432, scrcpy.ACTION_DOWN)
    # Mouseup
    client.control.touch(1272, 432, scrcpy.ACTION_UP)

def auto_upgrade(client):
    upgrade_skill3(client)
    upgrade_skill1(client)
    upgrade_skill2(client)
