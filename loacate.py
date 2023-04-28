import win32gui
from actions import LeftClick, RightClick
import time

# DOWN
# pos = (640, 600)

# UP
# pos = (640, 100)

# LEFT 
# pos = (400, 420)

# RIGHT
# pos = (900, 420)

# Attack
pos = (1100, 650)

time.sleep(5)

for i in range(5):
    LeftClick(pos)
    time.sleep(1)
