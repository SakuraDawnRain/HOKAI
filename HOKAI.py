import scrcpy
import cv2
from adbutils import adb
import os
from find import find_oppo, find_self

# If you already know the device serial
client = scrcpy.Client(device="DEVICE SERIAL")
# You can also pass an ADBClient instance to it

adb.connect("127.0.0.1:7555")
client = scrcpy.Client(device=adb.device_list()[0])

def attack():
    # Mousedown
    client.control.touch(1390, 765, scrcpy.ACTION_DOWN)
    # Mouseup
    client.control.touch(1390, 765, scrcpy.ACTION_UP)
    print("attack")

def on_frame(frame):
    # If you set non-blocking (default) in constructor, the frame event receiver 
    # may receive None to avoid blocking event.
    if frame is not None:
        # frame is an bgr numpy ndarray (cv2' default format)
        game = cv2.resize(frame, (640, 360))
        cv2.imshow("HOK", game)

        map = frame[0:285, 0:285]
        try:
            (self_x, self_y) = find_self(map)
            (oppo_x, oppo_y) = find_oppo(map)
            map = cv2.rectangle(map, (self_x+15, self_y+15), (self_x-15, self_y-15), (100, 255, 100), 3)
            map = cv2.rectangle(map, (oppo_x+15, oppo_y+15), (oppo_x-15, oppo_y-15), (100, 100, 255), 3)
            print((self_x-oppo_x)**2 + (self_y-oppo_y)**2)
            if (self_x-oppo_x)**2 + (self_y-oppo_y)**2 <1000:
                attack()
        except:
            print("error")
        cv2.imshow("map", map)

        # attack()
        
    cv2.waitKey(10)

client.add_listener(scrcpy.EVENT_FRAME, on_frame)
client.start()