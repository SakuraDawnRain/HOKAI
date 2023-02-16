import scrcpy
import cv2
from adbutils import adb

# If you already know the device serial
client = scrcpy.Client(device="DEVICE SERIAL")
# You can also pass an ADBClient instance to it

adb.connect("127.0.0.1:7555")
client = scrcpy.Client(device=adb.device_list()[0])

def on_frame(frame):
    # If you set non-blocking (default) in constructor, the frame event receiver 
    # may receive None to avoid blocking event.
    if frame is not None:
        # frame is an bgr numpy ndarray (cv2' default format)
        map = frame[0:285, 0:285]
        game = cv2.resize(frame, (640, 360))
        cv2.imshow("HOK", game)
        cv2.imshow("map", map)
        
    cv2.waitKey(10)

client.add_listener(scrcpy.EVENT_FRAME, on_frame)
client.start()