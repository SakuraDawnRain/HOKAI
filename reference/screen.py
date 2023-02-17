import cv2
import os
import subprocess

os.chdir("C:\\Program Files (x86)\\MuMu\\emulator\\nemu\\vmonitor\\bin")
os.system("adb_server.exe kill-server")
os.system("adb_server.exe connect 127.0.0.1:7555")

def attack():
    os.system('adb -s emulator-5554  shell input tap {} {}'.format(1390, 765))
    print("attack")

while(True):
    os.system("adb_server shell screencap /sdcard/screen.png")
    os.system("adb_server pull /sdcard/screen.png  C:\\code\\HOKAI\\screen.png")
    frame = cv2.imread("C:\\code\\HOKAI\\screen.png")
    cv2.waitKey(1)
    cv2.imshow("screen", frame)
    attack()
    
