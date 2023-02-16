import cv2
import os
import subprocess
import numpy as np

# https://blog.csdn.net/qinye101/article/details/120000500

def cmd(cmdStr: str):
    cmds = cmdStr.split(' ')
    proc = subprocess.Popen(
        cmds,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True
    )
    stdout, stderr = proc.communicate()
    # returncode 代表执行cmd命令报错
    if proc.returncode > 0:
        raise Exception(proc.returncode, stderr)
    return stdout

while(True):
    # cmd("adb_server shell screencap -p /sdcard/screen.png")
    # cmd("adb_server pull /sdcard/screen.png C:/code/HOKAI/screen.png")
    # frame = cv2.imread("C:\\code\\HOKAI\\screen.png")
    # cv2.waitKey(1)
    # cv2.imshow("screen", frame)
    byteImage = cmd('adb shell screencap -p').replace(b'\r\n', b'\n')
    # opencv读取内存图片
    frame = cv2.imdecode(np.asarray(bytearray(byteImage), dtype=np.uint8), cv2.IMREAD_COLOR)
    cv2.imshow("screen", cv2.resize(frame, (360, 640)))
    cv2.waitKey(1)