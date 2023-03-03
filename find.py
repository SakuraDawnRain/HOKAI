import cv2
import numpy as np

def find(mask):
    r, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c:cv2.arcLength(c, False), reverse=True)
    M = cv2.moments(contours[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)

def find_self(test_img):
    (test_B, test_G, test_R) = cv2.split(test_img)
    result = test_G[:]
    result[test_G<150] = 0
    result[test_R>150] = 0
    result[test_B>120] = 0
    result[result>0] = 255
    cv2.imshow("self", result)
    return find(result)

def find_oppo(test_img):
    (test_B, test_G, test_R) = cv2.split(test_img)
    result = test_G[:]
    result[test_G>150] = 0
    result[test_R<150] = 0
    result[test_B>120] = 0
    result[result>0] = 255
    cv2.imshow("oppo", result)
    return find(result)

