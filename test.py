import cv2
import numpy as np

def find(mask):
    r, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
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
    return find(result)

def find_oppo(test_img):
    (test_B, test_G, test_R) = cv2.split(test_img)
    result = test_G[:]
    result[test_G>150] = 0
    result[test_R<150] = 0
    result[test_B>120] = 0
    result[result>0] = 255
    return find(result)
# result = test_B[test_G<150 and test_R>150]
# result = test_G[test_G<150] = 0
# cv2.imshow("result", result)

# cv2.imshow("B", test_B)
# cv2.imshow("G", test_G)
# cv2.imshow("R", test_R)


self_hero_a = cv2.imread("./avatars/drj.webp")
oppo_hero_a = cv2.imread("./avatars/cwj.webp")
test_img = cv2.imread("./test2.png")

(x, y) = find_self(test_img)
img = cv2.rectangle(test_img, (x+15, y+15), (x-15, y-15), (100, 255, 100), 3)
(x, y) = find_oppo(test_img)
img = cv2.rectangle(img, (x+15, y+15), (x-15, y-15), (100, 100, 255), 3)
cv2.imshow("result", img)

cv2.waitKey(100000)

