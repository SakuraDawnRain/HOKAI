import numpy as np
import cv2
import os

img_path = "data\\raw\\054.png"
img_size = 285
patch_size = 32

name = img_path.split('\\')[-1][:3]
# pos_list = [(patch_size//4)*i for i in range((img_size//(patch_size//4))-3)]

pos_list = [(img_size-1-patch_size*i, patch_size*i) for i in range((img_size//patch_size)-1)]
pos_list += [(img_size-1-patch_size*i-patch_size, patch_size*i) for i in range((img_size//patch_size)-2)]
pos_list += [(img_size-1-patch_size*i, patch_size*i+patch_size) for i in range((img_size//patch_size)-2)]
pos_list.append((img_size-1-patch_size*5, patch_size*3))
pos_list.append((img_size-1-patch_size*3, patch_size*5))
# print(pos_list)

img = cv2.imread(img_path)
img = cv2.resize(img, (img_size, img_size))
# cv2.imshow("img", img)
# patches = []
count = 0
for (x, y) in pos_list:
    patch = img[x-patch_size:x, y:y+patch_size, :]
    cv2.imwrite(os.path.join("data", "temp", name+"_"+str(count)+".png"), patch)
    count += 1
        # patches.append(img[x:x+patch_size, y:y+patch_size, :])

