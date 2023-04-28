import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from models import GoogleNet

img_size = 285
patch_size = 32


# pos-(x, y) color-(b, g, r)
opponent_death_pos_bgr = [
    ((294, 92), (22, 1, 165)),
    ((341, 92), (20, 0, 168))
]
self_death_pos_bgr = [
    ((671, 12), (57, 61, 164)),
    ((661, 4), (50, 51, 123)),
    ((927, 15), (62, 67, 172)),
    ((940, 3), (46, 51, 119))
]

def check_color_similar(color1, color2, k=50):
    '''
    check if two color is similar
    return whether the b,g,r difference of the two colors are all smaller than k(k is parameter)
    '''
    return abs(color1[0]-color2[0])<k and abs(color1[1]-color2[1])<k and abs(color1[2]-color2[2])<k

def check_opponent_alive(img):
    for (pos, color) in opponent_death_pos_bgr:
        if not check_color_similar(color, img[pos[1]][pos[0]]):
            return True
    return False

def check_self_alive(img):
    for (pos, color) in self_death_pos_bgr:
        if not check_color_similar(color, img[pos[1]][pos[0]]):
            return True
    return False


def check_opponent_state(test_img):
    (test_B, test_G, test_R) = cv2.split(test_img)
    result = test_G[:]
    result[test_G>100] = 0
    result[test_R<100] = 0
    result[test_B>100] = 0
    result[test_R<test_G+test_B] = 0
    result[result>0] = 1
    return result.sum() < 20

def find_self(test_img):
    (test_B, test_G, test_R) = cv2.split(test_img)
    result = test_G[:]
    result[test_G<130] = 0
    result[test_R>130] = 0
    result[test_B>130] = 0
    result[result>0] = 255
    return result.sum() > 200

def find_oppo(test_img):
    (test_B, test_G, test_R) = cv2.split(test_img)
    result = test_G[:]
    result[test_G>100] = 0
    result[test_R<100] = 0
    result[test_B>100] = 0
    result[result>0] = 255
    return result.sum() > 200

class Detector:
    def __init__(self) -> None:
        # self.model = torchvision.models.resnet18()
        # self.model.fc = nn.Linear(in_features=512, out_features=4, bias=True)
        # self.model.load_state_dict(torch.load("ResNet18.pth", map_location=torch.device('cpu')))
        # self.model.eval()
        # self.trans = transforms.Compose([transforms.ToTensor(), transforms.Resize(224)]) 

        self.model = GoogleNet(3)
        self.model.load_state_dict(torch.load("checkpoints\\Inception_v4.pth", map_location=torch.device('cpu')))
        self.trans = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64))])

    def detect(self, img):
        self_pos_list = []
        enem_pos_list = []
        
        # both_pos_list = []
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pos_list = [(img_size-1-patch_size*i, patch_size*i) for i in range((img_size//patch_size)-1)]
        pos_list += [(img_size-1-patch_size*i-patch_size, patch_size*i) for i in range((img_size//patch_size)-2)]
        pos_list += [(img_size-1-patch_size*i, patch_size*i+patch_size) for i in range((img_size//patch_size)-2)]
        pos_list.append((img_size-1-patch_size*5, patch_size*3))
        pos_list.append((img_size-1-patch_size*3, patch_size*5))
        
        avail_map = np.zeros(img.shape)
        # print(len(pos_list))
        for (x, y) in pos_list:
            # print(x-patch_size, y+patch_size)
            avail_map[x-patch_size:x, y:y+patch_size, :] = 255
            patch = img[x-patch_size:x, y:y+patch_size, :]
            pred = self.detect_patch(patch)
            pred = int(pred.argmax(dim=-1))
            # if pred == 3:
            #     both_pos_list.append((x, y))
            self_flag = find_self(patch)
            enem_flag = find_oppo(patch)
            if pred == 2 and enem_flag:
                enem_pos_list.append((x, y))
            elif pred == 1 and self_flag:
                self_pos_list.append((x, y))

        # for x in pos_list:
        
        #     avail_map[img_size-1-x-patch_size:img_size-1-x, x:x+patch_size, :] = 255
        #     for y in pos_list:

        #         patch = img[x:x+patch_size, y:y+patch_size, :]
        #         self_flag = find_self(patch)
        #         enem_flag = find_oppo(patch)
                # print(x, y)
                # print(self_flag, oppo_flag)
                # if x+y>251 and x+y<284:
                # if x+y>240 and x+y<420 and (self_flag>0 or enem_flag>0):
                    # avail_map[x-patch_size:x, y:y+patch_size, :] = 255
                #     count +=1
                #     # print(img.shape)
                #     pred = self.detect_patch(patch)
                #     pred = int(pred.argmax(dim=-1))
                #     if pred == 3:
                #         both_pos_list.append((x, y))
                #     elif pred == 2:
                #         enem_pos_list.append((x, y))
                #     elif pred == 1:
                #         self_pos_list.append((x, y))
        # avail_map[284-patch_size:284, 0:0+patch_size, :] = 255
        # img[avail_map<1] = 0
        # cv2.imshow("avail", img)
        # print("search patch num", count)
        # if len(both_pos_list)>0:
        #     print("find both")
        if len(enem_pos_list)>0:
            print("find enemy")
        if len(self_pos_list)>0:
            print("find self")
        self_pos = np.mean(self_pos_list, axis=0) if len(self_pos_list)>0 else []
        enem_pos = np.mean(enem_pos_list, axis=0) if len(enem_pos_list)>0 else []
        # both_pos = np.mean(both_pos_list, axis=0) if len(both_pos_list)>0 else []
        result =  [self_pos, enem_pos]
        
        return result
                    

    def detect_patch(self, patch):
        patch = self.trans(patch)
        patch = patch.unsqueeze(0)
        with torch.no_grad():
            pred = self.model(patch)
        return pred[0]
    
# pos_list = [0, 32, 64, 96, 128, 160, 192, 224, 256, 288]    
# detector = Detector()
# img = cv2.imread("C:\\code\\HOKAI\\data\\024.png")
# result = detector.detect(img)
# self_pos = result[0]
# enemy_pos = result[1]
            # (self_x, self_y) = find_self(diff)
            # (oppo_x, oppo_y) = find_oppo(diff)
# print((self_pos[0]+15,self_pos[1]+15))
# img = cv2.rectangle(img, (int(self_pos[0]+15),int(self_pos[1]+15)), (int(self_pos[0]-15),int(self_pos[1]-15)), (100, 255, 100), 3)
# img = cv2.rectangle(img, (int(enemy_pos[0]+15),int(enemy_pos[1]+15)), (int(enemy_pos[0]-15),int(enemy_pos[1]-15)), (100, 255, 100), 3)
# img = cv2.rectangle(img, nenemy_pos+15, nenemy_pos-15, (100, 100, 255), 3)
# cv2.imshow("pos", img)
# cv2.waitKey(10000)
# for x in pos_list:
#     for y in pos_list:
#         cv2.putText(img, str(x+y), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
# cv2.imshow("pos", img)
# cv2.waitKey(10000)
# detector.detect(img)
# print("finish")

# model = torchvision.models.resnet18()
# model.fc = nn.Linear(in_features=512, out_features=4, bias=True)
# model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))

# from PIL import Image
# from torchvision import transforms

# trans = transforms.Compose([transforms.ToTensor(),
# 			 transforms.Resize(224)]) 
# img = Image.open("C:\\code\\HOKAI\\data\\3\\024_13.png")
# img = trans(img)
# img = img.unsqueeze(0)

# pred = model(img)
# print(pred)