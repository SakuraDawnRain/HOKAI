import cv2
import numpy as np
import json

# functions to make state and action data

def get_self_data(map):
    (test_B, test_G, test_R) = cv2.split(map)
    result = test_G[:]
    result[test_G<130] = 0
    result[test_R>130] = 0
    result[test_B>130] = 0
    result[result>0] = 255
    result[:100, :100] = 0
    result[-100:, -100:] = 0
    height, width = result.shape
    M  = cv2.getRotationMatrix2D((width/2,height/2), 45, 0.707)
    result = cv2.warpAffine(result, M, (width,height))
    result = cv2.resize(result, (32, 32))
    return np.sum(result, axis=1)

def get_oppo_data(map):
    (test_B, test_G, test_R) = cv2.split(map)
    result = test_G[:]
    result[test_G>100] = 0
    result[test_R<100] = 0
    result[test_B>100] = 0
    result[test_R<test_G+test_B] = 0
    result[result>0] = 255
    result[:100, :100] = 0
    result[-100:, -100:] = 0
    height, width = result.shape
    M  = cv2.getRotationMatrix2D((width/2,height/2), 45, 0.707)
    result = cv2.warpAffine(result, M, (width,height))
    result[:, :120] = 0
    result[:, -120:] = 0
    result = cv2.resize(result, (32, 32))
    return np.sum(result, axis=1)

def make_data(state, action, reward):
    data = {}
    data["state"] = state
    data["action"] = action
    data["reward"] = reward
    return data

def save_data(datalist, filename):
    with open(filename, 'w') as f:
        f.write(json.dumps(datalist))

def load_data(filename):
    with open(filename) as f:
        data = json.load(f)
    return data

def bool2int(action):
    result = []
    for i in action:
        result.append(1 if i else 0)
    return result

def process_data(data):
    data['action'] = bool2int(data['action'])
    return data

def get_processed_map(map):
    (test_B, test_G, test_R) = cv2.split(map)
    self_result = test_G[:]
    self_result[test_G<130] = 0
    self_result[test_R>130] = 0
    self_result[test_B>130] = 0
    self_result[self_result>0] = 255
    self_result[:100, :100] = 0
    self_result[-100:, -100:] = 0

    oppo_result = test_G[:]
    oppo_result[test_G>100] = 0
    oppo_result[test_R<100] = 0
    oppo_result[test_B>100] = 0
    oppo_result[test_R<test_G+test_B] = 0
    oppo_result[oppo_result>0] = 255
    oppo_result[:100, :100] = 0
    oppo_result[-100:, -100:] = 0

    grey_map = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)

    return np.stack([self_result, oppo_result, grey_map], axis=-1)