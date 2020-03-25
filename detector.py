import collections
import cv2
import torch
import numpy as np 
from models import CSPNet
from models.py_utils import pred
from utils import parse, parse_cfg, visualize

def get_test_input():
    img = cv2.imread('/home/zk/Desktop/0_Parade_Parade_0_465.jpg')
    img = cv2.resize(img, (336, 448)) # height, width, channel
    img = torch.tensor(img).float()
    img = img.permute(2, 0, 1).unsqueeze(0)
    img = img / 255.0
    return img


# test forward
model = CSPNet("configs/network_arch.cfg")
print('load model....')
model.load_state_dict(torch.load('CSP_Pytorch_e382_l0.pkl'))
print('done')

input = cv2.imread('D:\\0_Parade_Parade_0_519.jpg')
test_config = parse_cfg('configs/test.cfg')
test_config = test_config[0]
result = pred(input, model, test_config)
print(result.shape)
