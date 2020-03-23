# confidence threshold: 0.05
# generate bounding box according to center prediction and corresponding scale
# adust center location according to offset branch
# remap bounding box to original image size
# NMS

from .nms import soft_bbox_vote, format_img
import numpy as np
import torch
import cv2
import visdom
import sys
sys.path.append('.../')
from utils import visualize

def parse_wider_offset(Y, config):
    score = float(config['center_conf_thresh']) # 0.05
    soft_nms_thresh = float(config['soft_nms_thresh']) # 0.6
    soft_score = float(config['soft_score_thresh']) # 0.05
    down = int(config['down_scale']) # 4
    size_test = config['size_test']

    # tensorflow: (batch, height, widht, channel)
    # pytorch: (batch, channel, height, width)
    # Y[0] -> center map; Y[1] -> hw_map; Y[2] -> offset_map
    seman = Y[0][0, 0, :, :]
    height = Y[1][0, 0, :, :]
    width = Y[1][0, 1, :, :]
    offset_y = Y[2][0, 0, :, :]
    offset_x = Y[2][0, 1, :, :]
    y_c, x_c = np.where(seman > score) # confidence threshold
    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            w = np.exp(width[y_c[i], x_c[i]]) * down
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            x1, y1 = min(x1, size_test[1]), min(y1, size_test[0])
            boxs.append([x1, y1, min(x1 + w, size_test[1]), min(y1 + h, size_test[0]), s]) # (x1, y1, x2, y2)
   
    boxs = np.asarray(boxs, dtype=np.float32)
    boxs = soft_bbox_vote(boxs, thre=soft_nms_thresh, score=soft_score)
    return boxs


def detect_face(img, model, config, scale=1):
    # flip = False
    flip = int(config['flip'])
    # use scale to enlarge image for small faces while shrink image for large face
    img_h, img_w = img.shape[:2] # tensorflow [batch, heigt, width, channel] pytorch [batch, channel, height, width] opencv=> (height, width, channel)
    img_h_new, img_w_new = int(np.ceil(scale * img_h / 16) * 16), int(np.ceil(scale * img_w / 16) * 16) 
    # np.ceil(scale * img_h / 16) is the height the final feature map
    # np.ceil(scale * img_h) * 16 is the remapped new height of the input image
    scale_h, scale_w = img_h_new / img_h, img_w_new / img_w 

    # actual shrink / enlarge scale
    img_s = cv2.resize(img, (0, 0), fx=scale_w, fy=scale_h, interpolation=cv2.INTER_LINEAR)
    # img_s => height, width, channel
    # resize the img according to the actual scale
    config['size_test'] = [img_h_new, img_w_new]

    x_csp = format_img(img_s, config)

    # visualize('input', 'images', x_csp, 'img456')

    Y = model(x_csp, torch.cuda.is_available())
    # boxes = bbox_processing.parse_wider_offset(Y, config, score=0.05, nmsthre=0.6) # confidence thresh + soft nms
    boxes = parse_wider_offset(Y, config)
    if len(boxes) > 0:
        # 最下边长 scale > 12 (length > 1.308 mm) 的 bounding box 
        keep_index = np.where(np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) >= 12)[0]
        boxes = boxes[keep_index, :]
    if len(boxes) > 0:
        boxes[:, 0:4:2] = boxes[:, 0:4:2] / scale_w # 此时的 boxes 是在 img_h_new 上检测出来的，所以还要还原到原来的 img_h 上 
        boxes[:, 1:4:2] = boxes[:, 1:4:2] / scale_h
    else:
        boxes = np.empty(shape=[0, 5], dtype=np.float32)

    boxes_f = False
    if flip:
        img_sf = cv2.flip(img_s, 1)
        x_csp = format_img(img_sf, config)
        Y = model(x_csp, torch.cuda.is_available())
        # boxes = bbox_processing.parse_wider_offset(Y, config, score=0.05, nmsthre=0.6) # confidence thresh + soft nms
        boxes_f = parse_wider_offset(Y, config)
        if len(boxes_f) > 0:
            # 最下边长 scale > 12 (length > 1.308 mm) 的 bounding box 
            keep_index = np.where(np.minimum(boxes_f[:, 2] - boxes_f[:, 0], boxes_f[:, 3] - boxes_f[:, 1]) >= 12)[0]
            boxes = boxes_f[keep_index, :]
        if len(boxes_f) > 0:
            boxes_f[:, [0, 2]] = img_s.shape[1] - boxes_f[:, [2, 0]]
            boxes_f[:, 0:4:2] = boxes_f[:, 0:4:2] / scale_w # 此时的 boxes 是在 img_h_new 上检测出来的，所以还要还原到原来的 img_h 上 
            boxes_f[:, 1:4:2] = boxes_f[:, 1:4:2] / scale_h
        else:
            boxes_f = np.empty(shape=[0, 5], dtype=np.float32)
    return np.row_stack((boxes, boxes_f)) if boxes_f else boxes


def im_det_ms_pyramid(image, model, config, max_im_shrink):
    # shrink detecting and shrink only detect big face
    det_s = detect_face(image, model, config, 0.5)
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 64)[0]
    det_s = det_s[index, :]

    det_temp = detect_face(image, model, config, 0.75)
    index = np.where(np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) > 32)[0]
    det_temp = det_temp[index, :]
    det_s = np.row_stack((det_s, det_temp))

    det_temp = detect_face(image, model, config, 0.25)
    index = np.where(np.maximum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) > 96)[0]
    det_temp = det_temp[index, :]
    det_s = np.row_stack((det_s, det_temp))

    st = [1.25, 1.5, 1.75, 2.0, 2.25]
    for i in range(len(st)):
        if (st[i] <= max_im_shrink):
            det_temp = detect_face(image, model, config, st[i])
            # Enlarged images are only used to detect small faces.
            # 放大倍数越大，检测到的目标越小
            if st[i] == 1.25:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) < 128)[0]
                det_temp = det_temp[index, :]
            elif st[i] == 1.5:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) < 96)[0]
                det_temp = det_temp[index, :]
            elif st[i] == 1.75:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) < 64)[0]
                det_temp = det_temp[index, :]
            elif st[i] == 2.0:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) < 48)[0]
                det_temp = det_temp[index, :]
            elif st[i] == 2.25:
                index = np.where(
                    np.minimum(det_temp[:, 2] - det_temp[:, 0] + 1, det_temp[:, 3] - det_temp[:, 1] + 1) < 32)[0]
                det_temp = det_temp[index, :]
            det_s = np.row_stack((det_s, det_temp))
    return det_s
			

def generate_bbox(img, model, config):
    # 0x7fffffff 是 int32 类型数的最大值 
    # 为什么这么算呢? 
    max_im_shrink = (0x7fffffff / 577.0 / (img.shape[0] * img.shape[1])) ** 0.5  # the max size of input image
    shrink = max_im_shrink if max_im_shrink < 1 else 1
    det0 = detect_face(img, model, config)
    # print(det0)
    # det1 = im_det_ms_pyramid(img, model, config, max_im_shrink)
    # print(det1)
    # merge all test results via bounding box voting
    det = np.row_stack((det0, det1))
    # 去除小于 3 毫米的 bounding box
    keep_index = np.where(np.minimum(det[:, 2] - det[:, 0], det[:, 3] - det[:, 1]) >= 3)[0]
    det = det[keep_index, :]
    dets = soft_bbox_vote(det, thre=0.4)
    keep_index = np.where((dets[:, 2] - dets[:, 0] + 1) * (dets[:, 3] - dets[:, 1] + 1) >= 6 ** 2)[0]
    dets = dets[keep_index, :]
    return dets
