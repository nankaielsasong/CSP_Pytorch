import numpy as np
import torch
import cv2
import sys
import visdom
sys.path.append('...')
from utils import visualize

def format_img(img, config):
    """ formats an image for model prediction based on config """
    img = format_img_channels(img, config)
    return img 


def format_img_channels(img, config):
    """ formats the image channels based on config """
    # height, width, channel
    img = img.astype(np.float32)
    cfg = [float(x) for x in config['img_channel_mean'].split(',')]
    img[:, :, 0] -= cfg[0]
    img[:, :, 1] -= cfg[1]
    img[:, :, 2] -= cfg[2]
    img = torch.tensor(img).float()
    img = img.permute(2, 0, 1)
    img = torch.unsqueeze(img, 0)
    return img


def soft_bbox_vote(det, thre=0.35, score=0.05):
    if det.shape[0] <= 1:
        return det
    order = det[:, 4].ravel().argsort()[::-1]
    # numpy.argsort -> Returns the indices that would sort an array.
    # [::-1] 是将列表倒序的最简洁的方法
    det = det[order, :] # sort the boxes from higher score to lower score
    dets = []
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1) # height * width
        xx1 = np.maximum(det[0, 0], det[:, 0]) # 分数最高的 box 的 x1，所有 box 的 x1 => 取最大值 
        yy1 = np.maximum(det[0, 1], det[:, 1]) # 分数最高的 box 的 y1, 所有 box 的 y1 => 取最大值
        xx2 = np.minimum(det[0, 2], det[:, 2]) # 分数最高的 box 的 x2, 所有 box 的 x2 => 取最小值
        yy2 = np.minimum(det[0, 3], det[:, 3]) # 分数最高的 box 的 y2, 所有 box 的 y2 => 取最小值 
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter) # 分数最高的这个 box 和所有其他 box 的 IOU 的值

        # get needed merge det and delete these det
        merge_index = np.where(o >= thre)[0] # 和当前分数最高的 box 的重合度太大的 box 视作冗余，去除 
        det_accu = det[merge_index, :]
        det_accu_iou = o[merge_index]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1: # 没有要删除的与当前 bbx 冗余的 bbx
            try:
                dets = np.row_stack((dets, det_accu)) # 保留当前分数最高的 bounding box 并且不删除冗余
            except:
                dets = det_accu
            continue
        else: # 有要删除的和当前 bbx 冗余的 bbx => 结果由两部分组成，soft_det_accu(optional) + det_accu_sum
            soft_det_accu = det_accu.copy()
            soft_det_accu[:, 4] = soft_det_accu[:, 4] * (1 - det_accu_iou) # score * (1 - iou) => new_score
            soft_index = np.where(soft_det_accu[:, 4] >= score)[0] # new_sccore > score => keep
            soft_det_accu = soft_det_accu[soft_index, :]

            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4)) # bbx 的每个坐标值都和他对应的分数相乘
            max_score = np.max(det_accu[:, 4]) # 
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:]) 
            det_accu_sum[:, 4] = max_score # det_accu_sum = [sum(x1) / sum(score), sum(y1) / sum(score), sum(x2) / sum(score), sum(y2) / sum(score), max_score]

            if soft_det_accu.shape[0] > 0:
                det_accu_sum = np.row_stack((soft_det_accu, det_accu_sum))

            # in case dets is []
            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum
            

    order = dets[:, 4].ravel().argsort()[::-1]
    dets = dets[order, :]
    return dets # 返回结果是按照分数从高到低排列的 
