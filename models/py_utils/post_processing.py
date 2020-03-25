import numpy as np
import cv2
import torch
import math

def prepare_images(img, config):
    height, width = config['testsize']
    flip = int(config['flip'])
    channel_mean = [float(x) for x in config['channel_mean'].split(',')]
    imgs = []
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    imgs.append(img)
    if flip:
        img_f = cv2.flip(img, 1) # flip horizontally
        imgs.append(img_f)
    imgs = [x.astype(np.float32) for x in imgs]
    imgs = np.array(imgs)
    for i in range(3):
        imgs[:, :, :, i] -= channel_mean[i]
    imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)
    return imgs


def generate_bbox(pred, config):
    center_thresh = float(config['center_thresh'])
    center_cls, hw_regr, offset = pred
    # remove center points whose score is less than center_thresh
    center_inds = center_cls > center_thresh
    # generate bounding box
    center_cls = center_cls * center_inds
    hw_regr = hw_regr * center_inds
    offset = offset * center_inds
    bboxes = []
    inds = torch.nonzero(center_cls)
    for x in inds:
        batch, channel, height, width = x
        score = center_cls[batch, channel, height, width]
        scale_h = hw_regr[batch, 0, height, width]
        scale_w = hw_regr[batch, 1, height, width]
        # make offset on the center points
        # the type of offset is float, the type of center coordinate is integer
        offset_y = offset[batch, 0, height, width]
        offset_x = offset[batch, 1, height, width]
        # the first element stands for origin image if 0 otherwise for flipped image
        box = [0 if batch == 0 else 1,  width + offset_x, height + offset_y, math.exp(scale_w), math.exp(scale_h), score]
        bboxes.append(box)
    return np.array(bboxes)
    

def remap_bbox_to_input(bboxes, config):
    downsize = int(config['downsize'])
    testsize = config['testsize']
    bboxes[:, 3] = bboxes[:, 3] * downsize
    bboxes[:, 4] = bboxes[:, 4] * downsize
    bboxes[:, 1] = np.clip((bboxes[:, 1] + 0.5) * downsize - bboxes[:, 3] / 2, 0, testsize[1]) # x1
    bboxes[:, 2] = np.clip((bboxes[:, 2] + 0.5) * downsize - bboxes[:, 4] / 2, 0, testsize[0]) # y1
    bboxes[:, 3] = np.clip(bboxes[:, 1] + bboxes[:, 3], 0, testsize[1])
    bboxes[:, 4] = np.clip(bboxes[:, 2] + bboxes[:, 4], 0, testsize[0])
    return bboxes


def soft_nms(bboxes, score, thresh):
    soft_iou_thresh = thresh
    bbox_thresh = score
    sorted_inds = np.argsort(bboxes[:, 4])[::-1]
    sorted_bboxes = bboxes[sorted_inds]
    valid_inds = sorted_bboxes[:, 4] > bbox_thresh
    sorted_bboxes = sorted_bboxes[valid_inds]
    result = []
    len_r = len(sorted_bboxes)
    while len_r > 1:
        cur = sorted_bboxes[0]
        # calculate the iou between this box and the other boxes
        y1, x1 = sorted_bboxes[1:, 1], sorted_bboxes[1:, 0]
        top, left = y1, x1
        top[top < cur[1]] = cur[1]
        left[left < cur[0]] = cur[0]
        y2, x2 = sorted_bboxes[1:, 3], sorted_bboxes[1:, 2]
        bottom, right = y2, x2
        bottom[bottom > cur[3]] = cur[3]
        right[right > cur[2]] = cur[2]
        intersection = (right - left) * (bottom - top)
        union = (cur[2] - cur[0]) * (cur[3] - cur[1]) + (y2 - y1) * (x2 - x1) - intersection
        iou = intersection / union
        iou[iou < 0] = 0

        # decrease the score of bounding box whose iou with cur is more than a soft_iou_thresh
        scores = sorted_bboxes[1:, 4]
        decrease_inds = iou >= soft_iou_thresh
        scores[decrease_inds] = scores[decrease_inds] * (1 - iou[decrease_inds])
        sorted_bboxes[1:, 4] = scores
        bboxes = sorted_bboxes[1:, 4]
        sorted_inds = np.argsort(bboxes[:, 4])[::-1]
        sorted_bboxes = bboxes[sorted_inds]
        valid_inds = sorted_bboxes[:, 4] > bbox_thresh
        sorted_bboxes = sorted_bboxes[valid_inds]
        
        len_r = len(sorted_bboxes)
        result.append(cur)
    if len_r == 1:
        result.append(sorted_bboxes[0])
    
    if len(result) == 0:
        print("no bounding boxes after softnms")
        return np.empty(shape=[0, 5], dtype=np.float32)

    return np.array(result)


def nms(bboxes, config):
    # bboxes is a list of 6 elements [0/1, center_x, center_y, height, width, score]
    # sort the bboxes according to score
    origin_bboxes = bboxes[bboxes[:, 0] == 0, :]
    flipped_bboxes = bboxes[bboxes[:, 0] == 1, :]
    testsize = config['testsize']
    
    thresh = float(config['soft_iou_thresh'])
    score = float(config['bbox_score_thresh'])
    origin_bboxes = soft_nms(origin_bboxes, score, thresh)
    flipped_bboxes = soft_nms(flipped_bboxes, score, thresh)
    flipped_bboxes[:, [0, 2]] = testsize[1] - flipped_bboxes[:, [2, 0]]
    flipped_bboxes[:, [1, 3]] = testsize[0] - flipped_bboxes[:, [3, 1]]
    
    return np.vstack((origin_bboxes, flipped_bboxes))


def scaled_inference(img, model, config, scale=1):
    flip = int(config['flip'])
    minimum_hw = int(config['minimum_hw'])
    downsize = int(config['downsize'])
    height, width = img.shape[:2]
    # make sure the downsize is the scale betetween origin image and the output feature map
    # in_image is a little larger than the origin image => how to fix this offset
    # two ways: origin --scale--> scaled image --little adjust according to downsize--> actual_scale_h, actual_scale_w
    # origin --little adjust according to downsize--> --scale--> scaled image--> actual_scale_h, actual_scale_w
    # height * scale + offset, (height + offset) * scale
    scaled_height, scaled_width = height * scale, width * scale
    in_height, in_width = np.ceil(scaled_height / downsize) * downsize, np.ceil(scaled_width / downsize) * downsize
    actual_scale_h, actual_scale_w = in_height / height, in_width / width # 在 in_height 和 in_width 上检测出的 bounding box，还要再除 actual scale 才能得到在原图上的 bounding box
    
    config['testsize'] = [int(in_height), int(in_width)]
    imgs = prepare_images(img, config) # prepare input for network; numpy to tensor
    print(imgs.shape)
    pred = model(imgs, torch.cuda.is_available()) # flip (batch, channel, height, width)
    bboxes = generate_bbox(pred, config)
    if len(bboxes) == 0:
        print('generate 0 bounding boxes from output feature map')
        return np.empty(shape=[0, 5], dtype=np.float32)
    # center_x, center_y, scale_h, scale_w => x1, y1, x2, y2, score
    bboxes = remap_bbox_to_input(bboxes, config)
    # origin -> nms -> result1; flip -> nms -> transfer -> result2; thresh=0.6
    bboxes = nms(bboxes, config, in_height, in_width)
    # remap from input to origin, actual_scale_h, actual_scale_w; origin + flip
    keep_inds = np.minimum(bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]) >= minimum_hw
    bboxes = bboxes[keep_inds]
    bboxes[:, [0, 2]] /= actual_scale_w
    bboxes[:, [1, 3]] /= actual_scale_h

    return bboxes
    

def pyramid_inference(img, model, config):
    result = []
    scales = [float(x) for x in config['shrink_scales'].split(',')]
    limits = [int(x) for x in config['shrink_limits'].split(',')]
    for ind, s in enumerate(scales):
        print("pred on scale: {}".format(s))
        bboxes = scaled_inference(img, model, config, s)
        keep_inds = np.maximum(bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1) > limits[ind]
        bboxes = bboxes[keep_inds]
        result.append(bboxes)
    
    scales = [float(x) for x in config['enlarge_scales'].split(',')]
    limits = [int(x) for x in config['enlarge_limits'].split(',')]
    for ind, s in enumerate(scales):
        print("pred on scale: {}".format(s))
        bboxes = scaled_inference(img, model, config, s)
        keep_inds = np.minimum(bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1) < limits[ind]
        bboxes = bboxes[keep_inds]
        result.append(bboxes)
    
    return np.vstack(result)


def pred(img, model, config):
    # 不同尺度检测 + 图片翻转检测
    # 原始图片 --缩放 scale1--> 网络输入 --卷积 scale2--> 网络输出
    # 网络输出 * scale2  --> 网络输入 --> * scale1 映射到原图
    det_origin = scaled_inference(img, model, config, 1)
    det_pyramid = pyramid_inference(img, model, config)
    # result1 + result2 + result3 -> nms; thresh=0.4
    result = np.vstack((det_origin, det_pyramid))

    minimum_hw_1 = int(config['minimum_hw_1'])
    keep_inds = np.minimum(result[:, 2] - result[:, 0], result[:, 3] - result[:, 1]) >=  minimum_hw_1
    result = result[keep_inds]

    score = float(config['bbox_score_thresh'])
    thresh = float(config['soft_iou_thresh_1'])
    result = soft_nms(bboxes, score, thresh)

    minimum_area = int(config['minimum_area'])
    keep_inds = (dets[:, 2] - dets[:, 0] + 1) * (dets[:, 3] - dets[:, 1] + 1) >= minimum_area
    result = result[keep_index]
    return result