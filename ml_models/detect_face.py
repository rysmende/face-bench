from __future__ import annotations
import math
import torch
import numpy as np

from torch.nn import functional as F
from torchvision.ops import boxes as B

def get_bigger_face(boxes: np.ndarray) -> np.ndarray:
    '''
    Return the biggest box based on its area.

        Parameters:
            boxes (np.ndarray): Coordinates of bounding boxes
        Returns:
            box (np.ndarray): Coordinates of a bounding box
    '''
    return boxes[np.argmax(np.stack([
            (box[2] - box[0]) * (box[3] - box[1]) for box in boxes
        ]))]

def detect_face(
            img: torch.Tensor, pnet: torch.nn.Module, rnet: torch.nn.Module, 
            onet: torch.nn.Module, minsize: int = 20, 
            threshold: list[float] = [.6, .7, .7], factor: float = 0.709, 
            device: torch.device = torch.device('cpu')
        ) -> tuple:
    '''
    Return faces' coordinates and landmark points.

        Parameters:
            img (torch.Tensor): Input image
            pnet (torch.nn.Module): PNet model of MTCNN
            rnet (torch.nn.Module): RNet model of MTCNN
            onet (torch.nn.Module): ONet model of MTCNN
            minsize (int): Minimal size of scale pyramid
            threshold (list[float]): Score thresholds for P-, R- and ONet
            factor (float): Scaling factor of scale pyramid
            device (torch.device): Device for inference
        Returns:
            batch_boxes (np.ndarray): Coordinates of bounding boxes
            batch_probs (np.ndarray): Probabilities of bounding boxes
    '''
    imgs = img.unsqueeze(0) # When only one image.
    model_dtype = next(pnet.parameters()).dtype
    imgs = imgs.permute(0, 3, 1, 2).type(model_dtype) # Convert from NHWC to NHWC.
    batch_size = len(imgs)
    h, w = imgs.shape[2:4]
    m = 12.0 / minsize
    minl = min(h, w) * m

    # Create scale pyramid.
    scales = [m * factor**i for i in range(int(math.log(12 / minl, factor)) + 1)]

    # First stage.
    boxes = []
    image_inds = []
    scale_picks = []
    offset = 0
    for scale in scales:
        # Scale the image.
        im_data = F.interpolate(
            imgs, list(map(lambda x: int(x * scale + 1), (h, w))), mode='area'
        )
        im_data = (im_data - 127.5) * 0.0078125
        reg, probs = pnet(im_data)

        # Get bounding boxes.
        boxes_scale, image_inds_scale = generate_bounding_box(
            reg, probs[:, 1], scale, threshold[0]
        )
        boxes.append(boxes_scale)
        image_inds.append(image_inds_scale)

        # NMS boxes.
        pick = B.batched_nms(
            boxes_scale[:, :4], boxes_scale[:, 4], image_inds_scale, 0.5
        )
        scale_picks.append(pick + offset)
        offset += boxes_scale.shape[0]
    
    boxes = torch.cat(boxes, dim=0)
    image_inds = torch.cat(image_inds, dim=0)
    scale_picks = torch.cat(scale_picks, dim=0)

    # NMS within each scale + image
    boxes, image_inds = boxes[scale_picks], image_inds[scale_picks]

    # NMS within each image
    pick = B.batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
    boxes, image_inds = boxes[pick], image_inds[pick]

    regw  = boxes[:, 2] - boxes[:, 0]
    regh  = boxes[:, 3] - boxes[:, 1]
    qq1   = boxes[:, 0] + boxes[:, 5] * regw
    qq2   = boxes[:, 1] + boxes[:, 6] * regh
    qq3   = boxes[:, 2] + boxes[:, 7] * regw
    qq4   = boxes[:, 3] + boxes[:, 8] * regh
    boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).permute(1, 0)
    boxes = rerec(boxes)
    
    
    # Second stage
    if len(boxes) > 0:
        y, ey, x, ex = pad(boxes, w, h) # Pad boxes.
        # Scale the boxes.
        im_data = [
            F.interpolate(
                imgs[image_inds[k], :, y[k]-1:ey[k], x[k]-1:ex[k]].unsqueeze(0), 
                (24, 24), mode='area',
            ) for k in range(len(y)) if ey[k] > y[k] - 1 and ex[k] > x[k] - 1 
        ]

        im_data = torch.cat(im_data, dim=0)
        im_data = (im_data - 127.5) * 0.0078125
        out = rnet(im_data)

        out0 = out[0].permute(1, 0)
        out1 = out[1].permute(1, 0)
        score = out1[1, :]
        ipass = score > threshold[1]
        boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
        image_inds = image_inds[ipass]
        mv = out0[:, ipass].permute(1, 0)

        # NMS within each image
        pick = B.batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
        boxes, image_inds, mv = boxes[pick], image_inds[pick], mv[pick]
        boxes = bbreg(boxes, mv)
        boxes = rerec(boxes)

    # Third stage
    points = torch.zeros(0, 5, 2, device=device)
    if len(boxes) > 0:
        y, ey, x, ex = pad(boxes, w, h)
        im_data = [
            F.interpolate(
                imgs[image_inds[k], :, y[k]-1:ey[k], x[k]-1:ex[k]].unsqueeze(0), 
                size=(48, 48), mode='area',
            ) for k in range(len(y)) if ey[k] > y[k] - 1 and ex[k] > x[k] - 1
        ]       
        im_data = torch.cat(im_data, dim=0)
        im_data = (im_data - 127.5) * 0.0078125
        
        # This is equivalent to out = onet(im_data) to avoid GPU out of memory.
        out = onet(im_data)

        out0 = out[0].permute(1, 0)
        out1 = out[1].permute(1, 0)
        out2 = out[2].permute(1, 0)
        score = out2[1, :]
        points = out1
        ipass = score > threshold[2]
        points = points[:, ipass]
        boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
        image_inds = image_inds[ipass]
        mv = out0[:, ipass].permute(1, 0)

        w_i = boxes[:, 2] - boxes[:, 0] + 1
        h_i = boxes[:, 3] - boxes[:, 1] + 1
        points_x = w_i.repeat(5, 1) * points[:5, :] + boxes[:, 0].repeat(5, 1) - 1
        points_y = h_i.repeat(5, 1) * points[5:10, :] + boxes[:, 1].repeat(5, 1) - 1
        points = torch.stack((points_x, points_y)).permute(2, 1, 0)
        boxes = bbreg(boxes, mv)

        # NMS within each image using "Min" strategy
        pick = batched_nms_numpy(boxes[:, :4], boxes[:, 4], image_inds, 0.7, 'Min')
        boxes, image_inds, points = boxes[pick], image_inds[pick], points[pick]

    boxes = boxes.cpu().numpy()
    points = points.cpu().numpy()

    image_inds = image_inds.cpu()

    batch_boxes = [
        boxes [np.where(image_inds == b_i)].copy() for b_i in range(batch_size)
    ]

    batch_points = [
        points[np.where(image_inds == b_i)].copy() for b_i in range(batch_size)
    ]

    batch_boxes, batch_points = np.array(batch_boxes), np.array(batch_points)

    return batch_boxes, batch_points


def bbreg(boxes: torch.Tensor, reg: torch.Tensor) -> torch.Tensor:
    '''
    Tune bounding boxes' coordinates from regression results.
        
        Parameters:
            boxes (np.ndarray): Coordinates of bounding boxes
        Returns:
            boxes (np.ndarray): Coordinates of bounding boxes
    '''
    if reg.shape[1] == 1:
        reg = torch.reshape(reg, (reg.shape[2], reg.shape[3]))

    w  = boxes[:, 2] - boxes[:, 0] + 1
    h  = boxes[:, 3] - boxes[:, 1] + 1
    b1 = boxes[:, 0] + reg[:, 0] * w
    b2 = boxes[:, 1] + reg[:, 1] * h
    b3 = boxes[:, 2] + reg[:, 2] * w
    b4 = boxes[:, 3] + reg[:, 3] * h
    boxes[:, :4] = torch.stack([b1, b2, b3, b4]).permute(1, 0)

    return boxes


def generate_bounding_box(
            reg: torch.Tensor, probs: torch.Tensor, scale: float, thresh: float
        ) -> tuple:
    '''
    Generate bounding box from results of PNet.

        Parameters:
            reg (torch.Tensor): Regression results of PNet
            probs (torch.Tensor): Probabilities of each bounding box
            scale (float): Specific scale from scale pyramid
            thresh (float): Score threshold for PNet only
        Returns:
            box (torch.Tensor): Coordinates of bounding boxes
            image_inds (torch.Tensor): Indeces of images in batch
    '''
    stride = 2
    cellsize = 12

    reg = reg.permute(1, 0, 2, 3)

    mask = probs >= thresh
    mask_inds = mask.nonzero()
    image_inds = mask_inds[:, 0]
    score = probs[mask]
    reg = reg[:, mask].permute(1, 0)
    bb = mask_inds[:, 1:].type(reg.dtype).flip(1)
    q1 = ((stride * bb + 1) / scale).floor()
    q2 = ((stride * bb + cellsize - 1 + 1) / scale).floor()
    box = torch.cat([q1, q2, score.unsqueeze(1), reg], dim=1)
    
    return box, image_inds


def batched_nms_numpy(
            boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, 
            threshold: float, method: str
        ) -> torch.Tensor:
    '''
    Batched Non Maximum Suppression on Numpy.
    
    Strategy: in order to perform NMS independently per class, we add an offset
    to all the boxes.     
    
    The offset is dependent only on the class idx, and is large enough so that 
    boxes from different classes do not overlap.

        Parameters:
            boxes (torch.Tensor): Coordinates of bounding boxes
            scores (torch.Tensor): Probabilities of each bounding box
            idxs (torch.Tensor): Indeces of images in batch
            threshold (float): Score threshold for PNet only
            method (str): Method for choosing the region
        Returns:
            keep (torch.Tensor): Indeces of selected bounding boxes.
    '''
    device = boxes.device
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)
    
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    boxes_for_nms = boxes_for_nms.cpu().numpy()
    scores = scores.cpu().numpy()
    keep = nms_numpy(boxes_for_nms, scores, threshold, method)
    return torch.as_tensor(keep, dtype=torch.long, device=device)


def nms_numpy(
            boxes: np.ndarray, scores: np.ndarray, threshold: float, method: str
        ) -> np.ndarray:
    '''
    Non Maximum Suppression algorithm implemented on Numpy.

        Parameters:
            boxes (np.ndarray): Regression results of PNet
            scores (np.ndarray): Probabilities of each bounding box
            threshold (float): Score threshold for PNet only
            method (str): Method for choosing the region
        Returns:
            pick (np.ndarray): Indeces of selected bounding boxes.
    '''
    if boxes.size == 0:
        return np.empty((0, 3))

    x1 = boxes[:, 0].copy()
    y1 = boxes[:, 1].copy()
    x2 = boxes[:, 2].copy()
    y2 = boxes[:, 3].copy()
    s = scores
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]

        xx1 = np.maximum(x1[i], x1[idx]).copy()
        yy1 = np.maximum(y1[i], y1[idx]).copy()
        xx2 = np.minimum(x2[i], x2[idx]).copy()
        yy2 = np.minimum(y2[i], y2[idx]).copy()

        w = np.maximum(0.0, xx2 - xx1 + 1).copy()
        h = np.maximum(0.0, yy2 - yy1 + 1).copy()

        inter = w * h
        if method == 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o <= threshold)]

    pick = pick[:counter].copy()
    return pick


def pad(boxes: torch.Tensor, w: int, h: int) \
        -> tuple:
    '''
    Assure that bounding boxes' coordinates will not go out the image.

        Parameters:
            boxes (torch.Tensor): Coordinates of bounding boxes
            w (int): Width of input Image
            h (int): Height of input Image
        Returns:
            y  (np.ndarray): y0 coordinates of boxes
            ey (np.ndarray): y1 coordinates of boxes
            x  (np.ndarray): x0 coordinates of boxes
            ex (np.ndarray): x1 coordinates of boxes
    '''
    boxes = boxes.trunc().int().cpu().numpy()
    x = boxes[:, 0]
    y = boxes[:, 1]
    ex = boxes[:, 2]
    ey = boxes[:, 3]

    x[x < 1] = 1
    y[y < 1] = 1
    ex[ex > w] = w
    ey[ey > h] = h

    return y, ey, x, ex


def rerec(boxes: torch.Tensor) -> torch.Tensor:
    '''
    Make rectangle box more square-like.

        Parameters:
            boxes (torch.Tensor): Coordinates of bounding boxes
        Returns:
            boxes (torch.Tensor): Coordinates of bounding boxes
    '''
    h = boxes[:, 3] - boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    
    l = torch.max(w, h)
    boxes[:, 0]   = boxes[:, 0] + w * 0.5 - l * 0.5
    boxes[:, 1]   = boxes[:, 1] + h * 0.5 - l * 0.5
    boxes[:, 2:4] = boxes[:, :2] + l.repeat(2, 1).permute(1, 0)

    return boxes
