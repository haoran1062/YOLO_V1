# encoding:utf-8
import os, numpy as np, random, cv2
import torch

import torch.nn as nn
import torch.nn.functional as F

def compute_iou_matrix(bbox1, bbox2):
    '''
        input:
            tensor bbox1: [N, 4] (x1, y1, x2, y2)
            tensor bbox2: [M, 4]
        
        process:
            1. get two bbox max(left1, left2) and max(top1, top2) this is the Intersection's left-top point
            2. get two bbox min(right1, right2) and min(bottom1, bottom2) this is the Intersection's right-bottom point
            3. expand left-top/right-bottom list to [N, M] matrix
            4. Intersection W_H = right-bottom - left-top
            5. clip witch W_H < 0 = 0
            6. Intersection area = W * H
            7. IoU = I / (bbox1's area + bbox2's area - I)

        output:
            IoU matrix:
                        [N, M]

    '''
    if not isinstance(bbox1, torch.Tensor) or not isinstance(bbox2, torch.Tensor):
        print('compute iou input must be Tensor !!!')
        exit()
    N = bbox1.size(0)
    M = bbox2.size(0)
    b1_left_top = bbox1[:, :2].unsqueeze(1).expand(N, M, 2) # [N,2] -> [N,1,2] -> [N,M,2]
    b2_left_top = bbox2[:, :2].unsqueeze(0).expand(N, M, 2) # [M,2] -> [1,M,2] -> [N,M,2]

    left_top = torch.max(b1_left_top, b2_left_top)  # get two bbox max(left1, left2) and max(top1, top2) this is the Intersection's left-top point

    b1_right_bottom = bbox1[:, 2:].unsqueeze(1).expand(N, M, 2)
    b2_right_bottom = bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)

    right_bottom = torch.min(b1_right_bottom, b2_right_bottom)  # get two bbox min(right1, right2) and min(bottom1, bottom2) this is the Intersection's right-bottom point

    w_h = right_bottom - left_top   # get Intersection W and H
    w_h[w_h < 0] = 0    # clip -x to 0

    I = w_h[:, :, 0] * w_h[: ,: ,1] # get intersection area

    b1_area = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]) # [N, ]
    b1_area = b1_area.unsqueeze(1).expand_as(I) # [N, M]
    b2_area = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]) # [M, ]
    b2_area = b2_area.unsqueeze(0).expand_as(I) # [N, M]

    IoU = I / (b1_area + b2_area - I)   # [N, M] 

    return IoU

def convert_CxCyWH_to_X1Y1X2Y2(input_tensor, S, B, device):
    '''
        input:
            input_tensor:   [B, 4] B x bboxes with (CenterX, CenterY, W, H)
            B:  int Number B

        return:
            [B, 4] which with (Xleft, Ytop, Xright, Ybottom)
    '''
    # print('input tensor shape: ', input_tensor.size())
    assert input_tensor.size()[-1] == 4, 'convert position tensor must [n, 4], but this input last dim is %d'%(input_tensor.size()[-1])

    output_tensor = torch.FloatTensor(input_tensor.size(), device=device)
    output_tensor[:, :2] = input_tensor[:, :2] / (S) - 0.5 * input_tensor[:, 2:]
    output_tensor[:, 2:] = input_tensor[:, :2] / (S) + 0.5 * input_tensor[:, 2:]

    return output_tensor

def make_eval_tensor(batch_size, S, B, C, device='cpu'):
    nl = np.zeros((batch_size, S, S, B*5+C), np.float32)
    tnl = np.zeros((batch_size, S, S, B*5+C), np.float32)
    for x in range(S):
        for y in range(S):
            for b in range(B):
                if random.randint(0, 100) > 50:
                    nl[:, x, y, b] = np.random.ranf(1)
                if random.randint(0, 100) > 30:
                    tnl[:, x, y, b] = 1.
            nl[:, x, y, B:B+B*4] = np.random.ranf(size=B*4)
            tnl[:, x, y, B:B+B*4] = np.random.ranf(size=B*4)
    
    pred_tensor = torch.from_numpy(nl)
    target_tensor = torch.from_numpy(tnl)
    return pred_tensor.to(device), target_tensor.to(device)

def decoder(pred, grid_num=7, B=2, device='cpu', thresh=0.3, gt=False):
    '''
    pred (tensor) 1x7x7x30
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''
    boxes=[]
    cls_indexs=[]
    probs = []
    cell_size = 1./grid_num
    pred = pred.data
    pred = pred.squeeze(0) 
    contain1 = pred[:,:,0].unsqueeze(2)
    contain2 = pred[:,:,1].unsqueeze(2)
    contain = torch.cat((contain1,contain2),2)
    mask1 = contain > 0.1
    # mask1 = contain >= 0.
    mask2 = (contain==contain.max()) 
    mask = (mask1+mask2).gt(0)
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(B):
                if mask[i,j,b] == 1:
                    box = pred[i, j, B+b*4 : B+b*4+4]
                    contain_prob = torch.FloatTensor([pred[i,j,b]]).to(device)
                    
                    xy = torch.FloatTensor([j,i]).to(device)*cell_size 
                    box[:2] = box[:2]*cell_size + xy 
                    box_xy = torch.FloatTensor(box.size()).to(device)
                    box_xy[:2] = box[:2] - 0.5*box[2:]
                    box_xy[2:] = box[:2] + 0.5*box[2:]
                    max_prob, cls_index = torch.max(pred[i,j,5*B:],0)
                    # print(max_prob*contain_prob, cls_index)
                    if float((contain_prob*max_prob)[0]) > thresh:
                        boxes.append(box_xy.view(1,4))
                        cls_indexs.append(cls_index.view(1))
                        probs.append(contain_prob*max_prob)
    # print(boxes, cls_indexs, probs)
    if len(boxes) ==0:
        boxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes,0) #(n,4)
        probs = torch.cat(probs,0) #(n,)
        cls_indexs = torch.cat(cls_indexs,0) #(n,)
    
    nms_thresh = 1.0
    if not gt:
        nms_thresh=1.0
    keep = nms(boxes,probs, nms_thresh)
    return boxes[keep],cls_indexs[keep],probs[keep]
    

def nms(bboxes,scores,threshold=0.25):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1) * (y2-y1)

    _,order = scores.sort(0,descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)

def draw_debug_rect(img, bboxes, color=(0, 255, 0)):
    
    if isinstance(img, torch.Tensor):
        img = img.mul(255).byte()
        img = img.cpu().numpy()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img.shape)
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.tolist()
    def bbox_un_norm(img, bboxes):
        h, w, c = img.shape
        for bbox in bboxes:
            bbox[0] = int(bbox[0] * w)
            bbox[1] = int(bbox[1] * h)
            bbox[2] = int(bbox[2] * w)
            bbox[3] = int(bbox[3] * h)
        return bboxes

    if bboxes[0][0] < 1:
        bboxes = bbox_un_norm(img, bboxes)
    # print(bboxes)
    for box in bboxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=color,thickness=1)
    cv2.imshow('debug draw bboxes', img)
    cv2.waitKey(10000)
    
def cv_resize(img, resize=448):
    return cv2.resize(img, (resize, resize))

if __name__ == "__main__":
    b1 = [
        [10, 20, 100, 123],
        [200, 300, 300, 350]
    ]

    b2 = [
        [50, 60, 150, 120],
        [0, 10, 123, 150],
        [170, 190, 310, 400]
    ]

    nb1 = np.array(b1, np.float32)
    nb2 = np.array(b2, np.float32)

    tb1 = torch.from_numpy(nb1)
    tb2 = torch.from_numpy(nb2)

    iou = compute_iou_matrix(tb1, tb2)
    print(iou)
