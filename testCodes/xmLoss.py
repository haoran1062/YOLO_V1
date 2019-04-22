#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 
@Author: xmhan
@LastEditors: xmhan
@Date: 2019-04-04 14:31:32
@LastEditTime: 2019-04-11 19:25:17
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class YoloV1Loss(nn.Module):
    '''Loss of YoloV1
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, size_grid_cell=7, num_boxes=2, num_classes=20, lambda_coord=5, lambda_noobj=0.5):
        '''
        @description: 
        @param : 
            size_grid_cell: S in the paper
            num_boxes: B in the paper
            num_classes: C in the paper
            lambda_coord: weight to increase the loss from bounding box coordinate predictions
            lambda_noobj: weight to decrease the loss from confidence predictions for boxes that don’t contain objects
        @return: 
        '''
        super(YoloV1Loss, self).__init__()
        self.size_grid_cell = size_grid_cell
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord

        # adjust `lambda_noobj` according to `size_grid_cell` to prevent the imblance problem
        self.lambda_noobj = lambda_noobj * 7 / size_grid_cell 

    def convert_input_tensor_dim(self, in_tensor):
        out_tensor = torch.FloatTensor(in_tensor.size())
        out_tensor[:,:,:,:] = 0.
        out_tensor[:, :, :, 4] = in_tensor[:, :, :, 0]
        out_tensor[:, :, :, 9] = in_tensor[:, :, :, 1]
        out_tensor[:, :, :, :4] = in_tensor[:, :, :, 2:6]
        out_tensor[:, :, :, 5:9] = in_tensor[:, :, :, 6:10]
        out_tensor[:, :, :, 10:] = in_tensor[:, :, :, 10:]
        return out_tensor.cuda()

    def calc_iou(self, pred_boxes, target_boxes):
        '''
        @description: calculte iou matrix between two given bounding box matrices
        @param : 
            pred_boxes:     (tensor) size(Nx4)
            target_boxes:   (tensor) size(Mx4)
        @return: iou matrix, (tensor) size(NxM)
        '''
        N = pred_boxes.size()[0]
        M = target_boxes.size()[0]
        pred_boxes_tlbr = Variable(torch.FloatTensor(pred_boxes.size()))
        pred_boxes_tlbr[:, :2] = pred_boxes[:, :2] / self.size_grid_cell - pred_boxes[:, 2:] / 2
        pred_boxes_tlbr[:, 2:] = pred_boxes[:, :2] / self.size_grid_cell + pred_boxes[:, 2:] / 2
        
        target_boxes_tlbr = Variable(torch.FloatTensor(target_boxes.size()))
        target_boxes_tlbr[:, :2] = target_boxes[:, :2] / self.size_grid_cell - target_boxes[:, 2:] / 2
        target_boxes_tlbr[:, 2:] = target_boxes[:, :2] / self.size_grid_cell + target_boxes[:, 2:] / 2

        tl = torch.max(
            pred_boxes_tlbr[:, :2].unsqueeze(1).expand(N, M, 2),    # (N, 2) -> (N, 1, 2) -> (N, M, 2)
            target_boxes_tlbr[:, :2].unsqueeze(0).expand(N, M, 2),  # (M, 2) -> (1, M, 2) -> (N, M, 2)
        )

        br = torch.min(
            pred_boxes_tlbr[:, 2:].unsqueeze(1).expand(N, M, 2),    # (N, 2) -> (N, 1, 2) -> (N, M, 2)
            target_boxes_tlbr[:, 2:].unsqueeze(0).expand(N, M, 2),  # (M, 2) -> (1, M, 2) -> (N, M, 2)
        )

        wh = br - tl                                                # [N, M, 2]
        wh[wh < 0] = 0

        inner = wh[:, :, 0] * wh[:, :, 1]                           # [N, M]
        union_pred = ( (pred_boxes_tlbr[:, 2] - pred_boxes_tlbr[:, 0]) *
            (pred_boxes_tlbr[:, 3] - pred_boxes_tlbr[:, 1]) )       # [N,]
        union_target = ( (target_boxes_tlbr[:, 2] - target_boxes_tlbr[:, 0]) *
            (target_boxes_tlbr[:, 3] - target_boxes_tlbr[:, 1]) )   # [M,]
        
        union_pred = union_pred.unsqueeze(1).expand_as(inner)       # [N,] -> [N, 1] -> [N, M]
        union_target = union_target.unsqueeze(0).expand_as(inner)   # [M,] -> [1, M] -> [N, M]
        
        iou = inner / (union_pred + union_target - inner)
        return iou
    
    def forward(self, pred, target):
        '''
        @description: 
        @param : 
            pred: (tensor) size([batch_size, size_grid_cell, size_grid_cell, num_boxes*5+num_classes])
            target: (tensor) size([batch_size, size_grid_cell, size_grid_cell, 5+num_classes])
        @return: loss
        '''
        target = self.convert_input_tensor_dim(target)
        obj_mask = target[:, :, :, 4] > 0
        noobj_mask = target[:, :, :, 4] == 0
        obj_mask = obj_mask.unsqueeze(-1).expand_as(target)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target)

        len_encode = self.num_boxes * 5 + self.num_classes
        assert target.size()[-1] == len_encode

        # FIXME 1. compute mse classification loss
        obj_target = target[obj_mask].view(-1, len_encode)
        obj_target_boxes = obj_target[:, :self.num_boxes * 5].contiguous().view(-1, 5)
        obj_target_class = obj_target[:, self.num_boxes * 5:]

        obj_pred = pred[obj_mask].view(-1, len_encode)
        obj_pred_boxes = obj_pred[:, :self.num_boxes * 5].contiguous().view(-1, 5)
        obj_pred_class = obj_pred[:, self.num_boxes * 5:]

        obj_class_loss = F.mse_loss(obj_pred_class, obj_target_class, reduction='sum')

        # FIXME 2. compute noobj confidence loss
        # part 1: for noobj in grid cell level
        # XXX I think elementss of noobj_target_conf should be 0s
        # noobj_target = target[noobj_mask].view(-1, len_encode)
        # noobj_target_boxes = noobj_target[:, :self.num_boxes * 5].contiguous().view(-1, 5)
        # noobj_target_conf = noobj_target_boxes[:, 4]
        
        noobj_pred = pred[noobj_mask].view(-1, len_encode)
        noobj_pred_boxes = noobj_pred[:, :self.num_boxes * 5].contiguous().view(-1, 5)
        noobj_pred_conf = noobj_pred_boxes[:, 4]

        # noobj_conf_loss = F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction='sum')
        noobj_conf_loss = F.mse_loss(noobj_pred_conf, torch.zeros_like(noobj_pred_conf), reduction='sum')
        
        # FIXME 3. compute obj/noobj confidence loss and coordinate regression loss
        obj_coord_loss = 0
        obj_conf_loss = 0
        # noobj_conf_loss = 0
        for i in range(0, obj_target_boxes.size()[0], self.num_boxes):
            # size(S*S*B, 5)
            # ious = self.calc_iou(obj_pred_boxes[i:i+self.num_boxes] , obj_target_boxes[i])
            obj_pred_boxes_ = obj_pred_boxes[i:i+self.num_boxes]
            obj_target_boxes_ = obj_target_boxes[i].view(-1, 5)
            ious = self.calc_iou(obj_pred_boxes_[:, :4], obj_target_boxes_[:, :4])
            max_iou, max_idx = torch.max(ious, dim=0)
            # print(max_iou.item(), max_idx.item())

            # accumulate regression loss
            # obj_coord_loss += F.mse_loss(obj_pred_boxes_[max_idx, :2], obj_target_boxes_[max_idx, :2], reduction='sum')
            # obj_coord_loss += F.mse_loss(obj_pred_boxes_[max_idx, 2:4].sqrt(), 
            #     obj_target_boxes_[max_idx, 2:4].sqrt(), reduction='sum')
            obj_coord_loss += F.mse_loss(obj_pred_boxes_[max_idx, :2], obj_target_boxes_[:, :2], reduction='sum')
            obj_coord_loss += F.mse_loss(obj_pred_boxes_[max_idx, 2:4].sqrt(), 
                obj_target_boxes_[:, 2:4].sqrt(), reduction='sum')

            # accumulate obj confidence loss
            # BUG https://blog.csdn.net/caihh2017/article/details/85788966
            # obj_conf = obj_pred_boxes_[max_idx, 4] * max_iou.type(torch.DoubleTensor)
            # obj_conf_loss += F.mse_loss(obj_conf, torch.ones_like(obj_conf), reduction='sum')

            # https://github.com/xiongzihua/pytorch-YOLO-v1/blob/master/yoloLoss.py
            obj_conf = obj_pred_boxes_[max_idx, 4]
            # obj_conf_loss += F.mse_loss(obj_conf, max_iou, reduction='sum')
            # obj_conf_loss += F.mse_loss(obj_conf, max_iou.to('cuda:0'), reduction='sum')
            obj_conf_loss += F.mse_loss(obj_conf, max_iou.to(self.device), reduction='sum')

            # accumulate noobj confidence loss
            # part 2: for noobj in bounding box level within obj grid cell
            non_max_idx = torch.ones(self.num_boxes).byte()
            non_max_idx[max_idx] = 0
            noobj_conf = obj_pred_boxes_[non_max_idx, 4]
            noobj_conf_loss += F.mse_loss(noobj_conf, torch.zeros_like(obj_conf), reduction='sum')

        # print('obj_coord_loss: ', obj_coord_loss.item())
        # print('obj_conf_loss: ', obj_conf_loss.item())
        # print('noobj_conf_loss: ', noobj_conf_loss.item())
        # print('obj_class_loss: ', obj_class_loss.item())
        
        total_loss = (obj_class_loss + self.lambda_noobj * noobj_conf_loss + 
            self.lambda_coord * obj_coord_loss + obj_conf_loss)
        
        N = pred.size()[0]
        # return total_loss/N, obj_coord_loss/N, obj_conf_loss/N, noobj_conf_loss/N, obj_class_loss/N
        return total_loss/N

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolov1 = YoloV1Loss().to(device)
    
    import numpy as np
    np.random.seed(0)

    dummy = np.random.rand(10)
    # print(dummy)
    
    pred = np.random.rand(1, 7, 7, 30)
    target = np.random.rand(1, 7, 7, 30)
    target[:, :, :, 5:10] = target[:, :, :, :5]

    pred = torch.from_numpy(pred)
    target = torch.from_numpy(target)
    loss = yolov1(pred, target)    
    print(loss)