# encoding:utf-8
import numpy as np 
import torch

import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *

class YOLOLossV1(nn.Module):
    def __init__(self, _batch_size, _S, _B, _clsN, _l_coord=5., _l_noobj=0.5, _device='cuda:0'):
        super(YOLOLossV1, self).__init__()
        self.S = _S
        self.B = _B
        self.device = _device
        self.C = _clsN
        self.lambda_coord = _l_coord
        self.lambda_noobj = _l_noobj
        self.batch_size = _batch_size

    def forward(self, pred_tensor, target_tensor):
    
        # input tensor : [batch_szie, S, S, B*5+C]
        # for each cell S, the Tesnor define: [confidence x B, (x, y, w, h) x B, cls_N]

        # get coord or no Object mask from ground truth input:
        gt_coord_mask = target_tensor[:, :, :, 0] > 0                # [batch_size, S, S, 0]
        gt_no_obj_mask = target_tensor[:, :, :, 0] == 0              # [batch_size, S, S, 0]

        # flatten pred tensor
        coord_pred = pred_tensor[gt_coord_mask].view(-1, self.B*5+self.C)    # flatten to [-1,  B*(c, x, y, w, h) + C ]
        confidence_pred = coord_pred[:, :self.B].contiguous().view(-1)                     # top B is bboxes's confidence
        bboxes_pred = coord_pred[:, self.B: self.B*5].contiguous().view(-1, 4)                   # [B : B + B*4] is every bbox's [x, y, w, h]
        cls_pred = coord_pred[:, self.B*5: ]                         # [B*5 : ] is the class prop map

        # flatten ground truth tensor
        coord_target = target_tensor[gt_coord_mask].view(-1, self.B*5+self.C)
        # confidence_target = coord_target[:, :self.B].view(-1, self.B)   
        bboxes_target = coord_target[:, self.B:self.B*5].contiguous().view(-1, 4)
        cls_target = coord_target[:, self.B*5:]

        assert bboxes_pred.size() == bboxes_target.size(), 'bbox pred size != bbox target size !!!'
        # no object loss
        no_obj_pred = pred_tensor[gt_no_obj_mask].view(-1, self.B*5+self.C)
        no_obj_target = target_tensor[gt_no_obj_mask].view(-1, self.B*5+self.C)
        no_obj_mask = torch.ByteTensor(no_obj_pred.size())
        no_obj_mask.zero_()
        no_obj_mask[:, :self.B] = 1
        no_obj_contain_pred = no_obj_pred[no_obj_mask]
        no_obj_contain_target = no_obj_target[no_obj_mask]
        no_obj_loss = F.mse_loss(no_obj_contain_pred, no_obj_contain_target, size_average=False)


        # contain object loss
        coord_active_mask = torch.ByteTensor(bboxes_target.size()[0])
        coord_not_active_mask = torch.ByteTensor(bboxes_target.size()[0])
        coord_active_mask.zero_()
        coord_not_active_mask = 1
        bbox_target_IoUs = torch.zeros(bboxes_target.size()[0], device=self.device)

        for i in range(0, bboxes_pred.size()[0], self.B):
            pred_bboxes = bboxes_pred[i:i+self.B]

            pred_XY_bboxes = convert_CxCyWH_to_X1Y1X2Y2(pred_bboxes, self.S, self.B, self.device)
            target_bboxes = bboxes_target[i].view(-1, 4)
            target_XY_bboxes = convert_CxCyWH_to_X1Y1X2Y2(target_bboxes, self.S, self.B, self.device)
            IoUs = compute_iou_matrix(pred_XY_bboxes, target_XY_bboxes)

            max_IoU, max_index = IoUs.max(0)
            max_IoU = max_IoU.to(self.device)

            bbox_target_IoUs[i + max_index] = max_IoU

            coord_active_mask[i + max_index] = 1

        coord_not_active_mask -= coord_active_mask
        

        # hit object loss
        contain_loss = F.mse_loss(confidence_pred[coord_active_mask], bbox_target_IoUs[coord_active_mask], size_average=False)
        location_loss = F.mse_loss( torch.cat([ bboxes_pred[coord_active_mask][:, :2], torch.sqrt(bboxes_pred[coord_active_mask][:, 2:]) ]  ) , torch.cat([bboxes_target[coord_active_mask][:, :2], torch.sqrt(bboxes_target[coord_active_mask][:, 2:])] ), size_average=False )
        
        # not hit object loss
        bbox_target_IoUs[coord_not_active_mask] = 0
        not_contain_loss = F.mse_loss(confidence_pred[coord_not_active_mask], bbox_target_IoUs[coord_not_active_mask], size_average=False)

        # cls loss
        cls_loss = F.mse_loss(cls_pred, cls_target, size_average=False)

        total_loss = self.lambda_coord * location_loss + contain_loss + not_contain_loss + self.lambda_noobj * no_obj_loss + cls_loss
        total_loss /= self.batch_size

        return total_loss


if __name__ == "__main__":
    batch_size = 1
    B = 2
    S = 7
    clsN = 20
    device = 'cuda:0'
    device = 'cpu'
    pred_tensor, target_tensor = make_eval_tensor(batch_size, S, B, clsN, device)
    # pred_tensor = pred_tensor.to(device)
    # target_tensor = target_tensor.to(device)
    # print(pred_tensor.device)
    loss_layer = YOLOLossV1(batch_size, S, B, clsN, _device=device)
    loss_layer.to(device)
    total_loss = loss_layer.forward(pred_tensor, target_tensor)
    print(total_loss, total_loss.device)