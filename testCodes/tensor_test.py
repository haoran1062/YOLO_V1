# encoding:utf-8
import numpy as np, random
import torch

import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
from torch.autograd import Variable

def make_eval_tensor(batch_size, S, B, C):
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
    return pred_tensor, target_tensor

def convert_input_tensor_dim(in_tensor):
    out_tensor = torch.FloatTensor(in_tensor.size())
    out_tensor[:,:,:,:] = 0.
    out_tensor[:, :, :, 4] = in_tensor[:, :, :, 0]
    out_tensor[:, :, :, 9] = in_tensor[:, :, :, 1]
    out_tensor[:, :, :, :4] = in_tensor[:, :, :, 2:6]
    out_tensor[:, :, :, 5:9] = in_tensor[:, :, :, 6:10]
    out_tensor[:, :, :, 10:] = in_tensor[:, :, :, 10:]
    return out_tensor

def simluate_pred_tensor(pred_tensor, target_tensor, batch_size, S, B, C, lbd_coord=5., lbd_no_obj=0.5):
    
    print(pred_tensor.shape, target_tensor.shape)

    # input tensor : [batch_szie, S, S, B*5+C]
    # for each cell S, the Tesnor define: [confidence x B, (x, y, w, h) x B, cls_N]

    # get coord or no Object mask from ground truth input:
    gt_coord_mask = target_tensor[:, :, :, 0] > 0                # [batch_size, S, S, 0]
    gt_no_obj_mask = target_tensor[:, :, :, 0] == 0              # [batch_size, S, S, 0]

    # flatten pred tensor
    coord_pred = pred_tensor[gt_coord_mask].view(-1, B*5+C)    # flatten to [-1,  B*(c, x, y, w, h) + C ]
    confidence_pred = coord_pred[:, :B].contiguous().view(-1)                     # top B is bboxes's confidence
    bboxes_pred = coord_pred[:, B: B+B*4].contiguous().view(-1, 4)                   # [B : B + B*4] is every bbox's [x, y, w, h]
    cls_pred = coord_pred[:, B+B*4: ]                         # [B*5 : ] is the class prop map

    # flatten ground truth tensor
    coord_target = target_tensor[gt_coord_mask].view(-1, B*5+C)
    confidence_target = coord_target[:, :B].view(-1, B)   
    bboxes_target = coord_target[:, B:B*5].contiguous().view(-1, 4)
    cls_target = coord_target[:, B*5:]

    assert bboxes_pred.size() == bboxes_target.size(), 'bbox pred size != bbox target size !!!'
    # no object loss
    no_obj_pred = pred_tensor[gt_no_obj_mask].view(-1, B*5+C)
    no_obj_target = target_tensor[gt_no_obj_mask].view(-1, B*5+C)
    no_obj_mask = torch.ByteTensor(no_obj_pred.size())
    no_obj_mask.zero_()
    # print('no obj pred shape: ', no_obj_pred.shape, 'no obj target shape: ', no_obj_target.shape, 'no obj mask shape: ', no_obj_mask.shape)
    no_obj_mask[:, :B] = 1
    no_obj_contain_pred = no_obj_pred[no_obj_mask]
    no_obj_contain_target = no_obj_target[no_obj_mask]
    no_obj_loss = F.mse_loss(no_obj_contain_pred, no_obj_contain_target, size_average=False)

    # print(no_obj_loss)

    # contain object loss
    coord_active_mask = torch.ByteTensor(bboxes_target.size()[0])
    coord_not_active_mask = torch.ByteTensor(bboxes_target.size()[0])
    coord_active_mask.zero_()
    coord_not_active_mask = 1
    bbox_target_IoUs = torch.zeros(bboxes_target.size()[0])
    # print(bbox_target_IoUs.shape)
    for i in range(0, bboxes_pred.size()[0], B):
        pred_bboxes = bboxes_pred[i:i+B]

        pred_XY_bboxes = convert_CxCyWH_to_X1Y1X2Y2(pred_bboxes, S, B)
        target_bboxes = bboxes_target[i].view(-1, 4)
        target_XY_bboxes = convert_CxCyWH_to_X1Y1X2Y2(target_bboxes, S, B)
        IoUs = compute_iou_matrix(pred_XY_bboxes, target_XY_bboxes)
        # print(target_bboxes)

        max_IoU, max_index = IoUs.max(0)
        # print(IoUs)
        # print(pred_bboxes)
        # print(pred_XY_bboxes)

        bbox_target_IoUs[i + max_index] = max_IoU

        coord_active_mask[i + max_index] = 1

    coord_not_active_mask -= coord_active_mask
    # print(coord_active_mask)
    # print('bboxes pred tensor : ', bboxes_pred)
    # print(bbox_target_IoUs)
    # print(bboxes_pred.shape, bbox_target_IoUs.shape)
    # bbox_target_IoUs = torch.cat([bboxes_pred, bbox_target_IoUs.view(-1, 1)], 1)

    # print('confidence pred shape : ', confidence_pred.shape, 'coord active mask shape : ', coord_active_mask.shape, 'bbox_target_IoUs shape : ', bbox_target_IoUs.shape)
    
    # print(confidence_pred[coord_active_mask])

    # pred_active_bboxes = bboxes_pred[coord_active_mask]
    # print('pred_active bboxes: ')
    # print(pred_active_bboxes)

    # print('target active boxes: ')
    # print(bboxes_target[coord_active_mask])

    # hit object loss
    contain_loss = F.mse_loss(confidence_pred[coord_active_mask], bbox_target_IoUs[coord_active_mask], size_average=False)
    # print(contain_loss)
    # location_loss = F.mse_loss(bboxes_pred[coord_active_mask][:, :2], bboxes_target[coord_active_mask][:, :2], size_average=False) + F.mse_loss( torch.sqrt(bboxes_pred[coord_active_mask][:, 2:]), torch.sqrt(bboxes_target[coord_active_mask][:, 2:]), size_average=False )
    location_loss = F.mse_loss( torch.cat([ bboxes_pred[coord_active_mask][:, :2], torch.sqrt(bboxes_pred[coord_active_mask][:, 2:]) ]  ) , torch.cat([bboxes_target[coord_active_mask][:, :2], torch.sqrt(bboxes_target[coord_active_mask][:, 2:])] ), size_average=False )
    # print(location_loss)
    
    # not hit object loss
    bbox_target_IoUs[coord_not_active_mask] = 0
    not_contain_loss = F.mse_loss(confidence_pred[coord_not_active_mask], bbox_target_IoUs[coord_not_active_mask], size_average=False)

    # print(cls_pred.shape, cls_target.shape)
    # print(cls_pred)
    # print(cls_target)
    # cls loss
    cls_loss = F.mse_loss(cls_pred, cls_target, size_average=False)

    total_loss = lbd_coord * location_loss + 2*contain_loss + not_contain_loss + lbd_no_obj * no_obj_loss + cls_loss
    total_loss /= batch_size
    print('location loss : ', location_loss, 'contain loss : ', contain_loss, 'not contain loss: ', not_contain_loss  , 'no obj loss : ', no_obj_loss, 'cls loss : ', cls_loss)
    print(total_loss)
    


if __name__ == "__main__":

    batch_size = 1
    S = 7
    B = 2
    C = 20

    pred_tensor, target_tensor = make_eval_tensor(batch_size, S, B, C)

    simluate_pred_tensor(pred_tensor, target_tensor, batch_size, S, B, C)

    # o_pred_tensor = convert_input_tensor_dim(pred_tensor)
    # o_target_tensor = convert_input_tensor_dim(target_tensor)

    # print(o_pred_tensor.shape, o_target_tensor.shape)
    # print(o_pred_tensor[:, 0, 0, :10])
    # print(pred_tensor[:, 0, 0, :10])

    # [batch_szie, S, S, B*5+C]  [Conf1, Conf2, xywh1, xywh2, Cls] -> [Conf1, xywh1, Conf2, xywh2, Cls]
    
    

