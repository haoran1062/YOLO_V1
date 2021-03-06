# encoding:utf-8
import numpy as np, random
import torch

import torch.nn as nn
import torch.nn.functional as F
# from ..utils.utils import *
from torch.autograd import Variable
from OthYOLOLoss import yoloLoss
from LossModel import YOLOLossV1

def cv_resize(img):
    return cv2.resize(img, (448, 448))

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

def convert_CxCyWH_to_X1Y1X2Y2(input_tensor, S, B, device='cpu'):
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
    print(coord_active_mask)
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
    print('location loss : %.5f'% location_loss, 'contain loss : %.5f'% contain_loss, 'not contain loss: %.5f'% not_contain_loss  , 'no obj loss : %.5f'% no_obj_loss, 'cls loss : %.5f'% cls_loss)
    print(total_loss)
    


if __name__ == "__main__":

    from TestYOLODataLoader import yoloDataset
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    import cv2

    batch_size = 1
    S = 14
    B = 2
    C = 20
    device = 'cuda:0'

    transform = transforms.Compose([
        transforms.Lambda(cv_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_dataset = yoloDataset(list_file='2007_train.txt',train=False,transform = transform, device=device, little_train=True, S=14)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=4)
    test_iter = iter(test_loader)
    for iii in range(1):
        img, target_tensor = next(test_iter)

    pred_tensor, _target_tensor = make_eval_tensor(batch_size, S, B, C)

    # simluate_pred_tensor(pred_tensor, target_tensor, batch_size, S, B, C)

    o_pred_tensor = convert_input_tensor_dim(pred_tensor)
    o_target_tensor = convert_input_tensor_dim(target_tensor)

    print(50*'-' + 'YOLO Loss' + 50 * '-')
    YOLOLoss = yoloLoss(14, 2, 5., 0.5)
    YOLOLoss = YOLOLoss.to('cuda:0')
    YOLOLoss.forward(o_pred_tensor.cuda(), o_target_tensor.cuda())

    print(50*'-' + 'my YOLO Loss' + 50 * '-')
    myYOLOLoss = YOLOLossV1(1, 14, 2, 20)
    myYOLOLoss = myYOLOLoss.to('cuda:0')
    myYOLOLoss.forward(pred_tensor.cuda(), target_tensor.cuda())

    print('\n\n\n')

    print(o_pred_tensor.shape, o_target_tensor.shape)
    print(o_target_tensor[:, 0, 0, :10])
    print(target_tensor[:, 0, 0, :10])

    # [batch_szie, S, S, B*5+C]  [Conf1, Conf2, xywh1, xywh2, Cls] -> [Conf1, xywh1, Conf2, xywh2, Cls]
    
    

