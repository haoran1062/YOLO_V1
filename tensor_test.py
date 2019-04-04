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



def forward(pred_tensor, target_tensor, l_coord=5, l_noobj=.5):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,30)
        '''
        N = pred_tensor.size()[0]
        coo_mask = target_tensor[:,:,:,4] > 0
        noo_mask = target_tensor[:,:,:,4] == 0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)
        # print(coo_mask.shape, noo_mask.shape)
        # print(coo_mask)

        coo_pred = pred_tensor[coo_mask].view(-1,30)
        box_pred = coo_pred[:,:10].contiguous().view(-1,5) #box[x1,y1,w1,h1,c1]
        class_pred = coo_pred[:,10:]                       #[x2,y2,w2,h2,c2]
        
        coo_target = target_tensor[coo_mask].view(-1,30)
        box_target = coo_target[:,:10].contiguous().view(-1,5)
        class_target = coo_target[:,10:]

        # compute not contain obj loss
        noo_pred = pred_tensor[noo_mask].view(-1,30)
        noo_target = target_tensor[noo_mask].view(-1,30)
        noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size())
        noo_pred_mask.zero_()
        # print(noo_pred_mask.shape)
        noo_pred_mask[:,4]=1;noo_pred_mask[:,9]=1
        noo_pred_c = noo_pred[noo_pred_mask] #noo pred只需要计算 c 的损失 size[-1,2]
        noo_target_c = noo_target[noo_pred_mask]
        nooobj_loss = F.mse_loss(noo_pred_c,noo_target_c,size_average=False)

        #compute contain obj loss
        coo_response_mask = torch.cuda.ByteTensor(box_target.size())
        coo_response_mask.zero_()
        coo_not_response_mask = torch.cuda.ByteTensor(box_target.size())
        coo_not_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size())
        for i in range(0,box_target.size()[0],2): #choose the best iou box
            box1 = box_pred[i:i+2]
            box1_xyxy = torch.FloatTensor(box1.size())
            box1_xyxy[:,:2] = box1[:,:2]/14. -0.5*box1[:,2:4]
            box1_xyxy[:,2:4] = box1[:,:2]/14. +0.5*box1[:,2:4]
            box2 = box_target[i].view(-1,5)
            box2_xyxy = torch.FloatTensor(box2.size())
            box2_xyxy[:,:2] = box2[:,:2]/14. -0.5*box2[:,2:4]
            box2_xyxy[:,2:4] = box2[:,:2]/14. +0.5*box2[:,2:4]
            iou = compute_iou_matrix(box1_xyxy[:,:4],box2_xyxy[:,:4]) #[2,1]
            max_iou,max_index = iou.max(0)
            max_index = max_index.data
            # print(iou)
            # print(box1_xyxy[:, :4])
            # print(box1[:, :4])
            
            coo_response_mask[i+max_index]=1
            coo_not_response_mask[i+1-max_index]=1

            #####
            # we want the confidence score to equal the
            # intersection over union (IOU) between the predicted box
            # and the ground truth
            #####
            box_target_iou[i+max_index,torch.LongTensor([4])] = (max_iou).data
        # box_target_iou = Variable(box_target_iou).cuda()
        # print(coo_response_mask[:, 0])
        #1.response loss
        box_pred_response = box_pred[coo_response_mask].view(-1,5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1,5)
        box_target_response = box_target[coo_response_mask].view(-1,5)
        contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response_iou[:,4],size_average=False)
        loc_loss = F.mse_loss(box_pred_response[:,:2],box_target_response[:,:2],size_average=False) + F.mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]),size_average=False)
        #2.not response loss
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1,5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1,5)
        box_target_not_response[:,4]= 0
        #not_contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)
        
        #I believe this bug is simply a typo
        not_contain_loss = F.mse_loss(box_pred_not_response[:,4], box_target_not_response[:,4],size_average=False)

        #3.class loss
        class_loss = F.mse_loss(class_pred,class_target,size_average=False)

        total_loss = (l_coord*loc_loss + 2*contain_loss + not_contain_loss + l_noobj*nooobj_loss + class_loss)/N
        print('location loss : ', loc_loss, 'contain loss : ', contain_loss, 'not contain loss: ', not_contain_loss , 'no obj loss : ', nooobj_loss, 'cls loss : ', class_loss)

        print(total_loss)
    


if __name__ == "__main__":

    batch_size = 1
    S = 7
    B = 2
    C = 20

    pred_tensor, target_tensor = make_eval_tensor(batch_size, S, B, C)

    simluate_pred_tensor(pred_tensor, target_tensor, batch_size, S, B, C)

    o_pred_tensor = convert_input_tensor_dim(pred_tensor)
    o_target_tensor = convert_input_tensor_dim(target_tensor)

    # print(o_pred_tensor.shape, o_target_tensor.shape)
    # print(o_pred_tensor[:, 0, 0, :10])
    # print(pred_tensor[:, 0, 0, :10])

    forward(o_pred_tensor, o_target_tensor)

    # [batch_szie, S, S, B*5+C]  [Conf1, Conf2, xywh1, xywh2, Cls] -> [Conf1, xywh1, Conf2, xywh2, Cls]
    
    

