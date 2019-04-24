# encoding:utf-8
import numpy as np 
import torch

import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *

class YOLOLossV1(nn.Module):
    def __init__(self, _batch_size, _S, _B, _clsN, _l_coord=5., _l_noobj=0.5, _device='cuda:0', _logger=None, _vis=None):
        super(YOLOLossV1, self).__init__()
        self.S = _S
        self.B = _B
        self.device = _device
        self.C = _clsN
        self.lambda_coord = _l_coord
        self.lambda_noobj = _l_noobj
        self.batch_size = _batch_size
        self.logger = _logger
        self.vis = _vis

    def forward(self, pred_tensor, target_tensor):
        
        # input tensor : [batch_szie, S, S, B*5+C]
        # for each cell S, the Tesnor define: [confidence x B, (x, y, w, h) x B, cls_N]

        # get coord or no Object mask from ground truth input:
        gt_coord_mask = target_tensor[:, :, :, :self.B] == 1                # [batch_size, S, S, 0]

        # print(gt_coord_mask.shape)
        # print(gt_no_obj_mask.shape)
        # print(gt_coord_mask[:, :, :, 0])
        pred_cls = pred_tensor[:, :, :, self.B*5:]
        pred_cls = pred_cls[gt_coord_mask[:, :, :, 0]]

        gt_cls = target_tensor[:, :, :, self.B*5:]
        gt_cls = gt_cls[gt_coord_mask[:, :, :, 0]]
        # print(pred_cls)
        # print(gt_cls)

        cls_loss = F.mse_loss(pred_cls, gt_cls, reduction='sum')
        # print(cls_loss)

        gt_coord_mask_1 = gt_coord_mask[:, :, :, 0]
        # print(gt_coord_mask_1.dtype)
        # print(gt_coord_mask_1.shape)
        obj_center_indexs = torch.nonzero(gt_coord_mask_1)
        # print(obj_center_indexs)


        pred_confs = pred_tensor[:, :, :, :self.B]
        pred_bboxes = pred_tensor[:, :, :, self.B:self.B*5]

        # gt_confs = target_tensor[:, :, :, :self.B]
        gt_bboxes = target_tensor[:, :, :, self.B:self.B*5]

        contain_obj_mask = torch.ByteTensor(pred_confs.size())
        contain_obj_mask.zero_()
        not_contain_obj_mask = torch.ByteTensor(contain_obj_mask.size())
        not_contain_obj_mask[:, :, :, :] = 1
        pred_gt_IoUs = torch.zeros(contain_obj_mask.size())
        pred_gt_IoUs = pred_gt_IoUs.to(self.device)

        for now_center_index in obj_center_indexs:

            now_gt_bbox = gt_bboxes[now_center_index[0], now_center_index[1], now_center_index[2], :4]
            now_pred_bboxes = pred_bboxes[now_center_index[0], now_center_index[1], now_center_index[2], ].view(-1, 4)

            pred_XY_bboxes = convert_CxCyWH_to_X1Y1X2Y2(now_pred_bboxes, self.S, self.B, self.device)
            now_gt_bbox = now_gt_bbox.view(-1, 4)
            target_XY_bboxes = convert_CxCyWH_to_X1Y1X2Y2(now_gt_bbox, self.S, self.B, self.device)
            IoUs = compute_iou_matrix(pred_XY_bboxes, target_XY_bboxes)

            max_IoU, max_index = IoUs.max(0)
            max_index = max_index.to(self.device)
            contain_obj_mask[now_center_index[0], now_center_index[1], now_center_index[2], max_index] = 1
            max_IoU = max_IoU.to(self.device)
            pred_gt_IoUs[now_center_index[0], now_center_index[1], now_center_index[2], max_index] = max_IoU
        
        not_contain_obj_mask -= contain_obj_mask

        # print(contain_obj_mask[:, :, :, 0])
        # print(contain_obj_mask[:, :, :, 1])
        # print(contain_obj_mask)
        # print(not_contain_obj_mask[:, :, :, 0])
        # print(not_contain_obj_mask[:, :, :, 1])

        # print(pred_confs[contain_obj_mask])

        hit_obj_conf_loss = F.mse_loss(pred_confs[contain_obj_mask], pred_gt_IoUs[contain_obj_mask], reduction='sum')
        not_hit_obj_conf_loss = F.mse_loss(pred_confs[not_contain_obj_mask], pred_gt_IoUs[not_contain_obj_mask], reduction='sum')

        # print(hit_obj_conf_loss, not_hit_obj_conf_loss)
        pred_bboxes = pred_bboxes.view(-1, self.S, self.S, self.B, 4)
        gt_bboxes = gt_bboxes.view(-1, self.S, self.S, self.B, 4)

        # print(pred_bboxes[contain_obj_mask])
        # print('\n\n\n')
        # print(gt_bboxes[contain_obj_mask])

        hit_obj_location_loss = F.mse_loss( pred_bboxes[contain_obj_mask][:2], gt_bboxes[contain_obj_mask][:2], reduction='sum') + F.mse_loss( torch.sqrt(pred_bboxes[contain_obj_mask][2:]), torch.sqrt(gt_bboxes[contain_obj_mask][2:]), reduction='sum' )
        # print(hit_obj_location_loss)

        total_loss = self.lambda_coord * hit_obj_location_loss + hit_obj_conf_loss + self.lambda_noobj * not_hit_obj_conf_loss + cls_loss 
        total_loss /= self.batch_size

        if self.logger:
            self.logger.info('location loss : %.5f contain loss : %.5f not contain loss: %.5f classify loss : %.5f'%( hit_obj_location_loss.item() / self.batch_size, hit_obj_conf_loss.item() / self.batch_size, not_hit_obj_conf_loss.item() / self.batch_size, cls_loss.item() / self.batch_size) )
        else:
            print('location loss : %.5f'% (hit_obj_location_loss / self.batch_size), 'contain loss : %.5f'% (hit_obj_conf_loss / self.batch_size), 'not contain loss: %.5f'%(not_hit_obj_conf_loss / self.batch_size), 'classify loss : %.5f'%(cls_loss / self.batch_size) )
        # print('total loss: ', total_loss)
        if self.vis:
            self.vis.plot('location loss', hit_obj_location_loss.item() / self.batch_size)
            self.vis.plot('confidence loss', hit_obj_conf_loss.item() / self.batch_size)
            self.vis.plot('no object loss', not_hit_obj_conf_loss.item() / self.batch_size)
            self.vis.plot('classify loss', cls_loss.item() / self.batch_size)
        # exit()
        return total_loss


if __name__ == "__main__":
    from utils.YOLODataLoader import yoloDataset
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    batch_size = 1
    B = 2
    S = 7
    clsN = 20
    device = 'cuda:0'
    device = 'cpu'
    pred_tensor, target_tensor = make_eval_tensor(batch_size, S, B, clsN, device)

    transform = transforms.Compose([
        transforms.Lambda(cv_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    
    test_dataset = yoloDataset(list_file='2007_train.txt',train=False,transform = transform, device=device, little_train=True, S=S, test_mode=True)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    test_iter = iter(test_loader)
    for i in range(2):
        img, target = next(test_iter)

    loss_layer = YOLOLossV1(batch_size, S, B, clsN, _device=device)
    loss_layer.to(device)
    total_loss = loss_layer.forward(pred_tensor, target)
    print(total_loss, total_loss.device)

    