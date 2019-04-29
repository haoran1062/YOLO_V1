# encoding:utf-8
import os, cv2, logging, numpy as np, time, json, argparse
import torch

import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
from backbones.OriginResNet import resnet50
from backbones.OriginDenseNet import densenet121
from v1Loss import YOLOLossV1
from utils.YOLODataLoader import yoloDataset
from utils.utils import *
import multiprocessing as mp
from torchsummary import summary
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy
from utils.visual import Visual

parser = argparse.ArgumentParser(
    description='YOLO V1 Training params')
parser.add_argument('--config', default='configs/resnet_sgd_7x7.json')
args = parser.parse_args()

config_map = get_config_map(args.config)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
learning_rate = init_lr(config_map)

backbone_net = init_model(config_map)
backbone_net_p = nn.DataParallel(backbone_net.to(device), device_ids=config_map['gpu_ids'])
if config_map['resume_from_path']:
    backbone_net_p.load_state_dict(torch.load(config_map['resume_from_path']))

summary(backbone_net_p, (3, 448, 448), batch_size=config_map['batch_size'])

optimizer = torch.optim.SGD(backbone_net_p.parameters(), lr=learning_rate, momentum=0.99) # , weight_decay=5e-4)

if not os.path.exists(config_map['base_save_path']):
    os.makedirs(config_map['base_save_path'])

logger = create_logger(config_map['base_save_path'], config_map['log_name'])

my_vis = Visual(config_map['base_save_path'], log_to_file=config_map['vis_log_path'])

# backbone_net_p.load_state_dict(torch.load('densenet_sgd_S7_yolo.pth'))
lossLayer = YOLOLossV1(config_map['batch_size'], config_map['S'], config_map['B'], config_map['clsN'], config_map['lbd_coord'], config_map['lbd_no_obj'], with_mask=config_map['with_mask'], _logger=logger, _vis=my_vis)

backbone_net_p.train()

transform = transforms.Compose([
        transforms.Lambda(cv_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

train_dataset = yoloDataset(list_file=config_map['train_txt_path'], train=True, transform = transform, device=device, little_train=False, with_mask=True, S=config_map['S'])
train_loader = DataLoader(train_dataset,batch_size=config_map['batch_size'], shuffle=True, num_workers=4)
test_dataset = yoloDataset(list_file=config_map['test_txt_path'], train=False,transform = transform, device=device, little_train=False, with_file_path=True, with_mask=False, S=config_map['S'])
test_loader = DataLoader(test_dataset,batch_size=config_map['batch_size'],shuffle=False)#, num_workers=4)
data_len = int(len(test_dataset) / config_map['batch_size'])
logger.info('the dataset has %d images' % (len(train_dataset)))
logger.info('the batch_size is %d' % (config_map['batch_size']))

gt_test_map = prep_test_data(config_map['test_txt_path'], little_test=None)
gt_little_test_map = prep_test_data(config_map['test_txt_path'], little_test=config_map['little_val_data_len'])

num_iter = 0
best_mAP = 0.0
train_len = len(train_dataset)
train_iter = config_map['resume_epoch'] * len(train_loader)
last_little_mAP = 0.0

my_vis.img('label colors', get_class_color_img())

for epoch in range(config_map['resume_epoch'], config_map['epoch_num']):
    backbone_net_p.train()

    logger.info('\n\nStarting epoch %d / %d' % (epoch + 1, config_map['epoch_num']))
    logger.info('Learning Rate for this epoch: {}'.format(optimizer.param_groups[0]['lr']))

    epoch_start_time = time.clock()
    
    total_loss = 0.
    avg_loss = 0.
    
    for i,(images, target, mask_label) in enumerate(train_loader):
        # print('mask label : ', mask_label.shape, mask_label.dtype)
        it_st_time = time.clock()
        train_iter += 1
        learning_rate = learning_rate_policy(train_iter, epoch, learning_rate, config_map['lr_adjust_map'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        my_vis.plot('now learning rate', learning_rate)
        images = images.to(device)
        target = target.to(device)
        mask_label = mask_label.to(device)

        pred, p_mask = backbone_net_p(images)
        loss = lossLayer(pred, target, p_mask, mask_label)
        total_loss += loss.data.item()

        if my_vis and i % config_map['show_img_iter_during_train'] == 0:
            backbone_net_p.eval()
            img = un_normal_trans(images[0])
            bboxes, clss, confs = decoder(pred[0], grid_num=config_map['S'], device=device, thresh=0.15, nms_th=.45)
            bboxes = bboxes.clamp(min=0., max=1.)
            bboxes = bbox_un_norm(bboxes)
            img = draw_debug_rect(img.permute(1, 2 ,0), bboxes, clss, confs)
            my_vis.img('detect bboxes show', img)
            img = draw_classify_confidence_map(img, pred[0], config_map['S'], Color)
            my_vis.img('confidence map show', img)
            my_vis.img('mask gt', mask_label_2_img(mask_label[0].byte().cpu().numpy()))
            backbone_net_p.train()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        it_ed_time = time.clock()
        it_cost_time = it_ed_time - it_st_time
        if (i+1) % 5 == 0:
            avg_loss = total_loss / (i+1)
            logger.info('Epoch [%d/%d], Iter [%d/%d] expect end in %.2f min. Loss: %.4f, average_loss: %.4f, now learning rate: %f' %(epoch+1, config_map['epoch_num'], i+1, len(train_loader), it_cost_time * (len(train_loader) - i+1) // 60 , loss.item(), total_loss / (i+1), learning_rate))
            num_iter += 1
        
    epoch_end_time = time.clock()
    epoch_cost_time = epoch_end_time - epoch_start_time
    now_epoch_train_loss = total_loss / (i+1)
    my_vis.plot('train loss', now_epoch_train_loss)
    logger.info('Epoch {} / {} finished, cost time {:.2f} min. expect {} min finish train.'.format(epoch, config_map['epoch_num'], epoch_cost_time / 60, (epoch_cost_time / 60) * (config_map['epoch_num'] - epoch + 1)))

    #validation
    backbone_net_p.eval()
    now_little_mAP = 0.0
    test_mAP = 0.0

    now_little_mAP = run_test_mAP(backbone_net_p, deepcopy(gt_little_test_map), test_dataset, data_len, S=config_map['S'], logger=logger, show_img_iter = config_map["show_img_iter_during_val"], little_test=config_map['little_val_data_len'], vis=my_vis)
    
    # run full mAP cost much time, so when little mAP > thresh then run full test data's mAP 
    if now_little_mAP > last_little_mAP and now_little_mAP > config_map['run_full_test_mAP_thresh']:
        test_mAP = run_test_mAP(backbone_net_p, deepcopy(gt_test_map), test_dataset, data_len, S=config_map['S'], logger=logger, vis=my_vis)
        
    my_vis.plot('little mAP', now_little_mAP)
    my_vis.plot('mAP', test_mAP)
    last_little_mAP = now_little_mAP
    
    if test_mAP > best_mAP:
        best_mAP = test_mAP
        logger.info('get best test mAP %.5f' % best_mAP)
        torch.save(backbone_net_p.state_dict(),'%s/%s_S%d_best.pth'%(config_map['base_save_path'], config_map['backbone'], config_map['S']))
   
    torch.save(backbone_net_p.state_dict(),'%s/%s_S%d_last.pth'%(config_map['base_save_path'], config_map['backbone'], config_map['S']))