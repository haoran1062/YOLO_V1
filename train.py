# encoding:utf-8
import os, cv2, logging, numpy as np, time
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

def warmming_up_policy(now_iter, now_lr, stop_down_iter=1000):
    if now_iter <= stop_down_iter:
        now_lr += 0.000001
    return now_lr

def learning_rate_policy(now_iter, now_epoch, now_lr, lr_adjust_map, stop_down_iter=1000):
    now_lr = warmming_up_policy(now_iter, now_lr, stop_down_iter)
    if now_epoch in lr_adjust_map.keys():
        now_lr = lr_adjust_map[now_epoch]

    return now_lr

gpu_ids = [0]

device = 'cuda:0'
learning_rate = 0.0
num_epochs = 200
batch_size = 18
B = 2
S = 7
clsN = 20
lbd_coord = 5. 
lbd_no_obj = .1

lr_adjust_map = {
    1:0.001,
    50: 0.0001,
    100: 0.00001
}

backbone_type_list = ['densenet', 'resnet']
backbone_type = backbone_type_list[1]

if backbone_type == backbone_type_list[1]:
    backbone_net = resnet50()
    resnet = models.resnet50(pretrained=True)
    new_state_dict = resnet.state_dict()
    dd = backbone_net.state_dict()
    for k in new_state_dict.keys():
        if k in dd.keys() and not k.startswith('fc'):
            dd[k] = new_state_dict[k]
    backbone_net.load_state_dict(dd)
    batch_size = 16

if backbone_type == backbone_type_list[0]:
    backbone_net = densenet121()
    resnet = models.densenet121(pretrained=True)
    new_state_dict = resnet.state_dict()
    dd = backbone_net.state_dict()
    for k in new_state_dict.keys():
        if k in dd.keys() and not k.startswith('fc'):
            dd[k] = new_state_dict[k]
    backbone_net.load_state_dict(dd)

backbone_net_p = nn.DataParallel(backbone_net.to(device), device_ids=gpu_ids)
summary(backbone_net_p, (3, 448, 448), batch_size=batch_size)

# backbone_net_p.load_state_dict(torch.load('densenet_sgd_S7_yolo.pth'))
lossLayer = YOLOLossV1(batch_size, S, B, clsN, lbd_coord, lbd_no_obj)

backbone_net_p.train()

with_sgd = True
optimizer = torch.optim.SGD(backbone_net_p.parameters(), lr=learning_rate, momentum=0.99) # , weight_decay=5e-4)
opt_name = 'sgd'

if not with_sgd:
    optimizer = torch.optim.Adam(backbone_net_p.parameters(), lr=learning_rate, weight_decay=1e-8)
    opt_name = 'adam'

transform = transforms.Compose([
        transforms.Lambda(cv_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

base_save_path = '%s_%s_cellSize%d/'%(backbone_type, opt_name, S)
if not os.path.exists(base_save_path):
    os.makedirs(base_save_path)

log_name = 'train'
logger = create_logger(base_save_path, log_name)

data_base = 'datasets/'
train_data_name = data_base + 'train.txt'
# test_data_name = '2007_train.txt'
test_data_name =  data_base + '2007_test.txt'

train_dataset = yoloDataset(list_file=train_data_name, train=True, transform = transform, device=device, little_train=False, S=S)
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True, num_workers=4)
test_dataset = yoloDataset(list_file=test_data_name,train=False,transform = transform, device=device, little_train=False, with_file_path=True, S=S)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)#, num_workers=4)
data_len = int(len(test_dataset) / batch_size)
# data_len = len(test_dataset)
logger.info('the dataset has %d images' % (len(train_dataset)))
logger.info('the batch_size is %d' % (batch_size))

little_val_num = 500
gt_test_map = prep_test_data(test_data_name, little_test=None)
gt_little_test_map = prep_test_data(test_data_name, little_test=little_val_num)

num_iter = 0
best_mAP = 0.0
train_len = len(train_dataset)
train_iter = 0
last_little_mAP = 0.0

my_vis = Visual(base_save_path[:-1])

for epoch in range(num_epochs):
    backbone_net_p.train()

    logger.info('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    logger.info('Learning Rate for this epoch: {}'.format(optimizer.param_groups[0]['lr']))

    epoch_start_time = time.clock()
    
    total_loss = 0.
    avg_loss = 0.
    
    for i,(images,target) in enumerate(train_loader):
        it_st_time = time.clock()
        train_iter += 1
        learning_rate = learning_rate_policy(train_iter, epoch, learning_rate, lr_adjust_map)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        my_vis.plot('now learning rate', learning_rate)
        images = images.to(device)
        target = target.to(device)

        pred = backbone_net_p(images)
        loss = lossLayer(pred, target)
        total_loss += loss.data.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        it_ed_time = time.clock()
        it_cost_time = it_ed_time - it_st_time
        if (i+1) % 5 == 0:
            avg_loss = total_loss / (i+1)
            logger.info('Epoch [%d/%d], Iter [%d/%d] expect end in %.2f min. Loss: %.4f, average_loss: %.4f, now learning rate: %f' %(epoch+1, num_epochs, i+1, len(train_loader), it_cost_time * (len(train_loader) - i+1) // 60 , loss.item(), total_loss / (i+1), learning_rate))
            num_iter += 1
        
    epoch_end_time = time.clock()
    epoch_cost_time = epoch_end_time - epoch_start_time
    now_epoch_train_loss = total_loss / (i+1)
    my_vis.plot('train loss', now_epoch_train_loss)
    logger.info('Epoch {} / {} finished, cost time {:.2f} min. expect {} min finish train.'.format(epoch, num_epochs, epoch_cost_time / 60, (epoch_cost_time / 60) * (num_epochs - epoch + 1)))

    #validation
    backbone_net_p.eval()
    little_mAP = 0.0
    test_mAP = 0.0

    now_little_mAP = run_test_mAP(backbone_net_p, deepcopy(gt_little_test_map), test_dataset, data_len, logger=logger, little_test=little_val_num)
    
    if now_little_mAP > last_little_mAP:
        test_mAP = run_test_mAP(backbone_net_p, deepcopy(gt_test_map), test_dataset, data_len, logger=logger)
        
    my_vis.plot('little mAP', now_little_mAP)
    my_vis.plot('mAP', test_mAP)
    last_little_mAP = now_little_mAP
    
    if test_mAP > best_mAP:
        best_mAP = test_mAP
        logger.info('get best test mAP %.5f' % best_mAP)
        torch.save(backbone_net_p.state_dict(),'%s/%s_%s_S%d_best.pth'%(base_save_path, backbone_type, opt_name, S))
   
    torch.save(backbone_net_p.state_dict(),'%s/%s_%s_S%d_yolo.pth'%(base_save_path, backbone_type, opt_name, S))