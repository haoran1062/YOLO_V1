# encoding:utf-8
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable

# from backbones.ResNet import MyResNet, BasicBlock, BottleBlock
# from backbones.OthResNet import resnet50
from backbones.OriginResNet import resnet50
from backbones.OriginDenseNet import densenet121
from LossModel import YOLOLossV1
from utils.YOLODataLoader import yoloDataset
import numpy as np
from utils.utils import *
import multiprocessing as mp
from torchsummary import summary


def convert_input_tensor_dim(in_tensor):
        out_tensor = torch.FloatTensor(in_tensor.size())
        out_tensor[:,:,:,:] = 0.
        out_tensor[:, :, :, 0] = in_tensor[:, :, :, 4]
        out_tensor[:, :, :, 1] = in_tensor[:, :, :, 9]
        out_tensor[:, :, :, 2:6] = in_tensor[:, :, :, :4]
        out_tensor[:, :, :, 6:10] = in_tensor[:, :, :, 5:9]
        out_tensor[:, :, :, 10:] = in_tensor[:, :, :, 10:]
        return out_tensor.cuda()


if __name__ == "__main__":

    device = 'cuda:0'
    batch_size = 1
    B = 2
    S = 7
    # S = 14
    clsN = 20
    file_path = '2007_val.txt'
    file_path = '2007_train.txt'
    # file_path = '2007_test.txt'

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    un_norm = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()),
    transform = transforms.Compose([
        transforms.Lambda(cv_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_dataset = yoloDataset(list_file=file_path, train=False,transform = transform, device=device, little_train=True, S=S)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)#,num_workers=None)

    # YOLONet = MyResNet(BottleBlock)
    model_name = 'best.pth'
    model_name = 'resnet_adam_7.pth'
    model_name = 'resnet_sgd_7.pth'
    model_name = 'densenet_adamax_yolo.pth'
    model_name = 'densenet_sgd_S7_yolo.pth'
    gpu_ids = [0]
    # YOLONet = resnet50()
    YOLONet = densenet121()
    YOLONet = nn.DataParallel(YOLONet.to(device), device_ids=gpu_ids)

    YOLONet.load_state_dict(torch.load(model_name))
    # print(YOLONet)
    YOLONet = YOLONet.to(device)
    YOLONet.eval()



    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    un_normal_trans = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

    with torch.no_grad():
        for i, (images,target) in enumerate(test_loader):
            
            images = images.to(device)
            target = target.to(device)
            # print(images.shape, target.shape)
            
            pred = YOLONet(images)
            # print(pred.shape)
            # pred = convert_input_tensor_dim(pred)
            # for i in range(7):
            #     for j in range(7):
            #         print(pred[:, i:i+1, j:j+1, :])
            images = un_normal_trans(images.squeeze(0))

            bboxes, clss, confs = decoder(pred, grid_num=S, device=device, thresh=0.1) # 23456785)
            draw_debug_rect(images.permute(1, 2 ,0), bboxes, clss, confs)
            print(clss)
            # bboxes, clss, confs = decoder(target, grid_num=S, device=device, gt=True)
            # draw_debug_rect(images.permute(1, 2 ,0), bboxes, clss, confs, color=(255, 0, 0))
            if i > 128:
                exit()