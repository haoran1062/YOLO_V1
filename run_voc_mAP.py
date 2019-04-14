#encoding:utf-8
import os, numpy as np
from utils.utils import *



if __name__ == '__main__':
    #test_eval()
    from backbones.OriginResNet import resnet50
    from backbones.OriginDenseNet import densenet121
    from collections import defaultdict
    from tqdm import tqdm
    import torch 
    import torchvision.transforms as transforms
    from utils.YOLODataLoader import yoloDataset
    from torch.utils.data import DataLoader

    file_path = '2007_test.txt'
    # file_path = '2007_train.txt'
    # file_path = '2012_val.txt'
    # file_path = '2007_val.txt'
    debug_n = 100

    target = prep_test_data(file_path)
    
    #
    #start test
    #
    print('---start test---')

    backbone_type_list = ['densenet', 'resnet']
    backbone_type = backbone_type_list[0]

    if backbone_type == backbone_type_list[1]:
        model_name = 'resnet_sgd_7.pth'
        model_name = 'yolo.pth'
        YOLONet = resnet50()
    
    if backbone_type == backbone_type_list[0]:
        YOLONet = densenet121()
        model_name = 'densenet_sgd_yolo.pth'
        model_name = 'densenet_sgd_S7_best.pth'
        # model_name = 'yolo.pth'
        # model_name = 'densenet_adamax_yolo.pth'


    device = 'cuda:0'
    batch_size = 16
    count = 0
    S = 7

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    un_normal_trans = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    transform = transforms.Compose([
        transforms.Lambda(cv_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_dataset = yoloDataset(list_file=file_path, train=False,transform = transform, device=device, little_train=False, with_file_path=True, S=S)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)


    gpu_ids = [0]
    # YOLONet = resnet50()
    # YOLONet = densenet121()
    YOLONet = nn.DataParallel(YOLONet.to(device), device_ids=gpu_ids)
    YOLONet.load_state_dict(torch.load(model_name))
    YOLONet = YOLONet.to(device)
    YOLONet.eval()

    data_len = int(len(test_dataset) / batch_size)

    run_test_mAP(YOLONet, target, test_loader, data_len)
    


    
    