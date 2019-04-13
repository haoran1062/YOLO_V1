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
# from LossModel import YOLOLossV1
from v1Loss import YOLOLossV1
from testCodes.OthYOLOLoss import yoloLoss
from utils.YOLODataLoader import yoloDataset
import numpy as np
from utils.utils import *
import multiprocessing as mp
from torchsummary import summary

gpu_ids = [0]

device = 'cuda:0'
learning_rate = 0.001
num_epochs = 200
batch_size = 10
B = 2
S = 7
clsN = 20
lbd_coord = 5. 
lbd_no_obj = .5

# S = 14
backbone_type_list = ['densenet', 'resnet']
backbone_type = backbone_type_list[0]

if backbone_type == backbone_type_list[1]:
    backbone_net = resnet50()
    resnet = models.resnet50(pretrained=True)
    new_state_dict = resnet.state_dict()
    dd = backbone_net.state_dict()
    for k in new_state_dict.keys():
        print(k)
        if k in dd.keys() and not k.startswith('fc'):
            print('yes')
            dd[k] = new_state_dict[k]
    backbone_net.load_state_dict(dd)
if backbone_type == backbone_type_list[0]:
    backbone_net = densenet121()
    resnet = models.densenet121(pretrained=True)
    new_state_dict = resnet.state_dict()
    dd = backbone_net.state_dict()
    for k in new_state_dict.keys():
        print(k)
        if k in dd.keys() and not k.startswith('fc'):
            print('yes')
            dd[k] = new_state_dict[k]
    backbone_net.load_state_dict(dd)

# backbone_net.load_state_dict(torch.load('yolo.pth'))
# backbone_net = MyResNet(BottleBlock)

# backbone_net = backbone_net.to(device)
# print(backbone_net)
backbone_net_p = nn.DataParallel(backbone_net.to(device), device_ids=gpu_ids)
summary(backbone_net_p, (3, 448, 448), batch_size=batch_size)

#backbone_net.load_state_dict(torch.load('yolo.pth'))
lossLayer = YOLOLossV1(batch_size, S, B, clsN, lbd_coord, lbd_no_obj)
# lossLayer = yoloLoss(S, B, lbd_coord, lbd_no_obj)


backbone_net_p.train()
# net.train()
# different learning rate
params=[]
params_dict = dict(backbone_net_p.named_parameters())
for key,value in params_dict.items():
    if key.startswith('features'):
        params += [{'params':[value],'lr':learning_rate*1}]
    else:
        params += [{'params':[value],'lr':learning_rate}]

with_sgd = True
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)

if not with_sgd:
    optimizer = torch.optim.Adam(params,lr=learning_rate,weight_decay=1e-8)

transform = transforms.Compose([
        transforms.Lambda(cv_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

train_dataset = yoloDataset(list_file='2007_train.txt', train=True, transform = transform, device=device, little_train=True, S=S)
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True, num_workers=4)
test_dataset = yoloDataset(list_file='2007_train.txt',train=False,transform = transform, device=device, little_train=True, S=S)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False, num_workers=4)
print('the dataset has %d images' % (len(train_dataset)))
print('the batch_size is %d' % (batch_size))
logfile = open('log.txt', 'w')

num_iter = 0
best_test_loss = np.inf

for epoch in range(num_epochs):
    backbone_net_p.train()
    if with_sgd:
        if epoch == 1:
            learning_rate = 0.0005
        if epoch == 2:
            learning_rate = 0.00075
        if epoch == 3:
            learning_rate = 0.001
        # if epoch == 10:
        #     learning_rate = 0.01
        # if epoch == 20:
        #     learning_rate = 0.001
        if epoch == 150:
            learning_rate=0.0001
        if epoch == 180:
            learning_rate=0.00001
        # if epoch == 150:
        #     learning_rate = 0.000001


    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))
    
    total_loss = 0.
    
    for i,(images,target) in enumerate(train_loader):

        images = images.to(device)
        target = target.to(device)

        pred = backbone_net_p(images)
        loss = lossLayer(pred, target)
        total_loss += loss.data[0]
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 5 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' 
            %(epoch+1, num_epochs, i+1, len(train_loader), loss.data[0], total_loss / (i+1)))
            num_iter += 1
        # exit()

    #validation
    validation_loss = 0.0
    backbone_net_p.eval()
    with torch.no_grad():
        for i,(images,target) in enumerate(test_loader):

            images = images.to(device)
            target = target.to(device)
            
            pred = backbone_net_p(images)
            loss = lossLayer(pred, target)
            validation_loss += loss.data[0]
        validation_loss /= len(test_loader)
    print('now val loss : %.5f'%(validation_loss))
    
    if best_test_loss > validation_loss:
        best_test_loss = validation_loss
        print('get best test loss %.5f' % best_test_loss)
        torch.save(backbone_net_p.state_dict(),'best.pth')
    logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')  
    logfile.flush()      
    torch.save(backbone_net_p.state_dict(),'yolo.pth')