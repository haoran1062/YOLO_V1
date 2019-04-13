# encoding:utf-8
import os, sys, numpy as np, time 
import torch.nn as nn 

import torch
import torch.nn.functional as F
from torchvision import transforms, utils
from torchsummary import summary


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, BlockID, in_c, out_c, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # print('in channels : ', in_c)
        self.BlockID = BlockID
        self.add_module('Block_Conv1', nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False))
        self.add_module('Block_BN1', nn.BatchNorm2d(out_c))

        self.add_module('Block_Conv2', nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False))
        self.add_module('Block_BN2', nn.BatchNorm2d(out_c))

        self.add_module('Block_Relu', nn.ReLU(inplace=True))
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        res = x 
        H = self.Block_Conv1(x)
        H = self.Block_BN1(H)
        H = self.Block_Relu(H)

        H = self.Block_Conv2(H)
        H = self.Block_BN2(H)
        
        if self.downsample:
            res = self.downsample(res)
        H += res
        H = self.Block_Relu(H)

        return H


class BottleBlock(nn.Module):
    expansion = 4
    def __init__(self, BlockID, in_c, out_c, stride=1, downsample=None):
        super(BottleBlock, self).__init__()
        # print('in channels : ', in_c)
        self.BlockID = BlockID
        self.add_module('Block_Conv1', nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False))
        self.add_module('Block_BN1', nn.BatchNorm2d(out_c))

        self.add_module('Block_Conv2', nn.Conv2d(out_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False))
        self.add_module('Block_BN2', nn.BatchNorm2d(out_c))

        self.add_module('Block_Conv3', nn.Conv2d(out_c, out_c * 4, kernel_size=1, bias=False))
        self.add_module('Block_BN3', nn.BatchNorm2d(out_c * 4))

        self.add_module('Block_Relu', nn.ReLU(inplace=True))
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        res = x 
        H = self.Block_Conv1(x)
        H = self.Block_BN1(H)
        H = self.Block_Relu(H)

        H = self.Block_Conv2(H)
        H = self.Block_BN2(H)
        H = self.Block_Relu(H)

        H = self.Block_Conv3(H)
        H = self.Block_BN3(H)
        
        if self.downsample:
            res = self.downsample(res)
        H += res
        H = self.Block_Relu(H)

        return H


class MyResNet(nn.Module):
    def __init__(self, ResBlock, BackBoneStyle=50, S=7, B=2, C=20):
        super(MyResNet, self).__init__()
        self.in_channels = 64
        self.S = S
        self.B = B
        self.C = C 

        self.StageLayerNumMap = {
            18:[2, 2, 2, 2],
            34:[3, 4, 6, 3],
            50:[3, 4, 6, 3],
            101:[3, 4, 23, 3],
            152:[3, 8, 36, 3]
        }
        self.StageLayerNumberList = self.StageLayerNumMap[50]
        if BackBoneStyle not in self.StageLayerNumMap.keys():
            print('ResNet Layer not supported!!!')
            exit()
        else:
            self.StageLayerNumberList = self.StageLayerNumMap[BackBoneStyle]
        self.add_module('C1', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False))
        self.add_module('BN1', nn.BatchNorm2d(64))
        self.add_module('Relu', nn.ReLU(inplace=True))
        self.add_module('MaxPooling', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.C2 = self.build_ResBlock(BlockID=2, ResBlock=ResBlock, BlockNum=self.StageLayerNumberList[0], out_c=64)
        self.C3 = self.build_ResBlock(BlockID=3, ResBlock=ResBlock, BlockNum=self.StageLayerNumberList[0], out_c=128, stride=2)
        self.C4 = self.build_ResBlock(BlockID=4, ResBlock=ResBlock, BlockNum=self.StageLayerNumberList[0], out_c=256, stride=2)
        self.C5 = self.build_ResBlock(BlockID=5, ResBlock=ResBlock, BlockNum=self.StageLayerNumberList[0], out_c=512, stride=2)

        self.C6 = self.build_connect_layer(in_channels=2048)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self.kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                self.constant_init(m, 1)
            
    def build_connect_layer(self, in_channels):
        layer_list = []
        out_channels = self.B*5 + self.C
        det_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        layer_list.append(det_conv1)
        det_bn1 = nn.BatchNorm2d(out_channels)
        layer_list.append(det_bn1)
        # stride = 1
        if self.S not in [7, 14]:
            print('support cell be 7x7 or 14x14')
            exit()
        # if self.S == 14:
        #     stride = 2
        # print(stride)
        det_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        layer_list.append(det_conv2)
        det_bn2 = nn.BatchNorm2d(out_channels)
        layer_list.append(det_bn2)
        det_conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False)
        layer_list.append(det_conv3)
        det_bn3 = nn.BatchNorm2d(out_channels)
        layer_list.append(det_bn3)

        relu = nn.ReLU(inplace=True)
        layer_list.append(relu)
        
        return nn.Sequential(*layer_list)

    def constant_init(self, module, val, bias=0):
        nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def kaiming_init(self, module, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
        assert distribution in ['uniform', 'normal']
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, mode=mode, nonlinearity=nonlinearity)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def build_ResBlock(self, BlockID, ResBlock, BlockNum, out_c, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_c * ResBlock.expansion:
            downsample = nn.Sequential(
            nn.Conv2d(self.in_channels, out_c * ResBlock.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_c * ResBlock.expansion)
        )
        # print(downsample)
        layer_list = []
        layer_list.append(ResBlock((BlockID, 1), self.in_channels, out_c=out_c, stride=stride, downsample=downsample))
        self.in_channels = out_c * ResBlock.expansion
        # print(self.in_channels)
        for i in range(1, BlockNum):
            layer_list.append(ResBlock((BlockID, i+1), self.in_channels, out_c))

        return nn.Sequential(*layer_list)


    def forward(self, x):
        x = self.C1(x)
        x = self.BN1(x)
        x = self.Relu(x)
        x = self.MaxPooling(x)

        x = self.C2(x)

        x = self.C3(x)

        x = self.C4(x)

        x = self.C5(x)

        x = self.C6(x)
        x = F.sigmoid(x)
        x = x.permute(0, 2, 3, 1)

        # x = self.GlobalAveragePooling(x)
        # x = x.view(x.size(0), -1)
        # x = self.FC(x)
        
        return x



# if __name__ == "__main__":
#     tf = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#     device = 'cuda:0'

#     in_img = np.zeros((448, 448, 3), np.uint8)
#     t_img = transforms.ToTensor()(in_img)
#     t_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(t_img)
#     t_img.unsqueeze_(0)
#     t_img = t_img.to(device)
#     # t_img = tf(in_img)
#     # t_img
    
#     print(t_img.shape, t_img.device)
#     model = MyResNet(BottleBlock, BackBoneStyle=50, S=14, B=2, C=20)
#     # model = MyResNet(BasicBlock, BackBoneStyle=18)
#     model.to(device)
#     # model.forward(t_img)
#     # print(model)
#     summary(model, (3, 448, 448))
#     t = model.forward(t_img)
#     print(t.shape)
