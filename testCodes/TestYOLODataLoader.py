# encoding:utf-8
import os, sys, numpy as np, random, time, cv2
import torch

import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import imgaug as ia
from imgaug import augmenters as iaa
ia.seed(1)
class yoloDataset(data.Dataset):
    image_size = 448
    def __init__(self,list_file,train,transform, device, little_train=False, S=7):
        print('data init')
        
        self.train = train
        self.transform=transform
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.S = S
        self.B = 2
        self.C = 20
        self.device = device

        self.augmentation = iaa.Sometimes(0.5,
            iaa.SomeOf((1, 6), [
                iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
                iaa.Sharpen((0.1, 1.0)),       # sharpen the image
                iaa.GaussianBlur(sigma=(2., 3.5)),
                iaa.OneOf([
                    iaa.GaussianBlur(sigma=(2., 3.5)),
                    iaa.AverageBlur(k=(2, 5)),
                    iaa.BilateralBlur(d=(7, 12), sigma_color=(10, 250), sigma_space=(10, 250)),
                    iaa.MedianBlur(k=(3, 7)),
                ]),
                # iaa.Fliplr(1.0),
                # iaa.Flipud(1.0),

                iaa.AddElementwise((-50, 50)),
                iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),
                iaa.JpegCompression(compression=(80, 95)),

                iaa.Multiply((0.5, 1.5)),
                iaa.MultiplyElementwise((0.5, 1.5)),
                iaa.ReplaceElementwise(0.05, [0, 255]),
                iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                                children=iaa.WithChannels(2, iaa.Add((-10, 50)))),
                iaa.OneOf([
                    iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                                       children=iaa.WithChannels(1, iaa.Add((-10, 50)))),
                    iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                                       children=iaa.WithChannels(2, iaa.Add((-10, 50)))),
                ]),

            ], random_order=True)
        )

        torch.manual_seed(23)
        with open(list_file) as f:
            lines  = f.readlines()
        
        if little_train:
            lines = lines[:64]

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            
        self.num_samples = len(self.fnames)
    
    def get_boxes_labels(self, in_path):
        bboxes = []
        labels = []
        with open(in_path.replace('JPEGImages', 'labels').replace('jpg', 'txt'), 'r') as f:
            for line in f:
                ll = line.strip().split(' ')
                labels.append(int(ll[0]))
                x = float(ll[1])
                y = float(ll[2])
                w = float(ll[3])
                h = float(ll[4])
                bboxes.append([x, y, w, h])
        return torch.Tensor(bboxes), torch.LongTensor(labels)

    def __getitem__(self,idx):
        
        fname = self.fnames[idx]
        # fname = '/data/datasets/VOCdevkit/VOC2007/JPEGImages/000067.jpg'
        # print(fname)
        img = cv2.imread(fname)
        boxes, labels = self.get_boxes_labels(fname)

        if self.train:
            # TODO
            # add data augument
            img = self.augmentation.augment_image(img)
            # print(img.shape)
            # pass
        
        target = self.encoder(boxes,labels)# 7x7x30
        
        img = self.transform(img)
        # print(fname)
        # print(type(img))
        return img, target
        # return img.to(self.device), target.to(self.device)

    def __len__(self):
        return self.num_samples

    def encoder(self,boxes,labels):
        '''
        boxes (tensor) [[cx,cy,w,h],[]]
        labels (tensor) [...]
        return 7x7x30
        '''
        target = torch.zeros((self.S, self.S, self.B*5 + self.C))
        cell_size = 1./self.S
        # wh = boxes[:,2:]-boxes[:,:2]
        # cxcy = (boxes[:,2:]+boxes[:,:2])/2
        wh = boxes[:, 2:]
        cxcy = boxes[:, :2]
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample/cell_size).ceil()-1 #
            target[int(ij[1]),int(ij[0]),:self.B] = 1
            target[int(ij[1]),int(ij[0]),self.B*5 + int(labels[i])] = 1
            xy = ij*cell_size 
            delta_xy = (cxcy_sample -xy)/cell_size
            for b in range(self.B):
                target[int(ij[1]),int(ij[0]),self.B+b*4 : self.B+b*4+2] = delta_xy
                target[int(ij[1]),int(ij[0]),self.B+b*4+2 : self.B+b*4+4] = wh[i]
                
            
        return target


if __name__ == "__main__":

    from utils import decoder, draw_debug_rect, cv_resize

    transform = transforms.Compose([
        transforms.Lambda(cv_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    device = 'cpu'
    train_dataset = yoloDataset(list_file='2007_val.txt',train=False, transform = transform, device=device, little_train=True, S=14)
    train_loader = DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=0)
    train_iter = iter(train_loader)
    for i in range(1):
        img, target = next(train_iter)
    print(img.shape, target.shape)
    boxes, clss, confs = decoder(target)
    print(boxes, clss, confs)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    un_normal_trans = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    img = un_normal_trans(img.squeeze(0))
    draw_debug_rect(img.permute(1, 2 ,0), boxes)
    for i in range(14):
        for j in range(14):
            print(target[:, i:i+1, j:j+1, :])