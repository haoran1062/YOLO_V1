#encoding:utf-8
import os, numpy as np
from utils.utils import *

VOC_CLASSES = (   
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
Color = [[0, 0, 0],
                    [128, 0, 0],
                    [0, 128, 0],
                    [128, 128, 0],
                    [0, 0, 128],
                    [128, 0, 128],
                    [0, 128, 128],
                    [128, 128, 128],
                    [64, 0, 0],
                    [192, 0, 0],
                    [64, 128, 0],
                    [192, 128, 0],
                    [64, 0, 128],
                    [192, 0, 128],
                    [64, 128, 128],
                    [192, 128, 128],
                    [0, 64, 0],
                    [128, 64, 0],
                    [0, 192, 0],
                    [128, 192, 0],
                    [0, 64, 128]]
def voc_ap(rec,prec,use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0.,1.1,0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec>=t])
            ap = ap + p/11.

    else:
        # correct ap caculation
        mrec = np.concatenate(([0.],rec,[1.]))
        mpre = np.concatenate(([0.],prec,[0.]))

        for i in range(mpre.size -1, 0, -1):
            mpre[i-1] = np.maximum(mpre[i-1],mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

def voc_eval(preds,target,VOC_CLASSES=VOC_CLASSES,threshold=0.5,use_07_metric=False,):
    '''
    preds {'cat':[[image_id,confidence,x1,y1,x2,y2],...],'dog':[[],...]}
    target {(image_id,class):[[],]}
    '''
    aps = []
    for i,class_ in enumerate(VOC_CLASSES):
        pred = preds[class_] #[[image_id,confidence,x1,y1,x2,y2],...]
        if len(pred) == 0: #如果这个类别一个都没有检测到的异常情况
            ap = -1
            print('---class {} ap {}---'.format(class_,ap))
            aps += [ap]
            break
        #print(pred)
        image_ids = [x[0] for x in pred]
        confidence = np.array([float(x[1]) for x in pred])
        BB = np.array([x[2:] for x in pred])
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        npos = 0.
        for (key1,key2) in target:
            if key2 == class_:
                npos += len(target[(key1,key2)]) #统计这个类别的正样本，在这里统计才不会遗漏
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d,image_id in enumerate(image_ids):
            bb = BB[d] #预测框
            if (image_id,class_) in target:
                BBGT = target[(image_id,class_)] #[[],]
                for bbgt in BBGT:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(bbgt[0], bb[0])
                    iymin = np.maximum(bbgt[1], bb[1])
                    ixmax = np.minimum(bbgt[2], bb[2])
                    iymax = np.minimum(bbgt[3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    union = (bb[2]-bb[0]+1.)*(bb[3]-bb[1]+1.) + (bbgt[2]-bbgt[0]+1.)*(bbgt[3]-bbgt[1]+1.) - inters
                    if union == 0:
                        print(bb,bbgt)
                    
                    overlaps = inters/union
                    if overlaps > threshold:
                        tp[d] = 1
                        BBGT.remove(bbgt) #这个框已经匹配到了，不能再匹配
                        if len(BBGT) == 0:
                            del target[(image_id,class_)] #删除没有box的键值
                        break
                fp[d] = 1-tp[d]
            else:
                fp[d] = 1
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp/float(npos)
        prec = tp/np.maximum(tp + fp, np.finfo(np.float64).eps)
        #print(rec,prec)
        ap = voc_ap(rec, prec, use_07_metric)
        print('---class {} ap {}---'.format(class_,ap))
        aps += [ap]
    print('---map {}---'.format(np.mean(aps)))

def test_eval():
    preds = {'cat':[['image01',0.9,20,20,40,40],['image01',0.8,20,20,50,50],['image02',0.8,30,30,50,50]],'dog':[['image01',0.78,60,60,90,90]]}
    target = {('image01','cat'):[[20,20,41,41]],('image01','dog'):[[60,60,91,91]],('image02','cat'):[[30,30,51,51]]}
    voc_eval(preds,target,VOC_CLASSES=['cat','dog'])

def from_img_path_get_label_list(img_path, img_size=(448, 448)):
    '''
        return [ [label, x0, y0, x1, y1] ]
    '''
    label_path = img_path.replace('JPEGImages', 'labels').replace('jpg', 'txt')
    label_list = []
    with open(label_path, 'r') as f:
        for line in f:
            ll = line.strip().split(' ')
            label = int(ll[0])
            x = float(ll[1])
            y = float(ll[2])
            w = float(ll[3])
            h = float(ll[4])
            x0 = int( (x - 0.5*w) * img_size[0] )
            y0 = int( (y - 0.5*h) * img_size[1] )
            x1 = int( (x + 0.5*w) * img_size[0] )
            y1 = int( (y + 0.5*h) * img_size[1] )
            label_list.append([label, x0, y0, x1, y1])
    return label_list

def bbox_un_norm(bboxes, img_size=(448, 448)):
    (w, h) = img_size
    for bbox in bboxes:
        bbox[0] = int(bbox[0] * w)
        bbox[1] = int(bbox[1] * h)
        bbox[2] = int(bbox[2] * w)
        bbox[3] = int(bbox[3] * h)
    return bboxes

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
    file_path = '2007_train.txt'
    # file_path = '2007_val.txt'
    debug_n = 100

    target =  defaultdict(list)
    preds = defaultdict(list)
    image_list = [] #image path list

    f = open(file_path)
    lines = f.readlines()
    file_list = []
    for line in lines:
        file_list.append(line.strip())
        # splited = line.strip().replace('JPEGImages', 'labels').replace('jpg', 'txt')
        # file_list.append(splited)
    f.close()

    # if debug_n:
        # file_list = file_list[:debug_n]

    print('---prepare target---')
    img_size = (448, 448)
    bar = tqdm(total=len(file_list))
    for index,image_file in enumerate(file_list):
        # img = cv2.imread(image_file)

        image_id = image_file.split('/')[-1].split('.')[0]
        image_list.append(image_id)
        label_list = from_img_path_get_label_list(image_file)
        for i in label_list:
            
            class_name = VOC_CLASSES[i[0]]
            target[(image_id,class_name)].append(i[1:])
       
        bar.update(1)
    bar.close()
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
        model_name = 'densenet_best.pth'
        model_name = 'densenet_yolo.pth'

    device = 'cuda:0'
    batch_size = 1
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

    bar = tqdm(len(test_dataset))
    with torch.no_grad():
        for i, (images, now_target, fname) in enumerate(test_loader):
            # print(fname)
            # if i > debug_n:
                # print(preds)
                # break
            img_id = fname[0].split('/')[-1].split('.')[0]
            images = images.to(device)
            now_target = now_target.to(device)
            # print(images.shape, target.shape)
            
            pred = YOLONet(images)
            
            images = un_normal_trans(images.squeeze(0))

            bboxes, clss, confs = decoder(pred, grid_num=S, device=device, thresh=0.1, nms_th=0.5)# 23456785)
            # draw_debug_rect(images.permute(1, 2 ,0), bboxes, clss, confs, show_time=1000)
            # print(confs, bboxes, clss)
            bboxes = bbox_un_norm(bboxes)
            if len(confs) == 1 and confs[0].item() == 0. :
                continue
            for j in range(len(confs)):
                preds[VOC_CLASSES[clss[j].item()]].append( [img_id, confs[j].item(), int(bboxes[j][0].item()), int(bboxes[j][1].item()), int(bboxes[j][2].item()), int(bboxes[j][3].item()), ] )

            bar.update(1)
        bar.close()


        # cv2.imwrite('testimg/'+image_path,image)
        # count += 1
        # if count == 100:
        #     break
    
    print('---start evaluate---')
    voc_eval(preds,target,VOC_CLASSES=VOC_CLASSES, threshold=0.5, use_07_metric=True)