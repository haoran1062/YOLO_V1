# encoding:utf-8
import os, numpy as np, random, cv2, logging
import torch

import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm

def compute_iou_matrix(bbox1, bbox2):
    '''
        input:
            tensor bbox1: [N, 4] (x1, y1, x2, y2)
            tensor bbox2: [M, 4]
        
        process:
            1. get two bbox max(left1, left2) and max(top1, top2) this is the Intersection's left-top point
            2. get two bbox min(right1, right2) and min(bottom1, bottom2) this is the Intersection's right-bottom point
            3. expand left-top/right-bottom list to [N, M] matrix
            4. Intersection W_H = right-bottom - left-top
            5. clip witch W_H < 0 = 0
            6. Intersection area = W * H
            7. IoU = I / (bbox1's area + bbox2's area - I)

        output:
            IoU matrix:
                        [N, M]

    '''
    if not isinstance(bbox1, torch.Tensor) or not isinstance(bbox2, torch.Tensor):
        print('compute iou input must be Tensor !!!')
        exit()
    N = bbox1.size(0)
    M = bbox2.size(0)
    b1_left_top = bbox1[:, :2].unsqueeze(1).expand(N, M, 2) # [N,2] -> [N,1,2] -> [N,M,2]
    b2_left_top = bbox2[:, :2].unsqueeze(0).expand(N, M, 2) # [M,2] -> [1,M,2] -> [N,M,2]

    left_top = torch.max(b1_left_top, b2_left_top)  # get two bbox max(left1, left2) and max(top1, top2) this is the Intersection's left-top point

    b1_right_bottom = bbox1[:, 2:].unsqueeze(1).expand(N, M, 2)
    b2_right_bottom = bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)

    right_bottom = torch.min(b1_right_bottom, b2_right_bottom)  # get two bbox min(right1, right2) and min(bottom1, bottom2) this is the Intersection's right-bottom point

    w_h = right_bottom - left_top   # get Intersection W and H
    w_h[w_h < 0] = 0    # clip -x to 0

    I = w_h[:, :, 0] * w_h[: ,: ,1] # get intersection area

    b1_area = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]) # [N, ]
    b1_area = b1_area.unsqueeze(1).expand_as(I) # [N, M]
    b2_area = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]) # [M, ]
    b2_area = b2_area.unsqueeze(0).expand_as(I) # [N, M]

    IoU = I / (b1_area + b2_area - I)   # [N, M] 

    return IoU

def convert_CxCyWH_to_X1Y1X2Y2(input_tensor, S, B, device):
    '''
        input:
            input_tensor:   [B, 4] B x bboxes with (CenterX, CenterY, W, H)
            B:  int Number B

        return:
            [B, 4] which with (Xleft, Ytop, Xright, Ybottom)
    '''
    # print('input tensor shape: ', input_tensor.size())
    assert input_tensor.size()[-1] == 4, 'convert position tensor must [n, 4], but this input last dim is %d'%(input_tensor.size()[-1])

    output_tensor = torch.FloatTensor(input_tensor.size(), device=device)
    output_tensor[:, :2] = input_tensor[:, :2] / (S) - 0.5 * input_tensor[:, 2:]
    output_tensor[:, 2:] = input_tensor[:, :2] / (S) + 0.5 * input_tensor[:, 2:]

    return output_tensor

def make_eval_tensor(batch_size, S, B, C, device='cpu'):
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
    return pred_tensor.to(device), target_tensor.to(device)

def decoder(pred, grid_num=7, B=2, device='cpu', thresh=0.3, nms_th=0.5, gt=False):
    '''
    pred (tensor) 1x7x7x30
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''
    boxes=[]
    cls_indexs=[]
    probs = []
    cell_size = 1./grid_num
    pred = pred.data
    pred = pred.squeeze(0) 
    
    # pred = pred.unsqueeze(0)
    # print(pred.shape)
    contain1 = pred[:,:,0].unsqueeze(2)
    contain2 = pred[:,:,1].unsqueeze(2)
    contain = torch.cat((contain1,contain2),2)
    mask1 = contain > 0.0001
    # mask1 = contain >= 0.
    mask2 = (contain==contain.max()) 
    mask = (mask1+mask2).gt(0)
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(B):
                if mask[i,j,b] == 1:
                    box = pred[i, j, B+b*4 : B+b*4+4]
                    contain_prob = torch.FloatTensor([pred[i,j,b]]).to(device)
                    
                    xy = torch.FloatTensor([j,i]).to(device)*cell_size 
                    box[:2] = box[:2]*cell_size + xy 
                    box_xy = torch.FloatTensor(box.size()).to(device)
                    box_xy[:2] = box[:2] - 0.5*box[2:]
                    box_xy[2:] = box[:2] + 0.5*box[2:]
                    max_prob, cls_index = torch.max(pred[i,j,5*B:],0)
                    # print(max_prob*contain_prob, cls_index)
                    if float((contain_prob*max_prob)[0]) > thresh:
                        boxes.append(box_xy.view(1,4))
                        cls_indexs.append(cls_index.view(1))
                        probs.append(contain_prob*max_prob)
    # print(boxes, cls_indexs, probs)
    if len(boxes) ==0:
        boxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes,0) #(n,4)
        probs = torch.cat(probs,0) #(n,)
        cls_indexs = torch.cat(cls_indexs,0) #(n,)
    
    nms_thresh = 1.0
    if not gt:
        nms_thresh=nms_th
    keep = nms(boxes,probs, nms_thresh)
    return boxes[keep],cls_indexs[keep],probs[keep]
    

def nms(bboxes,scores,threshold=0.25):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1) * (y2-y1)

    _,order = scores.sort(0,descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)


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

def voc_eval(preds,target,VOC_CLASSES=VOC_CLASSES,threshold=0.5,use_07_metric=False, logger=None):
    '''
    preds {'cat':[[image_id,confidence,x1,y1,x2,y2],...],'dog':[[],...]}
    target {(image_id,class):[[],]}
    '''
    aps = []
    for i,class_ in enumerate(VOC_CLASSES):
        pred = preds[class_] #[[image_id,confidence,x1,y1,x2,y2],...]
        if len(pred) == 0: #如果这个类别一个都没有检测到的异常情况
            ap = -1
            if logger:
                logger.info('---class {} ap {}---'.format(class_,ap))
            else:
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
        if logger:
            logger.info('---class {} ap {}---'.format(class_,ap))
        else:
            print('---class {} ap {}---'.format(class_,ap))
        aps += [ap]
    mAP = np.mean(aps).item()
    if logger:
            logger.info('---map {}---'.format(mAP))
    else:
        print('---map {}---'.format(mAP))
    return mAP

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

def prep_test_data(file_path, little_test=None):
    target =  defaultdict(list)
    
    image_list = [] #image path list
    f = open(file_path)
    lines = f.readlines()
    file_list = []
    for line in lines:
        file_list.append(line.strip())
        

    f.close()

    if little_test:
        file_list = file_list[:little_test]

    print('---prepare target---')
    img_size = (448, 448)
    bar = tqdm(total=len(file_list))
    for index,image_file in enumerate(file_list):

        image_id = image_file.split('/')[-1].split('.')[0]
        image_list.append(image_id)
        label_list = from_img_path_get_label_list(image_file, img_size=img_size)
        for i in label_list:
            
            class_name = VOC_CLASSES[i[0]]
            target[(image_id,class_name)].append(i[1:])
       
        bar.update(1)
    bar.close()
    return target

def run_test_mAP(YOLONet, target, test_datasets, data_len, S=7, device='cuda:0', reversed=False, logger=None, little_test=None):
    preds = defaultdict(list)
    # bar = tqdm(total=data_len)

    with torch.no_grad():
        for i, (images, now_target, fname) in tqdm(enumerate(test_datasets)):
            if little_test:
                if i >= little_test:
                    break
            images = images.to(device)
            now_target = now_target.to(device)
            
            img_id = fname.split('/')[-1].split('.')[0]
            pred = YOLONet(images[None, :, :, :])
            if reversed:
                pred = convert_input_tensor_dim(pred)
            bboxes, clss, confs = decoder(pred, grid_num=S, device=device, thresh=0.005, nms_th=.45)
            bboxes = bboxes.clamp(min=0., max=1.)
            bboxes = bbox_un_norm(bboxes)
            if len(confs) == 1 and confs[0].item() == 0. :
                continue
            for j in range(len(confs)):
                preds[VOC_CLASSES[clss[j].item()]].append( [img_id, confs[j].item(), int(bboxes[j][0].item()), int(bboxes[j][1].item()), int(bboxes[j][2].item()), int(bboxes[j][3].item()), ] )
            
    if logger:
        logger.info('---start evaluate---')
    else:
        print('---start evaluate---')

    return voc_eval(preds,target,VOC_CLASSES=VOC_CLASSES, threshold=0.5, use_07_metric=False, logger=logger)

# def run_test_mAP(YOLONet, target, test_loader, data_len, S=7, device='cuda:0', logger=None):
#     preds = defaultdict(list)
#     bar = tqdm(total=data_len)
#     with torch.no_grad():
#         for i, (images, now_target, fname) in enumerate(test_loader):
            
#             images = images.to(device)
#             now_target = now_target.to(device)

#             pred_batch = YOLONet(images)
#             NN = pred_batch.shape[0]
#             for iii in range(NN):
#                 img_id = fname[iii].split('/')[-1].split('.')[0]
#                 pred = pred_batch[iii, :, :, :]
#                 pred = pred.unsqueeze(0)

#                 bboxes, clss, confs = decoder(pred, grid_num=S, device=device, thresh=0.0001, nms_th=1.)# 23456785)
#                 bboxes = bbox_un_norm(bboxes)
#                 if len(confs) == 1 and confs[0].item() == 0. :
#                     continue
#                 for j in range(len(confs)):
#                     preds[VOC_CLASSES[clss[j].item()]].append( [img_id, confs[j].item(), int(bboxes[j][0].item()), int(bboxes[j][1].item()), int(bboxes[j][2].item()), int(bboxes[j][3].item()), ] )

#             bar.update(1)
#         bar.close()
#     # print('\n'*5)
#     if logger:
#         logger.info('---start evaluate---')
#     else:
#         print('---start evaluate---')
#     return voc_eval(preds,target,VOC_CLASSES=VOC_CLASSES, threshold=0.5, use_07_metric=False, logger=logger)
    
def draw_debug_rect(img, bboxes, clss, confs, color=(0, 255, 0), show_time=10000):

    if isinstance(img, torch.Tensor):
        img = img.mul(255).byte()
        img = img.cpu().numpy()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img.shape)
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.tolist()
    def bbox_un_norm(img, bboxes):
        h, w, c = img.shape
        for bbox in bboxes:
            bbox[0] = int(bbox[0] * w)
            bbox[1] = int(bbox[1] * h)
            bbox[2] = int(bbox[2] * w)
            bbox[3] = int(bbox[3] * h)
        return bboxes

    if bboxes[0][0] < 1:
        bboxes = bbox_un_norm(img, bboxes)
    # print(bboxes)
    for i, box in enumerate(bboxes):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=color,thickness=2)
        cls_i = int(clss[i].item())
        cv2.putText(img, '%s %.2f'%(VOC_CLASSES[cls_i], confs[i]), (box[0], box[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, 10)
    cv2.imshow('debug draw bboxes', img)
    cv2.waitKey(show_time)
    
def cv_resize(img, resize=448):
    return cv2.resize(img, (resize, resize))

def create_logger(base_path, log_name):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    fhander = logging.FileHandler('%s/%s.log'%(base_path, log_name))
    fhander.setLevel(logging.INFO)

    shander = logging.StreamHandler()
    shander.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
    fhander.setFormatter(formatter) 
    shander.setFormatter(formatter) 

    logger.addHandler(fhander)
    logger.addHandler(shander)

    return logger

if __name__ == "__main__":
    b1 = [
        [10, 20, 100, 123],
        [200, 300, 300, 350]
    ]

    b2 = [
        [50, 60, 150, 120],
        [0, 10, 123, 150],
        [170, 190, 310, 400]
    ]

    nb1 = np.array(b1, np.float32)
    nb2 = np.array(b2, np.float32)

    tb1 = torch.from_numpy(nb1)
    tb2 = torch.from_numpy(nb2)

    iou = compute_iou_matrix(tb1, tb2)
    print(iou)
