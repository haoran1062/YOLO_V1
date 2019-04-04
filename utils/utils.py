# encoding:utf-8
import os, numpy as np, random
import torch

import torch.nn as nn
import torch.nn.functional as F

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
