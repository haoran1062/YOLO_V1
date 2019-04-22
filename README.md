# YOLO_V1
### ReadMe 
 - this repo is an easy implementation for YOLO V1 with **Pytorch 0.4** and **python3.6**
 - still debugging... now, **YOLO V1 loss**, **Resnet/Densenet backbone**, **data loader**, **eval/test voc mAP** finished. 
 - now I'm working on improve performance
  - find low mAP case of without warmming up
 - ##### performance
   - train with voc+ datasets (2007train 2007val 2012train 2012val)
   - test on **voc2007 test** 
     - backbone `densenet121`, with warmming up policy train epoch `103`, **mAP**: `0.6038`
     - backbone `resnet50`, train epoch `114`, **mAP**: `0.531` (`44` improve to `53` by add bbox affine augments)

 ### TODO
  - `find reason of low performance`



