# YOLO_V1
### ReadMe 
 - this repo is an easy implementation for YOLO V1 with **Pytorch 0.4** and **python3.6**
 - still debugging... now, **YOLO V1 loss**, **Resnet/Densenet backbone**, **data loader**, **eval/test voc mAP** finished. 
 - now I'm working on improve performance
 - ##### performance
   - train with voc+ datasets (2007train 2007val 2012train 2012val)
   - test on **voc2007 test** 
     - backbone `densenet121`, train epoch `23`, **mAP**: `0.417`
     - backbone `resnet50`, train epoch `100+`, **mAP**: `0.447`

 ### TODO
  - `find reason of low performance`
  - `soft-NMS/softer-NMS`
  - `train voc 2007`



