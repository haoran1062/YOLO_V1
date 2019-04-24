# YOLO_V1
### ReadMe 
 - this repo is an easy implementation for YOLO V1 with **Pytorch 0.4** and **python3.6**
 - still updating... now, **YOLO V1 loss**, **Resnet/Densenet backbone**, **data loader**, **eval/test voc mAP**, **show plots by visdom** finished. 
 - now I'm working on improve performance
  - find low mAP case of without warmming up
  - now performance is approach to origin paper(I use better backbone, so the performance should better than origin paper, still training)
 - ##### performance
   - train with voc+ datasets (2007train 2007val 2012train 2012val total ~`11k`)
   - test on **voc2007 test** 
     - backbone `densenet121`, with **warmming up policy** train epoch `103`, **mAP**: `0.6038`
     - backbone `resnet50`, with **warmming up policy** train epoch `92`, **mAP**: `0.632`

 ### TODO
  - `improve performance`
    - `optimize warmming up/learning rate policy`
    - `make better data augment policy`
  - `better code standards`
  - `add support cell size to 14x14 to get better performance`
  - `compress model/run on ARM`



