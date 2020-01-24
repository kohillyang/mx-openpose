# Requirements

```
tqdm
opencv-python
easydict
pycocotools
gluoncv
mxnet
```

# Prepare for train.
Example command: 
```bash
PYTHONPATH=. /data2/zyx/yks/anaconda3/bin/python3 /data3/zyx/yks/mx-openpose/scripts/train_gluon_cpm.py \
--dataset-root=/data3/zyx/yks/dataset/coco2017 --gpus=7,8 --disable-fusion --backbone=res50
```
you may want to change dataset root and gpus by yourself.


# Demo
After you have trained your own model or download the pretrained model, you can use `scripts/evaluate.py` to evaluate the model.

Example command:
```bash
PYTHONPATH=. /data2/zyx/yks/anaconda3/bin/python3 /data3/zyx/yks/mx-openpose/scripts/evaluate.py \
--resume=pretrained/resnet50-cpm-resnet-cropped-flipped_rotated-47-0.0.params \
--dataset-root="/data3/zyx/yks/dataset/coco2017" \
--gpus="0" --stage=0 --viz
```
Also, you may want to change resume, dataset root and gpus by yourself.

Example Results of our implementation:

![](figures/Figure_1.png)

# Results on val 2017
Our implementation(Dilated-Resnet50 as backbone, 21 epochs):
```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.561
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.788
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.610
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.544
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.596
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.600
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.803
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.641
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.555
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.666
```

mAP of the original model(converted from caffe):
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.590
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.810
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.643
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.575
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.623
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.630
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.824
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.675
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.582
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.699
```

mAP of the original model (re-train) 38 epochs.
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.560
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.788
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.601
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.554
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.582
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.598
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.801
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.638
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.563
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.653
```

If initialize parameters with Xavier.

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.564
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.787
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.610
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.555
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.588
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.601
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.800
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.641
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.564
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.658
```
The original pretrained model converted from Caffe can be downloaded from <https://drive.google.com/drive/folders/0BzffphMuhDDMV0RZVGhtQWlmS1U>, which is bought from [mxnet_Realtime_Multi-Person_Pose_Estimation](https://github.com/dragonfly90/mxnet_Realtime_Multi-Person_Pose_Estimation) by @dragonfly90.<br>.
