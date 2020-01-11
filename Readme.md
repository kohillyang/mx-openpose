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
python3 scripts/train_gluon_cpm.py --dataset-root="/data3/zyx/yks/dataset/coco2017" --gpus="7,8"
```
you may want to change dataset root and gpus by yourself.


# Demo
After you have trained your own model or download the pretrained model, you can use `scripts/evaluate.py` to evaluate the model.

Example command:
```bash
/data2/zyx/yks/anaconda3/bin/python3 scripts/evaluate.py --resume=output/cpm/resnet50-cpm-resnet-cropped-flipped_rotated-masked-26-0.0.params --dataset-root="/data3/zyx/yks/dataset/coco2017" --gpus="2" --viz
```
Also, you may want to change resume, dataset root and gpus by yourself.

Example Results of our implementation:

![](figures/Figure_1.png)

# Results on val 2017
Our implementation:
```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.532
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.765
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.569
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.515
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.566
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.572
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.784
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.607
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.526
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.638
```

mAP of the original model:
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


The original pretrained model converted from Caffe can be downloaded from <https://drive.google.com/drive/folders/0BzffphMuhDDMV0RZVGhtQWlmS1U>, which is bought from [mxnet_Realtime_Multi-Person_Pose_Estimation](https://github.com/dragonfly90/mxnet_Realtime_Multi-Person_Pose_Estimation) by @dragonfly90.<br>.
43:
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.515
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.738
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.559
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.555
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.484
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.543
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.747
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.589
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.565
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.516

42:
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.504
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.720
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.503
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.533
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.479
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.531
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.726
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.537
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.544
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.514
 
41:
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.528
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.736
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.546
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.566
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.499
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.559
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.747
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.579
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.581
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.533
    
40: 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.521
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.724
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.587
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.566
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.488
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.553
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.737
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.611
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.579
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.521

39
 
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.522
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.729
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.559
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.558
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.493
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.549
Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.737
Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.589
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.569
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.526

38:
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.515
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.735
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.554
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.566
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.473
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.544
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.747
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.589
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.577
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.505
None


