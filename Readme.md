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
Just run `python3 scripts/train_gluon_cpm.py`, you may want to change the coco path and gpus in configs.py


# Demo
After you have trained your own model or download the pretrained model, you can run `scripts/evaluate.py` to evaluate the model.
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
