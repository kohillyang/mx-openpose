# Requirements

`python3 -m pip install mxnet_cu100 -i https://pypi.tuna.tsinghua.edu.cn/simple --pre tqdm opencv-python easydict pycocotools gluoncv`

# Prepare for train.
Just run `python3 train_gluon_cpm.py`, you may want to change the coco path and gpus in configs.py


# Demo
After you have trained your own model or download the pretrained model, you can run `python3 demo_cpm.py` to evaluate the model.


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
