# Requirements

`python3 -m pip install mxnet_cu100 -i https://pypi.tuna.tsinghua.edu.cn/simple --pre tqdm opencv-python easydict pycocotools gluoncv`

# Prepare for train.
Just run `python3 train_gluon_cpm.py`, you may want to change the coco path and gpus in configs.py


# Demo
After you have trained your own model or download the pretrained model, you can run `python3 demo_cpm.py` to evaluate the model.


# Results on val 2017

Using three scales and no flipping, the following is the results, the mAP of the implementation after 24 epochs
on the first 50 images of val2017 is 0.396 lower than the original model(0.438).
I tried my best to find the reasons but I failed, please contact me if you have any ideas.

```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.396
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.704
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.387
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.365
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.443
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.444
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.728
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.450
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.378
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.537
```

Single scale mAP of the original model:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.438
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.747
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.456
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.407
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.495
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.492
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.770
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.513
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.418
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.597
```