# Requirements

`python3 -m pip install mxnet_cu100 -i https://pypi.tuna.tsinghua.edu.cn/simple --pre tqdm opencv-python easydict pycocotools gluoncv`

# Prepare for train.
Just run `python3 train_gluon_cpm_teacher`, you may want to change the coco path in configs.py


# Demo
After you have trained your own model or download the pretrained model, you can run `python3 demo_cpm.py` to evaluate the model.


# Results on val 2017


```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.282
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.584
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.249
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.198
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.403
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.342
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.634
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.326
 0Av0e0rage Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.207
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.536
```
