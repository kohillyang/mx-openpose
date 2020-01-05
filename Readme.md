# Requirements

`python3 -m pip install mxnet_cu100 -i https://pypi.tuna.tsinghua.edu.cn/simple --pre tqdm opencv-python easydict pycocotools gluoncv`

# Prepare for train.
Just run `python3 train_gluon_cpm.py`, you may want to change the coco path and gpus in configs.py


# Demo
After you have trained your own model or download the pretrained model, you can run `python3 demo_cpm.py` to evaluate the model.


# Results on val 2017

The following is the results after 24 epochs on the first 50 images of val2017, the mAP(single scale) is 0.300,
lower than the original model(0.340). I tried my best to find the reasons but I failed, please contact
me if you have any ideas.

```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.300
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.626
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.239
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.221
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.415
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.354
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.663
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.309
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.229
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.532
```

Single scale mAP of the original model:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.341
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.648
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.324
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.242
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.482
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.399
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.680
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.394
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.250
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.614
```