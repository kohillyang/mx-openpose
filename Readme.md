# Requirements

`python3 -m pip install mxnet_cu100 -i https://pypi.tuna.tsinghua.edu.cn/simple --pre tqdm opencv-python easydict pycocotools gluoncv`

# Prepare for train.
Just run `python3 train_gluon_cpm_teacher`, you may want to change the coco path in configs.py


# Demo
After you have trained your own model or download the pretrained model, you can run `python3 demo_cpm.py` to evaluate the model.

