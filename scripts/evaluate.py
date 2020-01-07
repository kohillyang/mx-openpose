import sys

sys.path.append("MobulaOP")
import tqdm
import mxnet as mx
import numpy as np
import os, cv2
import json

from models.cpm import CPMNet
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from configs import get_coco_config
from utils.heatpaf_parser import parse_heatpaf_py, parse_heatpaf_cxx
import mobula


def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :] * 0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :] * 0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :] * 0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :] * 0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def pad_image(img_ori, dshape=(368, 368)):
    fscale = 1.0 * dshape[0] / img_ori.shape[0]
    img_resized = cv2.resize(img_ori, dsize=(0, 0), fx=fscale, fy=fscale)
    image_padded, pad = padRightDownCorner(img_resized, 8, 128)
    return img_resized, image_padded, pad, fscale


if __name__ == '__main__':
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
    limbSeq = [[1, 8],
               [8, 9],
               [9, 10],
               [1, 11],
               [11, 12],
               [12, 13],
               [1, 2],
               [2, 3],
               [3, 4],
               [2, 16],
               [1, 5],
               [5, 6],
               [6, 7],
               [5, 17],
               [1, 0],
               [0, 14],
               [0, 15],
               [14, 16],
               [15, 17]]
    limbSeq = np.array(limbSeq)
    mobula.op.load('HeatPafParser', os.path.join(os.path.dirname(__file__), "utils/operator_cxx"))
    dshape = (512, 512)
    ctx_list = [mx.cpu(0)]
    config = get_coco_config()
    # baseDataSet = COCOKeyPoints(root=config.TRAIN.DATASET.coco_root, splits=("person_keypoints_val2017",))
    # val_transform = transforms.Compose([transforms.ImagePad(dshape)])
    # val_dataset = PafHeatMapDataSet(baseDataSet, config, val_transform)
    # number_of_keypoints = val_dataset.number_of_keypoints
    results = []
    image_ids = []
    annFile = os.path.join(config.TRAIN.DATASET.coco_root, 'annotations/person_keypoints_val2017.json')
    image_dir = os.path.join(config.TRAIN.DATASET.coco_root, 'val2017')
    cocoGt = COCO(annFile)
    cats = cocoGt.loadCats(cocoGt.getCatIds())
    catIds = cocoGt.getCatIds(catNms=['person'])
    # imgIds = cocoGt.getImgIds(catIds=catIds)

    # # net = DRN50_GCN(num_classes=val_dataset.number_of_keypoints + 2 * val_dataset.number_of_pafs)
    # # sym, _, _ = mx.model.load_checkpoint('pretrained/pose', 0)
    # net = CPMVGGNet(resize=False)
    # net.collect_params().load("pretrained/pose-0000.params")
    net = CPMNet(19, 19, resize=False)
    net.collect_params().load("output/cpm/resnet50-cpm-resnet-cropped-flipped_rotated-30-0.0.params")
    net.collect_params().reset_ctx(ctx_list)

    imgIds = cocoGt.getImgIds(catIds=catIds)
    for i in tqdm.tqdm(range(10)):
        image_id = imgIds[i]
        img = cocoGt.loadImgs(image_id)[0]
        image_path = os.path.join(image_dir, img['file_name'])
        # image_path = "figures/test2.jpg"
        image_id = img['id']
        image_ids.append(image_id)
        image_ori = cv2.imread(image_path)[:, :, ::-1]

        inputs = []
        for scale in [1]:
            image_resized = cv2.resize(image_ori, (0, 0), fx=scale, fy=scale)[np.newaxis]
            inputs.append(mx.nd.array(image_resized).as_in_context(ctx_list[0]))
        net_results = [net(x) for x in inputs]
        heatmaps = [mx.nd.contrib.BilinearResize2D(x[-1], width=image_ori.shape[1], height=image_ori.shape[0]).asnumpy()
                    for x in net_results]
        pafmaps = [mx.nd.contrib.BilinearResize2D(x[-2], width=image_ori.shape[1], height=image_ori.shape[0]).asnumpy()
                   for x in net_results]
        heatmap_mean = np.mean(heatmaps, axis=0)[0]
        pafmap_mean = np.mean(pafmaps, axis=0)[0]

        r1 = parse_heatpaf_cxx(heatmap_mean.astype(np.float64), pafmap_mean.astype(np.float64), limbSeq, image_id)
        r2 = parse_heatpaf_py(image_ori, heatmap_mean.transpose((1, 2, 0)), pafmap_mean.transpose((1, 2, 0)), limbSeq,
                              image_id)
        results.extend(r1)

    annType = ['segm', 'bbox', 'keypoints']
    annType = annType[2]  # specify type here
    prefix = 'person_keypoints' if annType == 'keypoints' else 'instances'
    cocoGt = COCO(annFile)
    cats = cocoGt.loadCats(cocoGt.getCatIds())
    catIds = cocoGt.getCatIds(catNms=['person'])
    imgIds = cocoGt.getImgIds(catIds=catIds)

    with open('evaluationResult.json', 'w') as outfile:
        json.dump(results, outfile)
    resJsonFile = 'evaluationResult.json'
    cocoDt2 = cocoGt.loadRes(resJsonFile)

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt2, annType)
    cocoEval.params.imgIds = image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    k = cocoEval.summarize()
    print(k)
