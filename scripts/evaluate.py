import cv2
import json
import os
import sys

import mxnet as mx
import numpy as np
import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from configs import get_coco_config
from datasets.cocodatasets import COCOKeyPoints
from models.cpm import CPMVGGNet

sys.path.append("MobulaOP")
from utils.heatpaf_parser import parse_heatpaf_py, parse_heatpaf_cxx
import mobula


if __name__ == '__main__':
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

    mobula.op.load('HeatPafParser', os.path.join(os.path.dirname(__file__), "../utils/operator_cxx"))
    ctx_list = [mx.gpu(0)]
    config = get_coco_config()
    baseDataSet = COCOKeyPoints(root=config.TRAIN.DATASET.coco_root, splits=("person_keypoints_val2017",))
    results = []
    image_ids = []
    annFile = os.path.join(config.TRAIN.DATASET.coco_root, 'annotations/person_keypoints_val2017.json')
    image_dir = os.path.join(config.TRAIN.DATASET.coco_root, 'val2017')
    cocoGt = COCO(annFile)
    cats = cocoGt.loadCats(cocoGt.getCatIds())
    catIds = cocoGt.getCatIds(catNms=['person'])
    imgIds = cocoGt.getImgIds(catIds=catIds)

    net = CPMVGGNet(resize=False)
    net.collect_params().load("pretrained/pose-0000.params")

    # net = CPMNet(19, 19, resize=False)
    # net.collect_params().load("output/cpm/resnet50-cpm-resnet-cropped-flipped_rotated-47-0.0.params")
    net.collect_params().reset_ctx(ctx_list)

    for i in tqdm.trange(len(imgIds)):
        image_id = imgIds[i]
        img = cocoGt.loadImgs(image_id)[0]
        image_path = os.path.join(image_dir, img['file_name'])
        # image_path = "figures/test2.jpg"
        image_id = img['id']
        image_ids.append(image_id)
        image_ori = cv2.imread(image_path)[:, :, ::-1]
        boxsize = 368
        scale_search = [0.5, 1, 1.5, 2]
        multiplier = [x * boxsize * 1.0 / image_ori.shape[0] for x in scale_search]
        heatmaps = []
        pafmaps = []
        for scale in multiplier:
            image_resized = cv2.resize(image_ori, (0, 0), fx=scale, fy=scale)
            fp = lambda x: x if x % 8==0 else x + 8 - x % 8
            image_resized_padded = np.zeros(shape=(fp(image_resized.shape[0]), fp(image_resized.shape[1]), image_resized.shape[2]), dtype=np.float32)
            image_resized_padded[:image_resized.shape[0], :image_resized.shape[1], :image_resized.shape[2]] = image_resized
            data = mx.nd.array(image_resized[np.newaxis]).as_in_context(ctx_list[0])
            net_results = net(data)
            heatmap = net_results[-1][0].asnumpy().transpose((1, 2, 0))
            pafmap = net_results[-2][0].asnumpy().transpose((1, 2, 0))
            heatmap = cv2.resize(heatmap, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
            pafmap = cv2.resize(pafmap, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:image_resized.shape[0], :image_resized.shape[1]]
            pafmap = pafmap[:image_resized.shape[0], :image_resized.shape[1]]
            heatmap = cv2.resize(heatmap, (image_ori.shape[1], image_ori.shape[0]), interpolation=cv2.INTER_CUBIC)
            pafmap = cv2.resize(pafmap, (image_ori.shape[1], image_ori.shape[0]), interpolation=cv2.INTER_CUBIC)
            heatmaps.append(heatmap)
            pafmaps.append(pafmap)
        heatmap_mean = np.mean(heatmaps, axis=0)
        pafmap_mean = np.mean(pafmaps, axis=0)

        if config.VAL.USE_CXX_HEATPAF_PARSER:
            r = parse_heatpaf_cxx(heatmap_mean.astype(np.float64), pafmap_mean.astype(np.float64), limbSeq, image_id)
        else:
            r = parse_heatpaf_py(image_ori, heatmap_mean, pafmap_mean, baseDataSet.skeleton, image_id)
        results.extend(r)

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
