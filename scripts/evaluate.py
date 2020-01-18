import cv2
import json
import os
import sys
import time
import argparse
import mxnet as mx
import numpy as np
import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from datasets.cocodatasets import COCOKeyPoints
from models.cpm import CPMVGGNet, CPMNet

sys.path.append("MobulaOP")
from utils.heatpaf_parser import parse_heatpaf_cxx
from utils.viz import show_keypoints
import mobula


def parse_args():
    parser = argparse.ArgumentParser(description='validate Openpose network')
    parser.add_argument('--dataset-root', help='coco dataset root contains annotations, train2017 and val2017.',
                        required=True, type=str)
    parser.add_argument('--gpus', help='The gpus used to validate the network.', required=False, type=str, default="0")
    parser.add_argument('--resume', help='params of the network.', required=True, type=str)
    parser.add_argument('--viz', help='Whether to visualize the inference result.', action="store_true")
    parser.add_argument('--caffe-model', help='Whether to evaluate the original caffe model.', action="store_true")
    parser.add_argument('--stage', help='Stage to evaluate, 0 is recommended for resnet50-cpm. '
                                        'and 5 is recommended for original pretrained model from caffe', required=True,
                        type=int, default=0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
    args = parse_args()
    ctx = [mx.gpu(int(i)) for i in [int(x) for x in str(args.gpus).split(",")] if int(i) >= 0]
    ctx_list = ctx if ctx else [mx.cpu()]
    baseDataSet = COCOKeyPoints(root=args.dataset_root, splits=("person_keypoints_val2017",))
    results = []
    image_ids = []
    annFile = os.path.join(args.dataset_root, 'annotations/person_keypoints_val2017.json')
    image_dir = os.path.join(args.dataset_root, 'val2017')
    cocoGt = COCO(annFile)
    cats = cocoGt.loadCats(cocoGt.getCatIds())
    catIds = cocoGt.getCatIds(catNms=['person'])
    imgIds = cocoGt.getImgIds(catIds=catIds)

    if args.caffe_model:
        net = CPMVGGNet(resize=False)
        net.collect_params().load(args.resume)
    else:
        net = CPMNet(19, 19, resize=False)
        net.collect_params().load(args.resume)
        net.hybridize()

    net.collect_params().reset_ctx(ctx_list)

    for i in tqdm.tqdm(range(len(imgIds))):
        image_id = imgIds[i]
        img = cocoGt.loadImgs(image_id)[0]
        image_path = os.path.join(image_dir, img['file_name'])
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
            heatmap = net_results[args.stage * 2 + 1][0].asnumpy().transpose((1, 2, 0))
            pafmap = net_results[args.stage * 2 + 0][0].asnumpy().transpose((1, 2, 0))
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

        from scipy.ndimage.filters import gaussian_filter
        heatmap_mean_transposed = np.array([gaussian_filter(heatmap_mean[:, :, i], sigma=3)
                                            for i in range(heatmap_mean.shape[2])])
        pafmap_mean_transposed = pafmap_mean.transpose((2, 0, 1))
        r = parse_heatpaf_cxx(heatmap_mean_transposed, pafmap_mean_transposed, baseDataSet.skeleton, image_id)
        if args.viz:
            show_keypoints(image_ori, [x["keypoints"] for x in r])
        results.extend(r)

    annType = ['segm', 'bbox', 'keypoints']
    annType = annType[2]  # specify type here
    prefix = 'person_keypoints' if annType == 'keypoints' else 'instances'
    cocoGt = COCO(annFile)
    cats = cocoGt.loadCats(cocoGt.getCatIds())
    catIds = cocoGt.getCatIds(catNms=['person'])
    imgIds = cocoGt.getImgIds(catIds=catIds)

    resJsonFile = 'output/evaluationResult{}.json'.format(time.time())
    with open(resJsonFile, 'w') as outfile:
        json.dump(results, outfile)
    cocoDt2 = cocoGt.loadRes(resJsonFile)

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt2, annType)
    cocoEval.params.imgIds = image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    k = cocoEval.summarize()
    print(k)
