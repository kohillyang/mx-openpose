import logging
import os
import sys
import tqdm
import easydict
import matplotlib.pyplot as plt
import mxnet as mx
import mxnet.autograd as ag
import numpy as np
from mxnet import gluon
import os, cv2, math
import json
from scipy.ndimage.filters import gaussian_filter
from datasets.cocodatasets import COCOKeyPoints
from datasets.dataset import PafHeatMapDataSet
from datasets.pose_transforms import default_train_transform, ImagePad
from models.drn_gcn import DRN50_GCN
from models.cpm import get_cpm_symbol
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO



def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad



def parse_heatpaf(oriImg, heatmap_avg, paf_avg, limbSeq, image_id=0, category_id=1, fscale=1.0):
    param = dict()
    param['thre1'] = 0.1
    param['thre2'] = 0.05
    all_peaks = []
    peak_counter = 0

    for part in range(19 - 1):
        x_list = []
        y_list = []
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > param['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        cid = list(range(peak_counter, peak_counter + len(peaks)))
        peaks_with_score_and_id = [peaks_with_score[i] + (cid[i],) for i in range(len(cid))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)
    # find connection in the specified sequence, center 29 is in the position 15
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]
    # the middle joints heatmap correpondence
    mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
              [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
              [55, 56], [37, 38], [45, 46]]

    connection_all = []
    special_k = []
    special_non_zero_index = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x - 19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        # print(k)
        # print(candA)
        # print('---------')
        # print(candB)
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    try:
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        # print('vec: ',vec)
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        # print('norm: ', norm)
                        vec = np.divide(vec, norm)
                        # print('normalized vec: ', vec)
                        startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                            np.linspace(candA[i][1], candB[j][1], num=mid_num)))
                        # print('startend: ', startend)
                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                          for I in range(len(startend))])
                        # print('vec_x: ', vec_x)
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                          for I in range(len(startend))])
                        # print('vec_y: ', vec_y)
                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        # print(score_midpts)
                        # print('score_midpts: ', score_midpts)
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                            0.5 * oriImg.shape[0] / norm - 1, 0)

                        # print('score_with_dist_prior: ', score_with_dist_prior)
                        criterion1 = len(np.nonzero(score_midpts > param['thre2'])[0]) > 0.8 * len(score_midpts)
                        # print('score_midpts > param["thre2"]: ', len(np.nonzero(score_midpts > param['thre2'])[0]))
                        criterion2 = score_with_dist_prior > 0

                        if criterion1 and criterion2:
                            # print('match')
                            # print(i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2])
                            connection_candidate.append(
                                [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])
                    except Exception as e:
                        import logging
                        logging.exception(e)
                        print('error rendering')
                    # print('--------end-----------')
            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            # print('-------------connection_candidate---------------')
            # print(connection_candidate)
            # print('------------------------------------------------')
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    # print('----------connection-----------')
                    # print(connection)
                    # print('-------------------------------')
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        # elif(nA != 0 or nB != 0):
        else:
            special_k.append(k)
            special_non_zero_index.append(indexA if nA != 0 else indexB)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))

    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            try:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(limbSeq[k]) - 1

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if (subset[j][indexB] != partBs[i]):
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        print("found = 2")
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])
            except:
                print("not link")

    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 3 or subset[i][-2] / subset[i][-1] < 0.2:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    r = []
    orderCOCO = [1, 0, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]
    for j in range(len(subset)):
        category_id = 1
        keypoints = np.zeros(51)
        score = 0
        for part in range(18):
            if part == 1:
                continue
            index = int(subset[j][part])
            if index > 0:
                realpart = orderCOCO[part] - 1
                if part == 0:
                    keypoints[realpart * 3] = candidate[index][0]
                    keypoints[realpart * 3 + 1] = candidate[index][1]
                    keypoints[realpart * 3 + 2] = 1
                    # score = score + candidate[index][2]
                else:
                    keypoints[(realpart) * 3] = candidate[index][0]
                    keypoints[(realpart) * 3 + 1] = candidate[index][1]
                    keypoints[(realpart) * 3 + 2] = 1
                    # score = score + candidate[index][2]

        keypoints_list = keypoints.tolist()
        current_dict = {'image_id': image_id,
                        'category_id': category_id,
                        'keypoints': keypoints_list,
                        'score': subset[j][-2]}
        r.append(current_dict)
    return r


def pad_image(img_ori, dshape=(368, 368)):
    fscale = 1.0 * dshape[0] / img_ori.shape[0]
    img_resized = cv2.resize(img_ori, dsize=(0, 0), fx=fscale, fy=fscale)
    image_padded, pad = padRightDownCorner(img_resized, 8, 128)
    return img_resized, image_padded, pad, fscale


if __name__ == '__main__':
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
    ctx_list = [mx.gpu(8)]
    baseDataSet = COCOKeyPoints(root="/data3/zyx/yks/dataset/coco2017", splits=("person_keypoints_val2017",))
    val_dataset = PafHeatMapDataSet(baseDataSet, default_train_transform)
    number_of_keypoints = val_dataset.number_of_keypoints
    # net = DRN50_GCN(num_classes=val_dataset.number_of_keypoints + 2 * val_dataset.number_of_pafs)
    # sym, _, _ = mx.model.load_checkpoint('pretrained/pose', 0)
    sym = get_cpm_symbol()
    net = mx.gluon.SymbolBlock(sym, mx.sym.var(name="data"))
    net.collect_params().load("pretrained/pose-0000.params")
    net.collect_params().reset_ctx(ctx_list)
    results = []
    image_ids = []
    annFile = '/data3/zyx/yks/dataset/coco2017/annotations/person_keypoints_val2017.json'
    cocoGt = COCO(annFile)
    cats = cocoGt.loadCats(cocoGt.getCatIds())
    catIds = cocoGt.getCatIds(catNms=['person'])
    imgIds = cocoGt.getImgIds(catIds=catIds)

    for i in tqdm.tqdm(range(min(len(val_dataset), 50))):
        # da = val_dataset[i]
        # image_id = val_dataset.baseDataSet[i][3]
        # image_path = val_dataset.baseDataSet[i][0]

        img = cocoGt.loadImgs(imgIds[i])[0]
        image_path = '/data3/zyx/yks/dataset/coco2017/val2017/' + img['file_name']
        image_id = img['id']
        image_ids.append(image_id)
        cimgRGB = cv2.imread(image_path)[:, :, ::-1]
        cscale = 368 * 1.0 / cimgRGB.shape[0]
        imageToTest = cv2.resize(cimgRGB, (0, 0), fx=cscale, fy=cscale, interpolation=cv2.INTER_CUBIC)

        imageToTest_padded, pad = padRightDownCorner(imageToTest, 8, 128)
        transposeImage = np.transpose(np.float32(imageToTest_padded[:, :, :]), (2, 0, 1)) / 256 - 0.5
        testimage = transposeImage

        result = net(mx.nd.array(testimage[np.newaxis]).as_in_context(ctx_list[0]))

        heatmap = np.moveaxis(result[1].asnumpy()[0], 0, -1)
        heatmap = cv2.resize(heatmap, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)  # INTER_LINEAR
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (cimgRGB.shape[1], cimgRGB.shape[0]), interpolation=cv2.INTER_CUBIC)

        pagmap = np.moveaxis(result[0].asnumpy()[0], 0, -1)
        pagmap = cv2.resize(pagmap, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
        pagmap = pagmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        pagmap = cv2.resize(pagmap, (cimgRGB.shape[1], cimgRGB.shape[0]), interpolation=cv2.INTER_CUBIC)

        r = parse_heatpaf(cimgRGB, heatmap, pagmap , val_dataset.baseDataSet.skeleton,
                          image_id=image_id, fscale=1.0)
        results.extend(r)

    annType = ['segm','bbox','keypoints']
    annType = annType[2]      #specify type here
    prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
    annFile = '/data3/zyx/yks/dataset/coco2017/annotations/person_keypoints_val2017.json'
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
    cocoEval.params.imgIds  = image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    k = cocoEval.summarize()
    print(k)