import mobula
import os
import math
import numpy as np

mobula.op.load('HeatPafParser', os.path.join(os.path.dirname(__file__), "operator_cxx"))


def parse_heatpaf_cxx(heatmap_avg, paf_avg, limbSeq, image_id=0, category_id=1, fscale=1.0):
    orderCOCO = [1, 0, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]
    orderCOCO = [x-1 for x in orderCOCO]
    orderCOCO = [orderCOCO.index(x) for x in range(17)]  # coco has 17 keypoints.

    keypoints, scores = mobula.op.HeatPafParser[np.ndarray](limbSeq)(heatmap_avg, paf_avg);
    indices_by_scores = np.where(scores > 0)[0]
    keypoints = keypoints[indices_by_scores][:, orderCOCO]
    scores = scores[indices_by_scores]
    r = []
    for subset, score in zip(keypoints, scores):
        current_dict = {'image_id': image_id,
                        'category_id': category_id,
                        'keypoints': subset.reshape(-1).tolist(),
                        'score': float(score)}
        r.append(current_dict)
    return r
