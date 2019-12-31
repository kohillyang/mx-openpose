import cv2
import os
from datasets.pose_transforms import PafHeatMapBaseDataSet
import numpy as np
import mxnet as mx
import mobula
import matplotlib.pyplot as plt


class PafHeatMapDataSet(PafHeatMapBaseDataSet):
    def __init__(self, base_dataset, transforms=None):
        super(PafHeatMapDataSet, self).__init__(base_dataset.skeleton[:, 0], base_dataset.skeleton[:, 1])
        self.baseDataSet = base_dataset
        self.transforms = transforms
        self.number_of_keypoints = self.baseDataSet.number_of_keypoints
        self.number_of_pafs = len(self.baseDataSet.skeleton)
        mobula.op.load('HeatGen', os.path.dirname(__file__))
        mobula.op.load('PAFGen', os.path.dirname(__file__))

    def __len__(self):
        return len(self.baseDataSet)

    def __getitem__(self, item):
        path, bboxes, joints, image_id = self.baseDataSet[item]
        image = cv2.imread(path)[:, :, ::-1]
        keypoints = joints[:, :, :2]
        availability = np.logical_and(joints[:, :, 0] > 0, joints[:, :, 1] > 0)
        availability = availability.astype(np.float32)
        if self.transforms is not None:
            image, bboxes, keypoints, availability = self.transforms(image, bboxes, keypoints, availability)
        joints = np.concatenate([keypoints, availability[:, :, np.newaxis]], axis=2)

        heatmap = mobula.op.HeatGen[np.ndarray]()(image.astype(np.float32), bboxes.astype(np.float32), joints.astype(np.float32))
        limb_sequence = self.baseDataSet.skeleton;
        pafmap = mobula.op.PAFGen[np.ndarray](limb_sequence)(image.astype(np.float32), bboxes.astype(np.float32), joints.astype(np.float32))
        heatmap_mask = self.genHeatmapMask(joints.astype(np.float32), heatmap)
        pafmap_mask = self.genPafmapMask(limb_sequence, joints.astype(np.float32), pafmap)
        return image, heatmap, heatmap_mask, pafmap, pafmap_mask

    def genHeatmapMask(self, joints, heatmaps):
        mask = np.ones_like(heatmaps)
        for i in range(len(joints)):
            for j in range(len(joints[0])):
                if joints[i, j, 2] > 0:
                    pass
                else:
                    mask[j][:] = 0
        return mask

    def genPafmapMask(self, limb_sequence, joints, pafmaps):
        mask = np.ones_like(pafmaps)
        for i in range(len(joints)):
            for j in range(len(limb_sequence)):
                if joints[i, limb_sequence[j, 0], 2] > 0 and joints[i, limb_sequence[j, 1], 2] > 0:
                    pass
                else:
                    mask[j][:] = 0
        return mask