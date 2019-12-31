import cv2
from datasets.pose_transforms import PafHeatMapBaseDataSet
import numpy as np
import mxnet as mx
import mobula


class PafHeatMapDataSet(PafHeatMapBaseDataSet):
    def __init__(self, base_dataset, transforms=None):
        super(PafHeatMapDataSet, self).__init__(base_dataset.skeleton[:, 0], base_dataset.skeleton[:, 1])
        self.baseDataSet = base_dataset
        self.transforms = transforms
        self.number_of_keypoints = self.baseDataSet.number_of_keypoints
        self.number_of_pafs = len(self.baseDataSet.skeleton)
        mobula.op.load('HeatGen')

    def __len__(self):
        return len(self.baseDataSet)

    def viz(self, image, heatmaps, pafmaps, pafmaps_masks):
        image = image.astype(np.uint8)
        import matplotlib.pyplot as plt
        fig0, axes = plt.subplots(int((heatmaps.shape[0]) / 4) + 1, 4, squeeze=False)
        for i in range(len(axes)):
            for j in range(len(axes[0])):
                n = i * len(axes[0]) + j
                if n < heatmaps.shape[0]:
                    image_can = image.copy()
                    image_can[:, :, 2] = heatmaps[n] * 255
                    axes[i, j].imshow(image_can)
        axes[-1, -1].imshow(image)
        paf_x, paf_y = pafmaps
        fig1, axes = plt.subplots(int((paf_x.shape[0]) / 4) + 1, 4, squeeze=False)
        for i in range(len(axes)):
            for j in range(len(axes[0])):
                n = i * len(axes[0]) + j
                if n < paf_x.shape[0]:
                    paf_norm = np.sqrt(paf_x[n] ** 2 + paf_y[n] ** 2)
                    image_can = image.copy()
                    image_can[:, :, 2] = paf_norm * 255
                    axes[i, j].imshow(image_can)
        axes[-1, -1].imshow(image)
        plt.figure()
        plt.imshow(pafmaps_masks[0].max(axis=0))
        plt.show()

    def __getitem__(self, item):
        path, bboxes, joints, image_id = self.baseDataSet[item]
        image = cv2.imread(path)[:, :, ::-1]
        keypoints = joints[:, :, :2]
        availability = np.logical_and(joints[:, :, 0] > 0, joints[:, :, 1] > 0)
        availability = availability.astype(np.float32)
        if self.transforms is not None:
            image, bboxes, keypoints, availability = self.transforms(image, bboxes, keypoints, availability)
        heatmap = mobula.op.HeatGen[np.ndarray]()(image.astype(np.float32), bboxes.astype(np.float32), joints.astype(np.float32))

if __name__ == '__main__':
    from datasets.cocodatasets import COCOKeyPoints

    baseDataSet = COCOKeyPoints(root="/data/coco", splits=("person_keypoints_val2017",))
    dataSet = PafHeatMapDataSet(baseDataSet)
    x = dataSet[0]
    for xx in x:
        print(xx.shape)
        print(xx.dtype)
    for img, heatmaps, heatmaps_masks, pafmaps, pafmaps_masks in dataSet:
        assert not np.any(np.isnan(pafmaps))
        assert not np.any(np.isnan(pafmaps_masks))
        dataSet.viz(img, heatmaps, pafmaps, pafmaps_masks)
