import cv2
from datasets.pose_transforms import PafHeatMapBaseDataSet, default_train_transform
import numpy as np


class PafHeatMapDataSet(PafHeatMapBaseDataSet):
    def __init__(self, base_dataset, transforms=None):
        super(PafHeatMapDataSet, self).__init__(base_dataset.skeleton[:, 0], base_dataset.skeleton[:, 1])
        self.baseDataSet = base_dataset
        self.transforms = transforms
        self.number_of_keypoints = self.baseDataSet.number_of_keypoints
        self.number_of_pafs = len(self.baseDataSet.skeleton)

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
        availability = joints[:, :, 2]
        if self.transforms is not None:
            image, bboxes, keypoints, availability = self.transforms(image, bboxes, keypoints, availability)
        img, heatmaps, heatmaps_masks, pafmaps, pafmaps_masks = self.generate_pafmap_heatmap(image, bboxes, keypoints,
                                                                                             availability)
        masks = self.masks_generator(image, bboxes, joints[:, :, :2], joints[:, :, 2])
        return img, heatmaps, heatmaps_masks * masks[np.newaxis], pafmaps, pafmaps_masks.max(axis=1, keepdims=True) #* masks[np.newaxis, np.newaxis]


if __name__ == '__main__':
    from datasets.cocodatasets import COCOKeyPoints

    baseDataSet = COCOKeyPoints(root="/data3/zyx/yks/dataset/coco2017", splits=("person_keypoints_val2017",))
    dataSet = PafHeatMapDataSet(baseDataSet, default_train_transform)
    x = dataSet[0]
    for xx in x:
        print(xx.shape)
        print(xx.dtype)
    for img, heatmaps, heatmaps_masks, pafmaps, pafmaps_masks in dataSet:
        assert not np.any(np.isnan(pafmaps))
        assert not np.any(np.isnan(pafmaps_masks))
        dataSet.viz(img, heatmaps, pafmaps, pafmaps_masks)
