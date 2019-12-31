import cv2
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
        mobula.op.load('HeatGen')
        mobula.op.load('PAFGen')

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
        axes[-1, -1].imshow(image.astype(np.float32))
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
        joints = np.concatenate([keypoints, availability[:, :, np.newaxis]], axis=2)

        heatmap = mobula.op.HeatGen[np.ndarray]()(image.astype(np.float32), bboxes.astype(np.float32), joints.astype(np.float32))
        limb_sequence = self.baseDataSet.skeleton;
        pafmap = mobula.op.PAFGen[np.ndarray](limb_sequence)(image.astype(np.float32), bboxes.astype(np.float32), joints.astype(np.float32))
        pafmap = pafmap.reshape(len(limb_sequence), 2, pafmap.shape[1], pafmap.shape[2])
        plt.imshow((pafmap[:, 0] **2 + pafmap[:, 1] **2).max(axis=0))
        plt.figure()
        plt.imshow(image.astype(np.uint8))
        plt.show()
        #
        # _, heatmaps, heatmaps_masks = self.heatmap_generator(image, keypoints, availability)
        # plt.figure()
        # plt.imshow(heatmaps.max(axis=0))
        # plt.show()
        return heatmap


if __name__ == '__main__':
    from datasets.cocodatasets import COCOKeyPoints
    import datasets.pose_transforms as transforms
    import easydict
    import os
    config = easydict.EasyDict()
    config.TRAIN = easydict.EasyDict()
    config.TRAIN.save_prefix = "output/gcn/"
    config.TRAIN.model_prefix = os.path.join(config.TRAIN.save_prefix, "resnet50-cpm-teachered-cropped")
    config.TRAIN.gpus = [1, 2]
    config.TRAIN.batch_size = 8
    config.TRAIN.optimizer = "SGD"
    config.TRAIN.lr = 5e-6
    config.TRAIN.momentum = 0.9
    config.TRAIN.wd = 0.0001
    config.TRAIN.lr_step = [8, 12]
    config.TRAIN.warmup_step = 100
    config.TRAIN.warmup_lr = config.TRAIN.lr * 0.1
    config.TRAIN.end_epoch = 26
    config.TRAIN.resume = None
    config.TRAIN.DATASET = easydict.EasyDict()
    config.TRAIN.DATASET.coco_root = "/data1/coco"
    config.TRAIN.TRANSFORM_PARAMS = easydict.EasyDict()

    # params for random cropping
    config.TRAIN.TRANSFORM_PARAMS.crop_size_x = 368
    config.TRAIN.TRANSFORM_PARAMS.crop_size_y = 368
    config.TRAIN.TRANSFORM_PARAMS.center_perterb_max = 40

    # params for random scale
    config.TRAIN.TRANSFORM_PARAMS.scale_min = 0.5
    config.TRAIN.TRANSFORM_PARAMS.scale_max = 1.1

    # params for putGaussianMaps
    config.TRAIN.TRANSFORM_PARAMS.sigma = 25

    # params for putVecMaps
    config.TRAIN.TRANSFORM_PARAMS.distance_threshold = 8
    train_transform = transforms.Compose([transforms.RandomScale(config), transforms.RandomCenterCrop(config)])

    baseDataSet = COCOKeyPoints(root="/data/coco", splits=("person_keypoints_val2017",))
    dataSet = PafHeatMapDataSet(baseDataSet, train_transform)
    for x in dataSet:
        pass
    x = dataSet[0]
    for xx in x:
        print(xx.shape)
        print(xx.dtype)
    # for img, heatmaps, heatmaps_masks, pafmaps, pafmaps_masks in dataSet:
    #     assert not np.any(np.isnan(pafmaps))
    #     assert not np.any(np.isnan(pafmaps_masks))
    #     dataSet.viz(img, heatmaps, pafmaps, pafmaps_masks)
