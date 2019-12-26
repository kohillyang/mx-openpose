from __future__ import division
import mxnet as mx
import numpy as np
import cv2


class GenGauMask(object):
    def __init__(self):
        pass

    def __call__(self, img, keypoints, availability):
        """
        :param img: The input image
        :param keypoints: A nxmx2 array representing keypoints in one image,
                n is number of instance and m is number of keypoints for each instance.
        :param visibility: A nxm array
        :return:
        """
        h, w = img.shape[0:2]
        nperson = keypoints.shape[0]
        nkeypoints = keypoints.shape[1]
        heatmaps = [np.zeros(shape=[h, w], dtype=np.float32) for _ in range(nkeypoints)]
        masks = [np.ones(shape=[h, w], dtype=np.float32) for _ in range(nkeypoints)]
        for n in range(nperson):
            for i in range(nkeypoints):
                if availability[n, i] > 0:
                    # x = min(int(round(w * keypoints[i, 0])), w - 1)
                    # y = min(int(round(h * keypoints[i, 1])), h - 1)
                    x = keypoints[n, i, 0]
                    y = keypoints[n, i, 1]
                    hm = self.CenterLabelHeatMap(w, h, x, y, sigma=7)
                    heatmaps[i][:] = np.max([heatmaps[i], hm], axis=0)
                else:
                    masks[i][:] = 0
        return img, np.array(heatmaps), np.array(masks)

    def CenterLabelHeatMap(self, img_width, img_height, c_x, c_y, sigma):
        Y1 = np.linspace(0, img_height - 1, img_height)
        X1 = np.linspace(0, img_width - 1, img_width)
        [X, Y] = np.meshgrid(X1, Y1)
        X = X - c_x
        Y = Y - c_y
        D2 = X * X + Y * Y
        E2 = 2. * sigma * sigma
        Exponet = D2 / E2
        heatmap = np.exp(-Exponet)
        return heatmap


class GenPafMaps(object):
    def __init__(self, paf_connection_first, paf_connection_second):
        self.distance_threshold = 7
        self.paf_connection_first = paf_connection_first  # type: list
        self.paf_connection_second = paf_connection_second  # type: list
        assert len(self.paf_connection_first) == len(self.paf_connection_second)

    def __call__(self, img, keypoints, availability):
        """
        :param img: The input image
        :param keypoints: A nxmx2 array representing keypoints in one image,
                n is number of instance and m is number of keypoints for each instance.
        :param visibility: A nxm array
        :return:
        """
        h, w = img.shape[:2]
        nperson = keypoints.shape[0]
        npaf = len(self.paf_connection_first)
        pafmaps_x = np.zeros(shape=(npaf, h, w), dtype=np.float32)
        pafmaps_y = np.zeros(shape=(npaf, h, w), dtype=np.float32)
        pafmap_masks = np.zeros(shape=(npaf, h, w), dtype=np.float32)

        for npaf, (px, py) in enumerate(zip(self.paf_connection_first, self.paf_connection_second)):
            for n in range(nperson):
                if availability[n, px] and availability[n, py]:
                    pafx, pafy, paf_mask = self.gen_paf_map(img, keypoints[n, px], keypoints[n, py])
                    pafmaps_x[npaf] += pafx
                    pafmaps_y[npaf] += pafy
                    pafmap_masks[npaf] += paf_mask
        norm_coff = pafmap_masks + 0.01
        pafmaps_x /= norm_coff
        pafmaps_y /= norm_coff
        pafmap_masks /= norm_coff
        return img, np.array([pafmaps_x, pafmaps_y]), pafmap_masks[np.newaxis]

    def gen_paf_map(self, img, p0, p1):
        """
        :param img: The input image
        :param p0: The first point
        :param p1: The second point
        :return:
        """
        h, w = img.shape[0:2]
        [mesh_x, mesh_y] = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))
        v1_x = mesh_x - p0[0]
        v1_y = mesh_y - p0[1]
        v2 = p1 - p0
        v2 /= np.sqrt(np.sum(v2 ** 2)) + .01

        outer_dot = v1_x * v2[1] - v1_y * v2[0]
        inner_dot = v1_x * v2[0] + v1_y * v2[1]
        # Distance between p0p1 and the target point
        distance = np.abs(outer_dot)

        v3_x = mesh_x - p1[0]
        v3_y = mesh_y - p1[1]
        inner_dot2 = v3_x * v2[0] + v3_y * v2[1]

        paf_x = np.zeros(shape=(h, w), dtype=np.float32)
        paf_y = np.zeros(shape=(h, w), dtype=np.float32)
        paf_mask = np.zeros(shape=(h, w), dtype=np.float32)
        # Points which lies between the two points and whose distance less than the threshold.
        indices = np.where(np.logical_and.reduce([inner_dot >= 0, inner_dot2 <= 0, distance < self.distance_threshold]))
        paf_x[indices] = v2[0]
        paf_y[indices] = v2[1]
        paf_mask[indices] = 1
        return paf_x, paf_y, paf_mask


class GenBBOXMask(object):
    def __init__(self):
        pass

    def __call__(self, img, bboxes, keypoints, availability):
        h, w = img.shape[:2]
        masks = np.zeros(shape=(h, w), dtype=np.float32)
        for x0, y0, x1, y1 in np.round(bboxes):
            masks[int(y0):int(y1), int(x0):int(x1)] = 1
        return masks


class PafHeatMapBaseDataSet(object):
    def __init__(self, paf_connection_first, paf_connection_second):
        self.heatmap_generator = GenGauMask()
        self.pafmap_generator = GenPafMaps(paf_connection_first, paf_connection_second)
        self.masks_generator = GenBBOXMask()

    def generate_pafmap_heatmap(self, img, bboxes, keypoints, availability):
        _, heatmaps, heatmaps_masks = self.heatmap_generator(img, keypoints, availability)
        _, pafmaps, pafmaps_masks = self.pafmap_generator(img, keypoints, availability)
        return img, heatmaps, heatmaps_masks, pafmaps, pafmaps_masks


class ImagePad(object):
    def __init__(self, dst_shape=(368, 368)):
        self.dst_shape = dst_shape

    def __call__(self, img_ori, bboxes, keypoints, availability):
        dshape = self.dst_shape
        fscale = min(dshape[0] / img_ori.shape[0], dshape[1] / img_ori.shape[1])
        img_resized = cv2.resize(img_ori, dsize=(0, 0), fx=fscale, fy=fscale)
        img_padded = np.zeros(shape=(int(dshape[0]), int(dshape[1]), 3), dtype=np.float32)
        img_padded[:img_resized.shape[0], :img_resized.shape[1], :img_resized.shape[2]] = img_resized
        keypoints = keypoints * fscale
        bboxes = bboxes * fscale
        return img_padded, bboxes, keypoints, availability


class Compose(object):
    def __init__(self, transforms=()):
        self.transforms = transforms

    def __call__(self, *args):
        for trans in self.transforms:
            args = trans(*args)
        return args


class RandomScale(object):
    def __init__(self, cfg):
        self.scale_min = cfg.TRAIN.TRANSFORM_PARAMS.scale_min
        self.scale_max = cfg.TRAIN.TRANSFORM_PARAMS.scale_max

    def __call__(self, img_ori, bboxes, keypoints, availability):
        bboxes = bboxes.copy()
        keypoints = keypoints.copy()
        availability = availability.copy()
        scale = np.random.random() * (self.scale_max - self.scale_min) + self.scale_min
        img_resized = cv2.resize(img_ori, (0, 0), fx=scale, fy = scale)
        bboxes[:, :4] *= scale
        keypoints *= scale
        return img_resized, bboxes, keypoints, availability


class RandomCenterCrop(object):
    def __init__(self, cfg):
        self.center_perterb_max = cfg.TRAIN.TRANSFORM_PARAMS.center_perterb_max  # type: float
        self.crop_size_x = cfg.TRAIN.TRANSFORM_PARAMS.crop_size_x
        self.crop_size_y = cfg.TRAIN.TRANSFORM_PARAMS.crop_size_y

    def __call__(self, img_ori, bboxes, keypoints, availability):
        bboxes = bboxes.copy()
        keypoints = keypoints.copy()
        availability = availability.copy()

        # Choose a bbox
        bbox_idx = np.random.randint(0, bboxes.shape[0])
        bbox = bboxes[bbox_idx]
        center_x = .5 * (bbox[0] + bbox[2])
        center_y = .5 * (bbox[1] + bbox[3])
        center_x += (np.random.random() * 2 - 1) * self.center_perterb_max
        center_y += (np.random.random() * 2 - 1) * self.center_perterb_max

        center_x = int(np.round(center_x))
        center_y = int(np.round(center_y))

        start_x = max(center_x - self.crop_size_x // 2, 0)
        start_y = max(center_y - self.crop_size_y // 2, 0)

        end_x = min(center_x + self.crop_size_x // 2, img_ori.shape[1])
        end_y = min(center_y + self.crop_size_y // 2, img_ori.shape[1])

        offset_x = center_x - self.crop_size_x // 2
        offset_y = center_y - self.crop_size_y // 2

        image_cropped = img_ori[start_y:end_y, start_x:end_x]
        image_cropped_padded = np.zeros(shape=(self.crop_size_y, self.crop_size_x, img_ori.shape[2]), dtype=np.float32)
        dst_start_x = start_x - offset_x
        dst_start_y = start_y - offset_y
        dst_end_x = dst_start_x + image_cropped.shape[1]
        dst_end_y = dst_start_y + image_cropped.shape[0]
        image_cropped_padded[dst_start_y:dst_end_y, dst_start_x:dst_end_x] = image_cropped
        bboxes[:, (0, 2)] -= offset_x
        bboxes[:, (1, 3)] -= offset_y
        keypoints[:, :, 0] -= offset_x
        keypoints[:, :, 1] -= offset_y
        for m in range(keypoints.shape[0]):
            for n in range(keypoints.shape[1]):
                x, y = keypoints[m, n]
                if not (0 <= x < image_cropped_padded.shape[1] and 0 <= y < image_cropped_padded.shape[0]):
                    availability[m, n] = 0

        return image_cropped_padded, bboxes, keypoints, availability