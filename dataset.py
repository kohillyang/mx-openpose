from pose_transforms import PafHeatMapBaseDataSet, default_train_transform
import numpy as np
import scipy.io as sio
import os
import cv2
import logging


class MPIIDataset(PafHeatMapBaseDataSet):
    def __init__(self, mat_path="/data3/zyx/yks/dataset/mpii/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat",
                 image_root="/data3/zyx/yks/dataset/mpii/images",
                 transforms=default_train_transform, debug=False):
        mid_1 = [0, 1, 2, 5, 4, 3, 10, 11, 8, 15, 14, 12, 13, 2, 3, 2, 3]
        mid_2 = [1, 2, 11, 4, 3, 13, 11, 12, 9, 14, 13, 13, 12, 3, 2, 13, 12]

        super(MPIIDataset, self).__init__(mid_1, mid_2)
        self._transforms = transforms
        self._mat_path = mat_path
        self._debug = debug
        all_objs = self.parse_mpii_mat(self._mat_path, image_root)
        print("parsing mpii finished, got {} images.".format(len(all_objs)))
        self.objs = all_objs
        self.number_of_keypoints = 16
        self.number_of_pafs = len(mid_1)
        self.mid_1 = mid_1
        self.mid_2 = mid_2

    def __getitem__(self, item):
        obj = self.objs[item]
        image_path = obj["img_path"]
        bboxes = []
        keypoints = np.zeros(shape=(len(obj["annoations"]), self.number_of_keypoints, 2))
        availability = np.zeros(shape=(len(obj["annoations"]), self.number_of_keypoints))

        part_id_composed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        for nrect, onerect in enumerate(obj["annoations"]):
            for x, y, pid, visible in onerect["annopoints"]:
                pid_mapped = part_id_composed.index(pid)
                keypoints[nrect, pid_mapped, :] = (x, y)
                availability[nrect, pid_mapped] = 1

        for keyps in keypoints:
            bboxes.append([np.min(keyps[:, 0]), np.min(keyps[:, 1]), np.max(keyps[:, 0]), np.max(keyps[:, 1])])
        bboxes = np.array(bboxes).astype(np.float32)
        ori_image = cv2.imread(image_path)[:, :, ::-1]
        if self._transforms is not None:
            ori_image, bboxes, keypoints, availability = self._transforms(ori_image, bboxes, keypoints, availability)
        masks = self.masks_generator(ori_image, bboxes, keypoints, availability)
        return self.generate_pafmap_heatmap(ori_image, bboxes, keypoints, availability) + (masks,)

    def __len__(self):
        return len(self.objs)

    def parse_mpii_mat(self, mat_path, images_path):
        import scipy.io as sio
        import os
        dmpi = sio.loadmat(mat_path, struct_as_record=False)
        import json
        all_img = []
        for img_index in range(dmpi['RELEASE'][0, 0].annolist.shape[1]):
            train_flag = dmpi['RELEASE'][0, 0].img_train[0, img_index]
            img_name = dmpi['RELEASE'][0, 0].annolist[0, img_index].image[0, 0].name[0]
            one_img = {}
            one_img['img_path'] = os.path.join(images_path, img_name)
            one_img['annoations'] = []
            for rect_index in range(dmpi['RELEASE'][0, 0].annolist[0, img_index].annorect.shape[1]):
                one_rect = {}
                one_rect['annopoints'] = []

                try:
                    one_rect_src = dmpi['RELEASE'][0, 0].annolist[0, img_index].annorect[0, rect_index]
                    # try:
                    #     one_rect['scale'] = float(one_rect_src.scale[0, 0])
                    #     scale = 1
                    #     one_rect['x1'] = int(one_rect_src.x1[0, 0] * scale)
                    #     one_rect['y1'] = int(one_rect_src.y1[0, 0] * scale)
                    #     one_rect['x2'] = int(one_rect_src.x2[0, 0] * scale)
                    #     one_rect['y2'] = int(one_rect_src.y2[0, 0] * scale)
                    # except AttributeError as e:
                    #     pass
                    if not hasattr(one_rect_src, "annopoints") or one_rect_src.annopoints.size == 0:
                        # print(one_rect_src.__dict__)
                        continue
                    for point_index in range(one_rect_src.annopoints[0, 0].point.shape[1]):
                        one_point_src = \
                        dmpi['RELEASE'][0, 0].annolist[0, img_index].annorect[0, rect_index].annopoints[0, 0].point[
                            0, point_index]
                        x = one_point_src.x[0, 0]
                        y = one_point_src.y[0, 0]
                        part_id = one_point_src.id[0, 0]
                        if hasattr(one_rect, "is_visible"):
                            is_visible_array = one_point_src.is_visible
                            if is_visible_array.size > 0:
                                is_visible = is_visible_array[0]
                                if isinstance(is_visible, np.ndarray):
                                    is_visible = is_visible[0]
                                    assert isinstance(is_visible, int)
                                else:
                                    assert isinstance(is_visible, str)
                                    is_visible = int(is_visible)
                            else:
                                is_visible = 0
                        else:
                            is_visible = 0
                        one_rect['annopoints'].append([int(x), int(y), int(part_id), bool(is_visible)])
                except AttributeError as e:
                    logging.exception(e)
                    pass
                except IndexError as e:
                    logging.exception(e)
                if len(one_rect['annopoints']) > 0:
                    one_img['annoations'].append(one_rect)
                else:
                    pass
            if len(one_img['annoations']) > 0:
                all_img.append(one_img)
        return all_img

    def viz(self, image, heatmaps, pafmaps, masks):
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
        axes[-1, -2].imshow(masks.squeeze())
        axes[-1, -1].imshow(image)
        plt.show()


if __name__ == '__main__':
    dataset = MPIIDataset()
    x= dataset[0]
    for xx in x:
        print(xx.shape)
        print(xx.dtype)
    for img, heatmaps, heatmaps_masks, pafmaps, pafmaps_masks, masks in dataset:
        dataset.viz(img, heatmaps, pafmaps, masks)
