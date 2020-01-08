import numpy as np
import matplotlib.pyplot as plt

coco_skeleton=np.array([[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                        [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]])-1


def show_keypoints(ori_image, keypoints, skeleton=coco_skeleton):
    c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
    plt.imshow(ori_image)
    sks = skeleton
    for kp in np.array(keypoints):
        x = kp[0::3]
        y = kp[1::3]
        v = kp[2::3]
        for sk in sks:
            if np.all(v[sk] > 0):
                plt.plot(x[sk], y[sk], linewidth=3, color=c)
        plt.plot(x[v > 0], y[v > 0], 'o', markersize=8, markerfacecolor=c, markeredgecolor='k', markeredgewidth=2)
        plt.plot(x[v > 1], y[v > 1], 'o', markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2)
    plt.show()
