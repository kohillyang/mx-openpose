import logging
import os
import sys
import tqdm
import time
import easydict
import pprint

import mxnet as mx
import mxnet.autograd as ag
import numpy as np
import matplotlib.pyplot as plt
from mxnet import gluon

from datasets.cocodatasets import COCOKeyPoints
from datasets.dataset import PafHeatMapDataSet
from models.cpm import CPMNet, CPMVGGNet
import datasets.pose_transforms as transforms
sys.path.append("MobulaOP")
import mobula
print(mobula.__path__)

@mobula.op.register
class BCELoss:
    def forward(self, y, target):
        return mx.nd.log(1 + mx.nd.exp(y)) - target * y

    def backward(self, dy):
        grad = mx.nd.sigmoid(self.X[0]) - self.X[1]
        # grad *= 1e-4
        self.dX[0][:] = grad * dy

    def infer_shape(self, in_shape):
        try:
            assert in_shape[0] == in_shape[1]
        except AssertionError as e:
            print(in_shape)
            raise e
        return in_shape, [in_shape[0]]


@mobula.op.register
class L2Loss:
    def forward(self, y, target):
        # return 2 * mx.nd.log(1 + mx.nd.exp(y)) - y - target * y
        return (y - target) ** 2

    def backward(self, dy):
        # grad = mx.nd.sigmoid(self.X[0])*2 - 1 - self.X[1]
        grad = 2 * (self.X[0] - self.X[1])
        # grad *= 1e-4
        self.dX[0][:] = grad * dy

    def infer_shape(self, in_shape):
        assert in_shape[0] == in_shape[1]
        return in_shape, [in_shape[0]]


@mobula.op.register
class BCEPAFLoss:
    def forward(self, y, target):
        return 2 * mx.nd.log(1 + mx.nd.exp(y)) - y - target * y

    def backward(self, dy):
        grad = mx.nd.sigmoid(self.X[0])*2 - 1 - self.X[1]
        self.dX[0][:] = grad * dy

    def infer_shape(self, in_shape):
        assert in_shape[0] == in_shape[1]
        return in_shape, [in_shape[0]]


def log_init(filename):
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '[%(asctime)s][%(filename)s][line: %(lineno)d][%(levelname)s] ## %(message)s')
    fh.setFormatter(formatter)
    formatter = logging.Formatter(
        '[%(filename)s][line: %(lineno)d][%(levelname)s] ## %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)


if __name__ == '__main__':
    os.environ["MXNET_USE_FUSION"]="0"
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
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

    os.makedirs(config.TRAIN.save_prefix, exist_ok=True)
    log_init(filename=config.TRAIN.model_prefix + "{}-train.log".format(time.time()))
    logging.info(pprint.pformat(config))

    # fix seed for mxnet, numpy and python builtin random generator.
    mx.random.seed(3)
    np.random.seed(3)

    ctx = [mx.gpu(int(i)) for i in config.TRAIN.gpus]
    ctx = ctx if ctx else [mx.cpu()]

    train_transform = transforms.Compose([transforms.RandomScale(config),
                                          transforms.RandomCenterCrop(config)])

    baseDataSet = COCOKeyPoints(root=config.TRAIN.DATASET.coco_root, splits=("person_keypoints_train2017",))
    train_dataset = PafHeatMapDataSet(baseDataSet, train_transform)

    # for img, heatmaps, heatmaps_masks, pafmaps, pafmaps_masks in train_dataset:
    #     assert not np.any(np.isnan(pafmaps))
    #     assert not np.any(np.isnan(pafmaps_masks))
    #     train_dataset.viz(img, heatmaps, pafmaps, pafmaps_masks)
    #
    # exit()
    net = CPMNet(19, 19)
    net_teacher = CPMVGGNet(resize=False)
    net_teacher.collect_params().load("pretrained/pose-0000.params")
    params = net.collect_params()
    for key in params.keys():
        if params[key]._data is None:
            default_init = mx.init.Zero() if "bias" in key or "offset" in key else mx.init.Normal()
            default_init.set_verbosity(True)
            if params[key].init is not None and hasattr(params[key].init, "set_verbosity"):
                params[key].init.set_verbosity(True)
                params[key].initialize(init=params[key].init, default_init=params[key].init)
            else:
                params[key].initialize(default_init=default_init)
    net.collect_params().reset_ctx(ctx)
    net_teacher.collect_params().reset_ctx(ctx)
    train_loader = mx.gluon.data.DataLoader(train_dataset, batch_size=config.TRAIN.batch_size,
                                            shuffle=True, num_workers=12, thread_pool=False, last_batch="discard")

    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=[len(train_loader) * x for x in config.TRAIN.lr_step],
                                                        warmup_mode="constant", factor=.1, base_lr=config.TRAIN.lr,
                                                        warmup_steps=config.TRAIN.warmup_step,
                                                        warmup_begin_lr=config.TRAIN.warmup_lr)
    if config.TRAIN.resume is not None:
        net.collect_params().load(config.TRAIN.resume)
    if config.TRAIN.optimizer == "SGD":
        trainer = mx.gluon.Trainer(
            net.collect_params(),
            'sgd',
            {'learning_rate': config.TRAIN.lr,
             'wd': config.TRAIN.wd,
             'momentum': config.TRAIN.momentum,
             'clip_gradient': None,
             'lr_scheduler': lr_scheduler
             })
    else:
        trainer = gluon.Trainer(
            net.collect_params(),
            'adam',
            {'learning_rate': config.TRAIN.lr,
             'lr_scheduler': lr_scheduler
             }
        )

    metric_dict = {}
    eval_metrics = mx.metric.CompositeEvalMetric()
    for i in range(6):
        metric_loss_heatmaps = mx.metric.Loss("batch_loss_heatmaps_stage_{}".format(i))
        metric_batch_loss_pafmaps = mx.metric.Loss("batch_loss_pafmaps_stage_{}".format(i))
        for child_metric in [metric_loss_heatmaps, metric_batch_loss_pafmaps]:
            eval_metrics.add(child_metric)
        metric_dict["stage{}_heat".format(i)] = metric_loss_heatmaps
        metric_dict["stage{}_paf".format(i)] = metric_batch_loss_pafmaps

    for epoch in range(config.TRAIN.end_epoch ):
        eval_metrics.reset()
        net.hybridize(static_alloc=True, static_shape=True)
        net_teacher.hybridize(static_alloc=True, static_shape=True)

        for batch_cnt, batch in enumerate(tqdm.tqdm(train_loader)):
            data_list = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            losses = []
            losses_dict = {}
            with ag.record():
                for data in data_list:
                    pafmaps, heatmaps = net_teacher(data)[-2:]
                    # plt.imshow(heatmaps[0][:-1].max(axis=0).asnumpy())
                    # plt.figure()
                    # plt.imshow(data[0].asnumpy().astype(np.uint8))
                    # plt.show()
                    y_hat = net(data)
                    for i in range(len(y_hat) // 2):
                        heatmap_prediction = y_hat[i * 2 + 1]
                        pafmap_prediction = y_hat[i * 2]
                        loss_heatmap = mx.nd.sum(L2Loss(heatmap_prediction,  heatmaps))
                        loss_pafmap = mx.nd.sum(L2Loss(pafmap_prediction, pafmaps))
                        losses.append(loss_heatmap)
                        losses.append(loss_pafmap)
                        losses_dict["stage_{}_heat".format(i)] = loss_heatmap
                        losses_dict["stage_{}_paf".format(i)] = loss_pafmap
            ag.backward(losses)
            trainer.step(1, ignore_stale_grad=False)
            for i in range(6):
                metric_dict["stage{}_heat".format(i)].update(None, losses_dict["stage_{}_heat".format(i)])
                metric_dict["stage{}_paf".format(i)].update(None, losses_dict["stage_{}_paf".format(i)])
            msg = ','.join(['{}={:.3f}'.format(w, v) for w, v in zip(*eval_metrics.get())])
            msg += ",lr={}".format(trainer.learning_rate)
            logging.info(msg)
            eval_metrics.reset()
        save_path = "{}-{}-{}.params".format(config.TRAIN.model_prefix, epoch, 0.0)
        net.collect_params().save(save_path)
        logging.info("Saved checkpoint to {}".format(save_path))
        trainer.save_states(config.TRAIN.model_prefix + "-trainer.states")

