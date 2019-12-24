import logging
import os
import sys
import tqdm
import time
import easydict

import mxnet as mx
import mxnet.autograd as ag
import numpy as np
from mxnet import gluon

from datasets.cocodatasets import COCOKeyPoints
from datasets.dataset import PafHeatMapDataSet
from datasets.pose_transforms import default_train_transform
from models.drn_gcn import DRN50_GCN

sys.path.append("MobulaOP")
import mobula
print(mobula.__path__)

@mobula.op.register
class SigmodCrossEntropyLoss:
    def forward(self, y, target):
        return mx.nd.log(1 + mx.nd.exp(y)) - target * y

    def backward(self, dy):
        grad = mx.nd.sigmoid(self.X[0]) - self.X[1]
        # grad *= 1e-4
        self.dX[0][:] = grad * dy

    def infer_shape(self, in_shape):
        assert in_shape[0] == in_shape[1]
        return in_shape, [in_shape[0]]


@mobula.op.register
class SigmodPafCrossEntropyLoss:
    def forward(self, y, target):
        return 2 * mx.nd.log(1 + mx.nd.exp(y)) - y - target * y

    def backward(self, dy):
        grad = mx.nd.sigmoid(self.X[0])*2 - 1 - self.X[1]
        # grad *= 1e-4
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
    config = easydict.EasyDict()
    config.TRAIN = easydict.EasyDict()
    config.TRAIN.save_prefix = "output/gcn/"
    config.TRAIN.model_prefix = os.path.join(config.TRAIN.save_prefix, "GCN-resnet50-")
    config.TRAIN.gpus = [8, 3]
    config.TRAIN.lr = 1e-4
    config.TRAIN.momentum = 1.9
    config.TRAIN.wd = 0.0001
    config.TRAIN.lr_step = [8, 12]
    config.TRAIN.warmup_step = 100
    config.TRAIN.warmup_lr = 1e-5
    config.TRAIN.end_epoch = 26
    config.TRAIN.loss_paf_weight = 1
    config.TRAIN.loss_heatmap_weight = 1
    config.TRAIN.resume = None
    os.makedirs(config.TRAIN.save_prefix, exist_ok=True)
    log_init(filename=config.TRAIN.model_prefix + "{}-train.log".format(time.time()))

    # fix seed for mxnet, numpy and python builtin random generator.
    mx.random.seed(3)
    np.random.seed(3)

    ctx = [mx.gpu(int(i)) for i in config.TRAIN.gpus]
    ctx = ctx if ctx else [mx.cpu()]

    baseDataSet = COCOKeyPoints(root="/data3/zyx/yks/dataset/coco2017", splits=("person_keypoints_train2017",))
    train_dataset = PafHeatMapDataSet(baseDataSet, default_train_transform)
    net = DRN50_GCN(num_classes=train_dataset.number_of_keypoints + 2 * train_dataset.number_of_pafs)

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

    train_loader = mx.gluon.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=12, thread_pool=False, last_batch="discard")

    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=[len(train_loader) * x for x in config.TRAIN.lr_step],
                                                        warmup_mode="constant", factor=.1, base_lr=config.TRAIN.lr,
                                                        warmup_steps=config.TRAIN.warmup_step, warmup_begin_lr=config.TRAIN.warmup_lr)
    if config.TRAIN.resume is not None:
        net.collect_params().load(config.TRAIN.resume)
    # trainer = mx.gluon.Trainer(
    #     net.collect_params(),  # fix batchnorm, fix first stage, etc...
    #     'sgd',
    #     {'learning_rate': config.TRAIN.lr,
    #      'wd': config.TRAIN.wd,
    #      'momentum': config.TRAIN.momentum,
    #      'clip_gradient': 1e-2,
    #      'lr_scheduler': lr_scheduler
    #      })
    trainer = gluon.Trainer(
        net.collect_params(),
        'adam',
        {'learning_rate': config.TRAIN.lr,
         #         'wd': args.wd, 'momentum': args.momentum
         'lr_scheduler': lr_scheduler
         }
    )

    metric_loss_heatmaps = mx.metric.Loss("loss_heatmaps")
    metric_loss_pafmaps = mx.metric.Loss("loss_pafmaps")
    metric_batch_loss_pafmaps = mx.metric.Loss("batch_loss_pafmaps")
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [metric_loss_heatmaps, metric_loss_pafmaps, metric_batch_loss_pafmaps]:
        eval_metrics.add(child_metric)
    for epoch in range(config.TRAIN.end_epoch ):
        metric_loss_heatmaps.reset()
        metric_loss_pafmaps.reset()
        for batch_cnt, batch in enumerate(tqdm.tqdm(train_loader)):
            data_list = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            heatmaps_list = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            heatmaps_masks_list = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
            pafmaps_list = gluon.utils.split_and_load(batch[3], ctx_list=ctx, batch_axis=0)
            pafmaps_masks_list = gluon.utils.split_and_load(batch[4], ctx_list=ctx, batch_axis=0)

            losses = []
            with ag.record():
                for data, heatmaps, heatmaps_masks, pafmaps, pafmaps_masks in zip(data_list, heatmaps_list, heatmaps_masks_list, pafmaps_list, pafmaps_masks_list):
                    number_of_keypoints = heatmaps.shape[1]
                    y_hat = net(data)
                    heatmap_prediction = y_hat[:, :number_of_keypoints]
                    pafmap_prediction = y_hat[:, number_of_keypoints:]
                    pafmap_prediction = pafmap_prediction.reshape(0, 2, -1, pafmap_prediction.shape[2], pafmap_prediction.shape[3])
                    loss_heatmap = config.TRAIN.loss_heatmap_weight * mx.nd.sum(SigmodCrossEntropyLoss(heatmap_prediction, heatmaps) * heatmaps_masks) / (mx.nd.sum(heatmaps_masks) + 0.001)
                    loss_pafmap = mx.nd.sum(((pafmap_prediction - pafmaps) ** 2) * pafmaps_masks) / (mx.nd.sum(pafmaps_masks) + 0.001)
                    # loss_pafmap = config.TRAIN.loss_paf_weight * mx.nd.sum((SigmodPafCrossEntropyLoss(pafmap_prediction, pafmaps)) * pafmaps_masks) / (mx.nd.sum(pafmaps_masks) + 0.001)

                    losses.append(loss_heatmap)
                    losses.append(loss_pafmap)
            ag.backward(losses)
            trainer.step(1, ignore_stale_grad=False)

            metric_loss_heatmaps.update(None, loss_heatmap)
            metric_loss_pafmaps.update(None, loss_pafmap)
            metric_batch_loss_pafmaps.update(None, loss_pafmap)
            msg = ','.join(['{}={:.3f}'.format(w, v) for w, v in zip(*eval_metrics.get())])
            msg += ",lr={}".format(trainer.learning_rate)
            logging.info(msg)
            metric_batch_loss_pafmaps.reset()
        save_path = "{}-{}-{}.params".format(config.TRAIN.model_prefix, epoch, 0.0)
        net.collect_params().save(save_path)
        trainer.save_states(config.TRAIN.model_prefix + "-trainer.states")

