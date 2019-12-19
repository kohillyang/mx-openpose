"""Train SSD"""
import logging

import mxnet as mx
import numpy as np
import mxnet.autograd as ag
import mxnet.ndarray as nd
from mxnet import gluon

from dataset import MPIIDataset
from models.drn_gcn import DRN50_GCN
import sys
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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    CLASSES = ('dial', 'needle')
    gpus = [3]
    epoch_count = 100000
    save_prefix = "output/gcn/"

    # fix seed for mxnet, numpy and python builtin random generator.
    mx.random.seed(3)
    np.random.seed(3)

    ctx = [mx.gpu(int(i)) for i in gpus]
    ctx = ctx if ctx else [mx.cpu()]

    train_dataset = MPIIDataset()
    net = DRN50_GCN(num_classes=train_dataset.number_of_keypoints + 2 * train_dataset.number_of_pafs)

    net.initialize(init=mx.init.Normal())
    net.collect_params().reset_ctx(ctx)

    train_loader = mx.gluon.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    trainer = gluon.Trainer(
        net.collect_params(),
        'adam',
        {'learning_rate': 1e-4,
         #         'wd': args.wd, 'momentum': args.momentum
         }
    )

    metric_loss_heatmaps = mx.metric.Loss("loss_heatmaps")
    metric_loss_pafmaps = mx.metric.Loss("loss_pafmaps")
    for epoch in range(epoch_count):
        metric_loss_heatmaps.reset()
        metric_loss_pafmaps.reset()
        for batch_cnt, batch in enumerate(train_loader):
            # img, heatmaps, heatmaps_masks, pafmaps, pafmaps_masks
            data_list = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            heatmaps_list = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            heatmaps_masks_list = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
            pafmaps_list = gluon.utils.split_and_load(batch[3], ctx_list=ctx, batch_axis=0)
            pafmaps_masks_list = gluon.utils.split_and_load(batch[4], ctx_list=ctx, batch_axis=0)
            masks_list = gluon.utils.split_and_load(batch[5], ctx_list=ctx, batch_axis=0)

            losses = []
            with ag.record():
                for data, heatmaps, heatmaps_masks, pafmaps, pafmaps_masks, masks in zip(
                        data_list, heatmaps_list, heatmaps_masks_list, pafmaps_list,
                        pafmaps_masks_list, masks_list):
                    number_of_keypoints = heatmaps.shape[1]
                    y_hat = net(data)
                    heatmap_prediction = y_hat[:, :number_of_keypoints]
                    pafmap_prediction = y_hat[:, number_of_keypoints:]
                    pafmap_prediction = pafmap_prediction.reshape(0, 2, -1, pafmap_prediction.shape[2], pafmap_prediction.shape[3])
                    loss_heatmap = mx.nd.sum(SigmodCrossEntropyLoss(heatmap_prediction, heatmaps) * heatmaps_masks * masks) / mx.nd.sum(heatmaps_masks * masks)
                    loss_pafmap = mx.nd.sum(((pafmap_prediction - pafmaps) ** 2) * pafmaps_masks * masks) / mx.nd.sum(pafmaps_masks * masks)
                    losses.append(loss_heatmap)
                    losses.append(loss_pafmap)
                    metric_loss_heatmaps.update(None, loss_heatmap)
                    metric_loss_heatmaps.update(None, loss_pafmap)
            ag.backward(losses)
            trainer.step(1, ignore_stale_grad=True)
