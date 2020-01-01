import easydict, os


def get_coco_config():
    config = easydict.EasyDict()
    config.TRAIN = easydict.EasyDict()
    config.TRAIN.save_prefix = "output/cpm/"
    config.TRAIN.model_prefix = os.path.join(config.TRAIN.save_prefix, "resnet50-cpm-teachered-cropped")
    config.TRAIN.gpus = [3, 8]
    config.TRAIN.batch_size = 8
    config.TRAIN.optimizer = "SGD"
    config.TRAIN.lr = 2e-5
    config.TRAIN.momentum = 0.9
    config.TRAIN.wd = 0.0001
    config.TRAIN.lr_step = [8, 12, 16]
    config.TRAIN.warmup_step = 100
    config.TRAIN.gamma = 1.0 / 3
    config.TRAIN.warmup_lr = config.TRAIN.lr * 0.1
    config.TRAIN.end_epoch = 26
    config.TRAIN.resume = None
    config.TRAIN.DATASET = easydict.EasyDict()
    config.TRAIN.DATASET.coco_root = "/data3/zyx/yks/dataset/coco2017"
    config.TRAIN.TRANSFORM_PARAMS = easydict.EasyDict()

    # params for random cropping
    config.TRAIN.TRANSFORM_PARAMS.crop_size_x = 368
    config.TRAIN.TRANSFORM_PARAMS.crop_size_y = 368
    config.TRAIN.TRANSFORM_PARAMS.center_perterb_max = 40

    # params for random scale
    config.TRAIN.TRANSFORM_PARAMS.scale_min = 0.5
    config.TRAIN.TRANSFORM_PARAMS.scale_max = 1.1

    # params for putGaussianMaps
    config.TRAIN.TRANSFORM_PARAMS.sigma = 7.0
    config.TRAIN.TRANSFORM_PARAMS.stride = 8

    # params for putVecMaps
    config.TRAIN.TRANSFORM_PARAMS.distance_threshold = 8
    return config
