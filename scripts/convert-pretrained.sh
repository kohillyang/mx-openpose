export PYTHONPATH=/data3/zyx/yks/mx-openpose/pretrained/caffe/python:$PYTHONPATH:
mmtoir -f caffe -n pretrained/VGG_ILSVRC_19_layers_deploy.prototxt -w pretrained/VGG_ILSVRC_19_layers.caffemodel -o caffe_vgg_IR
mmtocode -f mxnet --IRModelPath caffe_vgg_IR.pb --dstModelPath mxnet_vgg19.py --IRWeightPath vgg19.npy -dw mxnet_inception_v3-0000.params
