export PYTHONPATH=/data3/zyx/yks/mx-openpose/pretrained/caffe/python:$PYTHONPATH:
mmtoir -f caffe -n pretrained/VGG_ILSVRC_19_layers_deploy.prototxt -w pretrained/VGG_ILSVRC_19_layers.caffemodel -o caffe_vgg_IR
mmtocode -f mxnet --IRModelPath caffe_vgg_IR.pb --dstModelPath mxnet_vgg19.py --IRWeightPath caffe_vgg_IR.npy -dw mxnet_vgg19-0000.params
python -m mmdnn.conversion.examples.mxnet.imagenet_test -n mxnet_vgg19 -w mxnet_vgg19-0000.params --dump vgg19