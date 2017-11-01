import mxnet as mx
import numpy as np
from demo import padimg
from train import save_prefix
import cv2
import matplotlib.pyplot as plt
from modelCPMWeight import CPMModel_test
class DataBatch(object):
    def __init__(self, data, label, pad=0):
        self.data = [data]
        self.label = 0
        self.pad = pad
epoch = 20
batch_size = 1
sym  = CPMModel_test(False)
sym_load, newargs,aux_args = mx.model.load_checkpoint(save_prefix, epoch)
model = mx.mod.Module(symbol=sym, context=[mx.gpu(x) for x in [0]],                        
                    label_names=None)
model.bind(data_shapes=[('data', (batch_size, 3, max_img_shape[0], max_img_shape[1]))],for_training = True)
model.init_params(arg_params=newargs, aux_params=aux_args, allow_missing=False,allow_extra=False)

img = cv2.imread("figures/Figure_1.png")
img = padimg(img,368)
imgs_transpose = np.transpose(np.float32(img[:,:,:]), (2,0,1))/256 - 0.5
imgs_batch = DataBatch(mx.nd.array([imgs_transpose[:,:,:]]), 0)
model.forward(imgs_batch)
result = model.get_outputs()
heatmap = np.moveaxis(result[1].asnumpy()[0], 0, -1)
heatmap = cv2.resize(result[1], (368, 368), interpolation=cv2.INTER_CUBIC)   

plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(heatmap[:,:,14])
plt.show()