import mxnet as mx
import numpy as np
from demo import padimg
from train import save_prefix
#save_prefix = "/data1/yks/mxnet_ai/outputs/models/yks_pose"
import cv2
import matplotlib.pyplot as plt
from modelCPMWeight import CPMModel_test,CPMModel
class DataBatch(object):
    def __init__(self, data, label, pad=0):
        self.data = [data]
        self.label = 0
        self.pad = pad
epoch = 8700
batch_size = 1
sym  = CPMModel_test(False)
# mx.visualization.plot_network(CPMModel(),
# shape = {"data":(1,3,368,368),
#          "heatmaplabel":(1,15,46,46),
#          "heatweight":(1,15,46,46),
#          "partaffinityglabel":(1,26,46,46),
#          "vecweight":(1,26,46,46),
#          }                              
#                               
#                               ).view()
sym_load, newargs,aux_args = mx.model.load_checkpoint(save_prefix, epoch)
model = mx.mod.Module(symbol=sym, context=[mx.gpu(x) for x in [0]],                        
                    label_names=None)
model.bind(data_shapes=[('data', (batch_size, 3, 368, 368))],for_training = True)
model.init_params(arg_params=newargs, aux_params=aux_args, allow_missing=False,allow_extra=False)

img = cv2.imread("/home/kohill/mx_wildpose/sample_image/multiperson.jpg")
img = padimg(img,368)
imgs_transpose = np.transpose(np.float32(img[:,:,:]), (2,0,1))/256 - 0.5
imgs_batch = DataBatch(mx.nd.array([imgs_transpose[:,:,:]]), 0)
model.forward(imgs_batch)
result = model.get_outputs()
heatmap = np.moveaxis(result[-1].asnumpy()[0], 0, -1)
heatmap = cv2.resize(heatmap, (368, 368), interpolation=cv2.INTER_CUBIC)   

plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(heatmap[:,:,14])
plt.show()
