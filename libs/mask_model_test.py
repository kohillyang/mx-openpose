'''
Created on Nov 4, 2017

@author: kohill
'''
from __future__ import print_function
import json
import mxnet as mx
import numpy as np
import cv2
from demo import padimg
from train import save_prefix
import matplotlib.pyplot as plt
from modelCPMWeight import CPMModel_test
if __name__ == '__main__':
    epoch = 1400
    batch_size = 1
    sym  = CPMModel_test(False)
    sym_load, newargs,aux_args = mx.model.load_checkpoint(save_prefix, epoch)
    obj= json.load(open("libs/a.json","rb"))
    model = mx.mod.Module(symbol=sym, context=[mx.gpu(x) for x in [0]],                        
                        label_names=None)
    model.bind(data_shapes=[('data', (batch_size, 3, 368, 368))],for_training = True)
    model.init_params(arg_params=newargs, aux_params=aux_args, allow_missing=False,allow_extra=False)
    for one_img in obj:
        for rect in one_img['rects']:            
            ori_img = cv2.imread(one_img['path'])
            recti = list(map(lambda x:int(x),rect[:4]))
            ori_img_crop = ori_img[recti[1]:recti[3],recti[0]:recti[2],:]
            
            img = padimg(ori_img_crop,368)
            imgs_transpose = np.transpose(np.float32(img[:,:,:]), (2,0,1))/256 - 0.5
            imgs_batch = mx.io.DataBatch([mx.nd.array(imgs_transpose[np.newaxis,:,:,:])], label = None)
            model.forward(imgs_batch)
            result = model.get_outputs()
            heatmap = np.moveaxis(result[-1].asnumpy()[0], 0, -1)
            heatmap = cv2.resize(heatmap, (368, 368), interpolation=cv2.INTER_CUBIC)   
            
            plt.subplot(121)
            plt.imshow(img)
            plt.subplot(122)
            plt.imshow(heatmap[:,:,14])
            plt.show()
