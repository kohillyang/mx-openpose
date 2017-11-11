'''
Created on Nov 3, 2017

@author: kohill
'''
from __future__ import print_function
from __future__ import absolute_import
import sys
sys.path.append("../")
sys.path.append("./")

from cython.heatmap import putGaussianMaps
from cython.pafmap import putVecMaps

from Queue import Queue
import numpy as np
import mxnet as mx
import json,cv2,os
class config():
    sigma = 7.0
from random import randint
class DataBatchweight(object):
    def __init__(self, data, heatmaplabel, partaffinityglabel, heatweight, vecweight, pad=0):
        self.data = [data]
        self.label = [heatmaplabel, partaffinityglabel, heatweight, vecweight]
        self.pad = pad


def zero_batch(shape,batchsize = 1):
    data = mx.nd.array(np.zeros(shape= (batchsize,3,shape[0],shape[1])))
    heatmaplabel = mx.nd.array(np.zeros(shape= (batchsize,15,shape[0]/8,shape[1]/8)))
    heatweight = mx.nd.array(np.zeros(shape= (batchsize,15,shape[0]/8,shape[1]/8)))
    partaffinityglabel = mx.nd.array(np.zeros(shape= (batchsize,26,shape[0]/8,shape[1]/8)))
    vecweight = mx.nd.array(np.zeros(shape= (batchsize,26,shape[0]/8,shape[1]/8)))
    return DataBatchweight(data, heatmaplabel, partaffinityglabel, heatweight, vecweight)
    

class Ai_DataIter():
    def __init__(self,
                 batch_size = 16,
                 shape = [368,368],
                 json_path = "/data1/yks/dataset/ai_challenger/ai_challenger_keypoint_train_20170909/keypoint_train_annotations_20170909.json",
                 images_path = "/data1/yks/dataset/ai_challenger/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902/"
                 ):
        self.obj = json.load(open(json_path,"rb"))
        self.queue = Queue(maxsize = 10)
        self.images_path = images_path
        self.batchsize = batch_size
        self.shape = tuple(shape)
        self.index = 0
        self.numofparts = 14
        self.numoflinks = 13
    @property
    def provide_data(self):
        return [('data',(1,3,self.shape[0],self.shape[1]))]
    @property
    def provide_label(self):
        batch_size = self.batchsize
        numofparts = self.numofparts + 1
        numoflinks = self.numoflinks
        return   [
                    ('heatmaplabel', (batch_size, numofparts, 46, 46)),
                    ('partaffinityglabel', (batch_size, numoflinks * 2, 46, 46)),
                    ('heatweight', (batch_size, numofparts, 46, 46)),
                    ('vecweight', (batch_size, numoflinks * 2, 46, 46))] 
    def __len__(self):
        return len(self.obj)
    def __iter__(self):
        return self
    def __next__(self):


        transposeImage_batch = []
        heatmap_batch = []
        pagmap_batch = []
        heatweight_batch = []
        vecweight_batch = []
        for _ in range(self.batchsize):
            image, mask, heatmap, pagmap = self.get_next_data_label()
            self.index += 1
            maskscale = mask[0:368:8, 0:368:8]           
            heatweight = np.repeat(maskscale[np.newaxis, :, :], len(heatmap), axis=0)
            vecweight  = np.repeat(maskscale[np.newaxis, :, :], len(pagmap), axis=0)
            
            if randint(0,9) <=5:
                reserve_color = randint(0,2)
                image[:,:,(0,1,2)] = image[:,:,(reserve_color,reserve_color,reserve_color)]
            else:
                color_i,color_j,color_k = randint(0,2),randint(0,2),randint(0,2)
                image[:,:,(0,1,2)] = image[:,:,(color_i,color_j,color_k)] 
            transposeImage = np.transpose(np.float32(image), (2,0,1))/255 - 0.5
        
            
            transposeImage_batch.append(transposeImage)
            heatmap_batch.append(heatmap)
            pagmap_batch.append(pagmap)
            heatweight_batch.append(heatweight)
            vecweight_batch.append(vecweight)
            
        return mx.io.DataBatch(
            [mx.nd.array(transposeImage_batch)],
            [mx.nd.array(heatmap_batch),
            mx.nd.array(pagmap_batch),
            mx.nd.array(heatweight_batch),
            mx.nd.array(vecweight_batch)])

    def next(self):
        return self.__next__()
    def run(self):
        pass
    def get_next_data_label(self):
#         rindex = randint(0,len(self.obj)-1)
        oneimg = self.obj[self.index]
        img_path = os.path.join(self.images_path,oneimg['image_id']+".jpg")
        ori_img = cv2.imread(img_path)
        fscale_x = 1.0 * self.shape[1]/ori_img.shape[1]
        fscale_y = 1.0 * self.shape[0]/ori_img.shape[0]
        img = cv2.resize(ori_img,self.shape)
        keypoint_annotations = oneimg['keypoint_annotations']


        heat_map = list()
        stride = 8
        for _ in range(self.numofparts+1):
            heat_map.append(np.zeros((self.shape[0] / stride, 
                                      self.shape[1] / stride)))
        
        for key in keypoint_annotations:
            annotation = keypoint_annotations[key]
            assert len(annotation)//3 == self.numofparts
            for part_id in range(self.numofparts):
                x,y,v = annotation[part_id*3:part_id*3+3]
                if v <=2:
                    x *= fscale_x
                    y *= fscale_y
                    putGaussianMaps(heat_map[part_id], 
                                    self.shape[1], self.shape[0], 
                                    x,y,
                                    stride, 
                                    self.shape[1]//stride, 
                                    self.shape[0]//stride, config.sigma)                
        heat_map[self.numofparts] = np.max(heat_map[:-1],axis=0)
        mid_1 = [13,14,14,1, 2, 4, 5, 1, 7, 8, 4, 10, 11]
        mid_2 = [14,1,  4,2, 3, 5, 6, 7, 8, 9, 10,11, 12]
        thre = 1
    
        pag_map = list()
        for i in range(self.numoflinks*2):
            pag_map.append(np.zeros((self.shape[0] / stride, self.shape[1] / stride)))
    
        count = np.zeros_like(pag_map[0])
        for key in keypoint_annotations:
            keypoint_annotation = keypoint_annotations[key]
            for i in range(self.numoflinks):
                index0 = mid_1[i]-1 
                index1 = mid_2[i]-1 
                v0 = int(keypoint_annotation[index0 *3 + 2])
                v1 = int(keypoint_annotation[index1 *3 + 2])
                if v0 <=2 and v1 <= 2:                      
                    x0 = int(keypoint_annotation[index0 *3 + 0] * fscale_x)
                    y0 = int(keypoint_annotation[index0 *3 + 1] * fscale_y)   
                    x1 = int(keypoint_annotation[index1 *3 + 0] * fscale_x)
                    y1 = int(keypoint_annotation[index1 *3 + 1] * fscale_y)                  
                    putVecMaps(pag_map[2 * i], pag_map[2 * i + 1], count,
                               x0, y0, 
                               x1, y1,
                               stride, 46, 46, config.sigma, thre)
        mask = np.zeros(shape = self.shape,dtype = np.float32)
        human_annotations = oneimg['human_annotations']
        '''
        Todo: generate mask, ignore points which are not labeled.
        '''
        for key in human_annotations.keys():
            x0,y0,x1,y1 = np.array(human_annotations[key]).astype(np.float32)
            x0 *= fscale_x
            y0 *= fscale_y
            x1 *= fscale_x
            y1 *= fscale_y
            mask[int(y0):int(y1),int(x0):int(x1)] = 1
        return img,mask,heat_map,pag_map        
if __name__ == '__main__':
    ai_iter = Ai_DataIter()
    import matplotlib.pyplot as plt

    
    
    for i in range(len(ai_iter)):
        fig, axes = plt.subplots(3, 15, figsize=(45, 6),
                             subplot_kw={'xticks': [], 'yticks': []})
    
        fig.subplots_adjust(hspace=0.3, wspace=0.05)
        img,mask,heat_map,pag_map = ai_iter.get_next_data_label()
        for i in range(ai_iter.numofparts + 1):
            axes[0][i].imshow(heat_map[i])
        all_part = np.zeros_like(np.sqrt(pag_map[0 * 2]**2 + pag_map[0 * 2+1]**2 ))
        for i in range(ai_iter.numoflinks):
            axes[1][i].imshow(np.sqrt(pag_map[i * 2]**2 + pag_map[i * 2+1]**2 ))
            all_part += np.sqrt(pag_map[i * 2]**2 + pag_map[i * 2+1]**2 )
        axes[1][ai_iter.numoflinks].imshow(all_part) 
        axes[2][0].imshow(img[:,:,(2,1,0)])     
        axes[2][1].imshow(mask)          
        ai_iter.index += 1
        plt.show()
 