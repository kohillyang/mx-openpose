#!/usr/bin/python2
#encoding=utf-8
'''
Created on 2017年10月11日

@author: kohillyang
'''
# from __future__ import absolute_import
import sys,os,json,cv2,copy,sqlite3
from random import randint
import cPickle as pickle
import mxnet as mx
selfpathdir = os.path.dirname(__file__)



import numpy as np
from modelCPMWeight import numoflinks
import modelCPMWeight
numofparts = modelCPMWeight.numofparts -1 

# heatmap_index_map = [[12],[13],[0,3],[1,4],[2,5],[6,9],[7,10],[8,11],[14]]#14 means mean of all part
# pafmap_index_map =  [0,4,1,5,6,2,3,7,12,9,8,11,10]
'''
13 脖子
12 头顶 
11 右脚
10 右膝
9 右髋
8 左脚
7 左膝
6 左髋
5 右腕
4 右肘
3 右肩
2 左腕
1 左肘
0 左肩
'''

'''
0: 头顶-脖子  0
1： 脖子到左肩 4
2： 脖子到右肩 1
3： 左肩膀到左手肘 5
4： 左手肘到左手腕 6

5: 右肩膀到右手肘 2
6: 右手肘到右手腕 3
7： left shoulder - left hip 
8: left hip - left knee 12
9: left knee - left ankle 13
10: right shoulder-right hip 
11: right hip- right knee 9
12: right knee-right ankle 10
'''





class Ai_data_set(object):
    class DataBatchweight(object):
        def __init__(self, data, heatmaplabel, partaffinityglabel, heatweight, vecweight, pad=0):
            self.data = [data]
            self.label = [heatmaplabel, partaffinityglabel, heatweight, vecweight]
            self.pad = pad

    def __init__(self,batchsize,dbname ):
        self.dbname = dbname
        self.batchsize = batchsize
        self.conn = sqlite3.connect(self.dbname)
        self.cursor = self.conn.cursor()        
        self.cursor.execute(' SELECT COUNT(*) FROM DB0 ')
        self.count = int(self.cursor.fetchone()[0])
        print("info: dataset count",self.count)
    def __iter__(self):
        self.cur_batch = 0
        self.cursor.close()
        self.cursor = self.conn.cursor()        
        return self
    def __next__(self):
        r = []
        transposeImage_batch = []
        heatmap_batch = []
        pagmap_batch = []
        heatweight_batch = []
        vecweight_batch = []
        for _ in range(self.batchsize):
            self.cursor.execute(' SELECT * FROM DB0 limit {0} OFFSET {1}'.format(1,randint(0,self.count - 2)))
            row = self.cursor.fetchone()
            data  = row[1]
            r.append(pickle.loads(str(data)))            
            image, mask, heatmap, pagmap = pickle.loads(str(data))

            maskscale = mask[0:368:8, 0:368:8]           
            heatweight = np.repeat(maskscale[np.newaxis, :, :], len(heatmap), axis=0)
            vecweight  = np.repeat(maskscale[np.newaxis, :, :], len(pagmap), axis=0)
            
            transposeImage = np.transpose(np.float32(image), (2,0,1))/255 - 0.5
        
            self.cur_batch += 1
            
            transposeImage_batch.append(transposeImage)
            heatmap_batch.append(heatmap)
            pagmap_batch.append(pagmap)
            heatweight_batch.append(heatweight)
            vecweight_batch.append(vecweight)
            
        return Ai_data_set.DataBatchweight(
            mx.nd.array(transposeImage_batch),
            mx.nd.array(heatmap_batch),
            mx.nd.array(pagmap_batch),
            mx.nd.array(heatweight_batch),
            mx.nd.array(vecweight_batch))

    def next(self):
        return self.__next__()






            
