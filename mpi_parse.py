#!/usr/bin/python
# coding: utf-8

# In[124]:
import modelCPMWeight
from modelCPMWeight import numoflinks
numofparts = modelCPMWeight.numofparts -1 
from cython.heatmap import putGaussianMaps
from cython.pafmap import putVecMaps
from pprint import pprint
import cv2
import numpy as np
import sys,os,pickle
from random import randint
part2ours = [9,8,7,10,11,12,-1,14,-1,13,3,2,1,4,5,6]
img_train_size = 368

# In[125]:

def padimg(img,destsize):
    import cv2
    import numpy as np
    s = img.shape
#    print img.shape,destsize/s[1],destsize/s[1]

    if(s[0] > s[1]):
        img_d = cv2.resize(img,dsize = None,fx = 1.0 * destsize/s[0], fy = 1.0 * destsize/s[0])
        img_temp = np.ones(shape = (destsize,destsize,3),dtype=np.uint8) * 128
        sd = img_d.shape
        img_temp[0:sd[0],0:sd[1],0:sd[2]]=img_d
    else:
        img_d = cv2.resize(img,dsize = None,fx = 1.0 * destsize/s[1],fy = 1.0 * destsize/s[1])
        img_temp = np.ones(shape = (destsize,destsize,3),dtype=np.uint8) * 128 
        sd = img_d.shape
        img_temp[0:sd[0],0:sd[1],0:sd[2]]=img_d
    return img_temp


# In[126]:

def map2Ai(one_rect,fscale):
    one_person = np.zeros(shape = (numofparts,2))
    for onePoint in one_rect['annopoints']:
        x,y,part_id,isVisible = onePoint
        x *= fscale
        y *= fscale
        part_id = part2ours[part_id] -1 
        if part_id <= -1:
            continue
        one_person[part_id][0] = x 
        one_person[part_id][1] = y 
    return one_person
def genMask(oneimg,fscale):
    img = cv2.imread(oneimg['img_path'])
    mask = np.zeros(shape = (368,368),dtype = np.int32)
    for one_rect in oneimg['annoations']:
        one_person = map2Ai(one_rect,fscale)
        x0 = int(min(one_person,key = lambda x:x[0])[0])
        x1 = int(max(one_person,key = lambda x:x[0])[0])        
        y0 = int(min(one_person,key = lambda x:x[1])[1])
        y1 = int(max(one_person,key = lambda x:x[1])[1])
        # print x0,x1,y0,y1
        # pprint(one_rect)
        mask[y0:y1,x0:x1] = 1
    return mask


# In[127]:

def generateLabelMap(oneimg):

    ori_img = cv2.imread(oneimg['img_path'])
    ori_img_shape = ori_img.shape
    fscale = 368.0/max(ori_img_shape[0],ori_img_shape[1])
    img_pad = padimg(ori_img,368)
    thre = 0.5
    crop_size_width = 368
    crop_size_height = 368

    augmentcols = 368
    augmentrows = 368
    stride = 8
    grid_x = augmentcols / stride
    grid_y = augmentrows / stride
    sigma = 7.0
    
    heat_map = list()
    for i in range(numofparts+1):
        heat_map.append(np.zeros((crop_size_width / stride, crop_size_height / stride)))


    for one_rect in oneimg['annoations']:
        for onePoint in one_rect['annopoints']:
            x,y,part_id,isVisible = onePoint
            x *= fscale
            y *= fscale
            x = int(x)
            y = int(y)
            part_id = part2ours[part_id]-1
            if part_id <= -1:
                continue
            cv2.circle(heat_map[part_id],(x,y),8,(1,1,1),-1,8)

            putGaussianMaps(heat_map[part_id], 368, 368, 
                            x,y,
                            stride, grid_x, grid_y, sigma)
       
    ### put background channel
    #heat_map[numofparts] = heat_map[0]
    heat_map[numofparts] = np.max(heat_map[:-1],axis=0)
    # for g_y in range(heat_map[0].shape[0]):
    #     for g_x in range(heat_map[0].shape[1]):
    #         maximum=0
    #         for i in range(numofparts):
    #             if maximum<heat_map[i][g_y, g_x]:
    #                 maximum = heat_map[i][g_y, g_x]
    #         heat_map[numofparts][g_y,g_x]=max(maximum,0.0)
    

    mid_1 = [13,14,14,1, 2, 4, 5, 1, 7, 8, 4, 10, 11]
    mid_2 = [14,1,  4,2, 3, 5, 6, 7, 8, 9, 10,11, 12]
    thre = 1

    pag_map = list()
    for i in range(numoflinks*2):
        pag_map.append(np.zeros((46, 46)))

    count = np.zeros((46, 46))

    for one_rect in oneimg['annoations']:
        one_person = map2Ai(one_rect,fscale)
        for i in range(numoflinks):
            putVecMaps(pag_map[2 * i], pag_map[2 * i + 1], count,
                       one_person[mid_1[i]-1][0], one_person[mid_1[i]-1][1], 
                       one_person[mid_2[i]-1][0], one_person[mid_2[i]-1][1],
                       stride, 46, 46, sigma, thre)
    # for i in range(len(heat_map)):
    #     heat_map[i]= cv2.resize(heat_map[i],(46,46),interpolation=cv2.INTER_NEAREST)
    return img_pad,heat_map, pag_map,genMask(oneimg,fscale).astype(np.float32)


# In[128]:
def convertdataset2sqlite(filename = "dataset/mpi_inf_v2.db",maxcount = 9999999999):
    import sqlite3,json,cv2
    conn= sqlite3.connect(filename)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE if not exists DB0
            (ID TEXT PRIMARY KEY     NOT NULL,
            DATA        BLOB);''')  
    count = 0
    all_img = json.load(open("dataset/mpi_dataset.json","rb"))

    for oneimg in all_img:
        image,heatmap,pagmap,mask = generateLabelMap(oneimg)
        v = pickle.dumps([image, mask, heatmap, pagmap]) 
        cursor.execute("INSERT INTO DB0(ID,DATA) VALUES ( ?, ? )" ,
                        (str(oneimg['img_path']),sqlite3.Binary(buffer(v)),)  
                    )
        count += 1
        print(count,len(all_img),maxcount)
        if (count % 10)  == 0:
            conn.commit()  
        if (count > maxcount):
            return


if __name__ == "__main__":    
    convertdataset2sqlite()
    # get_ipython().magic(u'matplotlib inline')
    # import json,cv2,numpy as np,sys
    # import matplotlib.pyplot as plt
    # all_img = json.load(open("/data1/yks/dataset/openpose_dataset/mpi/a.json","rb"))
    # for oneimg in all_img[10:20]:
    #     img = cv2.imread(oneimg['img_path'])

    #     for one_rect in oneimg['annoations']:
    #         #cv2.rectangle(img,(one_rect['x1'],one_rect['y1']),(one_rect['x2'],one_rect['y2']),(0,255,255),-1,8)
    #         for point in one_rect['annopoints']:
    #             cv2.circle(img,(point[0],point[1]),12,(0,255,0),-1,8)
    #             cv2.putText(img, "{0}".format(point[2]), (point[0],point[1]), cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0));
    #     img = padimg(img,1024)
    #     fig = plt.gcf()
    #     fig.set_size_inches(16, 16)    
    #     plt.imshow(img)
    #     plt.show()

    #     img_pad,heat_map,pag_map,mask = generateLabelMap(oneimg)
    #     for heat in heat_map[13:]:
    #         fig = plt.gcf()
    #         heat = cv2.resize(heat,(368,368))
    #         fig.set_size_inches(16, 16)    
    #         plt.imshow(np.uint8(heat* 255))
    #         plt.show()
    #     fig = plt.gcf() 
    #     fig.set_size_inches(16, 16)    
    #     plt.imshow(np.uint8(mask* 255))
    #     plt.show()
            
