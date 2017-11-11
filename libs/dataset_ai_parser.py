import numpy as np
import json,cv2,pickle
from pprint import pprint
import matplotlib.pyplot as plt
from modelCPMWeight import numoflinks
import modelCPMWeight
from cython.heatmap import putGaussianMaps
from cython.pafmap import putVecMaps

numofparts = modelCPMWeight.numofparts -1 

jsonpath = "/data1/yks/dataset/ai_challenger/ai_challenger_keypoint_train_20170909/keypoint_train_annotations_20170909.json"
images_path = "/data1/yks/dataset/ai_challenger/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902/" 
obj = json.load(open(jsonpath,"rb"))
def imshow(x,y):
    fig = plt.gcf();fig.set_size_inches(8, 8);plt.title(x); plt.imshow(y);plt.show()
def genMask(human_annotations,fscale,img_path):
    img = cv2.imread(img_path)
    
    mask = np.zeros(shape = (368,368),dtype = np.int32)
    for key in human_annotations.keys():
        x0,x1,y0,y1 = (np.array(human_annotations[key]).astype(np.float32) * fscale).astype(np.int32)
        mask[y0:y1,x0:x1] = 1
    return mask

def padimg(img,destsize):
    import cv2
    import numpy as np
    
    s = img.shape
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
def generateLabelMap(imgid,human_annotations,keypoint_annotations):
    img_path = images_path + "{0}.jpg".format(imgid)
    ori_img = cv2.imread(img_path)
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
    for key in human_annotations:
        keypoint_annotation = keypoint_annotations[key]
        for i in range(len(keypoint_annotation)/3):
            x = keypoint_annotation[i*3 + 0]
            y = keypoint_annotation[i*3 + 1]
            v = keypoint_annotation[i*3 + 2]
            x *= fscale
            y *= fscale
            x = int(x)
            y = int(y)
            part_id = i
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
    for key in human_annotations:
        keypoint_annotation = keypoint_annotations[key]
        for i in range(numoflinks):
            index0 = mid_1[i]-1 
            index1 = mid_2[i]-1 
            x0 = int(keypoint_annotation[index0 *3 + 0] * fscale)
            y0 = int(keypoint_annotation[index0 *3 + 1] * fscale)   
            x1 = int(keypoint_annotation[index1 *3 + 0] * fscale)
            y1 = int(keypoint_annotation[index1 *3 + 1] * fscale)  

            putVecMaps(pag_map[2 * i], pag_map[2 * i + 1], count,
                       x0, y0, 
                       x1, y1,
                       stride, 46, 46, sigma, thre)
    # for i in range(len(heat_map)):
    #     heat_map[i]= cv2.resize(heat_map[i],(46,46),interpolation=cv2.INTER_NEAREST)
    return img_pad,heat_map, pag_map,genMask(human_annotations,fscale,img_path).astype(np.float32)
def pase_one_img(imgid,human_annotations,keypoint_annotations):
    pass
def has_all_keypoints(keypoints):
    r = True
    for i in range(len(keypoints)/3):
        v = keypoints[i*3 + 2]
        r &= v<=1
    return r
def convertdataset2sqlite(filename = "ai_inf_v1.db",maxcount = 9999999999):
    import sqlite3,json,cv2
    conn= sqlite3.connect(filename)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE if not exists DB0
            (ID TEXT PRIMARY KEY     NOT NULL,
            DATA        BLOB);''')  
    count = 0
    for oneimg in obj:
        human_annotations = oneimg['human_annotations']
        keypoint_annotations = oneimg['keypoint_annotations']
        all_has_all_points = True
        for key in human_annotations.keys():
            onehuman_rect = human_annotations[key]
            onehuman_keypoint = keypoint_annotations[key]
            all_has_all_points &= has_all_keypoints(onehuman_keypoint)
        if all_has_all_points:
            image,heatmap,pagmap,mask = generateLabelMap(oneimg['image_id'],human_annotations,keypoint_annotations)
   
            img_path = images_path + "{0}.jpg".format(oneimg['image_id'])

            v = pickle.dumps([image, mask, heatmap, pagmap]) 
            cursor.execute("INSERT INTO DB0(ID,DATA) VALUES ( ?, ? )" ,
                            (str(img_path),sqlite3.Binary(buffer(v)),)  
                        )
            count += 1
            print(count,len(obj),maxcount)
            if (count % 10)  == 0:
                conn.commit()  
            if (count > maxcount):
                return
if __name__ == "__main__":

    convertdataset2sqlite()