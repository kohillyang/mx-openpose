#!/usr/bin/python2
# encoding=utf-8
'''
Created on 2017年10月11日

@author: kohillyang
'''
import sys
from showdataset import Ai_data_set
import mxnet as mx
import numpy as np


from modelCPMWeight import CPMModel,numofparts,numoflinks
save_prefix  = "models/yks_pose"
def getModule(prefix=None , begin_epoch=0, batch_size=10,re_init = False,gpus = [1,5]):
    if re_init:
        print("reinit")
        pretrained_vgg19_yks_pose_prefix = "/data1/yks/mxnet_ai/outputs/models/yks_pose"
        pretrained_vgg19_yks_pose_epoch = 8600
        pre_resnet_prefix = "../mx-openpose-backup/models/yks_pose"
        pre_resnet_epoch = 5500
        _,arg_vgg_pose,aux_vgg_pose = mx.model.load_checkpoint(pretrained_vgg19_yks_pose_prefix,
                                                             pretrained_vgg19_yks_pose_epoch)
        sym_res,arg_res,aux_res = mx.model.load_checkpoint(pre_resnet_prefix,pre_resnet_epoch)      
        print(sym_res.list_arguments())
        sym = CPMModel(use_resnet = False)
        args = arg_vgg_pose
        aux = aux_vgg_pose

        from libs.config import resnet_keys
        for key in resnet_keys:
            try:
                args[key] = arg_res[key]
                # aux[key] = aux_res[key]
                print('using resnet key...',key)
            except KeyError as e:
                print (e,key)
    else:
        _,args,aux = mx.model.load_checkpoint(prefix, begin_epoch)
#         del args['conv5_1_CPM_L1_weight']
#         del args['conv5_1_CPM_L1_bias']
#         del args['Mconv1_stage2_L2_weight']
#         del args['Mconv1_stage2_L2_bias']
#         
#         for key in args.keys():
#             print(key,args[key].shape)
#             if args[key].shape == (128,169,7,7) or args[key].shape == (128,256,3,3) :
#                 del args[key]
        sym = CPMModel()
#         print(sym.list_arguments())
    model = mx.mod.Module(symbol=sym, context=[mx.gpu(x) for x in gpus],
                        label_names=['heatmaplabel',
                                'partaffinityglabel',
                                'heatweight',
                                'vecweight'])
    model.bind(data_shapes=[('data', (batch_size, 3, 368, 368))],
            label_shapes=[
                    ('heatmaplabel', (batch_size, numofparts, 46, 46)),
                    ('partaffinityglabel', (batch_size, numoflinks * 2, 46, 46)),
                    ('heatweight', (batch_size, numofparts, 46, 46)),
                    ('vecweight', (batch_size, numoflinks * 2, 46, 46))])
    model.init_params(arg_params=args, aux_params=aux, allow_missing=False,allow_extra = True)

    return model
def train(cmodel,train_data,begin_epoch,end_epoch,batch_size,save_prefix,single_train_count = 4):
    cmodel.init_optimizer(optimizer='rmsprop', optimizer_params=(('learning_rate', 1e-6 ), ))         
    for nbatch,data_batch in enumerate(train_data):
        current_batch = begin_epoch + nbatch 
        if current_batch >= end_epoch:
            print("info: finish training.")
            return
        if nbatch % 100 == 0:
            cmodel.save_checkpoint(save_prefix, current_batch)
            print ("save_checkpoint finished")
        if nbatch % 10 != 0:
            for _ in range(single_train_count):
                cmodel.forward(data_batch, is_train=True) 
                cmodel.backward()  
                cmodel.update()
        else:
            sumerror=0
            cmodel.forward(data_batch, is_train=True)       # compute predictions  
            prediction=cmodel.get_outputs()
            for i in range(6):
                lossiter = prediction[i*2 + 1].asnumpy()              
                cls_loss = np.sum(lossiter)/batch_size
                sumerror = sumerror + cls_loss
                print cls_loss,
            print ""
            for i in range(6):
                lossiter = prediction[i*2 + 0].asnumpy()              
                cls_loss = np.sum(lossiter)/batch_size
                sumerror = sumerror + cls_loss
                print cls_loss,
            print(current_batch,end_epoch,sumerror)
            print ""

            sumerror=0

            for _ in range(single_train_count):
                cmodel.forward(data_batch, is_train=True) 
                cmodel.backward()  
                cmodel.update()

            cmodel.forward(data_batch, is_train=True)       # compute predictions  
            prediction=cmodel.get_outputs()
            for i in range(6):
                lossiter = prediction[i*2 + 1].asnumpy()              
                cls_loss = np.sum(lossiter)/batch_size
                sumerror = sumerror + cls_loss
                print cls_loss,
            print ""
            for i in range(6):
                lossiter = prediction[i*2 + 0].asnumpy()              
                cls_loss = np.sum(lossiter)/batch_size
                sumerror = sumerror + cls_loss
                print cls_loss,
            print(current_batch,end_epoch,sumerror)
            print ""
         
            print("*******************************")
                
if __name__ == "__main__":

    start_epoch = 0
    batch_size = 24
    cpm_model = getModule(save_prefix,start_epoch,batch_size,False)
#     train_data = Ai_data_set(batch_size,"/data1/yks/mxnet_ai/ai_openpose/ai_inf_v1.db")
    from libs.dataset_parser import Ai_DataIter
    train_data = mx.io.PrefetchingIter( Ai_DataIter(batch_size = batch_size))
    train(cpm_model,train_data,start_epoch,9999,batch_size,save_prefix,1)





