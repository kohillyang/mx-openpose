# Introduction 

<div align="center">
<img src="https://github.com/kohillyang/mx-openpose/blob/master/figures/Figure_1.png"><br><br>
</div>
<div align="center">
<img src="https://github.com/kohillyang/mx-openpose/blob/master/figures/Figure_2.png"><br><br>
</div>
<div align="center">
<img src="https://github.com/kohillyang/mx-openpose/blob/master/figures/Figure_3.png"><br><br>
</div>
<div align="center">
<img src="https://github.com/kohillyang/mx-openpose/blob/master/figures/Figure_4.png"><br><br>
</div>

# Training Process.
Note for a bug, stage2 and stage 4's output are noise, which has already been fixed.

(click me to see the Training Process.)[http://oj5adp5xv.bkt.clouddn.com/trainng_process.png]


1. This is a mxnet-openpose implemention which is based on [mxnet_Realtime_Multi-Person_Pose_Estimation](https://github.com/dragonfly90/mxnet_Realtime_Multi-Person_Pose_Estimation) by @dragonfly90.<br>
2. I have tested this code under 4 K80 GPUS, after about 9000 iterations with batch 8, the loss can converge to about 90-110.<br>
3. Pretrained models are in <http://pan.baidu.com/s/1i5H2WHB>
4. You can get more information from the original [caffe version](https://github.com/CMU-Perceptual-Computing-Lab/openpose).


# Prepare for train.
1. run `mkdir model && mkdir models`
2. Prepare a python2 environment, and install packages of matplotlib, scipy, mxnet(>0.9), numpy.
3. Go into cython folder, run `python setup.py build_ext --inplace`
4. Download MPI dataset from <http://human-pose.mpi-inf.mpg.de>(mpii_human_pose_v1.tar.gz,12.1GB,mpii_human_pose_v1_u12_2.zip), extracting mpii_human_pose_v1_u12_2.zip you'll get a matlab mat file named `mpii_human_pose_v1_u12_1.mat`. Extracting mpii_human_pose_v1.tar.gz you'll get a folder named `imgaes`.
5. Run `python mpi_2_json.py --images=/data1/yks/dataset/openpose_dataset/mpi/images --annonations=/data1/yks/dataset/openpose_dataset/mpi/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat` to convert mpii_human_pose_v1_u12_1.mat from format mat to a json file which will be putted into folder `dataset`,you may need to change the MPI images path and the path of the annonations file.
6. Run `python mpi_parse.py` to generate mask, heatmap and some other labels, this operation will generate a sqlite db file named mpi_inf_v2.db in folder `dataset`.
7. Run `python train.py` to train the model.<br>


# Demo
After you have trained your own model or download the pretrained model, you can run `python2 demo.py --images=/data1/yks/dataset/openpose_dataset/mpi/images --prefix="models/yks_pose" --epoch=8600` to evaluate the model.
``
