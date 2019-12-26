import mxnet as mx

def get_cpm_symbol():
    data = mx.symbol.Variable(name='data')
    # data = mx.sym.transpose(data, (0, 3, 1, 2)) /256 -0.5
    conv1_1 = mx.symbol.Convolution(name='conv1_1', data=data, num_filter=64, pad=(1, 1), kernel=(3, 3), stride=(1, 1),
                                    no_bias=False)
    relu1_1 = mx.symbol.Activation(name='relu1_1', data=conv1_1, act_type='relu')
    conv1_2 = mx.symbol.Convolution(name='conv1_2', data=relu1_1, num_filter=64, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu1_2 = mx.symbol.Activation(name='relu1_2', data=conv1_2, act_type='relu')
    pool1_stage1 = mx.symbol.Pooling(name='pool1_stage1', data=relu1_2, pooling_convention='full', pad=(0, 0),
                                     kernel=(2, 2), stride=(2, 2), pool_type='max')
    conv2_1 = mx.symbol.Convolution(name='conv2_1', data=pool1_stage1, num_filter=128, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu2_1 = mx.symbol.Activation(name='relu2_1', data=conv2_1, act_type='relu')
    conv2_2 = mx.symbol.Convolution(name='conv2_2', data=relu2_1, num_filter=128, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu2_2 = mx.symbol.Activation(name='relu2_2', data=conv2_2, act_type='relu')
    pool2_stage1 = mx.symbol.Pooling(name='pool2_stage1', data=relu2_2, pooling_convention='full', pad=(0, 0),
                                     kernel=(2, 2), stride=(2, 2), pool_type='max')
    conv3_1 = mx.symbol.Convolution(name='conv3_1', data=pool2_stage1, num_filter=256, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu3_1 = mx.symbol.Activation(name='relu3_1', data=conv3_1, act_type='relu')
    conv3_2 = mx.symbol.Convolution(name='conv3_2', data=relu3_1, num_filter=256, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu3_2 = mx.symbol.Activation(name='relu3_2', data=conv3_2, act_type='relu')
    conv3_3 = mx.symbol.Convolution(name='conv3_3', data=relu3_2, num_filter=256, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu3_3 = mx.symbol.Activation(name='relu3_3', data=conv3_3, act_type='relu')
    conv3_4 = mx.symbol.Convolution(name='conv3_4', data=relu3_3, num_filter=256, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu3_4 = mx.symbol.Activation(name='relu3_4', data=conv3_4, act_type='relu')
    pool3_stage1 = mx.symbol.Pooling(name='pool3_stage1', data=relu3_4, pooling_convention='full', pad=(0, 0),
                                     kernel=(2, 2), stride=(2, 2), pool_type='max')
    conv4_1 = mx.symbol.Convolution(name='conv4_1', data=pool3_stage1, num_filter=512, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu4_1 = mx.symbol.Activation(name='relu4_1', data=conv4_1, act_type='relu')
    conv4_2 = mx.symbol.Convolution(name='conv4_2', data=relu4_1, num_filter=512, pad=(1, 1), kernel=(3, 3),
                                    stride=(1, 1), no_bias=False)
    relu4_2 = mx.symbol.Activation(name='relu4_2', data=conv4_2, act_type='relu')
    conv4_3_CPM = mx.symbol.Convolution(name='conv4_3_CPM', data=relu4_2, num_filter=256, pad=(1, 1), kernel=(3, 3),
                                        stride=(1, 1), no_bias=False)
    relu4_3_CPM = mx.symbol.Activation(name='relu4_3_CPM', data=conv4_3_CPM, act_type='relu')
    conv4_4_CPM = mx.symbol.Convolution(name='conv4_4_CPM', data=relu4_3_CPM, num_filter=128, pad=(1, 1), kernel=(3, 3),
                                        stride=(1, 1), no_bias=False)
    relu4_4_CPM = mx.symbol.Activation(name='relu4_4_CPM', data=conv4_4_CPM, act_type='relu')
    conv5_1_CPM_L1 = mx.symbol.Convolution(name='conv5_1_CPM_L1', data=relu4_4_CPM, num_filter=128, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu5_1_CPM_L1 = mx.symbol.Activation(name='relu5_1_CPM_L1', data=conv5_1_CPM_L1, act_type='relu')
    conv5_1_CPM_L2 = mx.symbol.Convolution(name='conv5_1_CPM_L2', data=relu4_4_CPM, num_filter=128, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu5_1_CPM_L2 = mx.symbol.Activation(name='relu5_1_CPM_L2', data=conv5_1_CPM_L2, act_type='relu')
    conv5_2_CPM_L1 = mx.symbol.Convolution(name='conv5_2_CPM_L1', data=relu5_1_CPM_L1, num_filter=128, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu5_2_CPM_L1 = mx.symbol.Activation(name='relu5_2_CPM_L1', data=conv5_2_CPM_L1, act_type='relu')
    conv5_2_CPM_L2 = mx.symbol.Convolution(name='conv5_2_CPM_L2', data=relu5_1_CPM_L2, num_filter=128, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu5_2_CPM_L2 = mx.symbol.Activation(name='relu5_2_CPM_L2', data=conv5_2_CPM_L2, act_type='relu')
    conv5_3_CPM_L1 = mx.symbol.Convolution(name='conv5_3_CPM_L1', data=relu5_2_CPM_L1, num_filter=128, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu5_3_CPM_L1 = mx.symbol.Activation(name='relu5_3_CPM_L1', data=conv5_3_CPM_L1, act_type='relu')
    conv5_3_CPM_L2 = mx.symbol.Convolution(name='conv5_3_CPM_L2', data=relu5_2_CPM_L2, num_filter=128, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=False)
    relu5_3_CPM_L2 = mx.symbol.Activation(name='relu5_3_CPM_L2', data=conv5_3_CPM_L2, act_type='relu')
    conv5_4_CPM_L1 = mx.symbol.Convolution(name='conv5_4_CPM_L1', data=relu5_3_CPM_L1, num_filter=512, pad=(0, 0),
                                           kernel=(1, 1), stride=(1, 1), no_bias=False)
    relu5_4_CPM_L1 = mx.symbol.Activation(name='relu5_4_CPM_L1', data=conv5_4_CPM_L1, act_type='relu')
    conv5_4_CPM_L2 = mx.symbol.Convolution(name='conv5_4_CPM_L2', data=relu5_3_CPM_L2, num_filter=512, pad=(0, 0),
                                           kernel=(1, 1), stride=(1, 1), no_bias=False)
    relu5_4_CPM_L2 = mx.symbol.Activation(name='relu5_4_CPM_L2', data=conv5_4_CPM_L2, act_type='relu')
    conv5_5_CPM_L1 = mx.symbol.Convolution(name='conv5_5_CPM_L1', data=relu5_4_CPM_L1, num_filter=38, pad=(0, 0),
                                           kernel=(1, 1), stride=(1, 1), no_bias=False)
    conv5_5_CPM_L2 = mx.symbol.Convolution(name='conv5_5_CPM_L2', data=relu5_4_CPM_L2, num_filter=19, pad=(0, 0),
                                           kernel=(1, 1), stride=(1, 1), no_bias=False)
    concat_stage2 = mx.symbol.Concat(name='concat_stage2', *[conv5_5_CPM_L1, conv5_5_CPM_L2, relu4_4_CPM])
    Mconv1_stage2_L1 = mx.symbol.Convolution(name='Mconv1_stage2_L1', data=concat_stage2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage2_L1 = mx.symbol.Activation(name='Mrelu1_stage2_L1', data=Mconv1_stage2_L1, act_type='relu')
    Mconv1_stage2_L2 = mx.symbol.Convolution(name='Mconv1_stage2_L2', data=concat_stage2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage2_L2 = mx.symbol.Activation(name='Mrelu1_stage2_L2', data=Mconv1_stage2_L2, act_type='relu')
    Mconv2_stage2_L1 = mx.symbol.Convolution(name='Mconv2_stage2_L1', data=Mrelu1_stage2_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage2_L1 = mx.symbol.Activation(name='Mrelu2_stage2_L1', data=Mconv2_stage2_L1, act_type='relu')
    Mconv2_stage2_L2 = mx.symbol.Convolution(name='Mconv2_stage2_L2', data=Mrelu1_stage2_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage2_L2 = mx.symbol.Activation(name='Mrelu2_stage2_L2', data=Mconv2_stage2_L2, act_type='relu')
    Mconv3_stage2_L1 = mx.symbol.Convolution(name='Mconv3_stage2_L1', data=Mrelu2_stage2_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage2_L1 = mx.symbol.Activation(name='Mrelu3_stage2_L1', data=Mconv3_stage2_L1, act_type='relu')
    Mconv3_stage2_L2 = mx.symbol.Convolution(name='Mconv3_stage2_L2', data=Mrelu2_stage2_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage2_L2 = mx.symbol.Activation(name='Mrelu3_stage2_L2', data=Mconv3_stage2_L2, act_type='relu')
    Mconv4_stage2_L1 = mx.symbol.Convolution(name='Mconv4_stage2_L1', data=Mrelu3_stage2_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage2_L1 = mx.symbol.Activation(name='Mrelu4_stage2_L1', data=Mconv4_stage2_L1, act_type='relu')
    Mconv4_stage2_L2 = mx.symbol.Convolution(name='Mconv4_stage2_L2', data=Mrelu3_stage2_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage2_L2 = mx.symbol.Activation(name='Mrelu4_stage2_L2', data=Mconv4_stage2_L2, act_type='relu')
    Mconv5_stage2_L1 = mx.symbol.Convolution(name='Mconv5_stage2_L1', data=Mrelu4_stage2_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage2_L1 = mx.symbol.Activation(name='Mrelu5_stage2_L1', data=Mconv5_stage2_L1, act_type='relu')
    Mconv5_stage2_L2 = mx.symbol.Convolution(name='Mconv5_stage2_L2', data=Mrelu4_stage2_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage2_L2 = mx.symbol.Activation(name='Mrelu5_stage2_L2', data=Mconv5_stage2_L2, act_type='relu')
    Mconv6_stage2_L1 = mx.symbol.Convolution(name='Mconv6_stage2_L1', data=Mrelu5_stage2_L1, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage2_L1 = mx.symbol.Activation(name='Mrelu6_stage2_L1', data=Mconv6_stage2_L1, act_type='relu')
    Mconv6_stage2_L2 = mx.symbol.Convolution(name='Mconv6_stage2_L2', data=Mrelu5_stage2_L2, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage2_L2 = mx.symbol.Activation(name='Mrelu6_stage2_L2', data=Mconv6_stage2_L2, act_type='relu')
    Mconv7_stage2_L1 = mx.symbol.Convolution(name='Mconv7_stage2_L1', data=Mrelu6_stage2_L1, num_filter=38, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mconv7_stage2_L2 = mx.symbol.Convolution(name='Mconv7_stage2_L2', data=Mrelu6_stage2_L2, num_filter=19, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    concat_stage3 = mx.symbol.Concat(name='concat_stage3', *[Mconv7_stage2_L1, Mconv7_stage2_L2, relu4_4_CPM])
    Mconv1_stage3_L1 = mx.symbol.Convolution(name='Mconv1_stage3_L1', data=concat_stage3, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage3_L1 = mx.symbol.Activation(name='Mrelu1_stage3_L1', data=Mconv1_stage3_L1, act_type='relu')
    Mconv1_stage3_L2 = mx.symbol.Convolution(name='Mconv1_stage3_L2', data=concat_stage3, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage3_L2 = mx.symbol.Activation(name='Mrelu1_stage3_L2', data=Mconv1_stage3_L2, act_type='relu')
    Mconv2_stage3_L1 = mx.symbol.Convolution(name='Mconv2_stage3_L1', data=Mrelu1_stage3_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage3_L1 = mx.symbol.Activation(name='Mrelu2_stage3_L1', data=Mconv2_stage3_L1, act_type='relu')
    Mconv2_stage3_L2 = mx.symbol.Convolution(name='Mconv2_stage3_L2', data=Mrelu1_stage3_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage3_L2 = mx.symbol.Activation(name='Mrelu2_stage3_L2', data=Mconv2_stage3_L2, act_type='relu')
    Mconv3_stage3_L1 = mx.symbol.Convolution(name='Mconv3_stage3_L1', data=Mrelu2_stage3_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage3_L1 = mx.symbol.Activation(name='Mrelu3_stage3_L1', data=Mconv3_stage3_L1, act_type='relu')
    Mconv3_stage3_L2 = mx.symbol.Convolution(name='Mconv3_stage3_L2', data=Mrelu2_stage3_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage3_L2 = mx.symbol.Activation(name='Mrelu3_stage3_L2', data=Mconv3_stage3_L2, act_type='relu')
    Mconv4_stage3_L1 = mx.symbol.Convolution(name='Mconv4_stage3_L1', data=Mrelu3_stage3_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage3_L1 = mx.symbol.Activation(name='Mrelu4_stage3_L1', data=Mconv4_stage3_L1, act_type='relu')
    Mconv4_stage3_L2 = mx.symbol.Convolution(name='Mconv4_stage3_L2', data=Mrelu3_stage3_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage3_L2 = mx.symbol.Activation(name='Mrelu4_stage3_L2', data=Mconv4_stage3_L2, act_type='relu')
    Mconv5_stage3_L1 = mx.symbol.Convolution(name='Mconv5_stage3_L1', data=Mrelu4_stage3_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage3_L1 = mx.symbol.Activation(name='Mrelu5_stage3_L1', data=Mconv5_stage3_L1, act_type='relu')
    Mconv5_stage3_L2 = mx.symbol.Convolution(name='Mconv5_stage3_L2', data=Mrelu4_stage3_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage3_L2 = mx.symbol.Activation(name='Mrelu5_stage3_L2', data=Mconv5_stage3_L2, act_type='relu')
    Mconv6_stage3_L1 = mx.symbol.Convolution(name='Mconv6_stage3_L1', data=Mrelu5_stage3_L1, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage3_L1 = mx.symbol.Activation(name='Mrelu6_stage3_L1', data=Mconv6_stage3_L1, act_type='relu')
    Mconv6_stage3_L2 = mx.symbol.Convolution(name='Mconv6_stage3_L2', data=Mrelu5_stage3_L2, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage3_L2 = mx.symbol.Activation(name='Mrelu6_stage3_L2', data=Mconv6_stage3_L2, act_type='relu')
    Mconv7_stage3_L1 = mx.symbol.Convolution(name='Mconv7_stage3_L1', data=Mrelu6_stage3_L1, num_filter=38, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mconv7_stage3_L2 = mx.symbol.Convolution(name='Mconv7_stage3_L2', data=Mrelu6_stage3_L2, num_filter=19, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    concat_stage4 = mx.symbol.Concat(name='concat_stage4', *[Mconv7_stage3_L1, Mconv7_stage3_L2, relu4_4_CPM])
    Mconv1_stage4_L1 = mx.symbol.Convolution(name='Mconv1_stage4_L1', data=concat_stage4, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage4_L1 = mx.symbol.Activation(name='Mrelu1_stage4_L1', data=Mconv1_stage4_L1, act_type='relu')
    Mconv1_stage4_L2 = mx.symbol.Convolution(name='Mconv1_stage4_L2', data=concat_stage4, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage4_L2 = mx.symbol.Activation(name='Mrelu1_stage4_L2', data=Mconv1_stage4_L2, act_type='relu')
    Mconv2_stage4_L1 = mx.symbol.Convolution(name='Mconv2_stage4_L1', data=Mrelu1_stage4_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage4_L1 = mx.symbol.Activation(name='Mrelu2_stage4_L1', data=Mconv2_stage4_L1, act_type='relu')
    Mconv2_stage4_L2 = mx.symbol.Convolution(name='Mconv2_stage4_L2', data=Mrelu1_stage4_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage4_L2 = mx.symbol.Activation(name='Mrelu2_stage4_L2', data=Mconv2_stage4_L2, act_type='relu')
    Mconv3_stage4_L1 = mx.symbol.Convolution(name='Mconv3_stage4_L1', data=Mrelu2_stage4_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage4_L1 = mx.symbol.Activation(name='Mrelu3_stage4_L1', data=Mconv3_stage4_L1, act_type='relu')
    Mconv3_stage4_L2 = mx.symbol.Convolution(name='Mconv3_stage4_L2', data=Mrelu2_stage4_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage4_L2 = mx.symbol.Activation(name='Mrelu3_stage4_L2', data=Mconv3_stage4_L2, act_type='relu')
    Mconv4_stage4_L1 = mx.symbol.Convolution(name='Mconv4_stage4_L1', data=Mrelu3_stage4_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage4_L1 = mx.symbol.Activation(name='Mrelu4_stage4_L1', data=Mconv4_stage4_L1, act_type='relu')
    Mconv4_stage4_L2 = mx.symbol.Convolution(name='Mconv4_stage4_L2', data=Mrelu3_stage4_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage4_L2 = mx.symbol.Activation(name='Mrelu4_stage4_L2', data=Mconv4_stage4_L2, act_type='relu')
    Mconv5_stage4_L1 = mx.symbol.Convolution(name='Mconv5_stage4_L1', data=Mrelu4_stage4_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage4_L1 = mx.symbol.Activation(name='Mrelu5_stage4_L1', data=Mconv5_stage4_L1, act_type='relu')
    Mconv5_stage4_L2 = mx.symbol.Convolution(name='Mconv5_stage4_L2', data=Mrelu4_stage4_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage4_L2 = mx.symbol.Activation(name='Mrelu5_stage4_L2', data=Mconv5_stage4_L2, act_type='relu')
    Mconv6_stage4_L1 = mx.symbol.Convolution(name='Mconv6_stage4_L1', data=Mrelu5_stage4_L1, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage4_L1 = mx.symbol.Activation(name='Mrelu6_stage4_L1', data=Mconv6_stage4_L1, act_type='relu')
    Mconv6_stage4_L2 = mx.symbol.Convolution(name='Mconv6_stage4_L2', data=Mrelu5_stage4_L2, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage4_L2 = mx.symbol.Activation(name='Mrelu6_stage4_L2', data=Mconv6_stage4_L2, act_type='relu')
    Mconv7_stage4_L1 = mx.symbol.Convolution(name='Mconv7_stage4_L1', data=Mrelu6_stage4_L1, num_filter=38, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mconv7_stage4_L2 = mx.symbol.Convolution(name='Mconv7_stage4_L2', data=Mrelu6_stage4_L2, num_filter=19, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    concat_stage5 = mx.symbol.Concat(name='concat_stage5', *[Mconv7_stage4_L1, Mconv7_stage4_L2, relu4_4_CPM])
    Mconv1_stage5_L1 = mx.symbol.Convolution(name='Mconv1_stage5_L1', data=concat_stage5, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage5_L1 = mx.symbol.Activation(name='Mrelu1_stage5_L1', data=Mconv1_stage5_L1, act_type='relu')
    Mconv1_stage5_L2 = mx.symbol.Convolution(name='Mconv1_stage5_L2', data=concat_stage5, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage5_L2 = mx.symbol.Activation(name='Mrelu1_stage5_L2', data=Mconv1_stage5_L2, act_type='relu')
    Mconv2_stage5_L1 = mx.symbol.Convolution(name='Mconv2_stage5_L1', data=Mrelu1_stage5_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage5_L1 = mx.symbol.Activation(name='Mrelu2_stage5_L1', data=Mconv2_stage5_L1, act_type='relu')
    Mconv2_stage5_L2 = mx.symbol.Convolution(name='Mconv2_stage5_L2', data=Mrelu1_stage5_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage5_L2 = mx.symbol.Activation(name='Mrelu2_stage5_L2', data=Mconv2_stage5_L2, act_type='relu')
    Mconv3_stage5_L1 = mx.symbol.Convolution(name='Mconv3_stage5_L1', data=Mrelu2_stage5_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage5_L1 = mx.symbol.Activation(name='Mrelu3_stage5_L1', data=Mconv3_stage5_L1, act_type='relu')
    Mconv3_stage5_L2 = mx.symbol.Convolution(name='Mconv3_stage5_L2', data=Mrelu2_stage5_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage5_L2 = mx.symbol.Activation(name='Mrelu3_stage5_L2', data=Mconv3_stage5_L2, act_type='relu')
    Mconv4_stage5_L1 = mx.symbol.Convolution(name='Mconv4_stage5_L1', data=Mrelu3_stage5_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage5_L1 = mx.symbol.Activation(name='Mrelu4_stage5_L1', data=Mconv4_stage5_L1, act_type='relu')
    Mconv4_stage5_L2 = mx.symbol.Convolution(name='Mconv4_stage5_L2', data=Mrelu3_stage5_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage5_L2 = mx.symbol.Activation(name='Mrelu4_stage5_L2', data=Mconv4_stage5_L2, act_type='relu')
    Mconv5_stage5_L1 = mx.symbol.Convolution(name='Mconv5_stage5_L1', data=Mrelu4_stage5_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage5_L1 = mx.symbol.Activation(name='Mrelu5_stage5_L1', data=Mconv5_stage5_L1, act_type='relu')
    Mconv5_stage5_L2 = mx.symbol.Convolution(name='Mconv5_stage5_L2', data=Mrelu4_stage5_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage5_L2 = mx.symbol.Activation(name='Mrelu5_stage5_L2', data=Mconv5_stage5_L2, act_type='relu')
    Mconv6_stage5_L1 = mx.symbol.Convolution(name='Mconv6_stage5_L1', data=Mrelu5_stage5_L1, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage5_L1 = mx.symbol.Activation(name='Mrelu6_stage5_L1', data=Mconv6_stage5_L1, act_type='relu')
    Mconv6_stage5_L2 = mx.symbol.Convolution(name='Mconv6_stage5_L2', data=Mrelu5_stage5_L2, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage5_L2 = mx.symbol.Activation(name='Mrelu6_stage5_L2', data=Mconv6_stage5_L2, act_type='relu')
    Mconv7_stage5_L1 = mx.symbol.Convolution(name='Mconv7_stage5_L1', data=Mrelu6_stage5_L1, num_filter=38, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mconv7_stage5_L2 = mx.symbol.Convolution(name='Mconv7_stage5_L2', data=Mrelu6_stage5_L2, num_filter=19, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    concat_stage6 = mx.symbol.Concat(name='concat_stage6', *[Mconv7_stage5_L1, Mconv7_stage5_L2, relu4_4_CPM])
    Mconv1_stage6_L1 = mx.symbol.Convolution(name='Mconv1_stage6_L1', data=concat_stage6, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage6_L1 = mx.symbol.Activation(name='Mrelu1_stage6_L1', data=Mconv1_stage6_L1, act_type='relu')
    Mconv1_stage6_L2 = mx.symbol.Convolution(name='Mconv1_stage6_L2', data=concat_stage6, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu1_stage6_L2 = mx.symbol.Activation(name='Mrelu1_stage6_L2', data=Mconv1_stage6_L2, act_type='relu')
    Mconv2_stage6_L1 = mx.symbol.Convolution(name='Mconv2_stage6_L1', data=Mrelu1_stage6_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage6_L1 = mx.symbol.Activation(name='Mrelu2_stage6_L1', data=Mconv2_stage6_L1, act_type='relu')
    Mconv2_stage6_L2 = mx.symbol.Convolution(name='Mconv2_stage6_L2', data=Mrelu1_stage6_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu2_stage6_L2 = mx.symbol.Activation(name='Mrelu2_stage6_L2', data=Mconv2_stage6_L2, act_type='relu')
    Mconv3_stage6_L1 = mx.symbol.Convolution(name='Mconv3_stage6_L1', data=Mrelu2_stage6_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage6_L1 = mx.symbol.Activation(name='Mrelu3_stage6_L1', data=Mconv3_stage6_L1, act_type='relu')
    Mconv3_stage6_L2 = mx.symbol.Convolution(name='Mconv3_stage6_L2', data=Mrelu2_stage6_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu3_stage6_L2 = mx.symbol.Activation(name='Mrelu3_stage6_L2', data=Mconv3_stage6_L2, act_type='relu')
    Mconv4_stage6_L1 = mx.symbol.Convolution(name='Mconv4_stage6_L1', data=Mrelu3_stage6_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage6_L1 = mx.symbol.Activation(name='Mrelu4_stage6_L1', data=Mconv4_stage6_L1, act_type='relu')
    Mconv4_stage6_L2 = mx.symbol.Convolution(name='Mconv4_stage6_L2', data=Mrelu3_stage6_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu4_stage6_L2 = mx.symbol.Activation(name='Mrelu4_stage6_L2', data=Mconv4_stage6_L2, act_type='relu')
    Mconv5_stage6_L1 = mx.symbol.Convolution(name='Mconv5_stage6_L1', data=Mrelu4_stage6_L1, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage6_L1 = mx.symbol.Activation(name='Mrelu5_stage6_L1', data=Mconv5_stage6_L1, act_type='relu')
    Mconv5_stage6_L2 = mx.symbol.Convolution(name='Mconv5_stage6_L2', data=Mrelu4_stage6_L2, num_filter=128, pad=(3, 3),
                                             kernel=(7, 7), stride=(1, 1), no_bias=False)
    Mrelu5_stage6_L2 = mx.symbol.Activation(name='Mrelu5_stage6_L2', data=Mconv5_stage6_L2, act_type='relu')
    Mconv6_stage6_L1 = mx.symbol.Convolution(name='Mconv6_stage6_L1', data=Mrelu5_stage6_L1, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage6_L1 = mx.symbol.Activation(name='Mrelu6_stage6_L1', data=Mconv6_stage6_L1, act_type='relu')
    Mconv6_stage6_L2 = mx.symbol.Convolution(name='Mconv6_stage6_L2', data=Mrelu5_stage6_L2, num_filter=128, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mrelu6_stage6_L2 = mx.symbol.Activation(name='Mrelu6_stage6_L2', data=Mconv6_stage6_L2, act_type='relu')
    Mconv7_stage6_L1 = mx.symbol.Convolution(name='Mconv7_stage6_L1', data=Mrelu6_stage6_L1, num_filter=38, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)
    Mconv7_stage6_L2 = mx.symbol.Convolution(name='Mconv7_stage6_L2', data=Mrelu6_stage6_L2, num_filter=19, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=False)

    return mx.symbol.Group([Mconv7_stage6_L1, Mconv7_stage6_L2])
