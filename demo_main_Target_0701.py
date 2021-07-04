 #!/usr/bin/env python
# -*- coding: utf-8 -*-



from __future__ import absolute_import
from six.moves import range

import os
import time
import numpy as np
from scipy import stats
from tqdm import tqdm

#   NNabla
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.ext_utils import get_extension_context  # GPU

#   Image
import cv2

#   Originals
from settings_20210601 import settings
import data_20210601 as dt

# plot
import matplotlib.pyplot as plt


# -------------------------------------------
#   Network for IQA
# ------------------------------------------
def network(input, mask, similarity, scope="", test=False, ch=2):

    def convblock(x, n, f_size=3, s=2,name=''):  #convolution　x:input　n:amount of channel　f_size: size of filter(kernel) s: width of stride
        r = PF.convolution(x, n, kernel=(f_size, f_size), pad=(f_size // 2, f_size // 2), stride=(s, s),
                           name=name)  # paddingは基本フィルタサイズの半分（切り捨て）　strideフィルタのずらし幅　name入れ物の名前
        return F.relu(r)  # 活性化関数

    def onebyone_conv(x, c, name = ''):
        r = PF.convolution(x, c, kernel = (1, 1), pad = (0, 0), name=name)
        return r

    def attention_mask(x, y, name=''):
        with nn.parameter_scope(name):
            query = onebyone_conv(x, x.shape[1] // 2, name='cnv1')
            key = onebyone_conv(y, x.shape[1] // 2, name='cnv2')
            val = onebyone_conv(y, x.shape[1] // 2, name='cnv3')

            z = F.batch_matmul(query, key, True, False)
            z = F.softmax(z)

            z = F.batch_matmul(z, val, False, True)
            z = onebyone_conv(z,x.shape[1], name='_cnv4')

            return F.sigmoid(z)


    with nn.parameter_scope(scope+'/cnn'):  # 64x64conv  with - nn.parameter-scopeのオブジェクトを用意

        K = 2
        L = 3
        ch_init = 16

        x = input
        y = mask

        for j in range(L):
            for i in range(K):
                x = PF.batch_normalization(x, name='bn_x'+ str(j + 1)+str(i + 1), batch_stat=not test)
                x = convblock(x, ch_init*(i+j*K+1), s=2, name='conv_x'+ str(j + 1) + str(i + 1))

            for i in range(K):
                y = PF.batch_normalization(y, name='bn_y'+ str(j + 1)+ str(i + 1), batch_stat=not test)
                y = convblock(y, ch_init*(i+1)*(j+1), s=2, name='conv_y'+ str(j + 1) + str(i + 1))

            z = attention_mask(x, y, name='att'+ str(j + 1) + str(i + 1))
            x = x + z

    return x #(20, 96, 4, 4)




def domain_transform_network(input, mask,  scope="", test=False, ch=2):
     def convblock(x, n, f_size=3, s=2,
                   name=''):  # convolution　x:input　n:amount of channel　f_size: size of filter(kernel) s: width of stride
         r = PF.convolution(x, n, kernel=(f_size, f_size), pad=(f_size // 2, f_size // 2), stride=(s, s),
                            name=name)  # paddingは基本フィルタサイズの半分（切り捨て）　strideフィルタのずらし幅　name入れ物の名前
         return F.relu(r)  # 活性化関数

     def onebyone_conv(x, c, name=''):
         r = PF.convolution(x, c, kernel=(1, 1), pad=(0, 0), name=name)
         return r

     def attention_mask(x, y, name=''):
         with nn.parameter_scope(name):
             query = onebyone_conv(x, x.shape[1] // 2, name='cnv1')
             key = onebyone_conv(y, x.shape[1] // 2, name='cnv2')
             val = onebyone_conv(y, x.shape[1] // 2, name='cnv3')

             z = F.batch_matmul(query, key, True, False)
             z = F.softmax(z)

             z = F.batch_matmul(z, val, False, True)
             z = onebyone_conv(z, x.shape[1], name='_cnv4')

             return F.sigmoid(z)

     with nn.parameter_scope(scope):  # 64x64conv  with - nn.parameter-scopeのオブジェクトを用意

         K = 2
         L = 3
         ch_init = 16

         x = input
         y = mask

         for j in range(L):
             for i in range(K):
                 x = PF.batch_normalization(x, name='bn_x' + str(j + 1) + str(i + 1), batch_stat=not test)
                 x = convblock(x, ch_init * (i + j * K + 1), s=2, name='conv_x' + str(j + 1) + str(i + 1))

             for i in range(K):
                 y = PF.batch_normalization(y, name='bn_y' + str(j + 1) + str(i + 1), batch_stat=not test)
                 y = convblock(y, ch_init * (i + 1) * (j + 1), s=2, name='conv_y' + str(j + 1) + str(i + 1))

             z = attention_mask(x, y, name='att' + str(j + 1) + str(i + 1))
             x = x + z

     return x #(20, 96, 4, 4)

def classifier(x, similarity, scope):

    # Fully connection

    with nn.parameter_scope(scope+'/fcn'):

        # similarity condition
        similarity = F.relu(PF.affine(similarity, x.size // x.shape[0] // 4, name='af1'))
        similarity = F.sigmoid(PF.affine(similarity, x.shape[1:], name='af2'))

        x = x + similarity

        # #x = F.mean(F.concatenate(x,y, axis=1), axis=(2, 3))
        x = F.mean(x, axis=(2, 3))

        # Affine Layer
        x = F.leaky_relu(PF.affine(x, (x.shape[1] // 2,), name='Affine1'), 0.01)

        # Affine Layer
        x = PF.affine(x, (1,), name='Affine2')

        return x



def discriminator(input, scope=''):
    def convblock(x, n, f_size=3, s=2,
                  name=''):  # convolution　x:input　n:amount of channel　f_size: size of filter(kernel) s: width of stride
        r = PF.convolution(x, n, kernel=(f_size, f_size), pad=(f_size // 2, f_size // 2), stride=(s, s),
                           name=name)  # paddingは基本フィルタサイズの半分（切り捨て）　strideフィルタのずらし幅　name入れ物の名前
        return F.relu(r)  # 活性化関数

    def onebyone_conv(x, c, name=''):
        r = PF.convolution(x, c, kernel=(1, 1), pad=(0, 0), name=name)
        return r

    def attention_mask(x, y, name=''):
        with nn.parameter_scope(name):
            query = onebyone_conv(x, x.shape[1] // 2, name='cnv1')
            key = onebyone_conv(y, x.shape[1] // 2, name='cnv2')
            val = onebyone_conv(y, x.shape[1] // 2, name='cnv3')

            z = F.batch_matmul(query, key, True, False)
            z = F.softmax(z)

            z = F.batch_matmul(z, val, False, True)
            z = onebyone_conv(z, x.shape[1], name='_cnv4')

            return F.sigmoid(z)

    with nn.parameter_scope(scope+'/fcn'):

        K = 2
        L = 3
        ch_init = 16

        x = input

        for j in range(L):
            for i in range(K):
                x = PF.batch_normalization(x, name='bn_x' + str(j + 1) + str(i + 1), batch_stat=not test)
                x = convblock(x, ch_init * (i + j * K + 1), s=2, name='conv_x' + str(j + 1) + str(i + 1))

        # Affine Layer
        x = F.leaky_relu(PF.affine(x, (x.shape[1]//2,), name='Affine1'), 0.01)

        # Affine Layer
        x = PF.affine(x, (1,), name='Affine2')

    return x #(20, 1)

def conditional_network(sim, var, scope=""):  # Fullconnectnetwork

    with nn.parameter_scope(scope):

        # Affine Layer : M -> 1
        x = F.concatenate(sim, var, axis=1)
        out = F.sigmoid(PF.affine(x, (64,), name='Affine_mid'))

        return out

def network2(input, scope=""):  # Fullconnectnetwork

    with nn.parameter_scope(scope):

        # Affine Layer : M -> 1
        c4 = F.sigmoid(PF.affine(input, (64,), name='Affine_mid2'))
        c5 = PF.affine(c4, (1,), name='Affine_out')

        return c5
# -------------------------------------------
#   Training
# -------------------------------------------
def train(args):
    """
    Training
    """

    ##  ~~~~~~~~~~~~~~~~~~~
    ##   Initial settings
    ##  ~~~~~~~~~~~~~~~~~~~

    #   Input Variable　      args. -> setting.
    Size = 256

    nn.clear_parameters()  # Clear　
    Input = nn.Variable([args.batch_size, 3, Size, Size])
    Input2 = nn.Variable([args.batch_size, 3, Size, Size])
    Mask = nn.Variable([args.batch_size, 3, Size, Size])
    Mask2 = nn.Variable([args.batch_size, 3, Size, Size])
    Trues = nn.Variable([args.batch_size, 1])  # True Value
    Similarity = nn.Variable([args.batch_size, 3])
    Variance = nn.Variable([args.batch_size, 3])


    #   Network Definition
    Name = "CNN"  # Name of scope which includes network models (arbitrary)
    Name2 = "DomainAdaptation"
    Name_dis = "Dis"

    # 係数読み込み
    # Source
    Feature_output = network(Input, Mask, Similarity, test=True, scope=Name)  #(20, 96, 4, 4)

    # with nn.parameter_scope(Name + '/fcn'):
    #     nn.load_parameters(os.path.join(args.model_save_path2, "network_fcn_80_param_{:04}.h5".format(args.retrain)))

    # Target (Domain Adaptation)
    Feature_output2 = domain_transform_network(Input2, Mask2, scope=Name2) #(20, 96, 4, 4)

    with nn.parameter_scope(Name + '/cnn'):
        nn.load_parameters(os.path.join(args.model_save_path, "network_cnn_80_param_{:04}.h5".format(args.epoch))) #SourceパラメータをSourceCNNに読み込み


    with nn.parameter_scope(Name2):
        nn.load_parameters(os.path.join(args.model_save_path, "network_cnn_80_param_{:04}.h5".format(args.epoch))) #SourceパラメータをTargetCNNに読み込み

    Feature_output.apply(need_grad=False)

    # Discriminator
    Dis1 = discriminator(Feature_output, scope=Name_dis) #(20, 1)
    Dis2 = discriminator(Feature_output2, scope=Name_dis) #(20, 1)

    # Loss Function
    loss_dis = F.mean(F.relu( 1-Dis1  )) + F.mean(F.relu( 1+Dis2 ))
    loss_gen = - F.mean(Dis2)


    #   Solver Setting
    solver = S.Adam(args.learning_rate_gen)  # Adam is used for solver　学習率の最適化
    solver2 = S.Adam(args.learning_rate_dis)  # Adam is used for solver　学習率の最適化
    solver.weight_decay(0.00001)  # Weight Decay for stable update
    solver2.weight_decay(0.00001)

    with nn.parameter_scope(Name2):  #TragetCNNパラメータを更新対象にセット
        solver.set_parameters(nn.get_parameters())

    with nn.parameter_scope(Name_dis):  #Discriminatorパラメータを更新対象にセット
        solver2.set_parameters(nn.get_parameters())

    #   Training Data Setting
    image_data, mask_data, mos_data, similarity, variance, target_image_data ,target_mask_data  = dt.data_loader(mask_out=True)  # Source
    batches = dt.create_batch_w_mask(image_data, mask_data, mos_data, similarity, variance, target_image_data ,target_mask_data, args.batch_size)
    del image_data, mask_data, mos_data, similarity, variance, target_image_data ,target_mask_data

    ##  ~~~~~~~~~~~~~~~~~~~
    ##   Learning
    ##  ~~~~~~~~~~~~~~~~~~~
    print('== Start Training ==')

    bar = tqdm(total=(args.epoch_domain_adaptation - args.epoch)*batches.iter_n, leave=False)
    bar.clear()
    cnt = 0
    loss_disp = True

    #   Load data
    if args.epoch > 0:  # 途中のエポック(retrain)から再学習
        with nn.parameter_scope(Name):
            print('Retrain from {0} Epoch'.format(args.epoch))
            with nn.parameter_scope('cnn'):
                nn.load_parameters(os.path.join(args.model_save_path, "network_cnn_80_param_{:04}.h5".format(args.epoch)))  #SourceパラメータをSourceCNNに読み込み
            solver.set_learning_rate(args.learning_rate_gen / np.sqrt(args.epoch))

        with nn.parameter_scope(Name2):
            nn.load_parameters(os.path.join(args.model_save_path, "network_cnn_80_param_{:04}.h5".format(args.epoch))) #SourceパラメータをTargetCNNに読み込み
            solver.set_learning_rate(args.learning_rate_gen / np.sqrt(args.epoch))


    ##  Training
    for i in range(args.epoch, args.epoch_domain_adaptation):  # args.retrain → args.epoch まで繰り返し学習

        bar.set_description_str('Epoch {0}/{1}:'.format(i + 1, args.epoch_domain_adaptation), refresh=False)  # プログレスバーに説明文を加える

        #   Shuffling
        batches.shuffle()

        ##  Batch iteration
        for j in range(batches.iter_n):  # バッチ学習

            cnt += 1

            #  Load Batch Data from Training data
            Input_npy, Mask_npy, Trues_npy, Similarity_npy, Variance_npy,Index_npy,  Input2_npy, Mask2_npy = next(batches)

            size_ = Input_npy.shape
            #Input.d = Input_npy.reshape([size_[0]*size_[1], size_[2], size_[3], size_[4]])
            Input.d = Input_npy
            Mask.d = Mask_npy
            Input2.d = Input2_npy
            Mask2.d = Mask2_npy


            # Generator
            solver.zero_grad()  # Initialize　 #   Initialize #勾配をリセット
            loss_gen.forward(clear_no_need_grad=True)
            loss_scale = 8
            loss_gen.backward(loss_scale, clear_buffer=True)
            solver.scale_grad(1. / loss_scale)
            solver.update()

            # Discriminator
            solver2.zero_grad()
            loss_dis.forward(clear_no_need_grad=True)  # Forward path　#順伝播
            loss_scale = 8
            loss_dis.backward(loss_scale, clear_buffer=True)  # Backward path　#誤差逆伝播法
            solver2.scale_grad(1. / loss_scale)
            solver2.update()

            # Progress
            if cnt % 10 == 0:
                bar.update(10)  # プログレスバーの進捗率を1あげる
                if loss_disp is not None:
                    bar.set_postfix_str('Loss. Dis={0:.3e}, Gen.={1:.3e}'.format(loss_dis.d, loss_gen.d), refresh=False)  # 実行中にloss_dispとSRCCを表示

        ## Save parameters
        if ((i + 1) % args.model_save_cycle) == 0 or (i + 1) == args.epoch_domain_adaptation:
            bar.clear()
            with nn.parameter_scope(Name2):
                nn.save_parameters(os.path.join(args.model_save_path, 'network_domain_adaptation_80_{:04}.h5'.format(i + 1))) #TragetCNNパラメータを保存
            with nn.parameter_scope(Name_dis):
                nn.save_parameters(os.path.join(args.model_save_path2, 'network_discriminator_80_{:04}.h5'.format(i + 1))) #Discriminatorパラメータを保存




# -------------------------------------------
#   Test
# -------------------------------------------
def test(args, mode='test'):
    """
    Training
    """
    M = 64
    ##  ~~~~~~~~~~~~~~~~~~~
    ##   Initial settings
    ##  ~~~~~~~~~~~~~~~~~~~

    #   Input Variable　変数定義
    nn.clear_parameters()  # Clear
    Input = nn.Variable([1, 3, 256, 256])  # Input
    #Input = nn.Variable([1, 6, 64, 64])  # Input
    Mask = nn.Variable([1, 3, 256, 256])  # Input
    Trues = nn.Variable([1, 1])  # True Value
    Similarity = nn.Variable([1, 3])
    Variance = nn.Variable([1, 3])
    Index = nn.Variable([1, 1])  # True Value


    #   Network Definition
    Name = "CNN"  # Name of scope which includes network models (arbitrary)
    Name2 = "DomainAdaptation"
    #Name_dis = "Dis"

    if mode == "test": #test
        Output = network(Input, Mask, Similarity, scope=Name, test=False)  # fullconnect
        Output = classifier(Output, Similarity, scope=Name)
        # conditional_info = conditional_network(sim=Similarity, var=Variance, scope=Name2)
        # postOutput = F.concatenate(preOutput, conditional_info, axis=1)
        # Output = network2(input=postOutput, scope=Name3)  # fullconnect

    else : #eval
        Output = domain_transform_network(Input, Mask, scope = Name2)
        Output = classifier(Output, Similarity, scope = Name)

    Loss_test = F.mean(F.absolute_error(Output, Trues))  # Loss Function (Squared Error) #誤差関数


    #   Load data　保存した学習パラメータの読み込み

    with nn.parameter_scope(Name + '/cnn'):
        nn.load_parameters(os.path.join(args.model_save_path, "network_cnn_80_param_{:04}.h5".format(args.epoch))) #SourceCNNパラメータをSourceCNNに読み込み

    with nn.parameter_scope(Name + '/fcn'):
        nn.load_parameters(os.path.join(args.model_save_path2, "network_fcn_80_param_{:04}.h5".format(args.epoch))) #SourceFCNパラメータをSourceFCNに読み込み

    #if mode != "test":
    with nn.parameter_scope(Name2):
        nn.load_parameters(os.path.join(args.model_save_path, "network_domain_adaptation_80_{:04}.h5".format(args.epoch_domain_adaptation))) #TargetCNNパラメータをTargetFCNに読み込み

    # Test Data Setting
    #image_data, mos_data, image_files = dt.data_loader(test=True)
    # image_data, mos_data= dt.data_loader(test=True)
    image_data, mask_data, mos_data, similarity, variance, target_image_data ,target_mask_data   = dt.data_loader(mode=mode,mask_out=True)
    batches = dt.create_batch_w_mask(image_data, mask_data, mos_data, similarity, variance, target_image_data ,target_mask_data,  1, test=True)
    del image_data, mask_data, mos_data, similarity, variance,  target_image_data ,target_mask_data


    truth = []
    result = []
    index = []

    for j in range(batches.iter_n):
        Input_npy, Mask_npy, Trues_npy, Similarity_npy, Variance_npy, Index_npy = next(batches)
        #Input.d = Input_npy[:,:,  0:64, 0:64]
        Input.d = Input_npy
        Mask.d = Mask_npy
        Trues.d = Trues_npy
        Similarity.d = Similarity_npy
        Variance.d = Variance_npy
        Index.d = Index_npy


        Loss_test.forward(clear_no_need_grad=True)
        Output.forward(clear_buffer=True)
        result.append(Output.d.copy())
        truth.append(Trues.d.copy())
        index.append(Index.d.copy())

    result = np.squeeze(np.array(result))
    mean = np.mean(result)
    truth = np.squeeze(np.array(truth))  # delete
    index = np.squeeze(np.array(index))  # delete

    #sort

    sorted_index = np.argsort(index)
    index = index[sorted_index]
    truth = truth[sorted_index]
    result = result[sorted_index]

    #calculate error
    np.set_printoptions(precision=3, floatmode='maxprec') #有効桁数を調整
    error =  result - truth

    # Evaluation of performance
    np.set_printoptions(precision=4, floatmode='maxprec')
    mae = np.average(np.abs(result - truth))
    SRCC, p1 = stats.spearmanr(truth, result)  # Spearman's Correlation Coefficient
    PLCC, p2 = stats.pearsonr(truth, result)

    np.set_printoptions(threshold=np.inf)
    #print("result: {}".format(result))
    #print("Trues: {}".format(truth))
    #print(np.average(result))

    print("img     truth       result       error")
    for i in range(args.eval_data_num):
        print("{}       {}   -->   {:.3g}       {:.3g}".format(i+1 , truth[i],result[i], error[i]))

    print("\n Model Parameter [epoch={0}]".format(args.epoch_domain_adaptation))
    print(" Mean Absolute Error with Truth: {0:.4f}".format(mae))
    print(" Speerman's Correlation Coefficient: {0:.5f}".format(SRCC))
    print(" Pearson's Linear Correlation Coefficient: {0:.5f}".format(PLCC))
    np.savetxt('result.csv', [np.array(truth).T, np.array(result).T], delimiter=',')
    #os.remove("./pkl/test_eval.pkl")  # add
    #os.remove("./pkl/test_image.pkl")  # add

    x = truth
    y = result
    plt.scatter(x, y)
    #plt.ylim([0, 5.0])
    plt.show()

    x = index + 1
    y = error
    plt.bar(x, y)
    plt.show()


if __name__ == '__main__':


    Mode  = "test"
    ctx = get_extension_context('cudnn', device_id=0, type_config="half")
    nn.set_default_context(ctx)
    #   Train
    if Mode=='test':
        test(settings())
    elif Mode=='eval':
        test(settings(),mode='eval')
    else:
        train(settings())

