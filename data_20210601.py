 #!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import math
import joblib
import glob
import re

import matplotlib.pyplot as plt


import cv2
import numpy as np
import numpy.random as rd

from settings_20210601 import settings

import os

#コントラスト類似度計算
import contrastSimilarity as cont

#ヒストグラム平坦化
import histgramEqualize as hist

import difference as diff

Number_of_images =  30

# -------------------------------------------
#   Load pkl files or ".jpg" & ".csv" files
# -------------------------------------------
def data_loader(mode='train', mask_out=False):
    """
    Read wav files or Load pkl files
	"""

    ##  Sort function for file name
    def numericalSort(value):
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    # Load settings
    args = settings()

    # Make folder
    if not os.path.exists(args.model_save_path):    # Folder of model
        os.makedirs(args.model_save_path)

    if not os.path.exists(args.pkl_path):           # Folder of train pkl
        os.makedirs(args.pkl_path)

    ex = "bmp"

    # File name
    if mode=='test':
        image_names = args.test_data_path + '/*.' + "png"
        eval_names = args.test_data_path + '/*.txt'
        pkl_image = args.pkl_path + '/test_image.pkl'
        pkl_eval = args.pkl_path + '/test_eval.pkl'
        pkl_mask = args.pkl_path + '/test_mask.pkl'
        pkl_target_image = args.pkl_path + '/target_image.pkl'
        pkl_target_mask = args.pkl_path + '/target_mask.pkl'
        mask_names = args.test_mask_path + '/*.' + "png"
        Number_of_images = args.test_data_num
        target_image_names = args.target_data_path + '/*.' + "png"
        target_mask_names = args.target_mask_path + '/*.' + "png"
    elif mode == 'eval':
        image_names = args.eval_data_path + '/*.' + "png"
        eval_names = args.eval_data_path + '/*.txt'
        pkl_image = args.pkl_path + '/eval_image.pkl'
        pkl_eval = args.pkl_path + '/eval_eval.pkl'
        pkl_mask = args.pkl_path + '/eval_mask.pkl'
        pkl_target_image = args.pkl_path + '/target_image.pkl'
        pkl_target_mask = args.pkl_path + '/target_mask.pkl'
        mask_names = args.eval_mask_path + '/*.' + "png"
        Number_of_images = args.eval_data_num
        target_image_names = args.target_data_path + '/*.' + "bmp"
        target_mask_names = args.target_mask_path + '/*.' + "bmp"
    else:
        image_names = args.train_data_path + '/*.' + "png" #すべてのtrainBMPファイルを読み込み
        eval_names  = args.train_data_path + '/*.txt' #すべてのtraintxtファイルを読み込み
        pkl_image   = args.pkl_path + '/train_image.pkl'
        pkl_eval    = args.pkl_path + '/train_eval.pkl'
        pkl_mask    = args.pkl_path + '/train_mask.pkl'
        pkl_target_image = args.pkl_path + '/target_image.pkl'
        pkl_target_mask = args.pkl_path + '/target_mask.pkl'
        mask_names  = args.train_mask_path  + '/*.' + "png"
        Number_of_images = args.train_data_num
        target_image_names = args.target_data_path + '/*.' + "bmp"
        target_mask_names = args.target_mask_path + '/*.' + "bmp"

    image_data = []
    mask_data = []
    target_image_data = []
    target_mask_data = []
    similarity  = []

    for target_image_file in sorted(glob.glob(target_image_names), key=numericalSort):
        mask = cv2.imread(target_image_file)
        # mask_data.append(mask.transpose(2, 0, 1))
        target_image_data.append(mask)

    for target_mask_file in sorted(glob.glob(target_mask_names), key=numericalSort):
        mask = cv2.imread(target_mask_file)
        # mask_data.append(mask.transpose(2, 0, 1))
        target_mask_data.append(mask)

    with open(pkl_target_image, 'wb') as f:  # Create clean pkl file
        joblib.dump(target_image_data, f, protocol=-1, compress=3)

    with open(pkl_target_mask, 'wb') as f:  # Create clean pkl file
        joblib.dump(target_mask_data, f, protocol=-1, compress=3)




    ##  ~~~~~~~~~~~~~~~~~~~
    ##   No pkl files
    ##    -> Read images & assesment values, and Create pkl files
    ##  ~~~~~~~~~~~~~~~~~~~
    if not (os.access(pkl_image, os.F_OK) and os.access(pkl_eval, os.F_OK) and os.access(pkl_mask, os.F_OK)):

        ##  Read Image files
        print(' Load bmp file...')

        # Get image data

        for image_file in sorted(glob.glob(image_names), key=numericalSort):
            img = cv2.imread(image_file)
            #image_data.append(img.transpose(2,0,1))
            image_data.append(img)

        #image_data = np.array(image_data)

        # Get evaluation data
        eval_data = []
        for imgage_file in sorted(glob.glob(eval_names), key=numericalSort):
            eval_data = np.expand_dims(np.loadtxt(glob.glob(eval_names)[0], delimiter=',', dtype='float'), axis=1)

        for mask_file in sorted(glob.glob(mask_names), key=numericalSort):
            mask = cv2.imread(mask_file)
                #mask_data.append(mask.transpose(2, 0, 1))
            mask_data.append(mask)

        #mask_data = np.array(mask_data)

        for target_image_file in sorted(glob.glob(target_image_names), key=numericalSort):
            mask = cv2.imread(target_image_file)
                #mask_data.append(mask.transpose(2, 0, 1))
            target_image_data.append(mask)

        for target_mask_file in sorted(glob.glob(target_mask_names), key=numericalSort):
            mask = cv2.imread(target_mask_file)
                #mask_data.append(mask.transpose(2, 0, 1))
            target_mask_data.append(mask)





        ##  Create Pkl files
        print(' Create Pkl file...')
        with open(pkl_image, 'wb') as f:        # Create clean pkl file
            joblib.dump(image_data, f, protocol=-1, compress=3)

        with open(pkl_eval, 'wb') as f:         # Create noisy pkl file
            joblib.dump(eval_data, f, protocol=-1, compress=3)

        with open(pkl_mask, 'wb') as f:  # Create clean pkl file
            joblib.dump(mask_data, f, protocol=-1, compress=3)


        with open(pkl_target_image, 'wb') as f:  # Create clean pkl file
            joblib.dump(target_image_data, f, protocol=-1, compress=3)

        with open(pkl_target_mask, 'wb') as f:  # Create clean pkl file
            joblib.dump(target_mask_data, f, protocol=-1, compress=3)


    else: #pklファイルの読み込み
        #if test  == False:  #train_pkl
        with open(pkl_image, 'rb') as f:        # Load image pkl file
            print(' Load Image Pkl...')
            image_data = joblib.load(f)

        with open(pkl_eval, 'rb') as f:         # Load evaluation pkl file
            print(' Load Evaluation Pkl...')
            eval_data = joblib.load(f)

        with open(pkl_mask, 'rb') as f:  # Load image pkl file
            print(' Load Mask Pkl...')
            mask_data = joblib.load(f)

        with open(pkl_target_image, 'rb') as f:  # Load image pkl file
            print(' Load Target Image Pkl...')
            target_image_data = joblib.load(f)

        with open(pkl_target_mask, 'rb') as f:  # Load image pkl file
            print(' Load Target Mask Pkl...')
            target_mask_data = joblib.load(f)



    #マスクとのコントラスト類似度を取得　(画像枚数,3) & ヒストグラム平坦化
    histimg = []
    histmask = []
    for i in range(Number_of_images):
        img1 = image_data[i]
        img2 = mask_data[i]
        similarity.append(cont.contrastSimilarity(img1, img2))
        histimg.append(hist.histgramEqualize(img1))
        histmask.append(hist.histgramEqualize(img2))
    similarity = np.array(similarity)

    histimg = np.array(histimg)
    histmask = np.array(histmask)

    #差分画像のRGBごとの分散を取得
    variance = []
    for i in range(Number_of_images):
        img1 = histimg[i]
        img2 = histmask[i]
        variance.append(diff.variance(img1, img2))

    variance = np.array(variance)
    # a = similarity[:, 0]
    # print(min(a))
    # n, bins, patches = plt.hist(a)
    # plt.xlabel("Values")
    # plt.ylabel("Frequency")
    # plt.title("Histogram")
    # plt.show()


    #varianceの型がおかしい

    #劣化画像とマスク画像を結合(Number_of_images, 6, 128, 128)
    #image_data = np.concatenate((histimg.transpose(0,3,1,2), histmask.transpose(0,3,1,2)), axis=1)
    image_data = histimg.transpose(0,3,1,2)
    mask_data  = histmask.transpose(0,3,1,2)




    target_image_data = np.array(target_image_data).transpose(0, 3, 1, 2)
    target_mask_data = np.array(target_mask_data).transpose(0, 3, 1, 2)


    if not mask_out:
        return image_data, eval_data, similarity, variance, target_image_data, target_mask_data
    if mask_out:
        return image_data, mask_data, eval_data, similarity, variance, target_image_data, target_mask_data



class create_batch:
    """
    Creating Batch Data for training
    """

    ## 	Initialization
    # def __init__(self, image, mos, sim, var, batches, test=False):
    def __init__(self, image, mos, sim, var,  batches,  test=False):

        # Data Shaping #.copyで値渡しを行う => 代入だと参照渡しになってしまう
        self.image  = image.copy()
        self.mos    = mos.copy()
        self.sim = sim.copy()
        self.var = var.copy()

        # Random index ( for data scrambling)

        ind = np.array(range(self.image.shape[0]))
        if not test:
            rd.shuffle(ind)

        # Parameters
        self.i = 0
        self.batch = batches
        self.iter_n = math.ceil(self.image.shape[0] / batches)     # Batch num for each 1 Epoch
        remain = self.iter_n * batches - self.image.shape[0]
        self.rnd = np.r_[ind, np.random.choice(ind, remain)] # Reuse beggining of data when not enough data

    def shuffle(self):
        self.i = 0
        rd.shuffle(self.rnd)

    def __iter__(self):
        return self

    def __len__(self):
        return self.iter_n

    ## 	Pop batch data
    def __next__(self):

        self.test = False
        index = self.rnd[self.i * self.batch: (self.i + 1) * self.batch]  # Index of extracting data
        self.i += 1
        if not self.test:
            return self.image[index], self.mos[index], self.sim[index], self.var[index]  # Image & MOS
        else:
            return self.image[index], self.sim[index], self.var[index]  # Image


class create_batch_w_mask(create_batch):

    def __init__(self, image_s, mask_s, mos, sim, var, image_t, mask_t, batches, test=False):
        super(create_batch_w_mask, self).__init__(image_s, mos, sim, var, batches, test)
        self.mask = mask_s
        self.image_t = image_t
        self.mask_t = mask_t

        ind_s = np.array(range(self.image.shape[0]))
        ind_t = np.array(range(self.image_t.shape[0]))

        self.iter_n = math.ceil(self.image.shape[0] / batches)  # Batch num for each 1 Epoch

        loop_num_s  =math.ceil(self.iter_n * batches / image_s.shape[0])
        loop_num_t = math.ceil(self.iter_n * batches / image_t.shape[0])

        red_s = []
        red_t = []
        if not test:
            for i in range(loop_num_s):
                red_s.extend(rd.permutation(ind_s))
            for i in range(loop_num_t):
                red_t.extend(rd.permutation(ind_t))
        else:
            for i in range(loop_num_s):
                red_s.extend(ind_s)
            for i in range(loop_num_t):
                red_t.extend(ind_t)

        rnd_s = red_s[:self.iter_n * batches]
        rnd_t = red_t[:self.iter_n * batches]

        self.rnd_s = np.array(rnd_s)
        self.rnd_t = np.array(rnd_t)

        print('OK')

    def __next__(self):

        self.test = False

        index_s = self.rnd_s[self.i * self.batch: (self.i + 1) * self.batch]  # Index of extracting data
        index_t = self.rnd_t[self.i * self.batch: (self.i + 1) * self.batch]  # Index of extracting data
        self.i += 1
        if not self.test:
            return self.image[index_s], self.mask[index_s], self.mos[index_s], self.sim[index_s], self.var[index_s], index_s, self.image_t[index_t], self.mask_t[index_t]  # Image & MOS
        else:
            return self.image[index_s], self.mask[index_s], self.mos[index_s], self.sim[index_s], self.var[index_s], index_s  # Image