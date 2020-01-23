
# coding: utf-8

# In[22]:

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import keras
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.utils import to_categorical
from keras import backend as K
import glob
import pickle
import numpy as np
import cv2
import random
import re
import shapefile
from scipy import stats
import argparse


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))



def parse_cmdline_args():
    parser = argparse.ArgumentParser(description='Parser for probability map prediction code')
    parser.add_argument('--prob_weight_path', type = str, default = '/data/zekunl/text_linking/heatmap/')
    parser.add_argument('--prediction_dir', type = str, default = '/data/zekunl/text_linking/predictions/')
    parser.add_argument('--gt_dir', type = str, default = '/data/zekunl/mydata/historical-map-groundtruth-25/')
    parser.add_argument('--word_coords_dir', type = str, default = '/data/zekunl/Geolocalizer/google_grouping/word_coords_list/')
    parser.add_argument('--output_dir', type = str, 
                        default ='/data/zekunl/text_linking/step2_predictions/' )
    
    parser.add_argument('--std_size', type = int, default = 256)
    parser.add_argument('--prob_thresh', type = float, default = 0.5)
    parser.add_argument('--MODE',action = 'store_true')
    
    parser.add_argument('--map_type', type = str, default = 'usgs')
    parser.add_argument('--WRITE_TO_FILE',action = 'store_true')
    parser.add_argument('--IF_SPECIFIC',action = 'store_true')
    parser.add_argument('--specific_map_name', type = str, default = '') # name without jpg appendix
    return (parser.parse_args())



# In[23]:

def unet(pretrained_weights = None,input_size = (256,256,4)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def rescale_padding(ori_img, std_size, fill_color = 0 ):
    
    height, width, c = ori_img.shape
    
    #print ori_img.shape
    # rescale and padding to std_size,std_size of ori_img \
    ########################################################################
    if height > width:
        scale = 1.0 * std_size / height
        padding = std_size - width * scale
    else:
        scale = 1.0 * std_size / width
        padding = std_size - height * scale

    scale_img = cv2.resize(ori_img, dsize=(0,0), fx = scale, fy = scale, interpolation=cv2.INTER_NEAREST)
    padding = np.int(round(padding))
     
    padsize_1 = np.int(np.floor( padding / 2 ))
    padsize_2 = np.int(padding - padsize_1)
    
    if c == 1:
        scale_img = np.expand_dims(scale_img, axis = -1)

    if height > width:
        padding_1 = np.ones((std_size, padsize_1, c), dtype = np.uint8) # h, w, c
        padding_2 = np.ones((std_size, padsize_2, c), dtype = np.uint8)
        padding_1 = np.array(fill_color) * padding_1
        padding_2 = np.array(fill_color) * padding_2
        
        std_img = np.concatenate((padding_1, scale_img, padding_2), axis = 1)
        
    else:
        padding_1 = np.ones((padsize_1, std_size, c), dtype= np.uint8) # h, w, c
        padding_2 = np.ones((padsize_2, std_size, c) , dtype = np.uint8)
        padding_1 = np.array(fill_color) * padding_1
        padding_2 = np.array(fill_color) * padding_2
        
        #print scale_img.shape, padding_1.shape, padding_2.shape
        std_img = np.concatenate((padding_1, scale_img, padding_2), axis = 0)
    ##############################################################################
    
    # make sure return size is exactly (std_size, std_size,)
    if std_img.shape[0] != std_size:
        std_img = std_img[:std_size,:,:]
    if std_img.shape[1] != std_size:
        std_img = std_img[:,:std_size,:]
        
    assert std_img.shape[0] == std_size, std_img.shape[1] == std_size
    #print std_img.shape
    return std_img

def get_mode_color(map_img):
    # produce background color for input map_img
    c1 = stats.mode(map_img[:,:,0].ravel())[0][0]
    c2 = stats.mode(map_img[:,:,1].ravel())[0][0]
    c3 = stats.mode(map_img[:,:,2].ravel())[0][0]
    return (c1, c2,c3)
    
    
def get_mean_color(map_img):
    # produce background color for input map_img
    c = tuple(np.mean(map_img, axis = (0,1)).astype('int'))
    
    #print c
    return c



def main(args):
    std_size = args.std_size
    prob_thresh = args.prob_thresh
    MODE = args.MODE # mode color or mean color

    prob_weight_path = args.prob_weight_path
    prediction_dir = args.prediction_dir
    gt_dir = args.gt_dir
    word_coords_dir = args.word_coords_dir
    output_dir = args.output_dir
    
    map_type = args.map_type
    WRITE_TO_FILE = args.WRITE_TO_FILE
    IF_SPECIFIC = args.IF_SPECIFIC
    specific_map_name = args.specific_map_name


    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


    model = unet()
    model.load_weights(prob_weight_path +map_type + '/magic.hdf5') # trained on specific map dataset

    if map_type == 'od':
        prefix = 'pred_10'
    else:
        prefix = 'pred_USGS'

    pred_paths = glob.glob(prediction_dir + prefix + '*.pkl')
    print pred_paths


    ############## processing one by one, could be speed up by batch prediction #####################

    for pred_path in pred_paths:
        # infer map image path and load map image
        map_name = os.path.basename(pred_path)[5:-4]

        if IF_SPECIFIC:
            if map_name != specific_map_name: ################### specify the one to process ####################
                continue

        map_path = gt_dir + map_name + '/' + map_name + '.jpg'
        map_img = cv2.imread(map_path)
        print map_img.shape

        word_coords_path = word_coords_dir +'pkl2_' +  map_name  + '.pkl'
        with open(word_coords_path, 'rb') as f:
            bbox_list = pickle.load(f)

        # load predictions of deep metric learning
        with open(pred_path, 'r') as f:
            pred_neighbors = pickle.load(f)


        step2_pred = {}

        for anchor_idx in range(len(pred_neighbors)):
            cur_pred_neighbors = pred_neighbors[anchor_idx]

            #print cur_pred_neighbors

            if len(cur_pred_neighbors) == 0:
                step2_pred[anchor_idx] = []
                continue # no predictions from last step
            if len(cur_pred_neighbors) == 1:
                step2_pred[anchor_idx] = cur_pred_neighbors
                continue # one prediction from last step
            if len(cur_pred_neighbors) == 2 and anchor_idx in cur_pred_neighbors:
                step2_pred[anchor_idx] = cur_pred_neighbors
                continue # two predictions but one is itself



            neighbor_locations = []
            anchor_loc = np.array(bbox_list[anchor_idx])
            neighbor_locations.extend(anchor_loc) # include itself first
            for neighbor_idx in cur_pred_neighbors:
                neighbor_locations.extend(np.array(bbox_list[neighbor_idx])) # include predicted neighbors

            neighbor_locations = np.array(neighbor_locations)

            #print 'neighbor_locations.shape',neighbor_locations.shape
            border_x_min, border_y_min = np.min(neighbor_locations, axis = 0).astype('int')
            border_x_max, border_y_max = np.max(neighbor_locations, axis = 0).astype('int')

            if border_x_max - border_x_min < 512:
                border_x_max = border_x_min + 512

            if border_y_max - border_y_min < 512:
                border_y_max = border_y_min + 512


            crop_img = map_img[border_y_min:border_y_max, border_x_min: border_x_max, :]
            if MODE:
                bg_color = get_mode_color(crop_img)
            else:
                bg_color = get_mean_color(crop_img)
            #print 'bg_color', bg_color
            std_img = rescale_padding(crop_img, std_size, fill_color = bg_color)

            #print recds[anchor_idx]
            #print 'crop_img.shape', crop_img.shape
            ##print anchor_idx, cur_pred_neighbors


            # get word mask
            mask_img = np.zeros((map_img.shape[0], map_img.shape[1],1)).astype('uint8') # h, w
            cv2.fillPoly(mask_img, [anchor_loc.astype('int')], color = (255))
            crop_mask = mask_img[border_y_min:border_y_max, border_x_min:border_x_max, :]
            std_mask = rescale_padding(crop_mask, std_size)


            input_x = np.concatenate([std_img, std_mask], axis = -1)
            pred = model.predict(np.expand_dims(input_x,0))[0]

            #print pred.shape

            # generate candidate mask 
            candidate_mask = np.zeros((map_img.shape[0], map_img.shape[1],1)).astype('uint8') # h, w
            for neighbor_idx in cur_pred_neighbors:
                neighbor_loc = np.array(bbox_list[neighbor_idx])
                cv2.fillPoly(candidate_mask, [neighbor_loc.astype('int')], color = neighbor_idx)

            crop_candidate = candidate_mask[border_y_min:border_y_max, border_x_min: border_x_max, :]
            std_candidate = rescale_padding(crop_candidate, std_size)

            # get argmax from overlapping candidate mask
            # put neighbor idx which has avg_cand_prob > threshold
            filter_mask = np.zeros((std_size, std_size,1)).astype('uint8') # h, w
            filtered_indices = []
            for neighbor_idx in cur_pred_neighbors:
                if neighbor_idx == anchor_idx:
                    continue # remove itself from comparison list
                indices = np.where(std_candidate == neighbor_idx) #get coordinates , get two arrays which are x, y coords
                avg_cand_prob =  np.sum(pred[indices]) / len(indices[0]) 
                if avg_cand_prob > prob_thresh:
                    filtered_indices.append(neighbor_idx)
                    # (3, n_indices) 3 because of h, w, c 3 axies. remove the third axis
                    cv2.fillPoly(filter_mask, [np.array([indices[1], indices[0]]).transpose(1,0).astype('int')], color = neighbor_idx)

            step2_pred[anchor_idx] = filtered_indices


        if WRITE_TO_FILE:
            output_path = output_dir + 'step2_pred_' + map_name + '.pkl'
            with open(output_path, 'w') as f:
                pickle.dump(step2_pred, f)


if __name__ == '__main__':
    args = parse_cmdline_args()
    print (args)
    main(args)
    
    

