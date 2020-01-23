# coding: utf-8
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import keras
from keras.layers import *
from keras.models import *
from keras.utils import to_categorical
from keras import layers
from keras import backend as K
import tensorflow as tf
import glob
import pickle
import numpy as np
import cv2
import random
import re
import sys
import argparse
from gensim.models import KeyedVectors
from numpy import dot
from numpy.linalg import norm

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# configure GPU usage
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))


def parse_cmdline_args():
    parser = argparse.ArgumentParser(description='Parser for deep metric learning prediction code')
    parser.add_argument('--preprocess_dir', type = str, default = '/data/zekunl/Geolocalizer/google_grouping/')
    parser.add_argument('--map_dir', type = str, default = '/data/zekunl/mydata/historical-map-groundtruth-25/')
    parser.add_argument('--word2vec_filepath', type = str, default = '../word2vec/glove.6B/glove.6B.50d.txt.word2vec')
    parser.add_argument('--pred_output_dir', type = str, default = '../predictions/')
    parser.add_argument('--map_type', type = str, default = 'od')
    parser.add_argument('--dml_weight_dir', type = str, default = '../scripts_v3/weights/')
    parser.add_argument('--dml_weight_name', type = str, 
                        default = '23_od_textLinking_text_loc_angle_size_cap_nontrainable_diffef1_best.hdf5')
    parser.add_argument('--capitalization_weights', type = str, 
                        default = '/data/zekunl/text_linking/style_learner/capitalization_real.hdf5')
    
    parser.add_argument('--ALPHA', type = int, default = 1)
    parser.add_argument('--max_batch_size', type = int, default = 256)
    return (parser.parse_args())
    


# code for removing special caracters, puctuations
# https://mlwhiz.com/blog/2019/01/17/deeplearning_nlp_preprocess/
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_text(x):
    x = str(x)
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, "")
    return x
def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x

def clean_string(in_string):
        return  clean_numbers(clean_text(in_string.lower().decode('utf-8', 'ignore')))


    
def trip_feat_loss(x):
    global ALPHA
    anchor, positive, negative = x

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
    
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), ALPHA)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss

def trip_feat_loss_shape(shapes):
    shape1, shape2, shape3 = shapes
    return (shape1[0], 1)

    
def my_visual_model():
    img_input = Input(shape=(224,224,3))
    
    x = layers.Conv2D(32, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)


    # Classification block
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(1024, activation='relu', name='fc1')(x)
    x = layers.Dense(500, activation='softmax', name='predictions')(x)
    
    model = keras.models.Model([img_input], [x], name='my_visual_model')
    
    return model

def feature_model():
    input_text = Input(shape = (50,))
    input_loc = Input(shape = (4,))
    
    t = input_text
    
    l = input_loc
    
    x = keras.layers.Concatenate(axis=-1)([ t, l])
    
    mix_representation = x
    
    model = keras.models.Model([input_text, input_loc], mix_representation)
    print (model.summary())
    
    return model

    
def clf_model():
    
    model = Sequential()
    model.add(Dense(128, input_shape=( ( 50 + 2 + 2 ) * 2,), activation = 'relu'))
    model.add(Dense(64,activation = 'relu'))
    model.add(Dense(16,activation = 'relu'))
    model.add(Dense(2, activation='softmax'))

    
    print (model.summary())

    return model


def overall_model(cap_weights):
    input_visual1 = Input(shape = (224,224,3))
    input_visual2 = Input(shape = (224,224,3))


    input_text1 = Input(shape = (50,))
    input_text2 = Input(shape = (50,))
    
    input_loc1 = Input(shape = (4,))
    input_loc2 = Input(shape = (4,))

    feat1 = keras.layers.Concatenate(axis=-1)([input_text1, input_loc1])
    feat2 = keras.layers.Concatenate(axis=-1)([input_text2, input_loc2])

    x = keras.layers.Concatenate(axis=-1)([feat1, feat2])
    
    y = clf_model()(x)
    #y = Dense(2, activation = 'softmax')(x)
    
    model = keras.models.Model([ input_visual1, input_text1, input_loc1, input_visual2, input_text2, input_loc2], [y], name='overall_model')
    
    return model

def dummy_loss_func(ones, real_loss):
    return real_loss + 0 * ones


def normalize_to_fixed_range(loc_array):
    for col in range(0, loc_array.shape[1]):
        mini = min(loc_array[:,col])
        maxi = max(loc_array[:,col])

        loc_array[:,col] = (loc_array[:,col] - (maxi + mini)/2.)/(maxi - mini)*2
    return loc_array

# from word_dir and location_dir, get the X, V, T, L components as the linkage prediction model input 
def get_google_validation_data_info(index,
        map_type = 'both',
        img_dir = '/data/zekunl/mydata/historical-map-groundtruth-25/',
        word_dir = "/data/zekunl/Geolocalizer/google_grouping/word_list/",
        location_dir = "/data/zekunl/Geolocalizer/google_grouping/word_coords_list/",
        num_val_maps = 3):
    
    all_map_list = [a[:-4] for a in os.listdir(word_dir)] 
    all_map_list = list(filter(lambda x: 'pkl2' in x, all_map_list))
    if map_type == 'both':
        map_list = all_map_list
    elif map_type == 'usgs':
        map_list = list(filter(lambda x: 'USGS' in x,  all_map_list))
    else :
        map_list = list(filter(lambda x: 'USGS' not in x,  all_map_list))

    map_list = map_list[-num_val_maps:]
    print ('map_list',map_list)
    
    with open(word_dir + map_list[index] + '.pkl', 'rb') as f:
        word_list = pickle.load(f)
        
    with open(location_dir + map_list[index] + '.pkl', 'rb') as f:
        bbox_list = pickle.load(f)
    
    assert len(word_list) == len(bbox_list)
    
    fi = map_list[index][5:]
        
    # construct text dict
    text_dict = {}
    # construct imgs_array
    # construct loc_array
    loc_array = []
    angle_list = []
    font_area_list = []
    
    img_path = img_dir + fi + '/' + fi + '.jpg'
    img = cv2.imread(img_path)
    imgs_array = []
    for i in range(0, len(bbox_list)):
        # populate text_dict
        text_dict[i] = word_list[i]
        
        cur_point = np.array(bbox_list[i])
        #print (cur_point)
        xmin1,ymin1 = np.min(cur_point, axis = 0)
        xmax1, ymax1 = np.max(cur_point, axis = 0)
        #print xmin1,ymin1,xmax1,ymax1

        # crop image
        xmin1 = int(xmin1)
        ymin1 = int(ymin1)
        xmax1 = int(xmax1)
        ymax1 = int(ymax1)

        margin =  0#5 #20

        xmin1 = xmin1 - margin
        ymin1 = ymin1 - margin
        xmax1 = xmax1 + margin
        ymax1 = ymax1 + margin

        #print(xmin1,ymin1,xmax1,ymax1)
        crop_img = img[ymin1:ymax1, xmin1:xmax1,:] #### <Y,X> IMPORTANT!!! ########## 
        crop_img = cv2.resize(crop_img, (224, 224))
        crop_img = crop_img/255.*2 - 1
        
        # compute angle
        ret = cv2.minAreaRect(np.expand_dims(cur_point,axis = 1).astype('int64'))
        (c_x,c_y),(wrect,hrect), angle = ret # angle in the range of [-90, 0)
        
        #print wrect, hrect
        if wrect < hrect:
            angle = angle + 180
        else:
            angle = angle + 90 
           
            
        # font area    
        font_area_list.append(wrect * hrect/len(text_dict[i]))
        angle_list.append(angle) 
    
        imgs_array.append(crop_img)
        loc_array.append([c_x,c_y])
        
    imgs_array = np.array(imgs_array)
    loc_array = normalize_to_fixed_range(np.array(loc_array))
    angle_array = np.expand_dims(np.array(angle_list),axis = 1) / 180. #[0,1]
    font_size_array = normalize_to_fixed_range( np.expand_dims(np.array(font_area_list),axis = 1) )
    
    return (fi, None , text_dict, imgs_array, loc_array , angle_array, font_size_array)



def main(args):
    preprocess_dir = args.preprocess_dir
    map_dir = args.map_dir
    filename = args.word2vec_filepath
    prediction_output = args.pred_output_dir
    map_type = args.map_type
    save_name = args.dml_weight_name
    cap_model_weights_path = args.capitalization_weights
    overall_model_weights_dir = args.dml_weight_dir
    #ALPHA = args.ALPHA
    max_batch_size = args.max_batch_size
    


    gensim_model = KeyedVectors.load_word2vec_format(filename, binary=False)

    if not os.path.isdir(prediction_output):
        os.makedirs(prediction_output)
    
    
    model = overall_model(cap_weights = cap_model_weights_path)
    model.summary()

    model.compile(optimizer='sgd',
        loss=['binary_crossentropy'],
        metrics=['acc'])


    model.load_weights(overall_model_weights_dir + save_name )

    map_wise_precisions = []
    map_wise_recalls = []
    for i in range(0, 3):
        pred_neighbors_dict = {}
        
        # get information for one map
        fi, _, text_dict, images, loc_array , angle_array, font_size_array = \
        get_google_validation_data_info(index = i, map_type = map_type ,
                             img_dir = map_dir, 
                             word_dir = preprocess_dir + '/word_list/', 
                             location_dir = preprocess_dir +'/word_coords_list/', 
                             num_val_maps = 3) 
        
        total_num_regions = len(text_dict)


        out_path = prediction_output + 'pred_' + fi + '.pkl'

        for anchor_idx in range(total_num_regions):
            anchor_img = images[anchor_idx]
            anchor_text = text_dict[anchor_idx]
            try:
                anchor_tf = gensim_model.wv[clean_string(anchor_text)]
            except Exception as e:
                #print (e)
                #print ('could not find corresponding text for anchor',anchor_idx, anchor_text)
                anchor_tf = np.random.randn(50,)

            anchor_loc = loc_array[anchor_idx]
            anchor_angle = angle_array[anchor_idx]
            anchor_fontsize = font_size_array[anchor_idx]



            OV = []
            OT = []
            OL = []
            OA = []
            OF = []

            Y_pred = []
            pred_values = []

            for other_idx in range(total_num_regions):
                other_img = images[other_idx]
                other_text = text_dict[other_idx]
                try:
                    other_tf = gensim_model.wv[clean_string(other_text)]
                except:
                    #print ('could not find corresponding text for other text', other_idx, other_text)
                    other_tf = np.random.randn(50,)

                other_loc = loc_array[other_idx]
                other_angle = angle_array[other_idx]
                other_fontsize = font_size_array[other_idx]

                OV.append(other_img)
                OT.append(other_tf)
                OL.append(other_loc)
                OA.append(other_angle)
                OF.append(other_fontsize)

                # evaluate once
                if (other_idx + 1 )% max_batch_size == 0 or (other_idx) == (total_num_regions -1 ): # Time to evaluate!

                    OV = np.array(OV)
                    OT = np.array(OT)
                    OL = np.array(OL)
                    OA = np.array(OA)
                    OF = np.array(OF)
                    OL = np.concatenate([OL, OA, OF], axis = 1)

                    XV = np.repeat(anchor_img[np.newaxis, :], OV.shape[0], axis = 0 )
                    XT = np.repeat(anchor_tf[np.newaxis, :], OV.shape[0], axis = 0 )
                    XL = np.repeat(anchor_loc[np.newaxis, :], OV.shape[0], axis = 0 )
                    XA = np.repeat(anchor_angle[np.newaxis, :], OV.shape[0], axis = 0 )
                    XF = np.repeat(anchor_fontsize[np.newaxis, :], OV.shape[0], axis = 0 )
                    XL = np.concatenate([XL, XA, XF], axis = 1)

                    # concatenate the anchor and other
                    y_pred = model.predict([ XV, XT,  XL, OV, OT, OL])

                    Y_pred.extend(np.argmax(y_pred, axis = 1)) # get label 1:yes 0:no
                    pred_values.extend(np.max(y_pred, axis = 1)) # get the values


                    # emptify other list
                    OV = []
                    OT = []
                    OL = []
                    OA = []
                    OF = []
                    #print (other_idx)



            Y_pred = np.array(Y_pred)
            pred_ones = np.where(Y_pred == 1)[0]
            pred_values = np.array(pred_values)[pred_ones]
            sort_indices = np.argsort(pred_values)[::-1]

            pred_neighbors_dict[anchor_idx] = pred_ones
            print (pred_ones)

        with open(out_path, 'wb') as f:
            pickle.dump(pred_neighbors_dict, f)




if __name__ == '__main__':
    args = parse_cmdline_args()
    print (args)
    main(args)
    
    

