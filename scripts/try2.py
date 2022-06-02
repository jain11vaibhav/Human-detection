#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import keras 
import numpy as np
from keras.models import Model
from random import shuffle
import cv2
import os
import pickle

import pandas as pd
import h5py
from time import time

import tensorflow as tf
from keras.utils import Sequence
from keras.models import Sequential
from keras.applications import ResNet50
from keras.applications import resnet50
# from keras.applications.resnet50 import preprocess_input
from keras.callbacks import TensorBoard
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


import warnings
from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
# from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs


# In[2]:


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# In[3]:


image_shape=(384,384,3)

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000):
    
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    # input_shape = _obtain_input_shape(input_shape,
    #                                   default_size=224,
    #                                   min_size=197,
    #                                   data_format=K.image_data_format(),
    #                                   include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = Dense(32,activation='relu')(x)
    x = layers.Conv2DTranspose(1,[5, 5],strides=(4,4),padding='same')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path, by_name=True)

        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model

# def person_detection():
#     bn_axis = 3
#     img_input = Input(shape=(image_shape[0],image_shape[1],3))
#     x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
#     x = layers.Conv2D(64, (7, 7),
#                       strides=(2, 2),
#                       padding='valid',
#                       kernel_initializer='he_normal',
#                       name='conv1')(x)
#     x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
#     x = layers.Activation('relu')(x)
#     x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
#     x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

#     x = conv_block(x, 3, [64, 64, 256], strides=(1, 1))
#     x = identity_block(x, 3, [64, 64, 256])
#     x = layers.Conv2DTranspose(1,[5, 5],strides=(4,4),padding='same')(x)
    
#     # x = identity_block(x, 3, [64, 64, 256])
#     # x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
#     # x = layers.Dense(8,activation = 'linear')(x)
#     model = Model(input = img_input,output = x)
#     # load weights
#     if weights_path:
#       model.load_weights(weights_path, by_name=True)
# #     model.summary()
#     return model


# In[4]:


checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    period=10)


# In[5]:


class MY_Generator(Sequence):
    def __init__(self, image_filenames, batch_size):
        self.image_filenames = image_filenames
        self.batch_size = batch_size

    def __len__(self):
        return len(self.image_filenames) 

    # def pickle_out(self,pixle_name):
         
    #     # print(img.shape)
    #     return(img)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx]
        batch_y = batch_x
        batch_y = "./ground_truth" + batch_y[10:]
        batch_y = batch_y[0:-4]+'.pickle'
        # print(batch_x,batch_y)
        # pickle=pickle_out(batch_y)
        pixle_name = batch_y
        obj=0
        with (open(pixle_name, "rb")) as openfile:
            while True:
                try:
                    obj=(pickle.load(openfile))
                except EOFError:
                    break
        
        x=np.zeros((image_shape[0],image_shape[1],1))
#         x= np.full((image_shape[0],image_shape[1],9),255)
        # print(obj[0]['height'])
        img=[]
        for i in range(32):
            z=x
	#             channel = i%9
            # channel=int((obj[i]['height'])/44)
            
            cv2.line(z,(int(obj[i]['center'][0]),(int(obj[i]['center'][1])-(int(obj[i]['height']/2)))),
            (int(obj[i]['center'][0]),(int(obj[i]['center'][1])+(int(obj[i]['height']/2)))),
                       (255,255,255),10)
            img.append(z)
        img=np.array(img)
        output=img
        # print(output)

	#         y1 = load_img(batch_x)

        
	#         y = img_to_array(y1)
        y = cv2.imread(batch_x)
        image = np.zeros((self.batch_size,384,384,3))
        for i in range(self.batch_size):
            image[i,:,:,:] = y[384*i:384*(i+1),:,:]    
        processed_image_resnet50 = resnet50.preprocess_input(image.copy())
        # processed_image_resnet50 = resnet50.preprocess_input(image.copy())
        # z = np.transpose(np.array([batch_y,batch_y2]))
        # print(processed_image_resnet50,pickle.shape)
	#         print(z,"captury",idx,batch_x[0])
        # print(image[:,192,:,0])
	
        return processed_image_resnet50,output


# In[6]:


a=[]
for i in os.listdir('./tdataset'):
    b=glob.glob('./tdataset/'+i+'/'+'*.jpg')
    a.extend(b)

# print(a)


# In[7]:


shuffle(a)
# print(a)
b=a[0:int(len(a)/8)]
c=a[int(len(a)/8):int(len(a)/4)]
a=a[int(len(a)/4):]


# In[8]:


model = ResNet50(include_top=False, weights='imagenet', input_shape=image_shape)


for layers in model.layers[:28]:
    # print(i,layers)
    layers.trainable=False

adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='mse', optimizer=adam, metrics=['accuracy','mae','mse'])  
model.summary()
# print(model.get_weights())


# In[9]:


batch_size = 32
my_training_batch_generator = MY_Generator(a, batch_size)
my_validation_batch_generator = MY_Generator(b, batch_size)


# In[10]:


tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit_generator(generator=my_training_batch_generator,
                                          steps_per_epoch=(len(a)),
                                          epochs=1,
                                          verbose=1,
                                          validation_data=my_validation_batch_generator,
                                          validation_steps=(len(b)),
                                          callbacks=[tensorboard, cp_callback]
#                                           use_multiprocessing=False,
#                                           workers=16,max_queue_size=32
                                          )
model.save_weights("./my_model_weights.h5")


# In[ ]:




