#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import keras 
import numpy as np
from keras import layers
from keras.models import Model
from keras.layers import Input
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
from keras.models import Sequential, Model 
from keras.applications.resnet50 import preprocess_input
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import TensorBoard
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


# In[2]:


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# In[3]:


image_shape=(384,384,3)

def identity_block(input_tensor,kernel_size, filters):
    filters1,filters2,filters3=filters
    bn_axis=3
    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

def conv_block(input_tensor,kernel_size,filters,strides=(2, 2)):
    filters1,filters2,filters3=filters
    bn_axis=3
    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal'
                    )(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, )(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis)(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def person_detection():
    bn_axis = 3
    img_input = Input(shape=(image_shape[0],image_shape[1],3))
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256])
    x = layers.Conv2DTranspose(1,[5, 5],strides=(4,4),padding='same')(x)
    
    # x = identity_block(x, 3, [64, 64, 256])
    # x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    # x = layers.Dense(8,activation = 'linear')(x)
    model = Model(input = img_input,output = x)
    # load weights
    # if weights_path:
    #   model.load_weights(weights_path, by_name=True)
#     model.summary()
    return model


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
                       (255,255,255),5)
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


model = person_detection()
model.compile(loss='mse', optimizer='adam', metrics=['accuracy','mae','mse'])  
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
                                          epochs=250,
                                          verbose=1,
                                          validation_data=my_validation_batch_generator,
                                          validation_steps=(len(b)),
                                          callbacks=[tensorboard, cp_callback]
#                                           use_multiprocessing=False,
#                                           workers=16,max_queue_size=32
                                          )


# In[ ]:




