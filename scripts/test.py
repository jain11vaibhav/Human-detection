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

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# keras.backend.clear_session
# session = InteractiveSession(config=config)

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
    x = layers.Conv2DTranspose(12,[5, 5],strides=(4,4),padding='same')(x)
    
    # x = identity_block(x, 3, [64, 64, 256])
    # x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    # x = layers.Dense(8,activation = 'linear')(x)
    model = Model(input = img_input,output = x)
    return model

# checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     checkpoint_path, verbose=1, save_weights_only=True,
#     period=10)




# def spit_data(img_folder):
#     x=os.listdir('./tdataset/'+img_folder)
#     x=shuffle(x)
#     x1=x[0:int(len(x)/50)]
#     x2=x[int(len(x)/50):len(x)]
#     return x1,x2

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
        
        x=np.zeros((image_shape[0],image_shape[1],12))
        # print(obj[0]['height'])
        img=[]
        for i in range(32):
            z=x
            channel=int(max((obj[i]['height']),(obj[i]['width']))/33)
            z[int(obj[i]['center'][0])][int(obj[i]['center'][1])][channel]=255
            img.append(z)
        img=np.array(img)
        output=img
        # print(output)

        y1 = load_img(batch_x)

        
        y = img_to_array(y1)
        # y = cv2.imread(batch_x)
        image = np.zeros((self.batch_size,384,384,3))
        for i in range(self.batch_size):
            image[i,:,:,:] = y[384*i:384*(i+1),:,:]    
        # processed_image_resnet50 = resnet50.preprocess_input(image.copy())
        # z = np.transpose(np.array([batch_y,batch_y2]))
        # print(processed_image_resnet50,pickle.shape)
#         print(z,"captury",idx,batch_x[0])
        # print(image[:,192,:,0])
        return image,output

a=[]
for i in os.listdir('./tdataset'):
    b=glob.glob('./tdataset/'+i+'/'+'*.jpg')
    a.extend(b)
    
shuffle(a)
# print(a)
b=a[0:int(len(a)/8)]
c=a[int(len(a)/8):int(len(a)/4)]
a=a[int(len(a)/4):]




model = person_detection()
model.compile(loss='mse', optimizer='adam', metrics=['accuracy','mae','mse'])  
model.summary()
batch_size = 32
my_training_batch_generator = MY_Generator(a, batch_size)
my_validation_batch_generator = MY_Generator(b, batch_size)


# tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit_generator(generator=my_training_batch_generator,
                                          steps_per_epoch=(len(a)),
                                          epochs=250,
                                          verbose=1,
                                          validation_data=my_validation_batch_generator,
                                          validation_steps=(len(b)),
                                          use_multiprocessing=False,
                                          workers=2,max_queue_size=32
                                          )

# for i in range(len)
# b='ground_truth'+a[7:]
# for i in range(image_shape[0]):
#     for j in  range(image_shape[1]):
# x1,x2=spit_data('tdataset/6/')
# model = person_detection()

# model.compile(loss='mse', optimizer='adam', metrics=['accuracy','mae','mse'])    

# for i in x1:
#     img=cv2.imread('/tdataset/6/'+i)
#     pickle=pickle_out('./ground_truth/6/'+i[0:(len(i)-3)]+'.pickle')
    

    


# model = person_detection()
# model.summary()

# pickle_out('1905.pickle')