import cv2
import shutil
import pandas as pd
import h5py
from time import time

import numpy as np
from numpy import genfromtxt

import keras
from keras.utils import Sequence
from keras.models import Sequential
from keras.models import load_model
from keras.applications import ResNet50
from keras.applications import resnet50
from keras.models import Sequential, Model 
from keras.applications.resnet50 import preprocess_input
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import TensorBoard
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
import tensorflow as tf
from keras.models import model_from_json


import os
import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def predict():
    if os.path.exists('./predict'):
        shutil.rmtree('./predict')
    os.mkdir('./predict')
    model = load_model('model.h5')
    # json_file = open('model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # model=model_from_json(loaded_model_json)
    # load weights into new model
    # model.load_weights("./weights/new.h5")
    model.load_weights("./backup/cp-0014.hdf5")
    print("Loaded model from disk")

    img = cv2.imread('1580.jpg')
    # img = cv2.imread('./dataset/1/3319.jpg')
    img = np.array(img, dtype=np.int32)
    img = img - 127
    img = img / 128

    image = np.zeros((32, 384, 384, 3))
    for i in range(32):
        image[i, :, :, :] = img[384*i:384*(i+1), :, :]

    # com = np.zeros((384,384*2,3))
    # com[:,:384,:] = img[0:384,:,:]
    # com[:,384:384*2,:] = img[384:384*2,:,:]
    # # com[:,384*2:384*3,:] = img[384*2:384*3,:,:]

    # new_image = cv2.resize(com, (384,384))
    # cv2.imwrite("test.jpg",(new_image + 1) * 128)

    # new_image = np.array([new_image])

    # xz = model.predict(new_image) * 255
    # cv2.imwrite("test_result.jpg",xz[0])


    xz = model.predict(image) * 255
    zz = np.zeros((384*32, 384, 1))
    for i in range(32):
        print(np.max(xz[i,:,:,:]))
        cv2.imwrite("./predict/"+str(i)+".jpg",xz[i,:,:,:])
        zz[384*i:384*(i+1),:,:] = xz[i,:,:,:] 
    cv2.imwrite("result.jpg",zz)

if __name__ == '__main__':
    predict()
