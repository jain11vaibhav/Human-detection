
import glob
import keras
import numpy as np
from keras.models import Model
from random import shuffle
import cv2
import os
import shutil
import pickle

import pandas as pd
import h5py
from time import time

import tensorflow as tf
from keras.utils import Sequence
from keras.models import Sequential
from keras.applications import resnet50
# from keras.applications.resnet50 import preprocess_input
from keras.callbacks import TensorBoard
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

import sys
import warnings
from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten, Dropout
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

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# import tensorflow as tf


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

K.clear_session()
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)



image_shape = (384, 384, 3)


def model2():
    model = Sequential()
    model.add(Conv2D(8, kernel_size=9, strides=(2, 2), activation='relu', input_shape=image_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(16, kernel_size=7, strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(32, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(48, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(layers.Conv2DTranspose(32, [10, 10], strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(layers.Conv2DTranspose(16, [12, 12], strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(layers.Conv2DTranspose(1, [14, 14], strides=(2, 2)))
    model.add(Activation('sigmoid'))
    return model

def model3():
    model = Sequential()
    model.add(Conv2D(8, kernel_size=13, strides=(2, 2), activation='relu', input_shape=image_shape,padding='valid'))
    model.add(BatchNormalization())
    # model.add(Activation('relu'))
    model.add(layers.LeakyReLU(alpha=0.1))

    model.add(Conv2D(16, kernel_size=11, strides=(2, 2), activation='relu', padding='valid'))
    model.add(BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    # model.add(Activation('relu'))

    model.add(Conv2D(32, kernel_size=9, strides=(2, 2), activation='relu', padding='valid'))
    model.add(BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    # model.add(Activation('relu'))

    model.add(Conv2D(64, kernel_size=7, strides=(2, 2), activation='relu', padding='valid'))
    model.add(BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    # model.add(Activation('relu'))

    model.add(Conv2D(128, kernel_size=5, strides=(2, 2), activation='relu', padding='valid'))
    model.add(BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    # model.add(Activation('relu'))

    model.add(layers.Conv2DTranspose(64, [5, 5], strides=(4, 4), padding='valid'))
    model.add(BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))

    model.add(layers.Conv2DTranspose(32, [8, 8], strides=(3, 3), padding='valid'))
    model.add(BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))

    model.add(layers.Conv2DTranspose(8, [10, 10], strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    # model.add(Activation('relu'))

    model.add(layers.Conv2DTranspose(8, [12, 12], strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    # model.add(Activation('relu'))

    model.add(layers.Conv2DTranspose(1, [14, 14], strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    return model

# def model4():
#     model = Sequential()
#     model.add(Conv2D(8, kernel_size=9, strides=(2, 2), activation='relu', input_shape=image_shape,padding='valid'))
#     model.add(BatchNormalization())
#     # model.add(Activation('relu'))
#     model.add(layers.LeakyReLU(alpha=0.1))

#     model.add(Conv2D(16, kernel_size=7, strides=(2, 2), activation='relu', padding='valid'))
#     model.add(BatchNormalization())
#     model.add(layers.LeakyReLU(alpha=0.1))
#     # model.add(Activation('relu'))

#     model.add(Conv2D(32, kernel_size=5, strides=(2, 2), activation='relu', padding='valid'))
#     model.add(BatchNormalization())
#     model.add(layers.LeakyReLU(alpha=0.1))
#     # model.add(Activation('relu'))

#     model.add(Conv2D(48, kernel_size=5, strides=(2, 2), activation='relu', padding='valid'))
#     model.add(BatchNormalization())
#     model.add(layers.LeakyReLU(alpha=0.1))

#     model.add(Conv2D(64, kernel_size=3, strides=(2, 2), activation='relu', padding='valid'))
#     model.add(BatchNormalization())
#     model.add(layers.LeakyReLU(alpha=0.1))
#     # model.add(Activation('relu'))

#     model.add(layers.Conv2DTranspose(32, [9, 9], strides=(4, 4), padding='same'))
#     model.add(BatchNormalization())
#     model.add(layers.LeakyReLU(alpha=0.1))
#     # model.add(Activation('relu'))

#     model.add(layers.Conv2DTranspose(16, [7, 7], strides=(4, 4), padding='same'))
#     model.add(BatchNormalization())
#     model.add(layers.LeakyReLU(alpha=0.1))
#     # model.add(Activation('relu'))

#     model.add(layers.Conv2DTranspose(8, [5, 5], strides=(2, 2), padding='same'))
#     model.add(BatchNormalization())
#     model.add(layers.LeakyReLU(alpha=0.1))

#     model.add(Conv2D(8, kernel_size=9, strides=(1, 1), activation='relu', padding='valid'))
#     model.add(BatchNormalization())
#     model.add(layers.LeakyReLU(alpha=0.1))

#     model.add(Conv2D(8, kernel_size=7, strides=(1, 1), activation='relu', padding='same'))
#     model.add(BatchNormalization())
#     model.add(layers.LeakyReLU(alpha=0.1))

#     model.add(Conv2D(8, kernel_size=5, strides=(1, 1), activation='relu', padding='same'))
#     model.add(BatchNormalization())
#     model.add(layers.LeakyReLU(alpha=0.1))

#     model.add(Conv2D(8, kernel_size=3, strides=(1, 1), activation='relu', padding='same'))
#     model.add(BatchNormalization())

#     model.add(Activation('sigmoid'))
#     return model

class MY_Generator(Sequence):
    def __init__(self, image_filenames, batch_size):
        self.image_filenames = image_filenames
        self.batch_size = batch_size

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx]
        batch_y = "./ground_truth" + batch_x[9:]
        pixle_name = '.' + batch_y.split('.')[0] + batch_y.split('.')[1] + '.pickle'
        with (open(pixle_name, "rb")) as openfile:
            while True:
                try:
                    obj = (pickle.load(openfile))
                except EOFError:
                    break
        img = []
        for i in range(32):
            gt = np.zeros((image_shape[0], image_shape[1], 1))
            z = np.zeros((image_shape[0], image_shape[1]))
            # channel = int((obj[i]['height'])/49)
            # cv2.circle(z, (int(obj[i]['center'][0]), int(obj[i]['center'][1])), int(image_shape[0]/32), (1, 0, 0), -1)
            cv2.rectangle(z,(int(obj[i]['center'][0])-25, int(obj[i]['center'][1])-int(int(obj[i]['height'])/2))
            ,(int(obj[i]['center'][0])+25, int(obj[i]['center'][1])+int(int(obj[i]['height'])/2)),1,-1)
            # cv2.imwrite("./tmp/"+str(i)+'.jpg',z)
            gt[:, :, 0] = z
            img.append(gt)
        # for j in range(32):
        #     os.mkdir("./tmp/"+str(j))
        #     for i in range(8):
        #         cv2.imwrite("./tmp/"+str(j)+'/'+str(i)+'.jpg',img[j][:,:,i] * 255)

        output = np.array(img)

        img = cv2.imread(batch_x)
        img = np.array(img, dtype=np.int32)
        img = img - 127
        img = img / 128

        image = np.zeros((self.batch_size, 384, 384, 3))

        for i in range(self.batch_size):
            image[i, :, :, :] = img[384*i:384*(i+1), :, :]

        return image, output


# def on_epoch_end(self, epoch, logs=None):
#   lr = float(K.get_value(self.model.optimizer.lr))
#   print("Learning rate:", lr)

def train():
    # For checkpoints
    if os.path.exists('./checkpoints'):
        shutil.rmtree('./checkpoints')
    os.mkdir('./checkpoints')
    checkpoint_path = "checkpoints/cp-{epoch:04d}.hdf5"
    checkpoint_path2 = "checkpoints/cp_best-{epoch:03d}.hdf5"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    earlyStopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, verbose=0, mode='min', restore_best_weights=True)
    cp_callback1 = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path2, verbose=1, save_weights_only=True, save_best_only=True, monitor="val_loss")
    cp_callback2 = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True, period=1)

    # Fetching the dataset
    dataset = []
    for i in os.listdir('./dataset'):
        b = glob.glob('./dataset/'+i+'/'+'*.jpg')
        dataset.extend(b)
    # shuffle(dataset)

    # Splitting the dataset into validation, training, and testing
    validation = dataset[0:int(len(dataset)/8)]
    # testing = dataset[int(len(dataset)/8):int(len(dataset)/4)]
    training = dataset[int(len(dataset)/8):]

    model=model3()

    adam = keras.optimizers.Adam(
        lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0002, amsgrad=False)

    model.compile(loss='mse', optimizer=adam,
                  metrics=['accuracy', 'mae', 'mse'])

    model.summary()
    model.save('model.h5')

    # model_json=model.to_json()
    # with open('model.json','w') as json_file:
    #     json_file.write(model_json)
    
    batch_size = 32
    my_training_batch_generator = MY_Generator(training, batch_size)
    my_validation_batch_generator = MY_Generator(validation, batch_size)
    # To include Tensorbaord
    # if os.path.exists('./logs'):
    #     shutil.rmtree('./logs')
    # os.mkdir('./logs')
    # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    model.fit_generator(generator=my_training_batch_generator,
                        steps_per_epoch=(len(training)),
                        epochs=10,
                        verbose=1,
                        validation_data=my_validation_batch_generator,
                        validation_steps=(len(validation)),
                        callbacks=[cp_callback1,
                                   cp_callback2, earlyStopping, ]
                        )
    model.save_weights("./weights/new.h5")


if __name__ == '__main__':
    train()
