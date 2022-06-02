import glob
import keras
import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
from time import time
from random import shuffle
from keras.utils import Sequence
from keras.models import load_model
from keras.applications import ResNet50
from keras.models import model_from_json
from keras.applications import resnet50
from keras.applications.imagenet_utils import preprocess_input
from train import MY_Generator 


from keras.callbacks import TensorBoard
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


image_shape = (384, 384, 3)


def train():
    model = load_model('model.h5')

    # json_file = open('model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # model=model_from_json(loaded_model_json)

    model.load_weights('./backup/cp-0014.hdf5')

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

    # to avoid the training of the resnet weights
    # for layers in model.layers[:15]:
    #     layers.trainable = False

    adam = keras.optimizers.Adam(
        lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00001, amsgrad=False)

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
    # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    model.fit_generator(generator=my_training_batch_generator,
                        steps_per_epoch=(len(training)),
                        epochs=80,
                        verbose=1,
                        validation_data=my_validation_batch_generator,
                        validation_steps=(len(validation)),
                        callbacks=[cp_callback1,
                                   cp_callback2, earlyStopping]
                        )
    model.save_weights("./weights/new.h5")

if __name__ == '__main__':
    train()
