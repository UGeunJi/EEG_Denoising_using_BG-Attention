import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers,Sequential
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from tensorflow.keras import datasets, layers, models, Input, Sequential

def Novel_CNN(datanum):
    model = tf.keras.Sequential()
    model.add(Input(shape=(datanum, 1)))

    model.add(layers.Conv1D(32, 3, activation = tf.nn.relu, padding = 'same', kernel_initializer = 'he_normal'))
    model.add(layers.Conv1D(32, 3, activation = tf.nn.relu, padding = 'same', kernel_initializer = 'he_normal'))
    model.add(layers.AveragePooling1D(pool_size= 2))

    model.add(layers.Conv1D(64, 3, activation = tf.nn.relu, padding = 'same', kernel_initializer = 'he_normal'))
    model.add(layers.Conv1D(64, 3, activation = tf.nn.relu, padding = 'same', kernel_initializer = 'he_normal'))
    model.add(layers.AveragePooling1D(pool_size= 2))

    model.add(layers.Conv1D(128, 3, activation = tf.nn.relu, padding = 'same', kernel_initializer = 'he_normal'))
    model.add(layers.Conv1D(128, 3, activation = tf.nn.relu, padding = 'same', kernel_initializer = 'he_normal'))
    model.add(layers.AveragePooling1D(pool_size= 2))

    model.add(layers.Conv1D(256, 3, activation = tf.nn.relu, padding = 'same', kernel_initializer = 'he_normal'))
    model.add(layers.Conv1D(256, 3, activation = tf.nn.relu, padding = 'same', kernel_initializer = 'he_normal'))
    model.add(layers.Dropout(0.5))
    model.add(layers.AveragePooling1D(pool_size = 2))

    model.add(layers.Conv1D(512, 3, activation = tf.nn.relu, padding = 'same', kernel_initializer = 'he_normal'))
    model.add(layers.Conv1D(512, 3, activation = tf.nn.relu, padding = 'same', kernel_initializer = 'he_normal'))
    model.add(layers.Dropout(0.5))
    model.add(layers.AveragePooling1D(pool_size = 2))

    model.add(layers.Conv1D(1024, 3, activation = tf.nn.relu, padding = 'same', kernel_initializer = 'he_normal'))
    model.add(layers.Conv1D(1024, 3, activation = tf.nn.relu, padding = 'same', kernel_initializer = 'he_normal'))
    model.add(layers.Dropout(0.5))
    model.add(layers.AveragePooling1D(pool_size = 2))

    model.add(layers.Conv1D(2048, 3, activation = tf.nn.relu, padding = 'same', kernel_initializer = 'he_normal'))
    model.add(layers.Conv1D(2048, 3, activation = tf.nn.relu, padding = 'same', kernel_initializer = 'he_normal'))
    model.add(layers.Dropout(0.5))
    #######
    model.add(layers.Flatten())
    #output1 = layers.Dense(2048)(flatten1)
    model.add(layers.Dense(datanum))
    #model = Model(inputs = inputs, outputs = output1)

    model.summary()
    return model



# 수정 전 code (error 뜸)

'''
def Novel_CNN(datanum):

    model = tf.keras.Sequential()
    inputs = Input(shape=(datanum,))
    conv1 = layers.Conv1D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv1 = layers.Conv1D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    pool1 = layers.AveragePooling1D(pool_size= 2)(conv1)

    conv2 = layers.Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv2 = layers.Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    pool2 = layers.AveragePooling1D(pool_size= 2)(conv2)

    conv3 = layers.Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv3 = layers.Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    pool3 = layers.AveragePooling1D(pool_size= 2)(conv3) #9

    conv4 = layers.Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv4 = layers.Conv1D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.AveragePooling1D(pool_size = 2)(drop4)  #13

    conv5 = layers.Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv5 = layers.Conv1D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    drop5 = layers.Dropout(0.5)(conv5)
    ###
    pool5 = layers.AveragePooling1D(pool_size = 2)(drop5)

    conv6 = layers.Conv1D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv6 = layers.Conv1D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    drop6 = layers.Dropout(0.5)(conv6)

    pool6 = layers.AveragePooling1D(pool_size = 2)(drop6)

    conv7 = layers.Conv1D(2048, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv7 = layers.Conv1D(2048, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    drop7 = layers.Dropout(0.5)(conv7)
    #######
    flatten1 = layers.Flatten()(drop7)
    #output1 = layers.Dense(2048)(flatten1)
    output1 = layers.Dense(1024)(flatten1)
    #model = Model(inputs = inputs, outputs = output1)

    model.summary()
    return model

'''
