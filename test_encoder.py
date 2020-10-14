# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:22:48 2020

@author: smail
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 


class Encoder:
    """Network that takes in MNIST digit and produces a single-dimensional latent variable.
    Consists of several convolution layers followed by fully-connected layers"""
    def __init__(self, training):
        """
        Arguments:
            training: Boolean of whether to use training mode or not. Matters for Batch norm layer
        """
        self.y_hat = None
        self.training = training
        self.bn = None

    def build(self, x, n_latent=1, name='y_hat'):
        """Convolutional and fully-connected layers to extract a latent variable from an MNIST image

        Arguments:
            x: Batch of MNIST images with dimension (n_batch, width, height)
            n_latent: Dimension of latent variable
            name: TensorFlow name of output
        """
        h = tf.expand_dims(x, axis=-1)  # Input to Conv2D needs dimension (n_batch, width, height, n_channels)
        h = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)(h)
        h = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(h)
        h = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)(h)
        h = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(h)
        h = tf.keras.layers.Flatten()(h)
        # h = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)(h)
        h = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)(h)
        h = tf.keras.layers.Dense(units=16, activation=tf.nn.relu)(h)
        h = tf.keras.layers.Dense(units=n_latent, name=name)(h)
        self.bn = tf.keras.layers.BatchNormalization()
        # Divide by 2 to make std dev close to 0.5 because distribution is uniform
        h = self.bn(h, training=self.training) / 2
        y_hat = tf.identity(h, name=name)
        return y_hat

    def __call__(self, x, n_latent=1, name='y_hat'):
        if self.y_hat is None:
            self.y_hat = self.build(x, n_latent, name)
        return self.y_hat
    
# training = tf.placeholder_with_default(True, [])
# print(type(training))
# for k,v in training.__dict__.items():
#     print(k)
#     print(v)
#     print()
import numpy as np
# encoder = Encoder(True)
batch_size,width,height = 2,27,27
# data = np.random.rand(batch_size,width,height)
# output = encoder(data)
# print(output.shape)

features, cities,lags = 10,10,5

from tensorflow.keras.layers import Conv2D, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import tensorflow
from tensorflow.keras.layers import BatchNormalization, Flatten

# def conv_plus_lstm(lags, features, cities, filters, kernSize):#model1
#     conv1_filters = 80
#     conv2_filters = 40
#     conv3_filters = 1
#     lstm1_nodes = 100
#     lstm2_nodes = 100
#     dense1_nodes = 100
#     number_cities = 2
    
#     input1 = Input(shape = (features, cities,lags))
#     block1 = Conv2D(conv1_filters, (kernSize, kernSize), padding = 'same', activation='relu')(input1)
#     block1 = BatchNormalization()(block1)
#     block1 = Conv2D(conv2_filters, (kernSize, kernSize), padding = 'same', activation='relu')(block1)
#     block1 = BatchNormalization()(block1)
#     block1 = Conv2D(conv3_filters, (kernSize, kernSize), padding = 'same', activation='relu')(block1)
#     block1 = BatchNormalization()(block1)
#     block1 = tensorflow.squeeze(block1,axis=-1)
#     block2 = LSTM(lstm1_nodes, return_sequences=True,name = "lstm1")(block1)
#     block2 = LSTM(lstm2_nodes, return_sequences=False,name = "lstm2")(block2)
#     block3 = Dense(dense1_nodes, activation='relu')(block2)
#     output1 = Dense(number_cities, activation='linear')(block3)
#     return Model(inputs=input1, outputs=output1)


class ConvPlusLSTM:
    def __init__(self, training):
        """
        Arguments:
            training: Boolean of whether to use training mode or not. Matters for Batch norm layer
        """
        self.y_hat = None
        self.training = training
        self.bn = None
        
    
        
    def build(self, x, name='y_hat'):
            conv1_filters = 5
            conv2_filters = 40
            conv3_filters = 1
            lstm1_nodes = 100
            lstm2_nodes = 100
            dense1_nodes = 100
            number_cities = 2
            # filters = 5
            kernSize = 7
            
            # input1 = Input(shape = (features, cities,lags))
            block1 = Conv2D(conv1_filters, (kernSize, kernSize), padding = 'same', activation='relu')(x)
            # self.bn = tf.keras.layers.BatchNormalization()
            # block1 = self.bn(block1,training=self.training)
            block1 = BatchNormalization()(block1)
            block1 = Flatten()(block1)
            # block1 = Conv2D(conv2_filters, (kernSize, kernSize), padding = 'same', activation='relu')(block1)
            # block1 = BatchNormalization()(block1)
            # block1 = Conv2D(conv3_filters, (kernSize, kernSize), padding = 'same', activation='relu')(block1)
            # block1 = BatchNormalization()(block1)
            # block1 = tensorflow.squeeze(block1,axis=-1)
            # block2 = LSTM(lstm1_nodes, return_sequences=True,name = "lstm1")(block1)
            # block2 = LSTM(lstm2_nodes, return_sequences=False,name = "lstm2")(block2)
            block1 = Dense(dense1_nodes, activation='relu')(block1)
            output1 = Dense(number_cities, activation='linear')(block1)
            return output1
            # return Model(inputs=input1, outputs=output1)
            
    def __call__(self, x, name='y_hat'):
        self.y_hat = self.build(x, name)
        return self.y_hat




data = np.random.rand(batch_size,features,cities,lags)
data = data.astype('float16') 
filters = 5
kernSize = 7

model = ConvPlusLSTM(True)
output = model(data)
print(output.shape)












