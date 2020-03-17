#!/usr/bin/env python
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, MaxPooling2D, Dropout, Dense, Flatten
#from tensorflow.keras.models import Model

def MyModel(input_shape):
    model = tf.keras.Sequential([
        Conv2D(64, input_shape=input_shape, kernel_size=(3,3), activation='relu', strides=1, padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.5),

        Conv2D(128, kernel_size=(3, 3), activation='relu', strides=2, padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        Conv2D(256, kernel_size=(3, 3), activation='relu', strides=1, padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(256, kernel_size=(3, 3), activation='relu', strides=2, padding='same'),
        BatchNormalization(),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),

        Dense(1, activation='sigmoid')
    ])

    return model
