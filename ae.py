from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Conv2DTranspose, Reshape, BatchNormalization, Activation, Dropout
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
import sys
import numpy as np
from PIL import Image

from ternary_ops import ternarize
from ternary_layers import TernaryDense, TernaryConv2D
from binary_ops import binary_tanh as binary_tanh_op
from binary_layers import BinaryDense, BinaryConv2D


def binary_tanh(x):
    return binary_tanh_op(x)

def ternary_tanh(x):
    x = K.clip(x, -1, 1)
    return ternarize(x)

H = 1
kernel_lr_multiplier = 'Glorot'
W_lr_multiplier = 'Glorot'
bias = False

class FPModel:
    def __init__(self):
        self.create_model()

    def create_model(self):
        input_img = Input(shape=(120,120,3))  # adapt this if using `channels_first` image data format


        x = TernaryConv2D(16, kernel_size=(4, 4), H=H, kernel_lr_multiplier=W_lr_multiplier, bias=bias,kernel_initializer='he_normal', padding='valid', strides=2)(input_img)
        x = BatchNormalization()(x)
        x = Activation(ternary_tanh)(x)

        x = TernaryConv2D(32, kernel_size=(4, 4), H=H, kernel_lr_multiplier=W_lr_multiplier, bias=bias,kernel_initializer='he_normal', padding='valid', strides=2)(x)
        x = BatchNormalization()(x)
        x = Activation(ternary_tanh)(x)

        x = TernaryConv2D(64, kernel_size=(4, 4), H=H, kernel_lr_multiplier=W_lr_multiplier, bias=bias,kernel_initializer='he_normal', padding='valid', strides=2)(x)
        x = BatchNormalization()(x)
        x = Activation(ternary_tanh)(x)

        x = TernaryConv2D(128, kernel_size=(3, 3), H=H, kernel_lr_multiplier=W_lr_multiplier, bias=bias,kernel_initializer='he_normal', padding='valid', strides=2)(x)
        x = BatchNormalization()(x)
        x = Activation(ternary_tanh)(x)

        x = Flatten()(x)
        x = TernaryDense(128, H=H, kernel_lr_multiplier=W_lr_multiplier,bias=bias,kernel_initializer='he_normal')(x)
        encoded = Activation(ternary_tanh)(x)

        x = Dense(4608)(encoded)
        x = Reshape([6, 6, 128])(x)
        x = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='valid')(x)
        x = Conv2DTranspose(32, (4, 4), strides=2, activation='relu', padding='valid')(x)
        x = Conv2DTranspose(16, (5, 5), strides=2, activation='relu', padding='valid')(x)
        decoded = Conv2DTranspose(3, (4, 4), strides=2, activation='sigmoid', padding='valid')(x)

        # x = Conv2D(16, (4, 4), activation='relu', padding='valid', strides=2)(input_img)
        # x = Conv2D(32, (4, 4), activation='relu', padding='valid', strides=2)(x)
        # x = Conv2D(64, (4, 4), activation='relu', padding='valid', strides=2)(x)
        # x = Conv2D(128, (3, 3), activation='relu', padding='valid', strides=2)(x)
        # x = Flatten()(x)
        # encoded = Dense(128)(x)
        #
        # x = Dense(4608)(encoded)
        # x = Reshape([6, 6, 128])(x)
        # x = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='valid')(x)
        # x = Conv2DTranspose(32, (4, 4), strides=2, activation='relu', padding='valid')(x)
        # x = Conv2DTranspose(16, (5, 5), strides=2, activation='relu', padding='valid')(x)
        # decoded = Conv2DTranspose(3, (4, 4), strides=2, activation='sigmoid', padding='valid')(x)

        self.encoder = Model(input_img, encoded)
        self.autoencoder = Model(input_img, decoded)

    def save(self, path):
        self.encoder.save(path)

    def load(self, path):
        self.encoder.load_weights(path)

    def encode(self, img):
        img = Image.fromarray(img)
        img = img.crop( ( 20, 0, 140 , 120 ) )
        img = np.asarray(img)

        img = np.array([img])
        img = img.astype('float32') / 255.
        enc = self.encoder.predict(img)
        return enc[0]
