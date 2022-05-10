import argparse
import sys
import os
import numpy as np
import warnings
import xml.etree.ElementTree as et
import pandas as pd
from tqdm import tqdm
import csv
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K
import os
from tensorflow.python.keras import backend as tensorflow_backend
from tensorflow.keras import layers

import tensorflow.keras.losses
projection_dim = 64
patch_size = 1
num_patches =28*28
class Patches(layers.Layer):
    def __init__(self, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        print(tf.shape(images)[1],'images')
        print(self.patch_size,'self.patch_size')
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        print(patches.shape,patch_dims,'patch_dims')
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size,
        })
        return config
    
    
class PatchEncoder(layers.Layer):
    def __init__(self, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection': self.projection,
            'position_embedding': self.position_embedding,
        })
        return config
    
    
dir_model = '/home/vahid/Documents/image_classification_transformer/output/model_1.h5'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

session = tf.compat.v1.Session(config=config)  # tf.InteractiveSession()
tensorflow_backend.set_session(session)
#tensorflow.keras.layers.custom_layer = PatchEncoder
#tensorflow.keras.layers.custom_layer = Patches
model = load_model(dir_model , compile=False,custom_objects = {"PatchEncoder": PatchEncoder, "Patches": Patches})

model.summary()
#if self.weights_dir!=None:
    #print('man burdayammmmaaa')
    #self.model.load_weights(self.weights_dir)
    

#self.img_height=self.model.layers[len(self.model.layers)-1].output_shape[1]
#self.img_width=self.model.layers[len(self.model.layers)-1].output_shape[2]
#self.n_classes=self.model.layers[len(self.model.layers)-1].output_shape[3]
