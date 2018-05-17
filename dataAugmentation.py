#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 13:27:10 2018

@author: csimm
"""

from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Dropout
from keras import backend as K
import time
import os, shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
K.set_image_dim_ordering('th')

theTime = time.strftime('%y%m%d_%H%M%S', time.localtime())

img_width, img_height = (96, 96)
input_tensor = Input(shape=(3, img_width, img_height))
# create the base pre-trained model
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

# add a global spatial average pooling layer
x = base_model.output
x = Flatten(input_shape=base_model.output_shape[1:])(x)
# let's add a fully-connected layer
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

print(len(model.layers))
#print(model.summary())
for layer in model.layers[:4]:
    layer.trainable = False
    
# compile model
model.compile(loss='binary_crossentropy',
          optimizer=optimizers.SGD(lr=0.001, momentum=0.9),
          metrics=['accuracy'])

print ('Model Compiled.')

train_data_dir = 'data'

# data augmentation in the training set
train_datagen = image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=0,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip = True,
        fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=10,
        class_mode='binary',
        save_to_dir='augmentedData',
#	shuffle=False,
        seed=1)

if os.path.exists('augmentedData'):
    shutil.rmtree('augmentedData') 
os.mkdir('augmentedData')

start_time = time.time()
model.fit_generator(
        train_generator,
        epochs=1,
#	shuffle=False,
        steps_per_epoch=5)    # number of training batches in an epoch

#data = train_generator.next()
print(train_generator.class_indices)
#print(data[1])
