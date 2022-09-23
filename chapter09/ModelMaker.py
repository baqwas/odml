#!/usr/bin/env python3
# ModelMaker.py
# odmlbook/BookSource/Chapter09/Chapter9_ModelMaker.py
# lmoroney
# cosmetic changes only by armw
# prerequisite packages using pip
# it will take a long time to install even silently
# python3 -m pip install -q tflite-model-maker
# neccessary packages
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2')
from tflite_model_maker import configs
from tflite_model_maker import ExportFormat
from tflite_model_maker import image_classifier
from tflite_model_maker import ImageClassifierDataLoader
import matplotlib.pyplot as plt
import os
# download pre-existing images
image_path = tf.keras.utils.get_file('flower_photos.tgz', 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')
# inform user on download folder path
print(image_path)
# review directory listing
os.listdir(image_path)
# prepare dataset
data = ImageClassifierDataLoader.from_folder(image_path)
# split dataset in training and testing sub-sets
train_data, test_data = data.split(0.9)
# classify training data
model = image_classifier.create(train_data)
# evaluate model
loss, accuracy = model.evaluate(test_data)
# export model
model.export(export_dir='/mm_flowers/')
# for iOS, labels need to added to the model
model.export(export_dir='/mm_flowers/', export_format = [ExportFormat.LABEL])
