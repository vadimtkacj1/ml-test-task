
import os
import sys
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from coefficient import dice_coefficient

seed = 2024
random.seed = seed
np.random.seed = seed
tf.seed = seed

image_size = 768
train_path = "C:\\Users\\Vlad\\Documents\\airbus-ship-detection\\train_v2"

model = tf.keras.models.load_model("unet_model.keras", custom_objects={'dice_coefficient': dice_coefficient})

test_path = 'C:\\Users\\Vlad\\Documents\\airbus-ship-detection\\test_v2'

img_path = os.path.join(test_path, os.listdir(test_path)[8])

def preprocess_input(image_path, target_size=(768, 768)):
    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0 
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

result = model.predict(preprocess_input(img_path), verbose=0)
result=(result > 0.3).astype(float)

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax = fig.add_subplot(1, 2, 1)
plt.imshow(np.squeeze(preprocess_input(img_path), axis=0));


ax = fig.add_subplot(1, 2, 2)
plt.imshow(np.squeeze(result, axis=0));
plt.show()