
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from coefficient import dice_coefficient
from models.unet import UNet
from DataGenerator import DataGenerator
import pandas as pd

seed = 2024
random.seed = seed
np.random.seed = seed
tf.seed = seed

csv_file = pd.read_csv("C:\\Users\\Vlad\\Documents\\airbus-ship-detection\\train_ship_segmentations_v2.csv")

image_size = 768
train_path = "C:\\Users\\Vlad\\Documents\\airbus-ship-detection\\train_v2"
epochs = 1
batch_size = 2

train_ids = next(os.walk(train_path))[2]

val_data_size = 10

valid_ids = train_ids[:val_data_size]
train_ids = train_ids[val_data_size:]

model = UNet(image_size)

metrics=["accuracy", dice_coefficient]

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss=tf.losses.binary_crossentropy

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.summary()

train_gen = DataGenerator(train_ids, train_path, csv_file, image_size=image_size, batch_size=batch_size)
valid_gen = DataGenerator(valid_ids, train_path, csv_file, image_size=image_size, batch_size=batch_size)

train_steps = len(train_ids) // batch_size
valid_steps = len(valid_ids) // batch_size

model.fit(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, 
                    epochs=epochs)

model.save("unet_model.keras")