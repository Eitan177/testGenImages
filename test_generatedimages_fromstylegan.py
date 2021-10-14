import pandas as pd
import tensorflow as tf
from glob import glob
import os
import numpy as np
import re
import h5py
import pdb
from keras.models import model_from_json

physical_devices = tf.config.experimental.list_physical_devices('GPU')
use_gpu_idx = np.array([4])
# set dynamic memory growth always
[tf.config.experimental.set_memory_growth(physical_devices[i], True) for i in use_gpu_idx]
# set available GPUs
tf.config.experimental.set_visible_devices([physical_devices[i] for i in use_gpu_idx], 'GPU')

os.chdir('/home/ehalper2/pathGan/gan/stylegan2-ada-pytorch/')
files=glob('[0-9]*_[A-Z]*_0831.hd5')
class TileLoader:
    def __init__(self):
        pass
        
    def __call__(self, filename, n):
        tiles_shape = [None, None]
        labels_shape = [None, ]

        tiles, labels = tf.py_function(lambda x: TileLoader.fetch_tiles(x, n, tiles_shape, labels_shape), inp=[filename], Tout=[tf.float32,  tf.int32])
        
        tiles.set_shape(tiles_shape)
        labels.set_shape(labels_shape)
        
        return tiles, labels
        
    @staticmethod
    def fetch_tiles(hd5_file, n, tiles_shape, labels_shape):
        
        h5_obj = h5py.File(hd5_file.numpy(),'r')
        nr=h5_obj['tiles_encoded'].shape[0]
        
        inds = np.sort(np.random.choice(nr, n, replace=False))
        tiles_all= h5_obj['tiles_encoded'][inds,:]

        diagnoses = np.repeat(np.int(re.sub('.+"','',re.sub('_.+','',str(h5_obj)))),n)
        return tiles_all, diagnoses  
 
ds = tf.data.Dataset.from_tensor_slices(files)
for ddd in ds:
  print(ddd)

ds_train = ds.map(lambda x: TileLoader()(x, n=36), num_parallel_calls=len(files))

ds_train = ds_train.unbatch().shuffle(buffer_size=len(files)*36, reshuffle_each_iteration=True) 

ds_test = ds.map(lambda x: TileLoader()(x, n=36), num_parallel_calls=len(files))
ds_test = ds_test.unbatch().shuffle(buffer_size=len(files)*36, reshuffle_each_iteration=True) 

ds_train=ds_train.batch(500, drop_remainder=True).repeat().prefetch(1)  
ds_test=ds_test.batch(500, drop_remainder=True).repeat().prefetch(1)   
 
model2 = tf.keras.models.Sequential([tf.keras.layers.Input(shape=[1920], dtype=np.float32),
                                  tf.keras.layers.Dropout(rate=0.2),
                                  tf.keras.layers.Dense(units=256, activation=tf.keras.activations.softplus),
                                  tf.keras.layers.Dropout(rate=0.2),
                                  tf.keras.layers.Dense(units=128, activation=tf.keras.activations.softplus),
                                  tf.keras.layers.Dense(units=33, activation=tf.keras.activations.softmax)])
model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),# loss=[tf.keras.losses.MeanSquaredError()])       
    loss=[tf.keras.losses.sparse_categorical_crossentropy], metrics=['mae', 'mse','accuracy'])
callbacks=tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=100, verbose=0,
    mode='min', baseline=None, restore_best_weights=True
)
model2.fit(ds_train, validation_data=ds_test, callbacks=callbacks, epochs=100,steps_per_epoch=20,validation_steps=5)




model_json = model2.to_json()
with open("1007stylegan_model_tcga_16max.json", "w") as json_file_896:
    json_file_896.write(model_json)
# serialize weights to HDF5
model2.save_weights('1007stylegan_model_tcga_16max.h5')

import matplotlib.pyplot as plt
plt.subplots(1,1)
plt.plot(model2.history.history['loss'],label='train')
plt.plot(model2.history.history['val_loss'],label='validation')
plt.legend()
plt.show()