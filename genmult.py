import os
import gc
import pandas as pd
import tensorflow as tf
from glob import glob
import numpy as np
import h5py
import multiprocessing
import pdb
from skimage.transform import rescale, resize, downscale_local_mean
from keras.models import model_from_json
from PIL import Image
from numba import cuda
physical_devices = tf.config.experimental.list_physical_devices('GPU')
use_gpu_idx = np.array([4])
print(use_gpu_idx)
# set dynamic memory growth always
[tf.config.experimental.set_memory_growth(physical_devices[i], True) for i in use_gpu_idx]
# set available GPUs
tf.config.experimental.set_visible_devices([physical_devices[i] for i in use_gpu_idx], 'GPU')

encoder_input_shape = (224, 224, 3)
tile_encoder = tf.keras.applications.DenseNet201(include_top=False, weights='imagenet', pooling='max', input_shape=encoder_input_shape)

def attempt_encode():
    tiles_encoded=tile_encoder(stacked_patches)

tile_encoded_shape = tuple(tile_encoder.outputs[0].shape[1:])

def worker(procnum, stacked_patches,return_dict):
    """worker function"""
    print(str(procnum) + " represent!")
    tiles_encoded=tile_encoder(stacked_patches)
    
    feature_sum_16_patches=tf.math.reduce_max(tf.reshape(tiles_encoded,(tile_use.shape[0],16,tiles_encoded.shape[-1])),axis=1)
    return_dict[procnum] = feature_sum_16_patches


os.chdir('/home/ehalper2/pathGan/gan/stylegan2-ada-pytorch')

class TileLoader:
    def __init__(self):
        pass
        
    def __call__(self, filename):
        tiles_shape = [None, None]
        labels_shape = [None, ]
        tiles, labels = tf.py_function(lambda x: TileLoader.fetch_tiles(x, tiles_shape, labels_shape), inp=[filename], Tout=[tf.float32,  tf.int32])
        
        tiles.set_shape(tiles_shape)
        labels.set_shape(labels_shape)
        
        return tiles, labels

    @staticmethod
    def fetch_tiles(ii, tiles_shape, labels_shape):
        
        ii=np.int(ii)
    #for ii in np.arange(0,32):
        print('/home/ehalper2/pathGan/gan/tcga/'+str(ii)+'_*')
        try:
            ppp=glob('/home/ehalper2/pathGan/gan/tcga/'+str(ii)+'_*')
        except:
            print('tried but cannot')

        #ppp=glob,glob('/home/ehalper2/pathGan/gan/tcga/'+str(ii)+'_*') 
        
        ppp=ppp[0]
        tile_multsize=[]
        tiles=[]
        r_ind = np.int(np.random.choice(5000000,1,replace=False))
        print(r_ind)
        print(ppp)
        os.system("python generate.py --outdir="+ppp+" --seeds="+str(r_ind)+"-"+str(r_ind+49)+" --class="+str(ii)+' --network=/home/ehalper2/pathGan/gan/stylegan2-ada-pytorch/tcga0_training/00001-tcga_d-cond-paper1024/network-snapshot-025000.pkl')
        print("python generate.py --outdir="+ppp+" --seeds="+str(r_ind)+"-"+str(r_ind+49)+" --class="+str(ii)+' --network=/home/ehalper2/pathGan/gan/stylegan2-ada-pytorch/tcga0_training/00001-tcga_d-cond-paper1024/network-snapshot-025000.pkl')
        
        for img_file in glob(ppp+'/*png'):
            
            img = Image.open(img_file)
            tile_multsize.append(resize(np.array(img)/(255,255,255),(896,896),anti_aliasing=True))
           
        os.system('rm '+ppp+'/*png')
        tiles = (np.stack([_ for _ in tile_multsize], axis=0) / np.array([1, 1, 1]) [np.newaxis, np.newaxis, np.newaxis, :]).astype(np.float32)    
        tile_multsize=list()
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []   
        for tt in np.arange(0,tiles.shape[0]):
              
            tile_use=tf.expand_dims(tiles[tt,:],axis=0)
            patched_tiles=tf.image.extract_patches(tile_use,sizes=[1,224,224,1],strides=[1,224,224,1],rates=[1,1,1,1],padding='VALID')
            # reshape the patches to 3d, 16 patches in 16 rows per tile
                        
            stacked_patches=tf.reshape(patched_tiles,(tile_use.shape[0]* 16,224,224,3))

            # no resize of patches
            #pdb.set_trace()  
            print('tt is '+str(tt))
            #ppp0 = multiprocessing.Process(target=worker, args=(tt, stacked_patches,return_dict))
            #print(tt)
            #jobs.append(ppp0)
            #ppp0.start() 
            #multiprocessing.Process(target=attempt_encode)
            
               
            tiles_encoded=tile_encoder(stacked_patches)
              
            # perform feature sum for every 16 patches, returning to tile dimension output
            feature_sum_16_patches=tf.math.reduce_max(tf.reshape(tiles_encoded,(tile_use.shape[0],16,tiles_encoded.shape[-1])),axis=1)
            #tf.keras.backend.clear_session()
            #gc.collect()
            pdb.set_trace()
            

            #for proc in jobs:

            #proc.join()
            #print('asedrwer')
            
            #feature_sum_conc=np.vstack(return_dict.values())    
            
            if tt==0:
               feature_sum_conc= feature_sum_16_patches.numpy()
            else:
                feature_sum_conc = np.vstack((feature_sum_conc,feature_sum_16_patches.numpy()))
        tiles=list()
        return feature_sum_conc, np.repeat(ii,feature_sum_conc.shape[0])    
ds=tf.data.Dataset.from_tensor_slices(np.arange(0,32))

ds_train = ds.map(lambda x: TileLoader()(x), num_parallel_calls=1)

ds_train = ds_train.unbatch().shuffle(buffer_size=32*50, reshuffle_each_iteration=True) 


ds_test = ds.map(lambda x: TileLoader()(x), num_parallel_calls=32)
ds_test = ds_test.unbatch().shuffle(buffer_size=32*50, reshuffle_each_iteration=True) 

ds_train=ds_train.batch(500, drop_remainder=True).repeat().prefetch(1)  
ds_test=ds_test.batch(500, drop_remainder=True).repeat().prefetch(1) 
 
model2 = tf.keras.models.Sequential([tf.keras.layers.Input(shape=[1920], dtype=np.float32),
                                  tf.keras.layers.Dropout(rate=0.2),
                                  tf.keras.layers.Dense(units=256, activation=tf.keras.activations.softplus),
                                  tf.keras.layers.Dropout(rate=0.2),
                                  tf.keras.layers.Dense(units=128, activation=tf.keras.activations.softplus),
                                  tf.keras.layers.Dense(units=32, activation=tf.keras.activations.softmax)])
model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),# loss=[tf.keras.losses.MeanSquaredError()])       
    loss=[tf.keras.losses.sparse_categorical_crossentropy], metrics=['mae', 'mse','accuracy'])
callbacks=tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=100, verbose=0,
    mode='min', baseline=None, restore_best_weights=True
)
model2.fit(ds_train, validation_data=ds_test, callbacks=callbacks, epochs=100,steps_per_epoch=20,validation_steps=5)