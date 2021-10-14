import numpy as np
import pandas as pd
import pdb
import openslide
import os
from glob import glob
import h5py
from tqdm import tqdm
from matplotlib import pyplot as plt
import tensorflow as tf
from glob import glob
from skimage.transform import rescale, resize, downscale_local_mean
from keras.models import model_from_json
from PIL import Image
# standard GPU setup
# get list of avaiable GPUs on system
physical_devices = tf.config.experimental.list_physical_devices('GPU')
use_gpu_idx = np.array([2])
# set dynamic memory growth always
[tf.config.experimental.set_memory_growth(physical_devices[i], True) for i in use_gpu_idx]
# set available GPUs
tf.config.experimental.set_visible_devices([physical_devices[i] for i in use_gpu_idx], 'GPU')
encoder_input_shape = (224, 224, 3)
tile_encoder = tf.keras.applications.DenseNet201(include_top=False, weights='imagenet', pooling='max', input_shape=encoder_input_shape)
tile_encoded_shape = tuple(tile_encoder.outputs[0].shape[1:])

tile_multsize, tile_buffer_cap = list(), 50


for ii in np.arange(1,32):

    print('/home/ehalper2/pathGan/gan/tcga/'+str(ii)+'_*')
    try:
        ppp=glob('/home/ehalper2/pathGan/gan/tcga/'+str(ii)+'_*')
    except:
        print('tried but cannot')

            
            
    ppp=ppp[0]

    hd5_filename = os.path.split(ppp+'_0831.hd5')[1]
    h5 = h5py.File(hd5_filename, mode='w')
    h5.create_dataset(name='tiles_encoded', shape=(0, ) + tile_encoded_shape, chunks=(1, ) + tile_encoded_shape, dtype=np.float32, maxshape=(None, ) + tile_encoded_shape, compression='gzip', compression_opts=4)

    os.chdir('/home/ehalper2/pathGan/gan/stylegan2-ada-pytorch')
    os.system('rm '+ppp+'/*png')
    os.system("python generate.py --outdir="+ppp+" --seeds="+str(0)+"-"+str(9999)+" --class="+str(ii)+' --network=/home/ehalper2/pathGan/gan/stylegan2-ada-pytorch/tcga0_training/00001-tcga_d-cond-paper1024/network-snapshot-025000.pkl')
    print("python generate.py --outdir="+ppp+" --seeds="+str(0)+"-"+str(9999)+" --class="+str(ii)+' --network=/home/ehalper2/pathGan/gan/stylegan2-ada-pytorch/tcga0_training/00001-tcga_d-cond-paper1024/network-snapshot-025000.pkl')
    for img_file in glob(ppp+'/*png'):
        img = Image.open(img_file)
        tile_multsize.append(resize(np.array(img)/(255,255,255),(896,896),anti_aliasing=True))
        if len(tile_multsize) > tile_buffer_cap:

            #h5 = h5py.File(hd5_filename, mode='a')
            h5['tiles_encoded'].resize(
                ((h5['tiles_encoded'].shape[0] + len(tile_multsize)),) + h5['tiles_encoded'].shape[1:])
            tiles = (np.stack(tile_multsize, axis=0)).astype(np.float32)
            # break tiles into 4x4 patches, 16 patches for each 896x896 tile
            patched_tiles=tf.image.extract_patches(tiles,sizes=[1,224,224,1],strides=[1,224,224,1],rates=[1,1,1,1],padding='VALID')
            # reshape the patches to 3d, 16 patches in 16 rows per tile
            stacked_patches=tf.reshape(patched_tiles,(tiles.shape[0]* 16,224,224,3))
            # no resize of patches
            tiles_encoded=tile_encoder(stacked_patches)
            # perform feature sum for every 16 patches, returning to tile dimension output
            feature_sum_16_patches=tf.math.reduce_max(tf.reshape(tiles_encoded,(tiles.shape[0],16,tiles_encoded.shape[-1])),axis=1)


            h5['tiles_encoded'][-len(tile_multsize):] = feature_sum_16_patches.numpy()
            tile_multsize=list()
            print('bunk')
            print(h5['tiles_encoded'].shape)

    h5.close()