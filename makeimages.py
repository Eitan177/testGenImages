from typing import Pattern
import numpy as np
import pandas as pd
import pdb
import gdown
import click
import os
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
import tensorflow as tf
from glob import glob
from skimage.transform import rescale, resize, downscale_local_mean
from keras.models import model_from_json
from PIL import Image
import streamlit as st
import urllib
from os.path import exists
from generate import *

file_exists = exists("network-snapshot-025000.pkl")
if file_exists != True:
    url="https://drive.google.com/u/0/uc?id=1XUYsXdSGVTTaQPR1TujMwRIVn94H6CuG&export=download"
    st.spinner('Downloading our pkl file model')
    gdown.download(url,"network-snapshot-025000.pkl")
make=st.button('make images')

if make:

    for ii in np.arange(1,2):
        generate_images(easing='linear',interpolation='linear',increment=.01,network_pkl='network-snapshot-025000.pkl',process='image',random_seed=0,diameter=100.0,scale_type='pad',seeds=[0,1,2,3],space='z',truncation_psi=1,noise_mode='const',outdir='.',class_idx=ii,size=False,frames=240,fps=24,start=0.0,stop=1.0,projected_w=None)
        ##os.system("python generate.py --outdir=. --seeds="+str(0)+"-"+str(9)+" --class="+str(ii)+' --network=network-snapshot-025000.pkl')


for mm in glob("*.png"):
    print(mm)
    im=Image.open(mm)
    st.image(im)
