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
import sys
from skimage.transform import rescale, resize, downscale_local_mean
from keras.models import model_from_json
from PIL import Image
import streamlit as st
import urllib
from os.path import exists
import time
from generate import *


file_exists = exists("network-snapshot-025000.pkl")
st.write('the network file exists: '+str(file_exists))
tcga_w_annot=pd.read_csv('tcga_labels_to_num.csv')
if file_exists != True:
    url="https://drive.google.com/u/0/uc?id=1XUYsXdSGVTTaQPR1TujMwRIVn94H6CuG&export=download"
    st.spinner('Downloading our pkl file model')
    gdown.download(url,"network-snapshot-025000.pkl")
labelmake=st.selectbox('what type of images do you want to generate',tcga_w_annot['type'].tolist())
nummake=int(tcga_w_annot['type_num'][tcga_w_annot['type']==labelmake])
num=st.select_slider('images to show',[1,2,3,4,5,6,7,8])
seeds =[int(ii) for ii in np.absolute(np.random.randn(num))*100]
generate_images(easing='linear',interpolation='linear',increment=.01,network_pkl='network-snapshot-025000.pkl',process='image',random_seed=0,diameter=100.0,scale_type='pad',seeds=seeds,space='z',truncation_psi=1,noise_mode='const',outdir='.',class_idx=nummake,size=False,frames=240,fps=24,start=0.0,stop=1.0,projected_w=None)
        ##os.system("python generate.py --outdir=. --seeds="+str(0)+"-"+str(9)+" --class="+str(ii)+' --network=network-snapshot-025000.pkl')

st.clear_cache()
for mm in glob("*.png"):
    print(mm)
    im=Image.open(mm)
    st.image(im)
os.system('rm *png')
sys.modules[__name__].__dict__.clear()
