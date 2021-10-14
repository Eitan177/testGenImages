from typing import Pattern
import numpy as np
import pandas as pd
import pdb

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

file_exists = exists("network-snapshot-025000.pkl")

if file_exists != True:
    url="https://livejohnshopkins-my.sharepoint.com/personal/ehalper2_jh_edu/Documents/network-snapshot-025000.pkl"
    urllib.request.urlretrieve(url,"network-snapshot-025000.pkl")
make=st.button('make images')


if make:

    for ii in np.arange(1,2):
        os.system("python generate.py --outdir=. --seeds="+str(0)+"-"+str(9)+" --class="+str(ii)+' --network=network-snapshot-025000.pkl')


for mm in glob("*.png"):
    print(mm)
    im=Image.open(mm)
    st.image(im)
