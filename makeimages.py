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
import generate

file_exists = exists("network-snapshot-025000.pkl")
if file_exists != True:
    url="https://drive.google.com/u/0/uc?id=1XUYsXdSGVTTaQPR1TujMwRIVn94H6CuG&export=download"
    st.spinner('Downloading our pkl file model')
    gdown.download(url,"network-snapshot-025000.pkl")
make=st.button('make images')


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--diameter', type=float, help='diameter of loops', default=100.0, show_default=True)
@click.option('--frames', type=int, help='how many frames to produce (with seeds this is frames between each step, with loops this is total length)', default=240, show_default=True)
@click.option('--fps', type=int, help='framerate for video', default=24, show_default=True)
@click.option('--increment', type=float, help='truncation increment value', default=0.01, show_default=True)
@click.option('--interpolation', type=click.Choice(['linear', 'slerp', 'noiseloop', 'circularloop']), default='linear', help='interpolation type', required=True)
@click.option('--easing',
              type=click.Choice(['linear', 'easeInOutQuad', 'bounceEaseOut','circularEaseOut','circularEaseOut2']),
              default='linear', help='easing method', required=True)
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--process', type=click.Choice(['image', 'interpolation','truncation','interpolation-truncation']), default='image', help='generation method', required=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--random_seed', type=int, help='random seed value (used in noise and circular loop)', default=0, show_default=True)
@click.option('--scale-type',
                type=click.Choice(['pad', 'padside', 'symm','symmside']),
                default='pad', help='scaling method for --size', required=False)
@click.option('--size', type=size_range, help='size of output (in format x-y)')
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--space', type=click.Choice(['z', 'w']), default='z', help='latent space', required=True)
@click.option('--start', type=float, help='starting truncation value', default=0.0, show_default=True)
@click.option('--stop', type=float, help='stopping truncation value', default=1.0, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)

if make:

    for ii in np.arange(1,2):
        generate_images('linear',.01,'network-snapshot-025000.pkl','image',0,100.0,'pad',seeds=[0,1,2,3],space='z',truncation_psi=1,noise_mode='const',outdir='.',class_idx=ii)
        ##os.system("python generate.py --outdir=. --seeds="+str(0)+"-"+str(9)+" --class="+str(ii)+' --network=network-snapshot-025000.pkl')


for mm in glob("*.png"):
    print(mm)
    im=Image.open(mm)
    st.image(im)
