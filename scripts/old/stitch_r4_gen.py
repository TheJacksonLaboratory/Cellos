#!/usr/bin/env python
#PBS -l nodes=1:ppn=4,walltime=24:00:00
#PBS -l mem=128GB
#PBS -N cells_segmentation

# coding: utf-8

# In[ ]


import numpy as np
import matplotlib
matplotlib.use('pdf')

import matplotlib.pyplot as plt
#import ipywidgets as widgets
import scipy.misc
from imageio import mimread
import imageio

from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift

from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_triangle
from scipy import ndimage as ndi
from skimage import data, restoration, filters, morphology
from skimage.morphology import label, remove_small_objects
from math import sqrt

import sys
import os

image_prefix = sys.argv[1]
well_loc = sys.argv[2]
output_dir = sys.argv[3]

#load images 
def load_images (directory, idx):
    image = np.zeros([1080,1080,idx])
    for i in range(1, (idx+1)):
        image_1 = directory.format(i)
        single_image_1 = scipy.misc.imread(image_1)  #(1080, 1080, this is a 2D array of each slice
        image[:, :, i-1] = single_image_1 #for the first looping i will get the 2d
    return image

def add_2channels(image1, image2, idx):
    image = np.zeros((1080, 1080 ,idx ,3))
    image[:,:,:,0] = image1/image1.max() #changed btn 0-1
    image[:,:,:,1] = image2/image2.max()  #multiplied by 2 bcse green color is faint
    return image

def stitch_row (img1, img2):
    img1_shift = img1[:,:-80,:]
    concat = np.concatenate((img1_shift,img2), axis =1)
    return concat 

def stitch_row_seg (img1, img2):
    img1_shift = img1[:,:-80,:]
    concat = np.concatenate((img1_shift,img2), axis =1)
    return concat

def preprocessing(img):
    img_grey= rgb2gray(img)
    binarized =np.where(img_grey>0.03, 1, 0)
    processed = morphology.binary_dilation(morphology.binary_opening(binarized.max(axis=2).astype(bool),selem=np.ones((8,8))),
                                            selem=np.ones((10,10))).astype(int)
    clean_img=img*processed[:,:,np.newaxis]

    return clean_img

def filter_image(image):
    #image_grey= rgb2gray(image)
    #image_2um_blur = np.zeros([1080, 1080,idx,3])
    image_2um_blur = np.zeros(image.shape)
    #sigma is the rayon of a triangle you are taking in this method, the bigger the sigma you take big objects
    image_2um_blur = gaussian(image_grey, sigma=0.5, mode = 'reflect',
                              multichannel=True , preserve_range=True) #, sigma=10
    #threshold triangle 
    mask_triangle_2um = image_2um_blur > threshold_triangle(image_2um_blur.flatten())
    #Remove small objects
    mask_image_removed_small_objects = remove_small_objects(mask_triangle_2um, min_size=8000)
    #label 
#    labeled_blobs_2um_big,_ = ndi.label(mask_image_removed_small_objects)

    return mask_image_removed_small_objects


image_2um_f16 = load_images (image_prefix+well_loc+'f16p{:02d}-ch3sk1fk1fl1.tiff', 100)
#image_2um_2_f16 = load_images (image_prefix+well_loc+'f16p{:02d}-ch1sk1fk1fl1.tiff', 100)
#image_2um_f16 = add_2channels(image_2um_1_f16, image_2um_2_f16, 100)
image_2um_f16_seg = preprocessing(image_2um_f16)
image_2um_f16_seg = filter_image(image_2um_f16_seg)

image_2um_f17 = load_images (image_prefix+well_loc+'f17p{:02d}-ch3sk1fk1fl1.tiff', 100)
#image_2um_2_f17 = load_images (image_prefix+well_loc+'f17p{:02d}-ch1sk1fk1fl1.tiff', 100)
#image_2um_f17 = add_2channels(image_2um_1_f17, image_2um_2_f17, 100)
image_2um_f17_seg = preprocessing(image_2um_f17)
image_2um_f17_seg = filter_image(image_2um_f17_seg)

image_2um_f18 = load_images (image_prefix+well_loc+'f18p{:02d}-ch3sk1fk1fl1.tiff', 100)
#image_2um_2_f18 = load_images (image_prefix+well_loc+'f18p{:02d}-ch1sk1fk1fl1.tiff', 100)
#image_2um_f18 = add_2channels(image_2um_1_f18, image_2um_2_f18, 100)
image_2um_f18_seg = preprocessing(image_2um_f18)
image_2um_f18_seg = filter_image(image_2um_f18_seg)

image_2um_f19 = load_images (image_prefix+well_loc+'f19p{:02d}-ch3sk1fk1fl1.tiff', 100)
i#mage_2um_2_f19 = load_images (image_prefix+well_loc+'f19p{:02d}-ch1sk1fk1fl1.tiff', 100)
#image_2um_f19 = add_2channels(image_2um_1_f19, image_2um_2_f19, 100)
image_2um_f19_seg = preprocessing(image_2um_f19)
image_2um_f19_seg = filter_image(image_2um_f19_seg)

image_2um_f20 = load_images (image_prefix+well_loc+'f20p{:02d}-ch3sk1fk1fl1.tiff', 100)
#image_2um_2_f20 = load_images (image_prefix+well_loc+'f20p{:02d}-ch1sk1fk1fl1.tiff', 100)
#image_2um_f20 = add_2channels(image_2um_1_f20, image_2um_2_f20, 100)
image_2um_f20_seg = preprocessing(image_2um_f20)
image_2um_f20_seg = filter_image(image_2um_f20_seg)

# row 4
image_2um_f18_17 = stitch_row(image_2um_f18,image_2um_f17)
image_2um_f18_16 = stitch_row(image_2um_f18_17, image_2um_f16)
image_2um_f20_19 = stitch_row(image_2um_f20, image_2um_f19)
image_2um_f20_16 = stitch_row(image_2um_f20_19, image_2um_f18_16)
np.save(output_dir+well_loc+'_stitched_row4_gen.npy',image_2um_f20_16)

image_2um_f18_17_seg = stitch_row_seg(image_2um_f18_seg,image_2um_f17_seg)
image_2um_f18_16_seg = stitch_row_seg(image_2um_f18_17_seg, image_2um_f16_seg)
image_2um_f20_19_seg = stitch_row_seg(image_2um_f20_seg, image_2um_f19_seg)
image_2um_f20_16_seg = stitch_row_seg(image_2um_f20_19_seg, image_2um_f18_16_seg)
np.save(output_dir+well_loc+'_stitched_row4_gen_seg.npy',image_2um_f20_16_seg)

#plt.figure(figsize=(10,17))
#plt.imshow(image_2um_f20_16.max(axis=2))
#plt.savefig("/projects/chuang-lab/mukasp/test_my_code/images_stitch_dataset2/output/r04c06_stitched_f20_16.pdf")

