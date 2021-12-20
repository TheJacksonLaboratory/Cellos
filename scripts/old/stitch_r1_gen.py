# coding: utf-8

# In[ ]


import numpy as np
import matplotlib
matplotlib.use('pdf')

import matplotlib.pyplot as plt
#import ipywidgets as widgets
from imageio import mimread
import imageio
import scipy.misc
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_triangle,median
from scipy import ndimage as ndi
from skimage import data, restoration, filters, morphology
from skimage.morphology import label, remove_small_objects
from math import sqrt

import sys
import os

image_prefix = str(sys.argv[1])
well_loc = str(sys.argv[2])
#number_of_stacks = int(sys.argv[2])
output_dir = str(sys.argv[3])
#z_step_value = float(sys.argv[3])

#image_prefix =/projects/liu-lab/ED_Lab_data/phenix_data_pk/pk_96_gfp_mcherry_test2_10292019_20x_5z_nodrug_v2__2019-10-29T15_57_33-Measurement 1/Images/
#well_loc = r02c06
#output_dir = /projects/chuang-lab/mukasp/test_my_code/images_stitch_dataset2/output/
#number_of_stacks =100 (hold on this one)
#image_prefix+'p{:02d}-ch1sk1fk1fl1.tiff'.format(i) 

#load images 
def load_images (directory, idx):
    image = np.zeros([1080,1080,idx])
    for i in range(1, (idx+1)):
        image_1 = directory.format(i)
        single_image_1 = scipy.misc.imread(image_1)  #(1080, 1080, this is a 2D array of each slice
        image[:, :, i-1] = single_image_1 #for the first looping i will get the 2d
    return image

def add_2channels(image1, image2, idx):
    image = np.zeros((1080, 1080 ,idx))
    image[:,:,:,0] = image1/image1.max() #changed btn 0-1
    image[:,:,:,1] = image2/image2.max()  #multiplied by 2 bcse green color is faint
    return image

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
    
    #image_2um_blur = median(image_grey)

    #threshold triangle 
    mask_triangle_2um = image_2um_blur > threshold_triangle(image_2um_blur.flatten())
    #Remove small objects
    mask_image_removed_small_objects = remove_small_objects(mask_triangle_2um, min_size=8000)
    #label 
#    labeled_blobs_2um_big,_ = ndi.label(mask_image_removed_small_objects)

    return mask_image_removed_small_objects

#fx for plotting
image_2um_f2 = load_images (image_prefix+well_loc+'f02p{:02d}-ch3sk1fk1fl1.tiff', 100)
#image_2um_2_f2= load_images (image_prefix+well_loc+'f02p{:02d}-ch1sk1fk1fl1.tiff', 100)
#image_2um_f2 = add_2channels(image_2um_1_f2, image_2um_2_f2, 100)
image_2um_f2_seg = preprocessing(image_2um_f2)
image_2um_f2_seg = filter_image(image_2um_f2_seg)

image_2um_f3 = load_images (image_prefix+well_loc+'f03p{:02d}-ch3sk1fk1fl1.tiff', 100)
#image_2um_2_f3 = load_images (image_prefix+well_loc+'f03p{:02d}-ch1sk1fk1fl1.tiff', 100)
#image_2um_f3 = add_2channels(image_2um_1_f3, image_2um_2_f3, 100)
image_2um_f3_seg = preprocessing(image_2um_f3)
image_2um_f3_seg = filter_image(image_2um_f3_seg)

image_2um_f4 = load_images (image_prefix+well_loc+'f04p{:02d}-ch3sk1fk1fl1.tiff', 100)
#image_2um_2_f4 = load_images (image_prefix+well_loc+'f04p{:02d}-ch1sk1fk1fl1.tiff', 100)
#image_2um_f4 = add_2channels(image_2um_1_f4, image_2um_2_f4, 100)
image_2um_f4_seg = preprocessing(image_2um_f4)
image_2um_f4_seg = filter_image(image_2um_f4_seg)

image_2um_f5 = load_images (image_prefix+well_loc+'f05p{:02d}-ch3sk1fk1fl1.tiff', 100)
#image_2um_2_f5 = load_images (image_prefix+well_loc+'f05p{:02d}-ch1sk1fk1fl1.tiff', 100)
#image_2um_f5 = add_2channels(image_2um_1_f5, image_2um_2_f5, 100)
image_2um_f5_seg = preprocessing(image_2um_f5)
image_2um_f5_seg = filter_image(image_2um_f5_seg)

image_2um_f6= load_images (image_prefix+well_loc+'f06p{:02d}-ch3sk1fk1fl1.tiff', 100)
#image_2um_2_f6= load_images (image_prefix+well_loc+'f06p{:02d}-ch1sk1fk1fl1.tiff', 100)
#image_2um_f6 = add_2channels(image_2um_1_f6, image_2um_2_f6, 100)
image_2um_f6_seg = preprocessing(image_2um_f6)
image_2um_f6_seg = filter_image(image_2um_f6_seg)

def stitch_row (img1, img2):
    img1_shift = img1[:,:-80,:]
    concat = np.concatenate((img1_shift,img2), axis =1)
    return concat

def stitch_row_seg (img1, img2):
    img1_shift = img1[:,:-80,:]
    concat = np.concatenate((img1_shift,img2), axis =1)
    return concat

image_2um_f45 = stitch_row(image_2um_f4,image_2um_f5)
image_2um_f46 = stitch_row(image_2um_f45, image_2um_f6)
image_2um_f23 = stitch_row(image_2um_f2, image_2um_f3)
image_2um_f26 = stitch_row(image_2um_f23, image_2um_f46)
np.save(output_dir+well_loc+'_stitched_row1_gen.npy',image_2um_f26)

image_2um_f45_seg = stitch_row_seg(image_2um_f4_seg,image_2um_f5_seg)
image_2um_f46_seg = stitch_row_seg(image_2um_f45_seg, image_2um_f6_seg)
image_2um_f23_seg = stitch_row_seg(image_2um_f2_seg, image_2um_f3_seg)
image_2um_f26_seg = stitch_row_seg(image_2um_f23_seg, image_2um_f46_seg)
np.save(output_dir+well_loc+'_stitched_row1_gen_seg.npy',image_2um_f26_seg)

#plt.figure(figsize=(10,17))
#plt.imshow(image_2um_f26.max(axis=2))
#plt.savefig(output_dir+well_loc+'_stitched_f26_gen_test2.pdf')
