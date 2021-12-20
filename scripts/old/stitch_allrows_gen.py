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
from scipy import misc
from imageio import mimread


from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift

from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_triangle
from scipy import ndimage as ndi
from collections import Counter
import pandas as pd
import sys
from collections import Counter, OrderedDict

from skimage import data, restoration, filters, morphology
#from ipywidgets import interact
#import getpass
from PIL import Image
from skimage.morphology import label, remove_small_objects
from matplotlib.colors import LinearSegmentedColormap
#, remove_small_objects
from math import sqrt
#from cStringIO import StringIO
from skimage.color import rgb2hed, rgb2gray
from skimage.exposure import rescale_intensity, histogram
from skimage.feature import blob_log
from skimage.draw import circle_perimeter


import sys
import os
import pickle
from skimage import measure
import zarr

image_prefix = sys.argv[1]
well_loc = sys.argv[2]
output_dir = sys.argv[3]

def preprocessing(img):
    img_grey= rgb2gray(img)
    binarized =np.where(img_grey>0.03, 1, 0)
    processed = morphology.binary_dilation(morphology.binary_opening(binarized.max(axis=2).astype(bool),selem=np.ones((8,8))),
                                            selem=np.ones((10,10))).astype(int)
    clean_img=img*processed[:,:,np.newaxis]

    return clean_img

def filter_image(image, idx):
    image_grey= rgb2gray(image)
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
    labeled_blobs_2um_big,_ = ndi.label(mask_image_removed_small_objects)

    return labeled_blobs_2um_big

from skimage.measure import regionprops
import matplotlib as mpl

def plot_labeled_segments(array3d, fix_overlay=True, annotate=True):
    '''
    function for plotting z-projection of labeled segmentations.

    arguments:
        array3d (np.ndarray): input 3D array of labels
        fix_overlay (bool): flag to decide how to treat segments which overlap 
            along z-axis. If this is set to True, the overlapping organoids will
            be shown with a different color. If flag is False, maximum of organoid labels
            along z-axis is presented. (default: True)
    '''
    plt.figure(figsize=(10,10));
    cmap = mpl.colors.ListedColormap(np.random.rand(256,3))

    if fix_overlay:
        tmp = 2**array3d
        tmp.sort(axis=2)
        tmp = tmp[:,:,0] + np.diff(tmp, axis=2).sum(axis=2)
        plt.imshow(np.log2(tmp), cmap=cmap)
    else:
        tmp = array3d.max(axis=2)
        plt.imshow(tmp, cmap=cmap)

    if annotate:
        for props in regionprops(tmp):
            y, x = props.centroid
            if fix_overlay:
                label = np.log2(props.label)
            else:
                label = props.label
            if label == int(label):
                label = int(label)
            if label>0:
                plt.text(x, y, label, fontsize=14, color='k', horizontalalignment='center', verticalalignment='center')

image_2um_f26 = np.load(output_dir+well_loc+'_stitched_row1_gen.npy')
image_2um_f117 = np.load(output_dir+well_loc+'_stitched_row2_gen.npy')
image_2um_f12_15 = np.load(output_dir+well_loc+'_stitched_row3_gen.npy')
image_2um_f20_16 = np.load(output_dir+well_loc+'_stitched_row4_gen.npy')
image_2um_f21_25 = np.load(output_dir+well_loc+'_stitched_row5_gen.npy')

image_2um_f26_seg = np.load(output_dir+well_loc+'_stitched_row1_gen_seg.npy')
image_2um_f117_seg = np.load(output_dir+well_loc+'_stitched_row2_gen_seg.npy')
image_2um_f12_15_seg = np.load(output_dir+well_loc+'_stitched_row3_gen_seg.npy')
image_2um_f20_16_seg = np.load(output_dir+well_loc+'_stitched_row4_gen_seg.npy')
image_2um_f21_25_seg = np.load(output_dir+well_loc+'_stitched_row5_gen_seg.npy')


# stitch column 12
image_2um_f26_shift = image_2um_f26[:-80,:,:]
image_2um_r12 = np.concatenate((image_2um_f26_shift,image_2um_f117),axis=0)

image_2um_f26_shift_seg = image_2um_f26_seg[:-80,:,:]
image_2um_r12_seg = np.concatenate((image_2um_f26_shift_seg,image_2um_f117_seg),axis=0)

#np.save("/projects/chuang-lab/mukasp/test_my_code/images_stitch_dataset2/output/r04c06_stitched_r12.npy",image_2um_r12)
del image_2um_f26
del image_2um_f26_shift
del image_2um_f117

del image_2um_f26_seg
del image_2um_f26_shift_seg
del image_2um_f117_seg
#plt.figure(figsize=(10,17))
#plt.imshow(image_2um_r12.max(axis=2))
#plt.savefig("/projects/chuang-lab/mukasp/test_my_code/images_stitch_dataset2/output/r04c06_stitched_r12.pdf")


# stitch column13
image_2um_r12_shift = image_2um_r12[:-80,:,:]
image_2um_r13 = np.concatenate((image_2um_r12_shift,image_2um_f12_15),axis=0)

image_2um_r12_shift_seg = image_2um_r12_seg[:-80,:,:]
image_2um_r13_seg = np.concatenate((image_2um_r12_shift_seg,image_2um_f12_15_seg),axis=0)

#np.save("/projects/chuang-lab/mukasp/test_my_code/images_stitch_dataset2/output/r04c06_stitched_r13.npy",image_2um_r13) 
del image_2um_r12
del image_2um_r12_shift
del image_2um_f12_15

del image_2um_r12_seg
del image_2um_r12_shift_seg
del image_2um_f12_15_seg

#plt.figure(figsize=(10,17))
#plt.imshow(image_2um_r13.max(axis=2))
#plt.savefig("/projects/chuang-lab/mukasp/test_my_code/images_stitch_dataset2/output/r04c06_stitched_r13.pdf")

# stitch column45
image_2um_f20_16_shift = image_2um_f20_16[:-80,:,:]
image_2um_r45 = np.concatenate((image_2um_f20_16_shift,image_2um_f21_25),axis=0)

image_2um_f20_16_shift_seg = image_2um_f20_16_seg[:-80,:,:]
image_2um_r45_seg = np.concatenate((image_2um_f20_16_shift_seg,image_2um_f21_25_seg),axis=0)

#np.save("/projects/chuang-lab/mukasp/test_my_code/images_stitch_dataset2/output/r04c06_stitched_r45.npy",image_2um_r45)

del image_2um_f20_16
del image_2um_f20_16_shift
del image_2um_f21_25

del image_2um_f20_16_seg
del image_2um_f20_16_shift_seg
del image_2um_f21_25_seg

#plt.figure(figsize=(10,17))
#plt.imshow(image_2um_r45.max(axis=2))
#plt.savefig("/projects/chuang-lab/mukasp/test_my_code/images_stitch_dataset2/output/r04c06_stitched_r45.pdf")

# stitch column15
image_2um_r13_shift = image_2um_r13[:-80,:,:]
image_2um_r15 = np.concatenate((image_2um_r13_shift,image_2um_r45),axis=0)
image_2um_r15_clean = preprocessing(image_2um_r15)
#np.save(output_dir+well_loc+'_stitched_r15_gen_clean.npy',image_2um_r15_clean)
zarr.save(output_dir+well_loc+'_stitched_r15_gen_clean.zarr',image_2um_r15_clean)

image_2um_r13_shift_seg = image_2um_r13_seg[:-80,:,:]
image_2um_r15_seg = np.concatenate((image_2um_r13_shift_seg,image_2um_r45_seg),axis=0)
#image_2um_r15_clean_seg = preprocessing(image_2um_r15_seg)


plt.figure(figsize=(10,17))
plt.imshow((image_2um_r15*2).max(axis=2))
plt.savefig(output_dir+well_loc+'_stitched_r15_gen.pdf')

plt.figure(figsize=(10,17))
plt.imshow((image_2um_r15_clean*2).max(axis=2))
plt.savefig(output_dir+well_loc+'_stitched_r15_gen_clean.pdf')

del image_2um_r15
del image_2um_r15_clean

labeled_blobs_2um_big,_ = ndi.label(image_2um_r15_seg)
#labeled_blobs_2um_big = filter_image(image_2um_r15_clean,100)

#del image_2um_r15_clean

props = measure.regionprops(labeled_blobs_2um_big)

l_coords=[]
for i in range (len(props)):
    coords = props[i].bbox
    l_coords.append(coords)

pickle.dump(l_coords, open(output_dir+well_loc+'_stitched_r15_gen_clean.p', "wb" ) )


#np.save(output_dir+well_loc+'_labeled_blobs_big_clean.npy',labeled_blobs_2um_big)

#def mask_image(image,n):
#    fig_ar_mask_2um = image == n #this number changes over time. it was 48, now it's 42
#    masked_image_2um = image * fig_ar_mask_2um
#
#    return masked_image_2um
#
#ef mask_one_cell_type(image1, image2):

plot_labeled_segments(labeled_blobs_2um_big, fix_overlay=False, annotate=True)
plt.savefig(output_dir+well_loc+'_labeled_numbered_orgaoids_clean.pdf')

one_counter = Counter (labeled_blobs_2um_big.flatten())

class OrderedCounter(Counter, OrderedDict):
     pass
counterlist = OrderedCounter(one_counter)

df = pd.DataFrame(list(counterlist.values()), index=list(counterlist.keys()), columns =['size'])

df['size_microns'] = df['size'].apply(lambda x:x*((6.45*6.45)*5))
writer = pd.ExcelWriter(output_dir+well_loc+'_excel_sheet_clean.xlsx')
df.to_excel(writer,'Sheet1')
writer.save()


plt.figure(figsize=(10,17))
plt.imshow(labeled_blobs_2um_big.max(axis=2), cmap =matplotlib.colors.ListedColormap(np.random.rand(256,3)))
plt.savefig(output_dir+well_loc+'_labeled_orgaoids_clean.pdf')

