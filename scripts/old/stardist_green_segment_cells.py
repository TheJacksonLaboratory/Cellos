# coding: utf-8

# In[ ]

from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import numpy as np
import matplotlib
matplotlib.use('pdf')
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist import Rays_GoldenSpiral
from stardist.matching import matching, matching_dataset
from stardist.models import Config3D, StarDist3D, StarDistData3D

np.random.seed(42)
lbl_cmap = random_label_cmap()

import sys
import os
import skimage as sk
import zarr
import pickle
from skimage import measure

file_name = sys.argv[1]
#image_prefix = sys.argv[1]
organID = int(sys.argv[2])
#output_dir = sys.argv[3]
input_dir = sys.argv[3]
output_dir = sys.argv[4]
#input_dir = /projects/chuang-lab/mukasp/test_my_code/images_stitch_dataset2/day2/output/
#output_dir = /projects/chuang-lab/mukasp/test_my_code/images_stitch_dataset2/day2/cells_seg/output/

l_coords = pickle.load( open(input_dir+file_name+'_stitched_r15_gen_clean.p', 'rb'))

image_2um_r15 = zarr.open(input_dir+file_name+'_stitched_r15_gen_clean.zarr')

axis_norm = (0,1,2) 

demo_model = False

if demo_model:
    print (
        "NOTE: This is loading a previously trained demo model!\n"
        "      Please set the variable 'demo_model = False' to load your own trained model.",
        file=sys.stderr, flush=True
    )
    model = StarDist3D.from_pretrained('3D_demo')
else:
    model = StarDist3D(None, name='stardist', basedir='/projects/chuang-lab/mukasp/cell_seg_project/code/models')
None;

IO1BC=image_2um_r15[l_coords[organID][0]:l_coords[organID][3],l_coords[organID][1]:l_coords[organID][4],:,1]

plt.figure(figsize=(5,5))
plt.imshow(IO1BC.max(axis=2), cmap ='gray')
plt.savefig(output_dir+file_name+'green_organoid{:d}.pdf'.format(organID))

one_orgaoid_blue_65_s = np.swapaxes(IO1BC,0,2)
one_orgaoid_blue_d5_42 = np.swapaxes(one_orgaoid_blue_65_s,1,2)

img = normalize(one_orgaoid_blue_d5_42, 1,99.8, axis=axis_norm)
labels, details = model.predict_instances(img)

plt.figure(figsize=(5,5))
plt.imshow(labels.max(axis=0), cmap =lbl_cmap)
plt.savefig(output_dir+file_name+'green_segment_cells{:d}.pdf'.format(organID))

#from collections import Counter, OrderedDict
import pandas as pd

#def get_dataframe(image):
#    one_counter = Counter(image.flatten())
#    class OrderedCounter(Counter, OrderedDict):
#        pass
#    counterlist = OrderedCounter(one_counter)
#    df = pd.DataFrame(list(counterlist.values()), index=list(counterlist.keys()), columns =['size'])
#    df_2 = df.reset_index()
#    df_2 = df_2.rename(columns={"index": "cells"})
#    df_2 = df_2.sort_values(by=['cells'])
#    df_2 = df_2.set_index('cells')

#    return df_2

#df_2 = get_dataframe(labels)
#df_2['organoids']=organID
#df_2['fluor']='GFP'

df =pd.DataFrame(measure.regionprops_table(labels, properties=('label','centroid', 'area','major_axis_length','minor_axis_length'))).set_index('label')
#df = pd.DataFrame(probs_red)
df['organoids']=organID
df['fluor']='GFP'

df.to_csv(output_dir+file_name+'_green_cells{:d}.csv'.format(organID))







