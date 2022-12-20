import glob
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('pdf')

#get a list of all masked(annotated) images, use images instead of mask to look at row images
Y = []
for filename in sorted(glob.glob('/projects/chuang-lab/mukasp/cell_seg_project/ChuangLab_organoids_analysis/docs/cellos_annotated_images/mask/*.tif')):
    im=imageio.mimread(filename)
    imarray= np.array(im)
    Y.append(imarray)

#plot and save pdf image
plt.imshow(Y[23].max(axis=0),cmap =matplotlib.colors.ListedColormap(np.random.rand(256,3))); plt.title('mask23 (XY slice)')
plt.savefig("mask23.pdf")