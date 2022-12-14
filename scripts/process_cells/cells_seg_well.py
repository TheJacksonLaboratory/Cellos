# segment cells and save csv file of cells in organoids per well
#imports 
from pathlib import Path
import zarr
import pandas as pd
from stardist.models import StarDist3D
from csbdeep.utils import normalize
from skimage import measure
import configparser
import argparse
import numpy as np

#function definitions
def load_stardist(stardist_path):
    demo_model = False
    if demo_model:
        print (
            "NOTE: This is loading a previously trained demo model!\n"
            "      Please set the variable 'demo_model = False' to load your own trained model.",
            #file=sys.stderr, flush=True
        )
        model = StarDist3D.from_pretrained('3D_demo')
    else:
        model = StarDist3D(None, name='stardist', basedir=Path(stardist_path))

    return model 

def load_images_rois(input_path,well_row,well_col):
    zimage = zarr.open(Path(input_path) / f"r{well_row:02}c{well_col:02}.zarr")
    rois = pd.read_csv(Path(input_path) / f"r{well_row:02}c{well_col:02}organoids.csv")
    return zimage,rois

def apply_stardist(stardist_path, zimage, rois, channel):
    model = load_stardist(stardist_path) 
    cells =pd.DataFrame()

    for row in rois.itertuples():
        organoid = zimage[:,row[5]:row[8],row[6]:row[9],row[7]:row[10]]
        img_org = organoid[int(channel),:,:,:]
        img = normalize(img_org, 1,99.8, axis= (0,1,2))
        labels, details = model.predict_instances(img)
        labels = np.moveaxis(labels, 0, -1)
        image = np.moveaxis(img_org, 0, -1)
        if labels.max() !=0:
            #uncomment to remove planar nuclei
            # for i in range(1,np.max(labels)+1):
            #     zmim=np.sum(np.max(1*(labels==i),axis=(0,1)))
            #     #print(zmim)
            #     if zmim<2:
            #         planar_cells = planar_cells+1
            #         labels[labels==i]=0
            df = pd.DataFrame(measure.regionprops_table(labels,intensity_image=image,
                                                        properties=("label","centroid","bbox", 
                                                                "area", "axis_major_length",
                                                                "axis_minor_length","area_bbox",
                                                                "extent", "area_filled","area_convex", 
                                                                "euler_number","extent", 
                                                                "intensity_max","intensity_mean","intensity_min",
                                                                "inertia_tensor_eigvals", 
                                                                "solidity"))).set_index('label') 
            df['organoids'] = row[1]
            df['fluor'] = channel
            cells = cells.append(df)
    return cells 

def parse_config(config_path):
    parser = configparser.ConfigParser()
    parser.read(config_path)
    input_path = parser['pipeline']['output_path']
    output_path = parser['cells_seg_well']['output_path']
    stardist_path = parser['cells_seg_well']['stardist_path']
    return (input_path,
            output_path,
            stardist_path)


#script 
def main(well_row, well_col, config_file):
    input_path, output_path, stardist_path = parse_config(config_file)
    zimage,rois = load_images_rois(input_path,well_row,well_col)
    red_segmented_cells = apply_stardist(stardist_path,zimage,rois,0 )
    green_segmented_cells = apply_stardist(stardist_path,zimage,rois,1 )
    segmented_cells = pd.concat([red_segmented_cells,green_segmented_cells], axis=0)
    segmented_cells.to_csv(Path(output_path) / f"r{well_row:02}c{well_col:02}cells.csv")
  

if __name__ == "__main__":
    desc = (" load zarr images and csv file of region of interest (organoids)"
            " output information about cells in each organoid.")
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-r', '--row', type=int, help='The row containing the well.')
    parser.add_argument('-c', '--col', type=int, help='The column containing the well.')
    parser.add_argument('config_file', help='Path to the config file'
                                            ' (see documentation).')
    args = parser.parse_args()
    main(args.row, args.col, args.config_file) 