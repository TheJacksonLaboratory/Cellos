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

#global variable definititon(all upper case variables)
STARDIST_PATH = '/projects/chuang-lab/mukasp/cell_seg_project/code/models'

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

def load_images(input_path,well_row,well_col,channel):
    image = zarr.open(Path(input_path) / f"r{well_row:02}c{well_col:02}.zarr")
    rois = pd.read_csv(Path(input_path) / f"r{well_row:02}c{well_col:02}organoids.csv")
    model = load_stardist(STARDIST_PATH) 
    cells =pd.DataFrame()
    for row in rois.itertuples():
        organoid = image[:,row[2]:row[5],row[3]:row[6],row[4]:row[7]]
        img = normalize(organoid, 1,99.8, axis= (0,1,2))
        labels, details = model.predict_instances(img[int(channel),:,:,:])
        #labels_green, details_green = model.predict_instances(img[0,:,:,:])
        if labels.max() != 0:
            df = pd.DataFrame(measure.regionprops_table(labels, 
                                                        properties=('label','centroid', 
                                                                'area', 'major_axis_length',
                                                                'minor_axis_length'))).set_index('label')
            df['organoids'] = row[1]
            df['fluor'] = channel
            cells = cells.append(df)
    return cells 

def parse_config(config_path):
    parser = configparser.ConfigParser()
    parser.read(config_path)
    input_path = parser['pipeline']['output_path']
    output_path = parser['cells_seg_well']['cells_path']
    return (input_path,
            output_path)


#script 
def main(well_row, well_col, config_file):
    input_path, output_path = parse_config(config_file)
    red_segmented_cells = load_images(input_path,well_row,well_col,0)
    green_segmented_cells = load_images(input_path,well_row,well_col,1)
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