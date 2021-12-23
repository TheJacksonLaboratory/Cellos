#Create a stitched well from the original files and save as zarr
# imports
import re
import zarr
import numpy as np
import xml.etree.ElementTree as ET
from numcodecs import Blosc
from pathlib import Path

#global variable definition(all upper case variables)
LAYOUT_25 = np.array([[ 2, 3, 4, 5, 6],
                      [11,10, 9, 8, 7],
                      [12,13, 1,14,15],
                      [20,19,18,17,16],
                      [21,22,23,24,25]])
PLANE_SIZE = (1080, 1080)
OVERLAP_X = 80
OVERLAP_Y = 80
WELL_ROW = 2
WELL_COLUMN = 2
PLATE_PATH = "/projects/liu-lab/ED_Lab_data/phenix_data_pk/pooja_96_gfp_mcherry_test11_10082021_20x_drug_d4_plate1__2021-10-08T13_09_49-Measurement 1/Images/"

# function definitions
def load_images(well,plate_path):
    return True


def parse_index_file(index_file_path, well_row, well_col):
    """Parse Index.idx.xml

    args:
        index_file_path : string or pathlike; Path to index file for plate
    returns:
        nfield : number of fields in well
        nplane : number of planes per image
        nchannel : number of channels per image
    """

    indexp = Path(index_file_path)
    tree = ET.parse(indexp)
    root = tree.getroot()
    wells = root.find("{http://www.perkinelmer.com/PEHH/HarmonyV5}Wells")
    target_well = None
    for well in wells:
        row = int(well.find("{http://www.perkinelmer.com/PEHH/HarmonyV5}Row").text)
        col = int(well.find("{http://www.perkinelmer.com/PEHH/HarmonyV5}Col").text)
        if row == well_row and col == well_col:
            target_well = well
    if not target_well:
        raise ValueError("This plate does not contain the indicated well")
    else:
        image_id_list = [image.attrib.get('id', None) for image in target_well]

    fields_list = []
    planes_list = []
    channels_list = []

    for image_id in image_id_list:
        if image_id:
            plane_match = re.search('(?<=P)[0-9]+', image_id)
            planes_list.append(int(plane_match.group(0)))

            field_match = re.search('(?<=F)[0-9]+', image_id)
            fields_list.append(int(field_match.group(0)))

            channel_match = re.search('(?<=R)[0-9]+', image_id)
            channels_list.append(int(channel_match.group(0)))

    fields_list = np.array(fields_list)
    planes_list = np.array(planes_list)
    channels_list = np.array(channels_list)

    assert fields_list.min() == 1
    assert planes_list.min() == 1
    assert channels_list.min() == 1

    return (fields_list.max(), planes_list.max(), channels_list.max())


def generate_zarr_container(zarr_path, nfield, nplane, nchannel, overlap, plane_size):
    """Generate an empty zarr of the appropriate size.

    args:
        zarr_path : pathlike; path to where zarr will be stored
        nfield : int; number of fields in the well
        nplane : int; number of planes per image
        nchannel : int; number of channels per image
        overlap : tuple; overlap between tiles (row [y], col [x])
        plane_size : tuple; shape of each plane (row [y], col [x])

    returns:
        zarr_con : zarr.core.Array
    """
    if nfield == 25:
        layout = LAYOUT_25
    else:
        raise ValueError(f'Wells with {nfield} fields are not supported')

    row_overlap_sum = (layout.shape[0] - 1) * overlap[0]
    col_overlap_sum = (layout.shape[1] - 1) * overlap[1]

    row_shape = (plane_size[0] * layout.shape[0]) - row_overlap_sum
    col_shape = (plane_size[1] * layout.shape[1]) - col_overlap_sum

    store = zarr.DirectoryStore(zarr_path)
    zarr_con = zarr.empty([nchannel, nplane, row_shape, col_shape],
                           dtype=np.uint16,
                           compressor=Blosc(cname='zstd',
                                            clevel=1,
                                            shuffle=Blosc.BITSHUFFLE),
                           read_only=False)

    return zarr_con




# script 
def main():
    nfield, nplane, nchannel = parse_index_file('Index.idx.xml',
                                                well_col=WELL_COLUMN,
                                                well_row=WELL_ROW)
    zarr_con = generate_zarr_container('test.zarr',
                                       nfield,
                                       nplane,
                                       nchannel,
                                       overlap=(OVERLAP_Y, OVERLAP_X),
                                       plane_size=PLANE_SIZE)

if __name__ == "__main__":
    main()
