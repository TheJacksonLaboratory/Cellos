#Create a stitched well from the original files and save as zarr
# imports
import re
import xml.etree.ElementTree as ET
from pathlib import Path

#global variable definition(all upper case variables)
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
        number of fields in well
        number of planes per image
        number of channels per image
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

# script 
def main():
    parse_index_file('Index.idx.xml', well_col=WELL_COLUMN, well_row=WELL_ROW)
    return True

if __name__ == "__main__":
    main()
