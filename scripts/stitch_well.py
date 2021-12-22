#Create a stitched well from the original files and save as zarr
# imports
import xml.etree.ElementTree as ET
from pathlib import Path

#global variable definition(all upper case variables)
OVERLAP_X = 80
OVERLAP_Y = 80
WELL_ROW = 1
WELL_COLUMN = 1
PLATE_PATH = "/projects/liu-lab/ED_Lab_data/phenix_data_pk/pooja_96_gfp_mcherry_test11_10082021_20x_drug_d4_plate1__2021-10-08T13_09_49-Measurement 1/Images/"

# function definitions
def load_images(well,plate_path):
    return True


def parse_index_file(index_file_path):
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
    for child in root:
        if child.tag == "{http://www.perkinelmer.com/PEHH/HarmonyV5}Wells":
            wells = child

    well = wells[0]
    for child in well:
        print(child.attrib)

    


# script 
def main():
    parse_index_file('Index.idx.xml')
    return True

if __name__ == "__main__":
    main()
