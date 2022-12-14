#Create a stitched well from the original files and save as zarr
# imports
import argparse
import configparser
import re
import zarr
import numpy as np
import xml.etree.ElementTree as ET
from numcodecs import Blosc
from pathlib import Path
from skimage import io, morphology, filters, measure
import pandas as pd
import matplotlib
matplotlib.use ('pdf')
import matplotlib.pyplot as plt
from skimage.morphology import square,binary_closing,dilation
import cv2

#global variable definition(all upper case variables)
#layout of 25 fields, change to the appropriate layout
LAYOUT_25 = np.array([[ 2, 3, 4, 5, 6],
                      [11,10, 9, 8, 7],
                      [12,13, 1,14,15],
                      [20,19,18,17,16],
                      [21,22,23,24,25]])

# function definitions
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


def generate_containers(zarr_path, layout, nplane, nchannel, overlap, plane_size):
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
        segment_con : np.array
    """
    row_overlap_sum = (layout.shape[0] - 1) * overlap[0]
    col_overlap_sum = (layout.shape[1] - 1) * overlap[1]

    row_shape = (plane_size[0] * layout.shape[0]) - row_overlap_sum
    col_shape = (plane_size[1] * layout.shape[1]) - col_overlap_sum

    store = zarr.DirectoryStore(zarr_path)
    zarr_con = zarr.empty([nchannel, nplane, row_shape, col_shape],
                           dtype=np.uint16,
                           store=store,
                           compressor=Blosc(cname='zstd',
                                            clevel=1,
                                            shuffle=Blosc.BITSHUFFLE),
                           overwrite=True,
                           read_only=False)

    segment_con = np.zeros([nplane, row_shape, col_shape], dtype=np.bool8)
    return zarr_con, segment_con


def get_filename(row, col, field, plane, channel):
    fn = f"r{row:02}c{col:02}f{field:02}p{plane:02}-ch{channel}sk1fk1fl1.tiff"
    return fn


def get_field_pixels(field, row, col, nplane, nchannel, images_path, plane_size):
    images_path = Path(images_path)
    orig_pixels_array = np.zeros([nchannel, nplane, *plane_size])
    for ch in range(1, nchannel + 1):
        for pl in range(1, nplane + 1):
            fn = get_filename(row, col, field, pl, ch)
            orig_pixels_array[ch - 1, pl - 1, ...] = io.imread(images_path / fn)
    return orig_pixels_array


def segment_image(image):
    img = image[:2, :, :, :]
    img_grey = (img[0, ...] * 0.2125) + (img[1, ...] * 0.7154)
    binarized = np.where(img_grey > 0.03, 1, 0)
    bin_opened = morphology.binary_opening(binarized.max(axis=0).astype(bool),
                                           selem=np.ones((8, 8)))
    processed = morphology.binary_dilation(bin_opened,
                                           selem=np.ones((10, 10),
                                                         dtype=int))
    clean_img = img * processed[np.newaxis, np.newaxis, :, :]
    image_grey = (clean_img[0, ...] * 0.2125) + (clean_img[1, ...] * 0.7154)
    image_blur = filters.gaussian(image_grey, sigma=0.5, mode = 'reflect',
                                      multichannel=True , preserve_range=True)
    thresh_value = filters.threshold_triangle(image_blur.flatten())
    mask_triangle = image_blur > thresh_value
    
    return mask_triangle

def calc_area(image):
    segments = image.max(axis=0)
    image_nbig = np.copy(segments)
    component_sizes = np.bincount(segments.ravel())
    too_big = component_sizes > 100000
    too_big_mask = too_big[segments]
    image_nbig[too_big_mask] = 0
    images_nbig_dilated_300 = dilation(image_nbig, square(300))
    image_nbig_dilated_closed_200 = binary_closing(images_nbig_dilated_300,
                                                   selem=square(200), out=None)
    image_area = cv2.countNonZero(image_nbig_dilated_closed_200.astype('int'))

    return image_nbig_dilated_closed_200, image_area

def parse_config(config_path):
    parser = configparser.ConfigParser()
    parser.read(config_path)
    plate_path = parser['pipeline']['plate_path']
    output_path = parser['pipeline']['output_path']
    config_dict = parser['stitch_well']
    plane_size_list = config_dict["plane_size"].strip('()').split(',')
    plane_size = tuple([int(x) for x in plane_size_list])
    return (plane_size,
            int(config_dict["overlap_x"]),
            int(config_dict["overlap_y"]),
            plate_path,
            output_path)

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
    cmap = matplotlib.colors.ListedColormap(np.random.rand(256,3))

    if fix_overlay:
        tmp = 2**array3d
        tmp.sort(axis=2)
        tmp = tmp[:,:,0] + np.diff(tmp, axis=2).sum(axis=2)
        plt.imshow(np.log2(tmp), cmap=cmap)
    else:
        tmp = array3d.max(axis=0)
        plt.imshow(tmp, cmap=cmap)

    if annotate:
        for props in measure.regionprops(tmp):
            y, x = props.centroid
            if fix_overlay:
                label = np.log2(props.label)
            else:
                label = props.label
            if label == int(label):
                label = int(label)
            if label>0:
                plt.text(x, y, label, fontsize=8, color='k', horizontalalignment='center', verticalalignment='center')

def get_channels(orig_zarr):
    new_zarr = np.zeros([101,5080,5080,3])
    new_zarr[:,:,:,1] = orig_zarr[:,:,:,0] /orig_zarr[:,:,:,0].max()
    new_zarr[:,:,:,0] = orig_zarr[:,:,:,1] /orig_zarr[:,:,:,1].max()

    return new_zarr

# script 
def main(well_row, well_col, config_file):
    plane_size, overlap_x, overlap_y, plate_path, output_path = parse_config(config_file)
    nfield, nplane, nchannel = parse_index_file(Path(plate_path) / 'Index.idx.xml',
                                                well_col=well_col,
                                                well_row=well_row)
    if nfield == 25:
        layout = LAYOUT_25
    else:
        raise ValueError(f'Wells with {nfield} fields are not supported') 

    zarr_con, segment_con = generate_containers(Path(output_path) / f"r{well_row:02}c{well_col:02}.zarr",
                                                layout,
                                                nplane,
                                                nchannel,
                                                overlap=(overlap_y,
                                                         overlap_x),
                                                plane_size=plane_size)

    for f in range(1, nfield + 1):
        print(f'Processing field: {f}')
        # Get field positions
        fposr, fposc = np.where(layout == f)
        rowstart = int((plane_size[0] - overlap_y) * fposr)
        colstart = int((plane_size[1] - overlap_x) * fposc)

        # Create array for field
        orig_pixel_array = get_field_pixels(f,
                                            well_row,
                                            well_col,
                                            nplane,
                                            nchannel,
                                            plate_path,
                                            plane_size)

        # Segment and insert into segment_con
        segmentation = segment_image(orig_pixel_array)
        segment_con[:, rowstart:rowstart+plane_size[0], colstart:colstart+plane_size[1]] = segmentation

        # Insert original into zarr_con
        zarr_con[:, :, rowstart:rowstart+plane_size[0], colstart:colstart+plane_size[1]] = orig_pixel_array

    #label segments and save rois
    label_segment = measure.label(segment_con, connectivity=3)
    label_segment = morphology.remove_small_objects(label_segment,connectivity=3,
                                            min_size=3000)

    #reorder organoid labels
    objects=np.ravel(label_segment)
    objects=objects[objects>0]
    labs=list(set(list(objects)))
    label_segment_order=np.zeros(np.shape(label_segment), dtype=np.int32)
    for i in range(len(labs)):
        label_segment_order[label_segment==labs[i]]=i+1

    #save lebeled organoids array as zarr
    zarr.save(Path(output_path) / f"r{well_row:02}c{well_col:02}organoids.zarr",label_segment_order)
  
    #save organoid labeled image
    plt.figure(figsize=(10,17))
    plt.imshow(label_segment_order.max(axis=0), cmap =matplotlib.colors.ListedColormap(np.random.rand(256,3)))
    plt.savefig(Path(output_path) / f"r{well_row:02}c{well_col:02}organoids.pdf")

    #save organoid labeled image with label numbers on top 
    plot_labeled_segments(label_segment_order,fix_overlay=False, annotate=True)
    plt.savefig(Path(output_path) / f"r{well_row:02}c{well_col:02}organoidsN.pdf")

    # calculate well area
    image_area, area = calc_area(label_segment_order)
    #save image area plot
    plt.figure(figsize=(10,10))
    plt.imshow(image_area)
    plt.savefig(Path(output_path) / f"r{well_row:02}c{well_col:02}area.pdf")

    #reorder axis of intensity image
    image = zarr_con[:2, :, :, :]
    image = np.moveaxis(image, 0, -1)

    # measure features of organoids
    df = pd.DataFrame(measure.regionprops_table(label_segment_order, intensity_image=image,
                                                            properties=("label","centroid","bbox", 
                                                                    "area", "major_axis_length",
                                                                    "area_bbox",
                                                                    "extent", "area_filled","area_convex", 
                                                                    "euler_number","extent", 
                                                                    "intensity_max","intensity_mean","intensity_min",
                                                                    "inertia_tensor_eigvals",
                                                                    "solidity"))).set_index('label') 

    df['well_area'] = area
    df.to_csv(Path(output_path) / f"r{well_row:02}c{well_col:02}organoids.csv")

    #plot the original image
    new_zarr = get_channels(image)
    plt.figure(figsize=(10,10))
    plt.imshow((new_zarr.max(axis=0))*3)
    plt.savefig(Path(output_path) / f"r{well_row:02}c{well_col:02}cells.pdf")
    
if __name__ == "__main__":
    desc = ("Stitch all of the fields from the well of an Opera Phenix plate and"
            " output regions of interest.")
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-r', '--row', type=int, help='The row containing the well.')
    parser.add_argument('-c', '--col', type=int, help='The column containing the well.')
    parser.add_argument('config_file', help='Path to the config file'
                                            ' (see documentation).')
    args = parser.parse_args()
    main(args.row, args.col, args.config_file)
#done