# ***Cellos***: High-throughput deconvolution of 3D organoid dynamics at cellular resolution for cancer pharmacology 

<p>
    <img src="docs/pipeline3d.png" width="1000"/>
</p>

- [Overview](#overview)
- [Data description](#data-description)
- [Installing the pipeline](#installing-the-pipeline)
- [Running the pipeline](#running-the-pipeline)

## Overview 

***Cellos*** (Cell and Organoid Segmentation) is a pipeline developed to perform high-throughput volumetric 3D segmentation and morphological quantification of organoids and their cells. ***Cellos*** segments organoids using classical algorithms and segments nuclei using our trained model based on Stardist-3D (https://github.com/stardist/stardist). 

## Data description
The image data used here were exported from the *PerkinElmer* Opera Phenix high content screening confocal microscope. The resulting folder contains subfolders with tiff files (Images) and xml files (metadata). Each tiff file was a single image from one well, one field, one plane and one channel. We developed an automatic protocol that organized all tiff files from same well and saved them as zarr arrays to minimize RAM and storage. All information for the images are deconvoluted from the respective metadata files. 

## Installing the pipeline
Currently, the pipeline uses a Python 3.7 environment. We provide a defined `requirements.txt` to install all packages and dependencies for a working environment on Rocky 9 Linux.  
We recommend creating a virtual environment for running the pipeline, for example using `conda`.
 
Installing the pipeline using `conda` to manage the Python version:  

```bash
git clone https://github.com/TheJacksonLaboratory/Cellos.git
cd Cellos #(make sure you are in the correct directory)
conda create -f environment.yml
```

This will use `conda` to create a Python 3.7 environment and then install all packages from PyPI using `pip` and the `requirements.txt` file.

If you prefer to install the pipeline dependencies into a pre-existing environment (e.g. `venv`), you can use:

```bash
pip install --require-hashes --no-deps -r requirements.txt
```
This will ensure you install the exact packages that we've tested.


## Running the pipeline
There are two main steps to run the pipeline: 1. Organanizing images and organoids segmentation. 2. Nuclei segmentation

> [!IMPORTANT]
> If you are using a virtual environment, ensure you have it activated!  
> For example, using `conda` as recommended, do:
> ```bash
> conda activate organoid
> ```

+ The process for running organizing images and organoids segmentation steps: 

      cd scripts/process_organoids/
      PYTHONPATH=~/anaconda3.1/envs/organoid/bin/python bash stitch_well.sh -r (add row number here) -c (add column number here) -f ../../config.example.cfg (when you want to process one well)
	  PYTHONPATH=~/anaconda3.1/envs/organoid/bin/python bash process_plate.sh -f ../../config.example.cfg (when you want to run multiple wells at the same time)

+ The process for running nuclei segmentation steps: 

      install tensorflow: pip install tensorflow
      cd scripts/process_cells/
      PYTHONPATH=~/anaconda3.1/envs/organoid/bin/python bash cells_seg_well.sh -r (add row number here) -c (add column number here) -f ../../config.example.cfg (when you want to process one well)
	  PYTHONPATH=~/anaconda3.1/envs/organoid/bin/python bash cells_process_plate.sh -f ../../config.example.cfg (when you want to run multiple wells at the same time)

## Demo

### Usage
We have made an example dataset with one well data publicly available,
the folder consists of images and .xml (metadata) file, and can be downloaded from: https://figshare.com/articles/dataset/cellos_data_zip/21992234 Once downloaded, it can be used as input example into the pipeline. The well row number=3 and column number=7. The image has 3 channels, channel1=EGFP, channel2=mCherry and channel3=brightfield. The expected results are under output folder.


> *Note*: you have to edit the config file to your needs. 
>
> | Parameter | Description | 
> |-----|-------------|
> | plate_path   | path to where your raw images are | 
> | output_path   | path to where the csv files and zarr arrays will be saved   | 
> | well_targets   | name number of rows and columns (row1,col1&#124;row2,col2) of wells to analyze   | 
> | plane_size   |  size of image of one field, one z-slice and one channel   | 
> | overlap_x and y   | overlapping pixels between two adjacent fields   | 
> | stardist_path   | path to the trained model for nuclei segmentation  | 
> | ...  | ...  | 


> ***Tip***: All of the steps implemented in our pipeline are optimized to run on high-performance computer (HPC) systems to take advantage of parallel processing and to carry out the steps that are computationally intensive. The most expensive step in the pipeline is to process the whole image in the steps to organize the image and segment organoids. To minimize this, if not needed, you can remove the step to calculate area of image that has organoids and to plot raw images. To run organoids segmentation step on demo data we used a wall-clock time of approximately 2 hours and 30 min on an HPC system using 1 core and 32 GB of memory and to run nuclei segmentation step we used a wall-clock time of approximately 25 min on HPC system using 4 cores and less than 5 GB of memory. 

