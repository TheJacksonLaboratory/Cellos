#!/bin/bash

#SBATCH --job-name="red_c03_cells_seg" 

#SBATCH -t 0-25:00:00

#SBATCH -N 1

#SBATCH -n 1

#SBATCH --mem=30GB

#SBATCH -p compute

#SBATCH -q batch

#SBATCH --array=1-16

cd $SLURM_SUBMIT_DIR

#FILE=$(head -n $PBS_ARRAYID r02c06_green_organoids_file.txt | tail -n 1)

FILE="r02c02"
input_dir="/projects/chuang-lab/mukasp/cell_seg_project/dataset11/day5/stitch_org_seg/plate1/output/"
output_dir="/projects/chuang-lab/mukasp/cell_seg_project/dataset11/day5/cell_seg/plate1/r02c02/output/"

STEP=50
START=$(( ( SLURM_ARRAY_TASK_ID - 1 ) * STEP + 1))
END=$(( START + STEP - 1 ))
END=$(( END > 757 ? 757 : END ))

for N in $( seq $START $END )
do
        /projects/chuang-lab/mukasp/anaconda3.1/bin/python stardist_red_segment_cells.py $FILE $N $input_dir $output_dir
done


