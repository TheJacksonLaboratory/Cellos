#!/bin/bash

#SBATCH --job-name="stitch_r02_seg" 

#SBATCH -t 0-25:00:00

#SBATCH -N 1

#SBATCH -n 1

#SBATCH --mem=250GB

#SBATCH -p compute

#SBATCH -q batch

#SBATCH --array=10-11


cd $SLURM_SUBMIT_DIR

input="/projects/liu-lab/ED_Lab_data/phenix_data_pk/pooja_96_gfp_mcherry_test11_10082021_20x_drug_d4_plate1__2021-10-08T13_09_49-Measurement 1/Images/"
well="r03c$SLURM_ARRAY_TASK_ID"
outdir="/projects/chuang-lab/mukasp/cell_seg_project/dataset11/day5/stitch_org_seg/plate1/output/"

/projects/chuang-lab/mukasp/anaconda3.1/bin/python stitch_r1_gen.py "$input" $well $outdir ; /projects/chuang-lab/mukasp/anaconda3.1/bin/python stitch_r2_gen.py "$input" $well $outdir ; /projects/chuang-lab/mukasp/anaconda3.1/bin/python stitch_r3_gen.py "$input" $well $outdir ; /projects/chuang-lab/mukasp/anaconda3.1/bin/python stitch_r4_gen.py "$input" $well $outdir ; /projects/chuang-lab/mukasp/anaconda3.1/bin/python stitch_r5_gen.py "$input" $well $outdir && /projects/chuang-lab/mukasp/anaconda3.1/bin/python stitch_allrows_gen_area.py "$input" $well $outdir 

