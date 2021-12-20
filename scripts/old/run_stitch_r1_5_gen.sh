#!/bin/bash

#PBS -l nodes=1:ppn=4

#PBS -l walltime=24:00:00

#PBS -q batch

#PBS -N stitch_r02

#PBS -t 6,9

cd $PBS_O_WORKDIR

input="/projects/liu-lab/ED_Lab_data/phenix_data_pk/pooja_96_gfp_mcherry_test4_20x_5z_nodrug__2020-02-10T14_57_54-Measurement 1/Images/"
well="r02c0$PBS_ARRAYID"
outdir="/projects/chuang-lab/mukasp/test_my_code/dataset4/mix_red_green/stitch_org_seg/output/"

python stitch_r1_gen.py "$input" $well $outdir ; python stitch_r2_gen.py "$input" $well $outdir ; python stitch_r3_gen.py "$input" $well $outdir ; python stitch_r4_gen.py "$input" $well $outdir ; python stitch_r5_gen.py "$input" $well $outdir && python stitch_allrows_gen.py "$input" $well $outdir 

