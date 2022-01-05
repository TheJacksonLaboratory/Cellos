#!/bin/bash

conf=../config.example.cfg

target_string=$(grep well_targets $conf)

IFS="="; read -a target_string_split <<<"$target_string"

wells=$(echo ${target_string_split[@]:1} | tr -d '[:space:]')

IFS="|"; read -a well_array <<<"$wells"

for well in ${well_array[@]}
do
 IFS=","; read -a coords <<<"$well"
 wellr=${coords[0]}
 wellc=${coords[1]}
 sbatch stitch_well.sh -r $wellr -c $wellc -f $conf 
done
