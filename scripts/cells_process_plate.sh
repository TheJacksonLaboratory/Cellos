#!/bin/bash
usage="$(basename "$0") [-h] [-f configfile] -- \
Run multiple stitch_well.sh submissions in parallel \
based on wells specified in config file.

where:
    -h  show this help text
    -f  path to config file
"
##Parse arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    -f) configfile="$2"; shift 2;;
    -h) echo "$usage"; exit 0;;
    -*) echo "unknown option: $1" >&2; exit 1;;
    **) break;;
  esac
done

# Error if missing arguments
if [ -z "$configfile" ]; then
  echo "ERROR: path to pipeline configfile required"
  echo "$usage"
  exit 1
fi

if [ -z "$PYTHONPATH" ]; then
  echo "ERROR: PYTHONPATH env variable must be set to run script"
  echo "$usage"
  exit 1
fi

target_string=$(grep well_targets $configfile)

IFS="="; read -a target_string_split <<<"$target_string"

wells=$(echo ${target_string_split[@]:1} | tr -d '[:space:]')

IFS="|"; read -a well_array <<<"$wells"

for well in ${well_array[@]}
do
 IFS=","; read -a coords <<<"$well"
 wellr=${coords[0]}
 wellc=${coords[1]}
 sbatch cells_seg_well.sh -r $wellr -c $wellc -f $configfile 
done