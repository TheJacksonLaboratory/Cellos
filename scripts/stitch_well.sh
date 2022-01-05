#!/bin/bash
#SBATCH --qos=batch
#SBATCH --partition=compute
#SBATCH --job-name=stitch_well
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=00:40:00

usage="$(basename "$0") [-h] [-r row -c column -f configfile] -- \
Stitch all fields in an Opera Phenix well and find ROIs. Note that
PYTHONPATH must be set in your environment to run the script.

where:
    -h  show this help text
    -r  row coordinate of well
    -c  column coordinate of well
    -f  path to config file
"

##Parse arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    -r) row="$2"; shift 2;;
    -c) column="$2"; shift 2;;
    -f) configfile="$2"; shift 2;;
    -h) echo "$usage"; exit 0;;
    -*) echo "unknown option: $1" >&2; exit 1;;
    **) break;;
  esac
done

# Error if missing arguments
if [ -z "$row" ]; then
  echo "ERROR: missing row coordinate"
  echo "$usage"
  exit 1
fi

if [ -z "$column" ]; then
  echo "ERROR: missing column coordinate"
  echo "$usage"
  exit 1
fi

if [ -z "$configfile" ]; then
  echo "ERROR: path to pipeline configfile required"
  echo "$usage"
  exit 1
fi

if [ ! -f "$configfile" ]; then
  echo "ERROR: $configfile does not exist"
  echo "$usage"
  exit 1
fi

if [ -z "$PYTHONPATH" ]; then
  echo "ERROR: PYTHONPATH env variable must be set to run script"
  echo "$usage"
  exit 1
fi

$PYTHONPATH stitch_well.py -r $row -c $column $configfile

