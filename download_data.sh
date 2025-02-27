#!/bin/bash

# Declare a dictionary with the Google Drive file IDs
declare -A dictionary
dictionary["elasticc_1"]="10M9YNrT58K5oVW0UdOlpSU5T-hg2-FL1"
dictionary["macho"]="10MYyQC_Eg6DFzV3ojUieooDINxCUwHsK"

# Check if the file name was provided as a parameter
if [ -z "${dictionary[$1]}" ]; then
  echo "File not found: $1"
  exit 1
fi

FILEID=${dictionary[$1]}
OUTDIR="data/lightcurves"
OUTFILE="$OUTDIR/$1.zip"  # Ensure the extension is correct for your file

# Create necessary directories
mkdir -p "$OUTDIR"

# Download the file using gdown
gdown "https://drive.google.com/uc?id=$FILEID" -O "$OUTFILE"
echo "Downloaded: $OUTFILE"

python3 -c "from scripts.utils import unzip; unzip('$OUTFILE', '$OUTDIR')"
