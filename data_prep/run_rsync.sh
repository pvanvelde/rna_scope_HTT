#!/bin/bash

# Define source and target paths for the two folders
SOURCE1="/media/grunwaldlab/SG Skyhawk AI 24TB/Q111 raw data/Q111_10slidesno2_june2025_new_illumination/exported"
TARGET1="/pi/grunwaldCHDI/data/RNAscope/q111_jun2025/Q111_10slidesno2_june2025_newillumination"

#SOURCE2="/media/grunwaldlab/SG Skyhawk AI 24TB/Q111 raw data/Q111_15slidesno1_june2025/exported"
#TARGET2="/pi/grunwaldCHDI/data/RNAscope/q111_jun2025/Q111_15slidesno1_june2025"

# Create target directories on the remote host if they do not exist
#ssh umass "mkdir -p $TARGET1 $TARGET2"
ssh umass "mkdir -p $TARGET1 "

# Run the rsync commands sequentially
rsync -avhP --ignore-existing "${SOURCE1}/" "umass:${TARGET1}/"
#rsync -avhP --ignore-existing "${SOURCE2}/" "umass:${TARGET2}/"