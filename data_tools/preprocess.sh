#!/bin/bash

# preprocess
# path
OUTPUT_DIR=/data/tmp/
# INPUT_DIR=/data/cleaner_machine/first_batch
COCO_JSON=${OUTPUT_DIR}/train.json

HDF5_DIR=${OUTPUT_DIR}/train_hdf5

IMAGE_SIZE=600
NUM_PER_FILE=3000

# label, convert to coco model
# python parse_label.py --output_path ${COCO_JSON} --input_dir ${INPUT_DIR} --input_size ${IMAGE_SIZE}


# # convert image to hdf5 file
# python create_hdf5.py --output_dir ${HDF5_DIR} \
    # --input_size ${IMAGE_SIZE} \
    # --num_images_per_file ${NUM_PER_FILE} \
    # --label_json ${COCO_JSON}


# visualize
python test.py --json_file ${COCO_JSON} --hdf5_dir ${HDF5_DIR}


# train




# postprocess

# convert to ONNX

# convert to MNN
