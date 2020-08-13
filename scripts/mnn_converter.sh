#!/bin/bash

MNN_HOME=${HOME}/Documents/MNN
MODEL_HOME=${HOME}/Models

# use ssd or centernet
MODEL_NAME=ssd
# MODEL_NAME=ctdet


${MNN_HOME}/build/MNNConvert -f ONNX --modelFile ${MODEL_HOME}/${MODEL_NAME}.onnx \
    --MNNModel ${MODEL_HOME}/${MODEL_NAME}.mnn --bizCode biz
