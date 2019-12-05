#!/bin/bash

# conda activate pytorch1.3

MODEL_PREFIX=model

python generate_onnx.py --out ${MODEL_PREFIX}.onnx


MNN_HOME=/home/indemind/Documents/MNN
MODEL_HOME=/home/indemind/Models/benchmark


${MNN_HOME}/build/MNNConvert -f ONNX --modelFile ./${MODEL_PREFIX}.onnx --MNNModel ${MODEL_HOME}/${MODEL_PREFIX}.mnn --bizCode biz

${MNN_HOME}/build/benchmark.out ${MODEL_HOME} 10 3 2
 
# quantize
# ${MNN_HOME}/build/quantized.out ./blazeface.mnn blazeface_quan.mnn ./mobilenetCaffeConfig.json
