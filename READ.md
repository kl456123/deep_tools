# Deeplearning ToolChains(for myself)
mainly about pytorch, tensorflow is nothing to hands on


## Dataset
it is different according to the framework where you work, like pytorch or tensorflow

### pytorch
* HDF5 format, convert all kinds of different labels and images to h5 dataset format,
    for me , just run `python data_tools/coco_preprocessor.py` is ok if use coco dataset to pretrain.
    It is obvious that it can extend to another dataset like voc or something else easliy.
    please check `coco_preprocessor.py` for example.

* data transform, sometime if the image is too big to load to GPU, you need to crop it first, then resize
    for me, just adding argument `use_crop=True` is ok when I run xxx_preprocessor.py.
    Note that you cannot resize it directly otherwise some objects will too small to detect(to match with any anchors)

### tensorflow
* well, tfrecord is ok. no any other options, just run `python data_tools/tf/create_tf_record.py`




## Model Selection
just select one from papers like mobilenet_v1 or something else,
do benchmark util it meet your need of lantency and precision,
just run `bash ./benchmark/convert.sh` to convert model to ONNX format, then

## Training

how to train faster, other than distributed training, please focus on something about data

* use utils from snippets/fly.py,
    like data prefetch, async cuda stream to load data from cpu to gpu

* map data to memory if RAM is larger than dataset, just run `bash ./scripts/mount_memory.sh`

## Utils
* watch your gpu on realtime, just run `bash ./scripts/gpu_monitor.sh`


## Deploy
use MNN inference framework to deploy pytorch model
1. convert model to onnx model
2. convert onnx model to mnn
3. add some postprocess code to get the final box
like nms, decode box (if you want to detect something)




