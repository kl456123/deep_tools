#!/bin/bash

python create_cleaner_machine_tf_record.py \
    --data_dir=/data/cleaner_machine/first_batch \
    --output_path=/data/tmp2/cleaner_machine.record \
    --label_map_path=/data/tmp2/cleaner_machine_label_map.pbtxt
