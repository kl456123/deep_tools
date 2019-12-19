#!/bin/bash

MOUNT_PATH=/data/tmp_memory/
SIZE=15G
sudo mount tmpfs ${MOUNT_PATH} -t tmpfs -o size=${SIZE}
