# -*- coding: utf-8 -*-

import torch
import argparse
from onnx_converter import MobileNetConverter as ONNXConverter


def parse_args():
    parser = argparse.ArgumentParser(description='generate onnx model file')
    parser.add_argument('--out', type=str, help='output path')
    parser.add_argument('--input_size', type=int, help='input image size')


    args = parser.parse_args()
    return args





def main():
    args = parse_args()
    input_names=[]
    output_names=[]

    # convert to onnx
    converter = ONNXConverter()
    converter.convert(args.out)

    # check
    # converter.check()



if __name__=='__main__':
    main()

