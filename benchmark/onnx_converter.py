# -*- coding: utf-8 -*-
import torch

class ONNXConverter(object):
    def __init__(self):
        self.onnx_path = ''


    def prepare_inputs(self, size=None):
        image_shape = [1,3, 224,224]
        if size is not None:
            image_shape[2] = size[1]
            image_shape[3] = size[0]
        default_image = torch.Tensor(*image_shape)
        inputs = []
        inputs.append(default_image)
        if len(inputs)==1:
            inputs = inputs[0]
        self.inputs = inputs
        return self

    def prepare_names(self):
        self.input_names = ['input']
        self.output_names = ['output']

    def prepare(self):
        self.prepare_names()
        self.prepare_inputs()

    def convert(self, model, onnx_path):
        #  import ipdb
        #  ipdb.set_trace()
        self.prepare()

        torch.onnx.export(
                model,
                self.inputs,
                onnx_path,
                verbose=True,
                output_names=self.output_names,
                input_names=self.input_names)

    def check(self, onnx_path=None):
        import onnx
        if onnx_path is None:
            onnx_path = self.onnx_path
        model = onnx.load(onnx_path)

        onnx.checker.check_model(model)

        print(onnx.helper.printable_graph(model.graph))
