# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, opts):
        self.layer = nn.Conv2d(10, 20, 3, 1, 1)

    def forward(self, x):
        return self.layer(x)


def create_model():
    opts = {}
    return Model(opts)


if __name__ == '__main__':
    # build model
    model = create_model()

    # load weights from file
    model_path = ''
    state_dict = torch.load(
        model_path,
        map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    # convert
    dummy_input = torch.randn(1, 3, 320, 320, device='cpu')
    input_names = ['input']
    output_names = ['cls_and_bbox', 'anchors']
    torch.onnx.export(
        model,
        dummy_input,
        'cleaner_machine.onnx',
        verbose=True,
        input_names=input_names,
        output_names=output_names)
