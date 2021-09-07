# -*- coding: utf-8 -*-
import cv2
import torch
import numpy as np
import onnx
import onnxruntime


# image to tensor
def img2tensor(image):
    img  = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float().cuda()
    img.div_(255).sub_(0.5).div_(0.5)
    return img
# tensor to numpy
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# load model
onnx_model = onnx.load("test.onnx")
# check model
onnx.checker.check_model(onnx_model)
# create an inference
ort_session = onnxruntime.InferenceSession("test.onnx")

# load image
image = cv2.imread('/path to image/*.jpg')
img = img2tensor(image)
# inference
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
feature = ort_session.run(None, ort_inputs)[0] # array
print(feature.shape)
