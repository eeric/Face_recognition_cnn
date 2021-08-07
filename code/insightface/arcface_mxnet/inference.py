#############################
# Modefied inference of mxnet
#https://github.com/deepinsight/insightface/issues/1417#issue-819308039
#https://github.com/deepinsight/insightface/issues/1417
#no tensor normalize,see https://github.com/deepinsight/insightface/issues/1417#issuecomment-894589411
#############################
import numpy as np
import cv2
import mxnet as mx
from collections import namedtuple

#import pkg_resources
#print("mxnet version:", pkg_resources.get_distribution("mxnet").version)

# convert array
def get_array(face_chip):
    face_chip = cv2.cvtColor(face_chip, cv2.COLOR_BGR2RGB)
    face_chip = face_chip.transpose(2, 0, 1)
    face_chip = face_chip[np.newaxis, :] # 4d
    array = mx.nd.array(face_chip)
    return array

# load mxnet weight
prefix = "./models/insightface/model"
sym, arg, aux = mx.model.load_checkpoint(prefix, 0)
# define mxnet
ctx = mx.cpu() # gpu_id = 0 ctx = mx.gpu(gpu_id)
mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,112,112))])
mod.set_params(arg, aux)
Batch = namedtuple('Batch', ['data'])
# read image
img = cv2.imread("face_chip.jpg")
array = get_array(img)
# inference
mod.forward(Batch([array]))
# feature
feat = mod.get_outputs()[0].asnumpy()
print(feat)
