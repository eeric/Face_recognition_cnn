## :memo: Facial Recognition from basics to advanced
> 1. **Facial_recogntion.md** for basics of facial recognition 
> 2. It contains a summary for occlusion robust facial recogntion systems.



REPO: https://github.com/deepinsight/insightface

# Training Face encoder for Face recognition
Summary of all the steps performed so far:
- Dataset Preparation
- Dataset Preprocessing
- Training Setup
- Model parameters/loss
- Model verification
- Model Finetuning options
- OSS Issues


## Dataset Preparation:

> - Datsets can be found @[Datasets-zoo](https://https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)
> - Dataset downloaded will be having the following files:
 ```Shell
    faces_emore/
       train.idx
       train.rec
       property
       lfw.bin
       cfp_fp.bin
       agedb_30.bin
```

The first three files are the training dataset while the last three files are verification sets.



### Dataset Asia celeb:

- The dataset will look like:
```Shell
    ms1m-retinaface-t1/
       train.idx
       train.rec
       property
       train.lst
       lfw.bin
       cfp_fp.bin
       agedb_30.bin
```
- train.idx and train.rec will be used to prepare images from this format
- property contains a single line with **(no_of_classes,w,h)**
- train.lst is list of image name, format "1\tpath\tidx_class"
- The last three files are binary verifcation files which the model will use for calculating accuracy on the fly.
- The given script can be used for converting rec files into images.
```

import numpy as np
import mxnet as mx
from mxnet import recordio
import matplotlib.pyplot as plt
import cv2
import os
path_imgidx = '/mnt/umair/faces_glintasia/train.idx' # path to train.rec
path_imgrec = '/mnt/umair/faces_glintasia/train.rec' # path to train.idx

imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
#%% 1 ~ 3804847
for i in range(3000):
        print(i)
        header, s = recordio.unpack(imgrec.read_idx(i+1))
        #print(str(header.label))
        #img = np.array(mx.image.imdecode(s))
        img = mx.image.imdecode(s).asnumpy()
        #print(type(img))
        path = os.path.join('images',str(header.label))
        if not os.path.exists(path):
                os.makedirs(path)
        path = os.path.join(path,str(i))
        #fig = plt.figure(frameon=False)
        #fig.set_size_inches(124,124)
        #ax = plt.Axes(fig, [0., 0., 1., 1.])
        #ax.set_axis_off()
        #fig.add_axes(ax)
        #ax.imshow(img, aspect='auto')
        #dpi=1
        #fname= str(i)+'jpg'
        #fig.savefig(fname, dpi)
        #plt.savefig(path+'.jpg',bbox_inches='tight',pad_inches=0)
        (b,g,r)=cv2.split(img)
        img = cv2.merge([r,g,b])
        #w,h = img.size
        print((img.shape))
        cv2.imwrite(path+'.jpg',img)

```
- After using this script, you will get a directory 'images' with images in their respective classes.
> :bulb: rec2img:https://github.com/deepinsight/insightface/issues/121
> 
>:bulb: rec2img cv2 : https://github.com/deepinsight/insightface/issues/65
>
### Conversion of images to .rec:
- If you have your custom dataset in the required format
- The above task can be accomplished by following the given links:
>
:bulb: Insight Face face2rec.py:  https://github.com/eeric/insightface/blob/master/src/data/face2rec2.py </br> error
https://github.com/apache/incubator-mxnet/blob/master/tools/im2rec.py </br> recon
> 
:bulb: img2rec: https://gluon-cv.mxnet.io/build/examples_datasets/recordio.html#sphx-glr-download-build-examples-datasets-recordio-py </br>
> 
:bulb:Useful https://github.com/deepinsight/insightface/issues/214#issuecomment-419812835
>
### list of images to .lst:
:bulb: dir2lst: https://github.com/eeric/insightface/blob/master/src/data/dir2lst_ytf.py </br>
>
e.g., 
>
1	/raid5data/dplearn/ms1m-retinaface-arcface/images/m.0107_f/109-FaceId-0.jpg	0
### property
>
e.g., 
>
93431,112,112
## Dataset Preprocessing:
- All face images are aligned by [MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html) and cropped to 112x112:
- Any public available *MTCNN* can be used to align the faces, and the performance should not change. We will improve the face normalisation step by full pose alignment methods recently.
## Training Face Encoder:

Follow the given steps for the training process:
### Setup Procedure:
1. Install `MXNet` with GPU support 
2. Your `MXNet` should be compatible with your `CUDA` version.
3. For installing `CUDA` and Nvidia Drivers follow :[Installing CUDA](https://www.pugetsystems.com/labs/hpc/How-To-Install-CUDA-10-1-on-Ubuntu-19-04-1405/)

```
pip install mxnet-cu{CUDA_VERSION}
```

2. Clone the InsightFace repository. We call the directory insightface as *`INSIGHTFACE_ROOT`*.

```
git clone --recursive https://github.com/deepinsight/insightface.git
```

3. Download the training set (`MS1M-Arcface`) and place it in *`$INSIGHTFACE_ROOT/datasets/`*. Each training dataset includes at least following 6 files:

```Shell
    faces_emore/
       train.idx
       train.rec
       property
       lfw.bin
       cfp_fp.bin
       agedb_30.bin
```

A custom dataset can also be used for training purpose, for producing a dataset follow the above instructions.

The first three files are the training dataset while the last three files are verification sets.

4. Train deep face recognition models.
In this part, we assume you are in the directory *`$INSIGHTFACE_ROOT/recognition/`*.
```Shell
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
```
- This part comes handy for training purpose.
- Make your own `config.py` files using `sample_config.py`
- Edit our `config.py` for setting dataset properties
>
Place and edit config file:
```Shell
cp sample_config.py config.py
vim config.py # edit dataset path etc..
```

`emore` is the dataset variable name that can be modified in `config.py`

(1). Train ArcFace with LResNet100E-IR.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network r100 --loss arcface --dataset emore
```
(2). Train CosineFace with LResNet50E-IR.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network r50 --loss cosface --dataset emore
```

(3). Train Softmax with LMobileNet-GAP.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network m1 --loss softmax --dataset emore
```


(4). Fine-tune the above Softmax model with Triplet loss.

```Shell
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --network m1 --loss triplet --lr 0.005 --pretrained ./models/m1-softmax-emore,1
```

## Verifying the model:

### Embedding Formation
- https://github.com/deepinsight/insightface/tree/master/deploy 
- `face_embedding.py` can be used for preparing embeddings
- `test.py` can be used for testing the distance between the formed embeddings

### 512-D Feature Embedding

In this part, we assume you are in the directory *`$INSIGHTFACE_ROOT/deploy/`*. The input face image should be generally centre cropped. We use *RNet+ONet* of *MTCNN* to further align the image before sending it to the feature embedding network.

1. Prepare a pre-trained model.
2. Put the model under *`$INSIGHTFACE_ROOT/models/`*. For example, *`$INSIGHTFACE_ROOT/models/model-r100-ii`*.
3. Run the test script *`$INSIGHTFACE_ROOT/deploy/test.py`*

### Pretrained Models

You can use `$INSIGHTFACE/src/eval/verification.py` to test all the pre-trained models.

**Please check [Model-Zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo) for more pretrained models.**



## :memo: Relevant issues on Insightface 

> :bulb: https://github.com/deepinsight/insightface
>
> :bulb: rec2img:https://github.com/deepinsight/insightface/issues/121
> 
>:bulb: rec2img cv2 : https://github.com/deepinsight/insightface/issues/65
>
> :bulb: Insight Face face2rec.py:  https://github.com/deepinsight/insightface/blob/master/src/data/face2rec2.py </br>
> 
> :bulb: img2rec: https://gluon-cv.mxnet.io/build/examples_datasets/recordio.html#sphx-glr-download-build-examples-datasets-recordio-py </br>
> 
> :bulb: Complete Summary : https://github.com/deepinsight/insightface/issues/791 </br>
