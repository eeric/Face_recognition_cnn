### img2rec.py生成的lst文件，顺序是乱序，不能用于triplet loss训练，只能按下面步骤：

（1）python3 make_lst.py --dataset-dir /your dataset forld path --list-file /your .lst path/train.lst --img-ext '.jpg'    ##生成.lst

（2）python3 face2rec2.py ./lst_file/full_mask --num-thread 8    ##生成.rec和.idx

face2rec2.py:   https://github.com/eeric/insightface/blob/master/src/data/face2rec2.py

### image-->bin
python lfw2pack.py --data-dir /path to/LFW --image-size 112,112 --output /path to/lfw.bin

### 使用DALI模式，需要特殊.rec格式

（1）生成.lst

格式：python im2rec.py --list /path to train(默认.lst) --recursive /path to image

如：python im2rec.py --list /mnt/data/dataset/train --recursive /mnt/data/face

（2）生成.rec和.idx

格式：python im2rec.py /path to train.lst --num-thread 16 --quality 100 /path to image

python im2rec.py /mnt/data/dataset/train.lst --num-thread 16 --quality 100 /mnt/data/face

im2rec.py: https://github.com/apache/incubator-mxnet/blob/master/tools/im2rec.py

*图片数量，4247,4557，另外有一张图像转rec格式的时候不成功，检查发现那张图片有可能只有jpeg头，这样当不使用pass_through模式（通常pass-through 默认为false，pass-through意思是不对图片进行解码）时，这张问题图像就不会直接转rec文件。

*用于DALI模式的rec格式文件，也能用于非DALI模式读取。
