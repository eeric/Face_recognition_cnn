img2rec.py生成的lst文件，顺序是乱序，不能用于triplet loss训练，只能按下面步骤：

（1）python3 make_lst.py --dataset-dir /your dataset forld path --list-file /your .lst path/train.lst --img-ext '.jpg'    ##生成.lst

（2）python3 face2rec2.py ./lst_file/full_mask --num-thread 8    ##生成.rec和.idx



