pip install fastai

from fastai.vision import *

%reload_ext autoreload
%autoreload 2
%matplotlib inline
import os
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
from fastai import *
from fastai.vision import *
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator
import warnings
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
warnings.filterwarnings('ignore')



#Set Parameter
EPOCHS = 25
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((256, 256))
image_size = 0
directory_root ='train'
width=256
height=256
depth=3


# # create a data generator
datagen = ImageDataGenerator()

# # load and iterate training dataset
train_it = datagen.flow_from_directory(directory_root, 
                                        class_mode='binary', batch_size=64)
                                        

np.random.seed(8)
bs = 64
tfms = get_transforms(flip_vert=True, max_warp=0)

data = ImageDataBunch.from_folder(directory_root, 
                                  valid_pct=0.2,
                                  train=".",
#                                   test="../test images",
                                  ds_tfms=tfms,
                                  size=224,bs=bs, 
                                  num_workers=0).normalize(imagenet_stats)

data.show_batch(rows=3, figsize=(10,8))
label_list = data.classes
label_list

# # Pre Processing
from fastai.metrics import error_rate # 1 - accuracy
learn = create_cnn(data, models.resnet34, metrics=accuracy)

defaults.device = torch.device('cuda') # makes sure the gpu is used
learn.fit_one_cycle(4)

# Find the learning rate
learn.unfreeze() # must be done before calling lr_find
learn.lr_find()
learn.recorder.plot()

# Lets save the model 
learn.save('plant-village-ai')
