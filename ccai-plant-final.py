#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install fastai


# In[3]:


from fastai.vision import *


# In[76]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
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


# In[77]:


#Set Parameter
EPOCHS = 25
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((256, 256))
image_size = 0
directory_root ='train'
directory_2= 'test_imgs'
width=256
height=256
depth=3


# In[78]:


# # create a data generator
datagen = ImageDataGenerator()


# In[79]:


# # load and iterate training dataset
train_it = datagen.flow_from_directory(directory_root, 
                                        class_mode='binary', batch_size=64)


# In[80]:


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


# In[81]:


data.show_batch(rows=3, figsize=(10,8))
label_list = data.classes
label_list


# In[82]:


# Training
from fastai.metrics import error_rate # 1 - accuracy
learn = create_cnn(data, models.resnet34, metrics=accuracy)


# In[11]:


learn.fit_one_cycle(1)


# In[12]:


learn.fit_one_cycle(4)


# In[13]:


# Find the learning rate
learn.unfreeze() # must be done before calling lr_find
learn.lr_find()
learn.recorder.plot()


# In[86]:


interp.plot_confusion_matrix(figsize=(28,17))


# In[15]:


learn.fit_one_cycle(8,max_lr=slice(1e-5,1e-4))


# In[52]:


# Lets save the model 
learn.save('plant-village-ai')


# In[90]:


#INTERPRETATION
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(4, figsize=(20,10))


# In[91]:



# Testing

np.random.seed(8)
bs = 64
tfms = get_transforms(flip_vert=True, max_warp=0)


test_set = ImageDataBunch.from_folder(directory_2, 
                                  valid_pct=0.2,
                                  train=".",
      #                            test=test_imgs,
                                  ds_tfms=tfms,
                                  size=224,bs=bs, 
                                  num_workers=0).normalize(imagenet_stats)



test_set.show_batch(rows=3, figsize=(10,8))

label_list_test = data.classes
label_list_test
# img = learn.data.train_ds[0][0]
# learn.predict(img)


# In[ ]:


# learn.data = (
#     ImageList
#         .from_csv(PATH, 'train_v2.csv', folder="train-jpg", suffix=".jpg")
#         .random_split_by_pct(0.2)
#         .label_from_df(sep=' ')
#         .add_test_folder('test-jpg')
#         .transform(tfms, size=256)
#         .databunch(bs=32)
#         .normalize(imagenet_stats)
# )


# In[92]:


logs_preds_test = learn.get_preds(ds_type = 'test_imgs')
logs_preds_test[0][0]


# In[ ]:


# data_test = ImageDataBunch.from_folder(logs_preds_test, 
                                  valid_pct=0.2)


# In[94]:


data_test.show_batch(rows=3, figsize=(10,8))
label_list_test = data_test.classes
label_list_test


# In[ ]:




