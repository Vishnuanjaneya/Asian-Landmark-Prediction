#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


# In[2]:


TF_MODEL_URL = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
LABEL_MAP_URL = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_asia_V1_label_map.csv'
IMAGE_SHAPE = (321, 321)


# In[3]:


df=pd.read_csv(LABEL_MAP_URL)


# In[5]:


classifer=tf.keras.Sequential([hub.KerasLayer(
    TF_MODEL_URL, 
    input_shape=IMAGE_SHAPE +(3,),
    output_key="predictions:logits"
)])


# In[7]:


classifer.summary()


# In[8]:


df.head(5)


# In[9]:


label_map=dict(zip(df.id, df.name))


# In[ ]:





# In[16]:


import cv2
img=cv2.imread('puri.jpg')
resize=cv2.resize(img, (640,480))
RGBimg=cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)
RGBimg=cv2.resize(RGBimg, (321,321))
RGBimg=np.array(RGBimg)/255
RGBimg=np.reshape(RGBimg, (1,321,321,3))
prediction=classifer.predict(RGBimg)


# In[18]:


result=label_map[np.argmax(prediction)]


# In[19]:


result


# In[20]:


def classifyimg(image):
    img=cv2.imread(image)
    resize=cv2.resize(img, (640,480))
    RGBimg=cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)
    RGBimg=cv2.resize(RGBimg, (321,321))
    RGBimg=np.array(RGBimg)/255
    RGBimg=np.reshape(RGBimg, (1,321,321,3))
    prediction=classifer.predict(RGBimg)
    return label_map[np.argmax(prediction)]


# In[21]:


classifyimg('puri.jpg')


# In[ ]:




