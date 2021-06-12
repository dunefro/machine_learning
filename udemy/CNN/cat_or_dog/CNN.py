#!/usr/bin/env python
# coding: utf-8

# # Convolution Neural Network

# ## Importing the libraries

# In[3]:


import tensorflow as tf
tf.__version__
# ImageDataGenerator library is used for image augmentation
from keras.preprocessing.image import ImageDataGenerator


# # Part 1 - Data Preprocessing

# ## Preprocessing the training set

# In[4]:


# rescaling -> multiplies the data with the given value. Here the value will be multiplied by 1/255 which means that all the value in the pixel will come down from [0,255] to [0,1]. This is done so as to bring down the higher pixel images to the same rate
# shear_range -> Tranformation that can be applied to the image. To know more https://www.geeksforgeeks.org/shearing-in-2d-graphics/ . Value 0.2 means Shx or Shy will be not more than 0.2
# zoom_range -> how much zooming is allowed
# horizontal_flip -> flipping  horizontally is allowed or not
train_datagen = ImageDataGenerator(rescale = 1./255 ,
                                   shear_range=0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True
                                  )
# target_size -> to reduce the image down to 64,64 matrix
# batch_size -> training set will be trained on the batch size of 32 images
# class_mode -> results will be binary, either a dog or a cat
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                          target_size = (64,64),
                                          batch_size = 32,
                                          class_mode = 'binary'
)


# ## Preprocessing the test set

# In[6]:


# The test image will be supplied as it is so it should not be subjected to any other change except rescaling because model is scaled for [0,1]
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64,64),
                                            batch_size = 32,
                                            class_mode = 'binary'

)


# # Part 2 - Building the CNN

# ## Initialising the CNN

# In[7]:


cnn = tf.keras.models.Sequential()


# ## Step 1 - Convolution

# In[8]:


# filters is the length of the feature map or the no of feature over which we wish to distinguish the input image. If we add more such layers ideally we should increse the no of filters
# kernel_size is the size of the feature detector
# rectifier activation function
# input image is converted to size 64,64,3

cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu' , input_shape = [64,64,3] ))


# ## Pooling

# In[9]:


# pool_size is the size of the matrix that is used to create the pooled feature map from the convoluted matrix
# strides is the no of steps the matrix should take so as to produce the max pool, 2 means 2 columns
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))


# ## Adding a second convolution layer

# In[10]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# ## Step 3 - Flattening 

# In[11]:


cnn.add(tf.keras.layers.Flatten())


# ## Step 4 - Full Connection

# In[12]:


cnn.add(tf.keras.layers.Dense(units=128, activation = 'relu'))


# ## Step 5 - Output Layer

# In[13]:


cnn.add(tf.keras.layers.Dense(units=1, activation = 'sigmoid'))


# # Part 3- Training the CNN

# ## compiling the CNN

# In[14]:


cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# ## Training the CNN on the training set and evaluating it on the Test Set

# In[15]:


cnn.fit(x = training_set , validation_data = test_set, epochs = 25)


# # Part 4 - Making a single prediction

# In[42]:


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_7.jpg' , target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image = test_image/255
result = cnn.predict(test_image)
training_set.class_indices

print(result)
if result[0][0] > 0.5:
  prediction = 'dog'
else:
  prediction = 'cat'
  


# In[43]:


print(prediction)


# In[ ]:




