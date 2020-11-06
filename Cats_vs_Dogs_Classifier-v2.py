#!/usr/bin/env python
# coding: utf-8

# In[31]:


from zipfile import ZipFile
import os

# Extract zip folder containing images of cats and dogs to a temporary file location

# UPDATE ACCORDINGLY
ZIP_NAME = 'kagglecatsanddogs_3367a.zip'
PARENT_DIR = 'Users/cats-or-dogs'

zf = ZipFile(ZIP_NAME, 'r')
print(zf)
zf.extractall(PARENT_DIR)
zf.close()


# In[32]:


# File/ Directory variables. UPDATE ACCORDINGLY

# Type name of classes extention
classA_ext = '\Cat'
classB_ext = '\Dog'

# Note that here '\PetImages' path already exists from the zip extraction and we are using this as the training directory i.e not creating/ moving for train
train_ext = '\PetImages'
TRAIN_DIR = PARENT_DIR + train_ext

classA_train_dir = TRAIN_DIR + classA_ext
classB_train_dir = TRAIN_DIR + classB_ext

# Here '\Validation-PetImages' does not exist and we will create it later on
valid_ext = '\Validation-PetImages'
VALIDATION_DIR = PARENT_DIR + valid_ext

classA_validation_dir = VALIDATION_DIR + classA_ext
classB_validation_dir = VALIDATION_DIR + classB_ext


# In[33]:


# Define a callback to exit training when accuracy reaches 99.9%
import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.995):
      print("\nReached 99.9% accuracy so cancelling training!")
      self.model.stop_training = True


# In[34]:


# Remove any files that are corrupt/ non-type
import os
os.sys.path
import sys
get_ipython().system('{sys.executable} -m pip install opencv-python')
import cv2
import imghdr
import os

# Create array of file names
classA_file_names = os.listdir(classA_train_dir)
classB_file_names = os.listdir(classB_train_dir)

# Make sure all files are of JPEG type and remove ones that are not
for file_name in classA_file_names:
    image_path = os.path.join(classA_train_dir, file_name)
    image = cv2.imread(image_path)
    img_type = imghdr.what(image_path)
    if img_type != "jpeg":
        os.remove(image_path)

for file_name in classB_file_names:
    image_path = os.path.join(classB_train_dir, file_name)
    image = cv2.imread(image_path)
    img_type = imghdr.what(image_path)
    if img_type != "jpeg":
        os.remove(image_path)


# In[35]:


# Create directory for validation data sets and move files into them
import shutil
import os
import os.path

try:
    # Create 'parent' validation directory
    os.mkdir(VALIDATION_DIR)
    # Create target directory to move cat files into (validaton set)
    os.mkdir(classA_validation_dir)
    # Create target directory to move dog files into (validaton set)
    os.mkdir(classB_validation_dir)
except OSError:
    pass

# Create array of file names with the remaining files (i.e. removed ones no longer exist)
classA_file_names = os.listdir(classA_train_dir)
classB_file_names = os.listdir(classB_train_dir)

# Define length of validation set to be 15% of training set
classA_valid_len = int(0.15 * len(classA_file_names))
classB_valid_len = int(0.15 * len(classB_file_names))

# Move files from training set (source) to validation set (target)
for file_name in classA_file_names[:classA_valid_len]:
    shutil.move(os.path.join(classA_train_dir, file_name), classA_validation_dir)

for file_name in classB_file_names[:classB_valid_len]:
    shutil.move(os.path.join(classB_train_dir, file_name), classB_validation_dir)


# In[26]:


# Create CNN model with 2DConv and maxpooling
import tensorflow as tf
import numpy as py
from tensorflow.keras import regularizers

model = tf.keras.models.Sequential([
    # First convolution layer
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)), #Picture is RGB therefore 3bytes per pixel Red, Blue, Green,
    tf.keras.layers.MaxPooling2D(2,2),
    # Second convolution layer                       
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Third convolution layer
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Flatten shape to input into DNN
    tf.keras.layers.Flatten(),
    # Hidden layer with 512 neurons
    tf.keras.layers.Dense(512, kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-3), activation='relu'),
    # Output later with 1 neuron (binary i.e. either cat or dog)
    tf.keras.layers.Dense(1, activation='sigmoid')
                           
])

# Observe layers summary of model
model.summary()


# In[27]:


from tensorflow.keras.optimizers import RMSprop

# Compile the CNN model
model.compile(
    loss='binary_crossentropy',
    optimizer = RMSprop(lr=0.0015),
    metrics=['accuracy'] )


# In[28]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Conduct pre-processing on the data to read and feed the images from the directories into the CNN

# Re-scale data as pixels have value of 0-255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Feed training dataset images in via batches of 425
train_generator = train_datagen.flow_from_directory (
    TRAIN_DIR, # Directory with training set images
    target_size=(300, 300), # Re-size target images
    batch_size = 145, #mini-batch of 250 to make CNN more efficient
    class_mode = 'binary'
)

# Feed validation dataset images in via batches of 75
valid_generator = validation_datagen.flow_from_directory (
     VALIDATION_DIR, #Directory with validation set images
     target_size=(300, 300), 
     batch_size = 75,
     class_mode = 'binary'
)


# In[29]:


callbacks = myCallback()

policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
tf.keras.mixed_precision.experimental.set_policy(policy)

# Train the dataset
history = model.fit(
    train_generator,
    steps_per_epoch=145,
    epochs=40,
    verbose = 1,
    validation_data=valid_generator,
    validation_steps = 25,
    callbacks=[callbacks]
)


# In[30]:


# PLOT LOSS AND ACCURACY
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")


plt.title('Training and validation loss')

# Desired output. Charts with training and validation metrics. No crash :)


# In[ ]:




