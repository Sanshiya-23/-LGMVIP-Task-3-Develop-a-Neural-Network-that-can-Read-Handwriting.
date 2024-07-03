#!/usr/bin/env python
# coding: utf-8

# # LGMVIP-Task 3-Develop a Neural Network that can Read Handwriting.

# # Step 1: Load and Preprocess the Data
# 
# First, we'll correct the file paths and load the MNIST data. Next, we will normalize the image data and convert the labels to categorical format.

# In[12]:


import numpy as np

#Load File paths
train_images_path = r'C:\Users\SANSHIYA\Downloads\archive (14)\train-images.idx3-ubyte'
train_labels_path = r'C:\Users\SANSHIYA\Downloads\archive (14)\train-labels.idx1-ubyte'
test_images_path = r'C:\Users\SANSHIYA\Downloads\archive (14)\t10k-images.idx3-ubyte'
test_labels_path = r'C:\Users\SANSHIYA\Downloads\archive (14)\t10k-labels.idx1-ubyte'

train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)
test_images = load_mnist_images(test_images_path)
test_labels = load_mnist_labels(test_labels_path)

print(f'Training data shape: {train_images.shape}')
print(f'Test data shape: {test_images.shape}')


# In[5]:


from tensorflow.keras.utils import to_categorical

#Normalize the images
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

#Convert labels to categorical format
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

print(f'Training labels shape: {train_labels.shape}')
print(f'Test labels shape: {test_labels.shape}')


# # Step 2: Build the CNN Model
# 
# Let's define and compile a CNN model using TensorFlow.

# In[6]:


import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# # Step 3: Train the Model
# 
# Let's train the CNN model on the training data.

# In[7]:


history = model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))


# # Step 4: Evaluate the Model
# 
# Finally, Let's evaluate the trained model on the test data.

# In[8]:


test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')


# # Step 5: Visualize Training History
# 
# Now, can plot the training and validation accuracy and loss over the epochs.

# In[9]:


import matplotlib.pyplot as plt

#training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

#training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

