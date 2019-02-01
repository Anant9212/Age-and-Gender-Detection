import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from random import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import keras
import glob
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from random import shuffle
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16


# initial parameters
epochs = 100
lr = 1e-3
batch_size = 64
img_dims = (96,96,3)

data = []
labels = []


train_man = r'C:\Users\kanan\Desktop\gender\man'
train_woman = r'C:\Users\kanan\Desktop\gender\woman'
train_m = [os.path.join(train_man,i) for i in os.listdir(train_man)]
train_w = [os.path.join(train_woman,i) for i in os.listdir(train_woman)]
train_images = train_m + train_w
shuffle(train_images)

for img in train_images:

    image = cv2.imread(img)
    image = image/255
    image = cv2.resize(image, (img_dims[0],img_dims[1]), interpolation=cv2.INTER_CUBIC)
    data.append(image)

    label = img.split('\\')[-2]
    if label == "woman":
        label = 1
    else:
        label = 0

    labels.append([label])
print(image.shape)

data = np.array(data, dtype="float")
labels = np.array(labels)

# split dataset for training and validation
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2,random_state=42)

# augmenting datset
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
horizontal_flip=False, fill_mode="nearest")

trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

vgg_16_model = VGG16(include_top=False, weights='imagenet', input_shape=(img_dims[0], img_dims[1], img_dims[2]))
vgg_16_model.summary()

print(type(vgg_16_model))

model = Sequential()
for layer in vgg_16_model.layers:
    model.add(layer)

model.summary()

for i in model.layers:
    i.trainable = False
#add top layer
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(2, activation='softmax'))

# compile the model
opt = Adam(lr=lr)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the model
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
                        validation_data=(testX,testY), steps_per_epoch=30,epochs=10)
