import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from tensorflow import keras
import tensorflow as tf
import pandas as pd
from sklearn.metrics import accuracy_score
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# train data preprocessing
f = open('dataKaggle/train.txt')
train_labels = []
train_images = []
cnt = 0
for i in f:
    inp = i.split(',')
    path = 'dataKaggle/train/' + inp[0]
    train_images.append(mpimg.imread(path))
    train_labels.append(int(inp[1]))
f.close()

# validation data preprocessing
f = open('dataKaggle/validation.txt')
validation_labels = []
validation_images = []
cnt = 0
for i in f:
    inp = i.split(',')
    path = 'dataKaggle/validation/' + inp[0]
    validation_images.append(mpimg.imread(path))
    validation_labels.append(int(inp[1]))
f.close()
validation_labels_exp = validation_labels

# test image preprocessing
f = open('dataKaggle/test.txt')
test_labels = []
test_images = []
paths = []
cnt = 0
for i in f:
    inp = i.split()
    paths.append(inp[0])
    test_images.append(mpimg.imread('dataKaggle/test/' + inp[0]))
f.close()

# processing
train_images = np.asarray(train_images)
validation_images = np.asarray(validation_images)
test_images = np.asarray(test_images)

# rehsaping the images to accomodate a monochrome channel and fit the CNN model
train_images = np.reshape(train_images, (train_images.shape[0], 32, 32, 1))
validation_images = np.reshape(
    validation_images, (validation_images.shape[0], 32, 32, 1))
test_images = np.reshape(test_images, (test_images.shape[0], 32, 32, 1))


train_labels = keras.utils.to_categorical(train_labels)
validation_labels = keras.utils.to_categorical(validation_labels)

# training the CNN model
model = keras.models.Sequential()
model.add(keras.layers.Input((32, 32, 1)))

# setting the convolution layers
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(150, 5, activation='relu'))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(150, 5, activation='relu'))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Flatten())
# 93% accuracy without double relu and sigmoid layers and 0.3 Dropout
# 94,22% accuracy with current configuration
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(700, 'relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(700, 'sigmoid'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(700, 'relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(700, 'sigmoid'))
# output layer
model.add(keras.layers.Dense(9, 'softmax'))

#compiling the model with adamax optimizer
model.compile(keras.optimizers.Adamax(),
              loss='categorical_crossentropy', metrics=['acc'])

# model.fit(train_images, train_labels, epochs=100, verbose=2, validation_data=(
#      validation_images, validation_labels), use_multiprocessing=True, batch_size=128)

#generating the model weights by taking them from the epoch with the best accuracy
checkpoint_filepath = './luana2/kaggle'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

# model.fit(train_images, train_labels, epochs=100, verbose=2, validation_data=(
#      validation_images, validation_labels), use_multiprocessing=True, batch_size=128, callbacks=[model_checkpoint_callback])

model.load_weights(checkpoint_filepath)


# 89-90% accuracy with relu + sigmoid layers and Dropout 0.15
# model = keras.models.Sequential()
# model.add(keras.layers.Input((32, 32, 1)))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.Conv2D(150, 3, activation='relu'))
# model.add(keras.layers.MaxPooling2D())
# # model.add(keras.layers.Dropout(0.15))
# model.add(keras.layers.Conv2D(150, 3, activation='relu'))
# model.add(keras.layers.MaxPooling2D())
# model.add(keras.layers.Flatten())
# # model.add(keras.layers.Dense(700, 'tanh'))
# # model.add(keras.layers.Dropout(0.15))
# # tanh relu sigmoid - 88% accuracy
# model.add(keras.layers.Dense(700, 'relu'))
# model.add(keras.layers.Dense(700, 'sigmoid'))
# model.add(keras.layers.Dense(9, 'softmax'))
# model.compile(keras.optimizers.Adamax(),
#               loss='categorical_crossentropy', metrics=['acc'])


#generating the CSVs
test_labels_pred = model.predict_classes(test_images)
data = []
for i in range(len(paths)):
    data.append([paths[i], test_labels_pred[i]])

tabel = pd.DataFrame(data, columns=['id', 'label'])
dataFrame = tabel.set_index(['id'])
dataFrame.to_csv("kaggle3.csv")


# confusion matrix
def confusion_matrix(y_true, y_pred):
    conf_matrix = np.zeros((9, 9))
    for i in range(len(y_true)):
        conf_matrix[y_true[i], y_pred[i]] += 1
    print(conf_matrix)


# accuracy
validation_labels_pred = model.predict_classes(validation_images)
print(accuracy_score(validation_labels_exp, validation_labels_pred))

confusion_matrix(validation_labels_exp, validation_labels_pred)
