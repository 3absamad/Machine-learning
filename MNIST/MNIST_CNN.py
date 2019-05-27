from __future__ import absolute_import, division, print_function

# TensorFlow and keras
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np

# LOAD THE DATA
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
print('Training set size:', train_labels.shape[0])
print('Image Size', train_images.shape)
print('Test set size:', test_labels.shape[0])

# SCALE THE DATA
train_images = train_images/255
test_images = test_images/255

# BUILDING THE MODEL
BATCH_SIZE = 64
EPOCHS = 5
INPUT_SHAPE = (28, 28, 1)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test Set Accuracy:', test_acc)

predictions = model.predict(test_images)
print('Prediction: ', np.argmax(predictions[9]))
print('Actual: ', test_labels[9])

# Save the model
model.save('MNIST_CNN.h5')

