from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# LOAD THE DATA
images = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = images.load_data()


# EXPLORE THE DATA
plt.figure()
plt.imshow(train_images[1])
plt.show()

# SCALE THE DATA
train_images = train_images/255
test_images = test_images/255

# PLOT AGAIN
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
plt.show()

# BUILDING THE MODEL
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, batch_size=64, epochs=10)

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test Set Accuracy:', test_acc)

predictions = model.predict(test_images)
print('Prediction: ', np.argmax(predictions[0]))
print('Actual: ', test_labels[0])
