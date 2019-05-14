from __future__ import absolute_import, division, print_function

# Import Tensorflow and helper libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load the data-set from the keras library
images = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = images.load_data()

# Check the Training/Test set size
print('Training set size:', train_labels.shape[0])
print('Test set size:', test_labels.shape[0])

# Plot a random image from the training data-set, also view the image before scaling/normalization.
plt.figure()
plt.imshow(train_images[np.random.randint(1, 59999)])
plt.show()

# Scale the images of the entire data-set to normalize inputs between [0-1]
train_images = train_images/255
test_images = test_images/255

# Plot random images along with respective labels from the data-set
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    j = np.random.randint(1, 59999)
    plt.imshow(train_images[j], cmap=plt.cm.binary)
    plt.xlabel(train_labels[j])
plt.show()

# Build the NN model:
# Model consists of input layer with 28*28 neurons (784 inputs), then another layer of 128 neurons with ReLU activation,
# and at last an output layer with softmax units for detecting probabilities of the input from 0-9
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model, and use ADAM optimization algorithm, and evaluation metric set to accuracy
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Summary of our sequential model
model.summary()

# Train the model, using batch size of 64 through 10 epochs
model.fit(train_images, train_labels, batch_size=64, epochs=10)

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test Set Accuracy:', test_acc)

predictions = model.predict(test_images)
print('Prediction: ', np.argmax(predictions[10]))
print('Actual: ', test_labels[10])
