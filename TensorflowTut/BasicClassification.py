# TensorFlow and Keras import:
import tensorflow as tf
from tensorflow import keras

# Helper libraries:
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# Load Fashion MNIST data:
fashion_mnist = keras.datasets.fashion_mnist
(train_img, train_lab), (test_img, test_lab) = fashion_mnist.load_data()

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Data Preprocessing:
train_img = train_img/255.0
test_img = test_img/255.0

# Building the neural network:
# TODO: Try different settings to see effects
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer
    keras.layers.Dense(units=128, activation=tf.nn.relu),  # deep layer
    keras.layers.Dense(units=10, activation=tf.nn.softmax)  # output layer
])

# Setting up the model and training it:
# TODO: Try different settings to see effects
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_img, train_lab, epochs=5, verbose=2)

# Evaluate performance on test set:
test_loss, test_acc = model.evaluate(test_img, test_lab)
print('Test accuracy: {}'.format(test_acc))

# Predicted labels and classes on test set:
output_array = model.predict(test_img)                          # probabilities
predicted_lab = np.argmax(output_array, 1)                      # class numbers
predicted_classes = [class_names[lab] for lab in predicted_lab]  # class names

# To make a prediction about a single image, first turn it into a batch:
# img = (np.expand_dims(img,0)
# A batch is an ndarray with 3 dimensions (3rd is 0 in this case)

# Functions for plotting:
def plot_img(i,output_array, true_lab, img):
    """ Plot one image with annotations"""
    output_array, true_lab, img = output_array[i], true_lab[i], img[i]
    plt.grid(0)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_lab = np.argmax(output_array)
    if predicted_lab == true_lab:
        color = 'green'
    else:
        color = 'red'

    # :2.0f means at least 2 characters of a float, with 0 decimals.
    # link: https://pyformat.info/
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_lab],
                                         100*np.max(output_array),
                                         class_names[true_lab]),
               color=color
               )


def plot_output_array(i, output_array, true_lab):
    """ Plot the probabilities of each class"""
    output_array, true_lab = output_array[i], true_lab[i]
    plt.grid(0)
    plt.xticks([])
    plt.yticks([])

    thisplot = plt.bar(range(10), output_array, color="#777777")
    plt.ylim([0,1])
    predicted_lab = np.argmax(output_array)

    thisplot[predicted_lab].set_color('red')
    thisplot[true_lab].set_color('green')


# Plot some images:
num_rows = 5
num_cols = 3
num_imgs = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_imgs):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_img(i, output_array, test_lab, test_img)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_output_array(i,output_array,test_lab)
plt.show()
