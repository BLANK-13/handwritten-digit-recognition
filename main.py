import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# mnist = tf.keras.datasets.mnist

# # x = images, y = labels
# (x_train, y_train), (x_test, y_test) = mnist.load_data() 

# #normalize the data to 0-1 for every pixel.
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)
# # done with pre processing.

# #neural network model
# model = tf.keras.models.Sequential() # basic sequential model
# model.add(tf.keras.layers.Flatten(input_shape= (28,28))) # input layer. what we basically do is flatten the 28x28 image into a 1x784 array no grid.
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # hidden layer. 128 neurons. activation function is relu.
# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # output layer. 10 neurons. activation function is softmax.

# #Compile the model and choose the optimizer and loss function.
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# #train the model
# model.fit(x_train, y_train, epochs=3)

# model.save('handwritten_digit_recognition.model')



# loading the model instead of training it everytime, because we saved it before.

model = tf.keras.models.load_model('handwritten_digit_recognition.model')

# loss, acc = model.evaluate(x_test, y_test)

# print("Loss: ", loss)
# print("Accuracy: ", acc)


sample_number = 1
while os.path.isfile(f'my_test_data/sample{sample_number}.png'):
    try:
        img = cv2.imread(f'my_test_data/sample{sample_number}.png', cv2.IMREAD_GRAYSCALE)
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"Prediction: {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print('Something went wrong !')
    finally:
        sample_number += 1