import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np


# Data
(_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
test_images = test_images / 255.0
test_labels = to_categorical(test_labels)


# Create model
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
img = "/path/to/image"
try:
	img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
	img = cv2.resize(img, (28, 28))
	img = img / 255.0
	img = img.reshape(1, 28, 28)
except Exception as e:
print(f"Error loading or preprocessing the image: {str(e)}")
exit()

# Predict
predictions = model.predict(user_image)
predicted_digit = np.argmax(predictions)
print(f"Predicted Digit: {predicted_digit}")handwriting.py