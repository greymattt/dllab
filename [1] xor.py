import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# create model
model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=1000, verbose=0)

# Evaluate
loss, accuracy = model.evaluate(X, y)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

# Predict
predictions = model.predict(X)
print('Predictions:')
for i in range(4):
  print(f'Input: {X[i]}, Predicted Output:{round(predictions[i][0])}')