# Algorithm:
# 1. Initialize RNN weights and biases randomly.
# 2. Preprocess text data: tokenize, create fixed-length sequences, convert to numerical indices.
# 3. Training loop:
# 	a. For each epoch, start with zero hidden state.
# 	b. For each sequence, calculate loss and update weights using backpropagation.
# 	c. Keep track of total loss for the epoch.
# 4. Word prediction:
# 	a. Begin with a seed sequence.
# 	b. Set hidden state to zeros.
# 	c. Predict next word's probability using RNN.
# 	d. Choose predicted word based on probability distribution.
# 5. Evaluate trained RNN's performance using metrics like perplexity.
# 6. Fine-tune hyperparameters (learning rate, architecture) if needed.

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample text data
text = "This is a sample text used for word prediction using RNN. Given a sequence of words, the RNN predicts the next word."

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Create input sequences and labels
sequences = tokenizer.texts_to_sequences([text])[0]
input_sequences = []
for i in range(1, len(sequences)):
	n_gram_sequence = sequences[:i+1]
	print(n_gram_sequence)
	input_sequences.append(n_gram_sequence)


# Pad sequences to have the same length
max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences,
maxlen=max_sequence_length, padding='pre')

# Split data into input and output
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# Convert y to one-hot encoding
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Create and compile the RNN model
model = tf.keras.Sequential([
tf.keras.layers.Embedding(total_words, 100,
input_length=max_sequence_length-1),
tf.keras.layers.SimpleRNN(100),
tf.keras.layers.Dense(total_words, activation='softmax')])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100, verbose=1)

# Function to predict the next word
def predict_next_word(seed_text, num_words_to_generate):
	for _ in range(num_words_to_generate):
		token_list = tokenizer.texts_to_sequences([seed_text])[0]
		token_list = pad_sequences([token_list],
		maxlen=max_sequence_length-1, padding='pre')
		predicted = model.predict(token_list, verbose=0)
		predicted_word_index = np.argmax(predicted)
		predicted_word = tokenizer.index_word[predicted_word_index]
		seed_text += " " + predicted_word
	return seed_text

# Generate predictions
seed_text = "This is"
predicted_text = predict_next_word(seed_text, num_words_to_generate=5)

# Print the predicted text
print(predicted_text)