# Using LSTM

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Data
texts = ["This movie is great!", "I didn't like this book.", "Awesome experience!"]
labels = [1, 0, 1]
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
sequences = pad_sequences(sequences, maxlen=20, padding='post', truncating='post')

# Create model
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=16, input_length=20))
model.add(LSTM(16))
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(sequences, labels, epochs=10)

# Predict
test_texts = ["This is a fantastic product!", "Waste of money."]
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_sequences = pad_sequences(test_sequences, maxlen=20, padding='post', truncating='post')
predictions = model.predict(test_sequences)

print(predictions)

# (or) Using textblob

from textblob import TextBlob
import numpy as np

user_input = "Text"

blob = TextBlob(user_input)
sentiment = blob.sentiment
polarity = sentiment.polarity
subjectivity = sentiment.subjectivity


def get_sentiment_label(polarity):
	if polarity > 0:
		return "Positive"
	elif polarity < 0:
		return "Negative"
	else:
		return "Neutral"

sentiment_label = get_sentiment_label(polarity)
print("Sentiment Analysis:")
print(f"Text: {user_input}")
print(f"Polarity: {polarity}")
print(f"Subjectivity: {subjectivity}")
print(f"Sentiment: {sentiment_label}")