# Algorithm:
# STEP 1: Importing all the essential libraries.
# STEP 2: Loading the tokenizer and model.
# STEP 3: Uploading the wav file.
# STEP 4: Loading the path location of the wav file.
# STEP 5: Adjusting sample rate and output.
# STEP 6: Training the model.
# STEP 7: Converting the wav file to text format.

import speech_recognition as sr

recognizer = sr.recognizer()
path = ""

try:
	with sr.AudioFile(path) as source:
		audio = recognizer.record(source)
	text = recognizer.recognize_google(audio)
	print(text)
except:
	print("Error")