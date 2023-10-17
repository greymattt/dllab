# Algorithm:
# STEP 1: Importing all the essential libraries.
# STEP 2: Loading the tokenizer and model.
# STEP 3: Uploading the wav file.
# STEP 4: Loading the path location of the wav file.
# STEP 5: Adjusting sample rate and output.
# STEP 6: Training the model.
# STEP 7: Converting the wav file to text format.

import torch
import librosa
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from IPython.display import Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer


tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
file_name = '/content/my-audio.wav'

Audio(file_name)
data = wavfile.read(file_name)
framerate = data[0]
sounddata = data[1]
time = np.arange(0,len(sounddata))/framerate
input_audio, _ = librosa.load(file_name, sr=16000)
input_values = tokenizer(input_audio, return_tensors="pt").input_values
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)[0]
print(transcription)